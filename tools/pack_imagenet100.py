#!/usr/bin/env python3
"""
pack_imagenet100.py -- turn 129,395 loose JPEGs into one packed uint8 memmap,
with frozen splits and a fingerprint that every downstream run is bound to.

Why pack
--------
Decoding is not the bottleneck on a 24-core machine. Three other things are:

  1. `dataload_frac` is a recorded telemetry column, and one of the five the
     playbook calls out as unrecoverable after the fact. Against a memmap it
     measures the model. Against 129k JPEG opens it measures whatever else the
     machine happened to be doing, and every timing column inherits that noise.
  2. NTFS small-file I/O over 129k files per epoch is slow and, worse, variable.
  3. The same artifact mounts on Kaggle as a Dataset, so the notebooks run
     unchanged in both places.

What it writes
--------------
    images_256.u8    (N, 256, 256, 3) uint8   ~23.7 GiB
    labels.npy       (N,) int16
    manifest.json    pack version, class list, per-index source path, counts
    splits.json      val / train / holdout index sets, and the policy that made them
    fingerprint.txt  sha256 binding all of the above

`fingerprint` participates in `config_hash`, so a repacked dataset invalidates
every run trained against the old one -- loudly, at gate time, rather than
silently at analysis time when two per-sample tables turn out to be indexed
against different images. Presence of a table is not evidence it is comparable
to another one (rule 5, applied to the dataset itself).

Design notes that are not obvious
---------------------------------
* **Every image is packed, including any that fail to decode.** N must not
  depend on decode success, or one corrupt file shifts every index after it and
  silently re-labels the rest of the dataset. Failures are written as zeros,
  recorded in `manifest["failures"]`, and *excluded from every split*, so they
  can never enter a measurement.

* **Workers write into the memmap directly** rather than returning arrays.
  Returning them would push ~24 GiB through pickle for no reason.

* **Resumable at chunk granularity.** The job takes 20-40 minutes; being unable
  to resume it would be its own small version of D-19.

* **Dry run first.** 32 images are packed to a scratch file, read back, and
  checked before the real job starts (rule 1). A shape or mode bug found here
  costs a second; found at image 120,000 it costs the whole pass.

Usage
-----
    python tools/pack_imagenet100.py --src "C:/Users/.../New folder" --out D:/msc_data/in100
    python tools/pack_imagenet100.py --src ... --out ... --verify     # re-check an existing pack
    python tools/pack_imagenet100.py --src ... --out ... --workers 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# A handful of ImageNet JPEGs are truncated. Pillow raises on them by default;
# every published ImageNet pipeline sets this. We still record which ones needed
# it, because "we quietly repaired 40 images" is a fact about the dataset.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Frozen parameters. Changing any of these changes the fingerprint, which is
# the point -- it must be impossible to change how the data is built without
# invalidating the runs built on it.
# ---------------------------------------------------------------------------
PACK_VERSION = "in100-pack-1"
STORED_RES = 256           # shorter side resized to this, then centre-cropped square
VAL_PER_CLASS = 100        # -> 10,000 val, matching CIFAR-100's test-set size so
                           #    rho_seed is estimated at a comparable n
HOLDOUT_N = 15_000         # slice OF train, augmentation off; carries EL2N and
                           #    forgetting events, so Q4 runs on it (DC-8).
                           #    15k also clears the Learn-then-Test requirement of
                           #    ~14,979 samples at eps=0.01, delta=0.05 -- CIFAR
                           #    could not and had to settle for eps=0.03.
SPLIT_SEED_VAL = 12345
SPLIT_SEED_HOLDOUT = 67890
CHUNK = 512
RESAMPLE = Image.BICUBIC


# ---------------------------------------------------------------------------
def build_index(src: Path):
    """Deterministic (path, label) list. Classes sorted by WNID, files sorted
    within a class. Sorted twice on purpose: the order IS the index space, and
    an index space that depends on directory-listing order is not reproducible
    across machines or filesystems."""
    train = src / "train"
    if not train.is_dir():
        raise SystemExit(f"no train/ directory under {src}")
    classes = sorted(d.name for d in train.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class folders under {train}")

    names = {}
    cm = src / "class_mappings.json"
    if cm.exists():
        names = json.loads(cm.read_text(encoding="utf-8"))

    paths, labels = [], []
    per_class = {}
    for ci, wnid in enumerate(classes):
        fs = sorted(p.name for p in (train / wnid).iterdir() if p.is_file())
        per_class[wnid] = len(fs)
        for f in fs:
            paths.append(f"{wnid}/{f}")
            labels.append(ci)
    return classes, names, paths, np.asarray(labels, dtype=np.int16), per_class


def _load_one(src_train: Path, rel: str):
    """shorter-side -> STORED_RES, centre-crop square, RGB uint8 HWC.

    Squaring here rather than at train time is what makes the memmap a fixed
    stride. The cost is that RandomResizedCrop later samples inside the central
    square rather than the full frame -- a mild reduction in augmentation
    diversity, applied identically to every architecture, and stated in the data
    card. It cannot manufacture a family effect because it is a constant.
    """
    with Image.open(src_train / rel) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w < h:
            nw, nh = STORED_RES, max(STORED_RES, round(h * STORED_RES / w))
        else:
            nh, nw = STORED_RES, max(STORED_RES, round(w * STORED_RES / h))
        im = im.resize((nw, nh), RESAMPLE)
        left, top = (nw - STORED_RES) // 2, (nh - STORED_RES) // 2
        im = im.crop((left, top, left + STORED_RES, top + STORED_RES))
        a = np.asarray(im, dtype=np.uint8)
    if a.shape != (STORED_RES, STORED_RES, 3):
        raise ValueError(f"{rel}: got {a.shape}")
    return a


def _pack_chunk(args):
    """Runs in a worker. Opens its own memmap handle -- on Windows the pool
    spawns rather than forks, so a handle opened in the parent is not inherited
    and would either crash or, far worse, silently serve zeros."""
    mm_path, n_total, src_train, start, rels = args
    mm = np.memmap(mm_path, dtype=np.uint8, mode="r+",
                   shape=(n_total, STORED_RES, STORED_RES, 3))
    failures = []
    for j, rel in enumerate(rels):
        try:
            mm[start + j] = _load_one(Path(src_train), rel)
        except Exception as e:                       # noqa: BLE001
            mm[start + j] = 0
            failures.append({"index": start + j, "path": rel,
                             "error": f"{type(e).__name__}: {str(e)[:160]}"})
    mm.flush()
    del mm
    return start, len(rels), failures


# ---------------------------------------------------------------------------
def dry_run(src_train: Path, rels, out: Path):
    """Rule 1: one synthetic-scale batch through the ENTIRE path -- decode,
    resize, crop, memmap write, memmap read-back, dtype and range check --
    before committing to 24 GiB of work. A partial dry run just moves where the
    bug hides, so this uses the same functions the real pass does."""
    t0 = time.time()
    probe = out / "_dryrun.u8"
    n = min(32, len(rels))
    mm = np.memmap(probe, dtype=np.uint8, mode="w+",
                   shape=(n, STORED_RES, STORED_RES, 3))
    del mm
    _, _, fails = _pack_chunk((str(probe), n, str(src_train), 0, rels[:n]))
    back = np.memmap(probe, dtype=np.uint8, mode="r",
                     shape=(n, STORED_RES, STORED_RES, 3))
    assert back.shape == (n, STORED_RES, STORED_RES, 3), back.shape
    assert back.dtype == np.uint8, back.dtype
    nonzero = int((back.reshape(n, -1).max(axis=1) > 0).sum())
    mean = float(back[:4].mean())
    del back
    probe.unlink(missing_ok=True)
    if nonzero < n - len(fails):
        raise SystemExit(f"dry run: {n - nonzero} of {n} images decoded to all zeros")
    if not (5.0 < mean < 250.0):
        raise SystemExit(f"dry run: mean pixel {mean:.1f} is implausible")
    print(f"[DRY]  {n} images -> memmap -> read back OK "
          f"({nonzero} non-zero, mean {mean:.1f}, {time.time()-t0:.1f}s, "
          f"{len(fails)} decode failures)")
    return True


def make_splits(labels: np.ndarray, n_classes: int, exclude: set):
    """val: VAL_PER_CLASS per class, stratified. train: the rest.
    holdout: HOLDOUT_N sampled FROM train -- it is not held out of training, it
    is training data evaluated with augmentation off, which is the only way
    EL2N and forgetting events are defined on it (see DC-8 and D-11)."""
    rng = np.random.default_rng(SPLIT_SEED_VAL)
    val = []
    for c in range(n_classes):
        idx = np.flatnonzero(labels == c)
        idx = np.array([i for i in idx if i not in exclude], dtype=np.int64)
        if len(idx) < VAL_PER_CLASS:
            raise SystemExit(f"class {c} has only {len(idx)} usable images, "
                             f"need {VAL_PER_CLASS} for val")
        val.append(rng.choice(idx, size=VAL_PER_CLASS, replace=False))
    val = np.sort(np.concatenate(val))

    all_idx = np.arange(len(labels), dtype=np.int64)
    train = np.setdiff1d(all_idx, np.union1d(val, np.fromiter(exclude, np.int64,
                                                              len(exclude))))
    rng2 = np.random.default_rng(SPLIT_SEED_HOLDOUT)
    holdout = np.sort(rng2.choice(train, size=min(HOLDOUT_N, len(train)),
                                  replace=False))
    return val, train, holdout


def fingerprint(classes, val, holdout) -> str:
    """Binds pack version, stored resolution, class identity and both split
    index sets. train is the complement, so it is implied.

    This goes into config_hash. Two runs that disagree about which 10,000 images
    are val produce per-sample tables that are aligned by index and meaningless
    to correlate -- the failure would look exactly like a real result."""
    h = hashlib.sha256()
    h.update(PACK_VERSION.encode())
    h.update(str(STORED_RES).encode())
    h.update("|".join(classes).encode())
    h.update(np.asarray(val, dtype=np.int64).tobytes())
    h.update(np.asarray(holdout, dtype=np.int64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
def verify(out: Path, n_probe: int = 512) -> int:
    """Re-read the pack and check it. Called after packing and available
    standalone -- because 'the job finished' and 'the data is right' are
    different claims, and only the second one matters."""
    man = json.loads((out / "manifest.json").read_text())
    n = man["count"]
    mm = np.memmap(out / "images_256.u8", dtype=np.uint8, mode="r",
                   shape=(n, STORED_RES, STORED_RES, 3))
    labels = np.load(out / "labels.npy")
    bad_idx = {f["index"] for f in man.get("failures", [])}

    rng = np.random.default_rng(7)
    probe = rng.choice(n, size=min(n_probe, n), replace=False)
    zeros = [int(i) for i in probe
             if int(mm[i].max()) == 0 and int(i) not in bad_idx]
    means = np.array([mm[i].mean() for i in probe])

    splits = json.loads((out / "splits.json").read_text())
    val = np.asarray(splits["val"], dtype=np.int64)
    hold = np.asarray(splits["holdout"], dtype=np.int64)
    train = np.asarray(splits["train"], dtype=np.int64)

    problems = []
    if zeros:
        problems.append(f"{len(zeros)} sampled images are all-zero but not "
                        f"recorded as decode failures: {zeros[:5]}")
    if len(np.intersect1d(val, train)):
        problems.append("val and train overlap")
    if not np.isin(hold, train).all():
        problems.append("holdout is not a subset of train")
    if len(labels) != n:
        problems.append(f"labels has {len(labels)} rows, pack has {n}")
    vc = np.bincount(labels[val], minlength=man["n_classes"])
    if vc.min() != VAL_PER_CLASS or vc.max() != VAL_PER_CLASS:
        problems.append(f"val is not stratified: per-class {vc.min()}..{vc.max()}")
    fp = fingerprint(man["classes"], val, hold)
    if fp != man["fingerprint"]:
        problems.append(f"fingerprint mismatch: recomputed {fp[:16]} vs "
                        f"stored {man['fingerprint'][:16]}")

    print(f"[VERIFY] n={n}  classes={man['n_classes']}  "
          f"val={len(val)} train={len(train)} holdout={len(hold)}")
    print(f"[VERIFY] sampled {len(probe)} images: mean pixel "
          f"{means.mean():.1f} (min {means.min():.1f}, max {means.max():.1f})")
    print(f"[VERIFY] decode failures recorded: {len(bad_idx)}")
    print(f"[VERIFY] fingerprint {man['fingerprint']}")
    for p in problems:
        print(f"[VERIFY] *** {p}")
    print("[VERIFY] " + ("OK" if not problems else f"{len(problems)} PROBLEM(S)"))
    return len(problems)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train/ and class_mappings.json")
    ap.add_argument("--out", required=True, help="destination for the pack (~24 GiB)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--verify", action="store_true", help="only re-verify an existing pack")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return 1 if verify(out) else 0

    classes, names, rels, labels, per_class = build_index(src)
    n = len(rels)
    nbytes = n * STORED_RES * STORED_RES * 3
    print(f"[PACK] {n} images, {len(classes)} classes, "
          f"{min(per_class.values())}-{max(per_class.values())} per class")
    print(f"[PACK] target {nbytes/2**30:.1f} GiB at {STORED_RES}px -> {out}")

    free_b = shutil.disk_usage(out).free
    if free_b < nbytes * 1.05:
        print(f"[PACK] not enough space: {free_b/2**30:.1f} GiB free, "
              f"need {nbytes*1.05/2**30:.1f} GiB")
        return 1

    if not dry_run(src / "train", rels, out):
        return 1

    mm_path = out / "images_256.u8"
    prog_path = out / "_progress.json"
    done = set()
    if mm_path.exists() and prog_path.exists():
        st = json.loads(prog_path.read_text())
        if st.get("pack_version") == PACK_VERSION and st.get("count") == n:
            done = set(st.get("done", []))
            print(f"[PACK] resuming: {len(done)} chunks already written")
    if not mm_path.exists():
        mm = np.memmap(mm_path, dtype=np.uint8, mode="w+",
                       shape=(n, STORED_RES, STORED_RES, 3))
        del mm

    chunks = [(str(mm_path), n, str(src / "train"), s, rels[s:s + CHUNK])
              for s in range(0, n, CHUNK) if s not in done]
    todo = sum(len(c[4]) for c in chunks)
    failures, t0, seen = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_pack_chunk, c) for c in chunks]
        for k, fut in enumerate(as_completed(futs), 1):
            start, cnt, fails = fut.result()
            done.add(start)
            failures += fails
            seen += cnt
            if k % 20 == 0 or k == len(futs):
                el = time.time() - t0
                rate = seen / max(el, 1e-9)
                print(f"[PACK] {seen}/{todo}  ({100*seen/max(todo,1):5.1f}%)  "
                      f"{rate:6.0f} img/s  "
                      f"eta {(todo - seen)/max(rate,1e-9)/60:5.1f} min", flush=True)
                prog_path.write_text(json.dumps(
                    {"pack_version": PACK_VERSION, "count": n,
                     "done": sorted(done)}))

    np.save(out / "labels.npy", labels)
    bad = {f["index"] for f in failures}
    val, train, holdout = make_splits(labels, len(classes), bad)
    fp = fingerprint(classes, val, holdout)

    (out / "manifest.json").write_text(json.dumps({
        "pack_version": PACK_VERSION,
        "stored_res": STORED_RES,
        "count": n,
        "n_classes": len(classes),
        "classes": classes,
        "class_names": {c: names.get(c, "") for c in classes},
        "per_class_counts": per_class,
        "source": str(src),
        "resize_policy": f"shorter side -> {STORED_RES} (bicubic), centre crop "
                         f"{STORED_RES}x{STORED_RES}, RGB",
        "paths": rels,
        "failures": failures,
        "fingerprint": fp,
        "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=1))

    (out / "splits.json").write_text(json.dumps({
        "policy": {
            "val": f"{VAL_PER_CLASS} per class, stratified, rng seed {SPLIT_SEED_VAL}",
            "train": "complement of val, minus decode failures",
            "holdout": f"{HOLDOUT_N} sampled FROM train, rng seed "
                       f"{SPLIT_SEED_HOLDOUT}; NOT excluded from training -- it is "
                       f"training data evaluated with augmentation off",
        },
        "val": val.tolist(), "train": train.tolist(), "holdout": holdout.tolist(),
    }))
    (out / "fingerprint.txt").write_text(fp + "\n")
    prog_path.unlink(missing_ok=True)

    print(f"[PACK] done in {(time.time()-t0)/60:.1f} min, "
          f"{len(failures)} decode failures")
    print(f"[PACK] fingerprint {fp}")
    return 1 if verify(out) else 0


if __name__ == "__main__":
    sys.exit(main())
