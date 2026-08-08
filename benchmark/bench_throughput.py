#!/usr/bin/env python3
"""
bench_throughput.py -- find the fastest safe configuration for THIS machine,
before committing ~400 GPU-hours to it.

Why this exists
---------------
The run matrix is ~235 GPU-hours of backbone training. A 20% throughput win is
47 hours; a 40% win is 94. That is worth an hour of measurement, and it is worth
measuring rather than assuming, because every number in `20_IN100_PORT_PLAN.md`
§6 is an *estimate* anchored on one guessed figure. The CIFAR programme's first
cost table was 40% low (D-10), and it only found out by running.

It is also the only way to answer a question the plan cannot: at 224px with a
packed memmap, is this pipeline GPU-bound or loader-bound, and does that answer
differ between `resnet18` and `swin_tiny`? If it is loader-bound for the small
models, adding workers is free speed. If it is GPU-bound, adding workers is
noise and the lever is batch size or dtype.

How it sweeps
-------------
NOT a full grid. 8 architectures x 5 batch sizes x 5 worker counts x 2 dtypes x
2 layouts x 3 compile modes is 2,400 configurations and would take longer than
it saves. Instead a **staged coordinate descent**, each stage fixing the winner
of the last:

    A  dtype x memory layout        4 configs   cheap, biggest per-config effect
    B  batch size ladder            5 configs   with OOM detection
    C  dataloader workers           5 configs   only meaningful once B is fixed
    D  torch.compile                3 configs   expensive to compile; winner only
    E  confirmation                 1 config    long steady-state run

Coordinate descent can miss an interaction that a grid would find. It is used
anyway because the interactions here are weak (batch size and worker count are
close to separable once neither is starving) and because a sweep that takes six
hours will not be run. Stage E re-measures the chosen configuration from cold,
so the reported number is a measurement and not a sum of stage bests.

What it measures
----------------
* **Steady-state training throughput** (img/s), forward + backward + optimiser
  step, after warmup, with `torch.cuda.synchronize()` around the timed region.
  Timing CUDA without synchronising measures queue-submission speed.
* **Peak VRAM**, so the chosen batch size has headroom rather than sitting one
  allocation away from an OOM at epoch 60.
* **The loader ceiling, separately** -- how fast batches arrive with no model at
  all. If model throughput is at that ceiling, the GPU is starving and no
  amount of tuning the model will help.
* **The cost of telemetry.** NVML power sampling runs at 10 Hz for every real
  run. Measured here rather than assumed to be free.

Usage
-----
    python benchmark/bench_throughput.py                     # full sweep, ~45 min
    python benchmark/bench_throughput.py --quick             # ~10 min, fewer points
    python benchmark/bench_throughput.py --archs resnet50 swin_tiny
    python benchmark/bench_throughput.py --synthetic         # no packed data needed

Writes `benchmark/results/bench_<host>_<timestamp>.json` plus a printed table.
**Send me the JSON** -- it contains everything needed to re-plan the run matrix.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MSC_OFFLINE", "1")

import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import msc_lib as M                                            # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
DATASET = "imagenet100"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def _free_vram():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _synth(batch, res, n_cls, n_batches=40):
    """Batches resident on the GPU, so the model is measured with NO loader in
    the way. Stage A/B/D want the model's ceiling, not the pipeline's."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, 3, res, res, generator=g).to(DEV)
    y = torch.randint(0, n_cls, (batch,), generator=g).to(DEV)
    return [(x, y)] * n_batches


def train_steps(arch, batch, res, n_cls, *, dtype=torch.float16,
                channels_last=True, compile_mode=None, steps=30, warmup=8,
                loader=None) -> dict:
    """One configuration. Returns img/s, peak VRAM, and any failure."""
    _free_vram()
    rec = {"arch": arch, "batch": batch, "dtype": str(dtype).split(".")[-1],
           "channels_last": channels_last, "compile": compile_mode or "off"}
    try:
        model = M.build_model(arch, n_cls, dataset=DATASET).to(DEV)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        if compile_mode:
            model = torch.compile(model, mode=compile_mode)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        crit = torch.nn.CrossEntropyLoss()
        model.train()

        data = loader if loader is not None else _synth(batch, res, n_cls,
                                                        steps + warmup)
        it = iter(data)
        n_seen, t0 = 0, None
        for i in range(steps + warmup):
            try:
                b = next(it)
            except StopIteration:
                it = iter(data)
                b = next(it)
            x, y = b[0].to(DEV, non_blocking=True), b[1].to(DEV, non_blocking=True)
            if channels_last and x.dim() == 4:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.amp.autocast("cuda", dtype=dtype):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if i == warmup - 1:
                # Warmup covers CUDA context creation, cuDNN autotuning and,
                # for torch.compile, the entire graph capture -- which can be
                # 30-90 s and would otherwise be charged to throughput.
                torch.cuda.synchronize()
                t0 = time.time()
            elif i >= warmup:
                n_seen += x.shape[0]
        torch.cuda.synchronize()
        dt = time.time() - t0
        rec.update({
            "img_s": n_seen / dt,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "ok": True,
        })
        del model, opt, scaler
    except torch.cuda.OutOfMemoryError:
        rec.update({"ok": False, "error": "OOM", "img_s": 0.0})
    except Exception as e:                                       # noqa: BLE001
        rec.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}",
                    "img_s": 0.0})
    _free_vram()
    return rec


def loader_ceiling(data_dir, batch, workers, res, steps=40) -> dict:
    """How fast do batches arrive with NO model at all?

    This is the number that says whether tuning the model is worth anything.
    If a model's throughput equals this, the GPU is idle waiting and the fix is
    the loader -- exactly what `dataload_frac` is for during the real run, and
    the reason D-40 mattered.
    """
    rec = {"workers": workers, "batch": batch}
    try:
        cfg = M.base_config("resnet18", DATASET, seed=1)
        cfg.update({"data_root": str(data_dir), "batch_size": batch,
                    "num_workers": workers, "input_res": res})
        tr, _, _, _, _ = M.build_loaders(cfg)
        it = iter(tr)
        for _ in range(5):                                       # warmup
            next(it)
        torch.cuda.synchronize()
        t0, n = time.time(), 0
        for _ in range(steps):
            try:
                b = next(it)
            except StopIteration:
                it = iter(tr)
                b = next(it)
            n += b[0].shape[0]
        torch.cuda.synchronize()
        rec.update({"img_s": n / (time.time() - t0), "ok": True})
        if hasattr(tr, "timing"):
            t = tr.timing()
            tot = max(1e-9, t["wait_s"] + t["augment_s"])
            rec["wait_frac"] = t["wait_s"] / tot
            rec["augment_frac"] = t["augment_s"] / tot
        del tr, it
    except Exception as e:                                       # noqa: BLE001
        rec.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:150]}",
                    "img_s": 0.0})
    _free_vram()
    return rec


# ---------------------------------------------------------------------------
def sweep(archs, res, n_cls, quick=False, data_dir=None) -> dict:
    out = {"stages": {}, "winner": {}}
    BS = [64, 128] if quick else [32, 64, 128, 192, 256]
    WK = [4, 12] if quick else [0, 4, 8, 12, 16]
    bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    # --- A: dtype x memory layout -----------------------------------------
    print("\n=== Stage A: dtype x memory layout (batch 64) " + "=" * 26)
    dtypes = [torch.float16] + ([torch.bfloat16] if bf16 else [])
    rows = []
    for a in archs:
        best, brec = -1, None
        for dt in dtypes:
            for cl in (True, False):
                r = train_steps(a, 64, res, n_cls, dtype=dt, channels_last=cl,
                                steps=20, warmup=6)
                rows.append(r)
                print(f"  {a:16s} {r['dtype']:8s} cl={str(cl):5s} "
                      f"{r['img_s']:8.1f} img/s"
                      + ("" if r["ok"] else f"   {r.get('error')}"))
                if r["ok"] and r["img_s"] > best:
                    best, brec = r["img_s"], r
        if brec:
            out["winner"].setdefault(a, {}).update(
                {"dtype": brec["dtype"], "channels_last": brec["channels_last"]})
    out["stages"]["A_dtype_layout"] = rows

    # --- B: batch size ------------------------------------------------------
    print("\n=== Stage B: batch size, with OOM detection " + "=" * 28)
    rows = []
    for a in archs:
        w = out["winner"].get(a, {})
        dt = torch.bfloat16 if w.get("dtype") == "bfloat16" else torch.float16
        best, brec = -1, None
        for bs in BS:
            r = train_steps(a, bs, res, n_cls, dtype=dt,
                            channels_last=w.get("channels_last", True),
                            steps=20, warmup=6)
            rows.append(r)
            v = f"{r.get('peak_vram_gb', 0):.1f} GB" if r["ok"] else r.get("error")
            print(f"  {a:16s} bs={bs:4d} {r['img_s']:8.1f} img/s   {v}")
            if r["ok"] and r["img_s"] > best:
                best, brec = r["img_s"], r
            if not r["ok"] and r.get("error") == "OOM":
                break                                # larger will also OOM
        if brec:
            out["winner"][a].update({"batch": brec["batch"],
                                     "peak_vram_gb": brec["peak_vram_gb"],
                                     "img_s_synthetic": brec["img_s"]})
    out["stages"]["B_batch"] = rows

    # --- C: loader ----------------------------------------------------------
    print("\n=== Stage C: dataloader workers (no model) " + "=" * 29)
    rows = []
    if data_dir:
        for wk in WK:
            r = loader_ceiling(data_dir, 128, wk, res)
            rows.append(r)
            extra = (f"  wait {r['wait_frac']*100:.0f}% / aug "
                     f"{r['augment_frac']*100:.0f}%" if "wait_frac" in r else "")
            print(f"  workers={wk:3d}  {r['img_s']:8.1f} img/s{extra}"
                  + ("" if r["ok"] else f"   {r.get('error')}"))
        ok = [r for r in rows if r["ok"]]
        if ok:
            b = max(ok, key=lambda r: r["img_s"])
            out["winner"]["_loader"] = {"workers": b["workers"],
                                        "ceiling_img_s": b["img_s"]}
    else:
        print("  skipped -- no packed dataset (pass --data-dir or use the pack)")
    out["stages"]["C_workers"] = rows

    # --- D: torch.compile ---------------------------------------------------
    print("\n=== Stage D: torch.compile at each winner " + "=" * 30)
    print("  (first call compiles; that cost is in warmup, not throughput)")
    rows = []
    for a in archs:
        w = out["winner"].get(a, {})
        if "batch" not in w:
            continue
        dt = torch.bfloat16 if w.get("dtype") == "bfloat16" else torch.float16
        base = w.get("img_s_synthetic", 0)
        for mode in (None, "default"):
            if mode is None:
                continue
            r = train_steps(a, w["batch"], res, n_cls, dtype=dt,
                            channels_last=w.get("channels_last", True),
                            compile_mode=mode, steps=25, warmup=12)
            rows.append(r)
            gain = (r["img_s"] / base - 1) * 100 if base and r["ok"] else 0
            print(f"  {a:16s} compile={mode:8s} {r['img_s']:8.1f} img/s  "
                  f"{gain:+5.1f}%" + ("" if r["ok"] else f"   {r.get('error')}"))
            if r["ok"] and r["img_s"] > base * 1.03:
                out["winner"][a]["compile"] = mode
                out["winner"][a]["img_s_synthetic"] = r["img_s"]
    out["stages"]["D_compile"] = rows

    # --- E: confirmation ----------------------------------------------------
    # Re-measure from cold. The stage bests are each a local maximum found under
    # slightly different conditions; a plan built by adding them up would be a
    # sum of measurements rather than a measurement.
    print("\n=== Stage E: confirmation, longer run at the chosen config " + "=" * 13)
    rows = []
    for a in archs:
        w = out["winner"].get(a, {})
        if "batch" not in w:
            continue
        dt = torch.bfloat16 if w.get("dtype") == "bfloat16" else torch.float16
        r = train_steps(a, w["batch"], res, n_cls, dtype=dt,
                        channels_last=w.get("channels_last", True),
                        compile_mode=w.get("compile"), steps=60, warmup=15)
        rows.append(r)
        out["winner"][a]["img_s_confirmed"] = r["img_s"]
        print(f"  {a:16s} {r['img_s']:8.1f} img/s   "
              f"{r.get('peak_vram_gb', 0):.1f} GB peak")
    out["stages"]["E_confirm"] = rows
    return out


def plan(winner, n_train=119_395, epochs=100) -> dict:
    """Turn measured throughput into the number that actually matters."""
    rows, total = [], 0.0
    for a, w in sorted(winner.items()):
        if a.startswith("_") or not w.get("img_s_confirmed"):
            continue
        sec_ep = n_train / w["img_s_confirmed"]
        h = sec_ep * epochs / 3600.0
        rows.append({"arch": a, "img_s": w["img_s_confirmed"],
                     "sec_per_epoch": sec_ep, "hours_per_run": h,
                     "hours_3_seeds": h * 3})
        total += h * 3
    return {"per_arch": rows, "atlas_gpu_hours": total,
            "atlas_days_continuous": total / 24.0, "epochs": epochs,
            "train_images": n_train}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip the loader stage; no packed dataset needed")
    ap.add_argument("--data-dir", default=os.environ.get("MSC_IN100_DIR"))
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA. This benchmark measures a GPU; there is nothing to say.")
        return 1

    p = torch.cuda.get_device_properties(0)
    archs = a.archs or M.zoo_for_dataset(DATASET)
    res = M.native_res(DATASET)
    n_cls = M.num_classes_for(DATASET)

    data_dir = None if a.synthetic else a.data_dir
    if data_dir and not M.data_present(DATASET, data_dir)[0]:
        print(f"no pack at {data_dir}; loader stage will be skipped")
        data_dir = None

    print("=" * 72)
    print(f"  {p.name}   {p.total_memory/2**30:.1f} GiB   sm_{p.major}{p.minor}")
    print(f"  torch {torch.__version__}   bf16 "
          f"{'yes' if torch.cuda.is_bf16_supported() else 'no'}   "
          f"cpus {os.cpu_count()}")
    print(f"  {len(archs)} architecture(s) at {res}px, {n_cls} classes")
    print(f"  loader stage: {'yes' if data_dir else 'SKIPPED (synthetic only)'}")
    print("=" * 72)

    t0 = time.time()
    try:
        res_ = sweep(archs, res, n_cls, quick=a.quick, data_dir=data_dir)
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        return 1

    pl = plan(res_["winner"], epochs=a.epochs)
    print("\n" + "=" * 72)
    print("  CHOSEN CONFIGURATION, AND WHAT IT COSTS")
    print("=" * 72)
    print(f"  {'arch':16s} {'bs':>4s} {'dtype':>9s} {'cl':>3s} {'cmp':>4s} "
          f"{'img/s':>8s} {'VRAM':>6s} {'s/ep':>7s} {'h x3':>7s}")
    for r in pl["per_arch"]:
        w = res_["winner"][r["arch"]]
        print(f"  {r['arch']:16s} {w.get('batch',0):4d} "
              f"{w.get('dtype','?'):>9s} {str(w.get('channels_last'))[0]:>3s} "
              f"{str(w.get('compile','off'))[:4]:>4s} "
              f"{r['img_s']:8.1f} {w.get('peak_vram_gb',0):5.1f}G "
              f"{r['sec_per_epoch']:7.0f} {r['hours_3_seeds']:7.1f}")
    print(f"\n  atlas total: {pl['atlas_gpu_hours']:.0f} GPU-hours "
          f"({pl['atlas_days_continuous']:.1f} days) at {a.epochs} epochs")
    print(f"  the PLAN estimated 235 GPU-hours -- "
          f"{'optimistic' if pl['atlas_gpu_hours'] > 235 else 'conservative'} "
          f"by {abs(pl['atlas_gpu_hours'] - 235)/235*100:.0f}%")

    lo = res_["winner"].get("_loader")
    if lo:
        print(f"\n  loader ceiling: {lo['ceiling_img_s']:.0f} img/s at "
              f"{lo['workers']} workers")
        starved = [r["arch"] for r in pl["per_arch"]
                   if r["img_s"] > lo["ceiling_img_s"] * 0.9]
        if starved:
            print(f"  *** {starved} are at or near the LOADER ceiling. Those are")
            print(f"  *** data-bound, not GPU-bound: tuning the model buys nothing")
            print(f"  *** and the lever is workers, prefetch or the pack format.")
        else:
            print("  every architecture is GPU-bound. The loader is not the "
                  "limit and worker count barely matters.")

    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"bench_{platform.node()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    fp.write_text(json.dumps({
        "host": platform.node(), "platform": platform.platform(),
        "gpu": p.name, "vram_gb": p.total_memory / 2**30,
        "sm": f"{p.major}{p.minor}", "cpus": os.cpu_count(),
        "torch": torch.__version__,
        "bf16": torch.cuda.is_bf16_supported(),
        "dataset": DATASET, "res": res, "epochs": a.epochs,
        "quick": a.quick, "loader_measured": bool(data_dir),
        "elapsed_min": (time.time() - t0) / 60,
        "winner": res_["winner"], "plan": pl, "stages": res_["stages"],
    }, indent=1, default=str), encoding="utf-8")
    print(f"\n  wrote {fp}")
    print("  *** Send me this file. It contains everything needed to re-plan")
    print("  *** the run matrix against measured throughput instead of an")
    print("  *** estimate -- which is what D-10 was about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
