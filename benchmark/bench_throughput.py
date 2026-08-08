#!/usr/bin/env python3
"""
bench_throughput.py -- measure training throughput without going near the edge.

WHAT WAS REMOVED, AND WHY IT WAS NEVER NEEDED
---------------------------------------------
The previous version crashed a workstation by climbing a batch-size ladder to
256 at 224px on a 20 GB card that also drives the display. Windows fired a TDR
reset and the machine hung.

The ladder is gone. Not made safer -- **gone** -- because it was the dangerous
stage AND a scientifically useless one:

    All eight architectures must train at the SAME batch size, or batch size
    joins accuracy and family as a confounded variable. Learning rate is scaled
    linearly from a reference batch, so a per-architecture batch "winner" would
    mean eight different recipes and the seed-reliability comparison would be
    measuring the recipe as much as the architecture.

So there was never anything to do with a per-architecture optimum. What is
actually needed is: how fast does each architecture train at the batch size we
are going to use, and does that batch fit with room to spare.

Both are answerable at a **fixed, conservative batch of 64**, and headroom for
larger batches is then **predicted from measured peak VRAM** rather than probed
by trying to run out of memory. Activation memory is close to linear in batch
size, so measuring 64 tells you what 128 will cost without ever allocating it.

Predict, don't probe. That is the whole change.

ALSO REMOVED
------------
* `torch.compile` -- the thing most likely to hang on Windows. Triton support
  is unreliable and a compile can sit for minutes with no output.
* The separate confirmation stage -- folded into the one measurement.
* The 192-batch ceiling -- the cap is now 64 and `--batch` will not exceed 96.

WHAT REMAINS, AND WHY EACH IS SAFE
----------------------------------
* **dtype x memory layout at batch 64** -- ~24 short runs. Real 10-40% effects,
  and at batch 64 even `vgg16` sits at a few GB.
* **the loader ceiling** -- measured with NO model on the GPU at all. Near-zero
  VRAM. This is the stage that says whether the small architectures are
  data-bound, which no amount of model tuning would tell you.

SAFETY, unchanged from the rebuild and still in force
-----------------------------------------------------
* `set_per_process_memory_fraction` (default **0.50**, was 0.70)
* every configuration in its own subprocess -- a child cannot take the parent
* hard timeout, killing the process TREE
* results written after every configuration

Usage
-----
    python benchmark/bench_throughput.py --plan-only        # prints, runs nothing
    python benchmark/bench_throughput.py --quick            # ~6 min
    python benchmark/bench_throughput.py --data-dir "D:\\msc_data\\in100"
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve()
OUT = BENCH.parent / "results"
DATASET = "imagenet100"

SAFE_BATCH = 64          # what every architecture is measured at
MAX_BATCH = 96           # --batch will not exceed this, whatever you pass
VRAM_FRAC = 0.50         # half the card. The display keeps the rest.
TIMEOUT = 120            # a 26-step run at batch 64 is ~15-30s even for vgg16
MAX_WORKERS = 8
STEPS, WARMUP = 24, 6


# ===========================================================================
# CHILD -- one configuration, one JSON line, exit.
# ===========================================================================
def child(spec: dict) -> int:
    os.environ.setdefault("MSC_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(ROOT / "src"))
    out = dict(spec)
    try:
        import torch
        import msc_lib as M

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA in child")
        # The safety valve: an oversized allocation raises a catchable Python
        # OOM instead of starving the display driver into a TDR reset.
        torch.cuda.set_per_process_memory_fraction(float(spec["vram_frac"]), 0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        res, n_cls = spec["res"], spec["n_cls"]

        if spec["kind"] == "loader":
            cfg = M.base_config("resnet18", DATASET, seed=1)
            cfg.update({"data_root": spec["data_dir"], "batch_size": spec["batch"],
                        "num_workers": spec["workers"], "input_res": res})
            tr, _, _, _, _ = M.build_loaders(cfg)
            it = iter(tr)
            for _ in range(4):
                next(it)
            torch.cuda.synchronize()
            t0, n = time.time(), 0
            for _ in range(spec["steps"]):
                try:
                    b = next(it)
                except StopIteration:
                    it = iter(tr)
                    b = next(it)
                n += b[0].shape[0]
            torch.cuda.synchronize()
            out["img_s"] = n / (time.time() - t0)
            if hasattr(tr, "timing"):
                t = tr.timing()
                tot = max(1e-9, t["wait_s"] + t["augment_s"])
                out["wait_frac"] = t["wait_s"] / tot
                out["augment_frac"] = t["augment_s"] / tot
            out["ok"] = True
            del it, tr
            return _emit(out)

        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[spec["dtype"]]
        model = M.build_model(spec["arch"], n_cls, dataset=DATASET).cuda()
        if spec["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
        crit = torch.nn.CrossEntropyLoss()
        model.train()

        b = spec["batch"]
        x = torch.randn(b, 3, res, res, device="cuda")
        y = torch.randint(0, n_cls, (b,), device="cuda")
        if spec["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)

        n_seen, t0 = 0, None
        for i in range(spec["steps"] + spec["warmup"]):
            with torch.amp.autocast("cuda", dtype=dt):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if i == spec["warmup"] - 1:
                torch.cuda.synchronize()
                t0 = time.time()
            elif i >= spec["warmup"]:
                n_seen += b
        torch.cuda.synchronize()
        out.update({"img_s": n_seen / (time.time() - t0),
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                    "params_m": M.count_parameters(model) / 1e6,
                    "ok": True})
    except Exception as e:                                       # noqa: BLE001
        n = type(e).__name__
        oom = "OutOfMemory" in n or "out of memory" in str(e).lower()
        out.update({"ok": False, "img_s": 0.0,
                    "error": "OOM" if oom else f"{n}: {str(e)[:160]}"})
    return _emit(out)


def _emit(d: dict) -> int:
    print("@@RESULT@@" + json.dumps(d, default=str))
    sys.stdout.flush()
    return 0


# ===========================================================================
# PARENT -- never touches CUDA.
# ===========================================================================
def _kill_tree(proc):
    try:
        import psutil
        p = psutil.Process(proc.pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except Exception:                                    # noqa: BLE001
                pass
        p.kill()
    except Exception:                                            # noqa: BLE001
        try:
            proc.kill()
        except Exception:                                        # noqa: BLE001
            pass


def run_isolated(spec: dict, timeout: int) -> dict:
    cmd = [sys.executable, str(BENCH), "--child", json.dumps(spec)]
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
        try:
            so, se = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(p)
            p.communicate()
            return {**spec, "ok": False, "img_s": 0.0,
                    "error": f"TIMEOUT {timeout}s (killed)"}
    except Exception as e:                                       # noqa: BLE001
        return {**spec, "ok": False, "img_s": 0.0,
                "error": f"spawn failed: {type(e).__name__}: {e}"}
    for line in (so or "").splitlines():
        if line.startswith("@@RESULT@@"):
            r = json.loads(line[len("@@RESULT@@"):])
            r["wall_s"] = time.time() - t0
            return r
    tail = " | ".join((se or "").strip().splitlines()[-3:])
    return {**spec, "ok": False, "img_s": 0.0,
            "error": "no result from child: " + tail[:200]}


class Results:
    def __init__(self, path: Path, header: dict):
        self.path, self.doc = path, {**header, "configs": [], "winner": {}}
        self.flush()

    def add(self, r: dict):
        self.doc["configs"].append(r)
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=1, default=str),
                       encoding="utf-8")
        os.replace(tmp, self.path)


def headroom(peak_gb: float, batch: int, total_gb: float,
             frac: float = 0.60) -> dict:
    """What batch would fit, PREDICTED from a measurement rather than probed.

    Activation memory is close to linear in batch size, and parameters and
    optimiser state are constant. So a measurement at 64 says what 128 costs
    without ever allocating it -- which is the whole point, because allocating
    it is what took the machine down.

    Deliberately conservative: `frac` of total VRAM, and the result is labelled
    an estimate. It is a starting point to verify, never a licence to jump
    straight to the largest number it prints.
    """
    if not peak_gb or peak_gb <= 0:
        return {}
    budget = total_gb * frac
    per_img = peak_gb / max(1, batch)
    est = int(budget / per_img) if per_img > 0 else 0
    fits = [b for b in (32, 64, 96, 128, 192, 256) if b <= est]
    return {"peak_gb_at_batch": peak_gb, "gb_per_image": per_img,
            "budget_gb": budget, "largest_estimated_batch": est,
            "safe_batches": fits}


def sweep(args, archs, res, n_cls, R: Results) -> dict:
    win, T, V = {}, args.timeout, args.vram_frac
    B = args.batch

    def spec(a, **kw):
        return {"kind": "model", "arch": a, "res": res, "n_cls": n_cls,
                "batch": B, "dtype": "float16", "channels_last": True,
                "steps": STEPS, "warmup": WARMUP, "vram_frac": V, **kw}

    # One batch size for everybody. Variants are dtype and memory layout only,
    # both of which are throughput choices and neither of which changes the
    # recipe -- unlike batch size, which does.
    variants = [("float16", True)]
    if not args.quick:
        variants.append(("float16", False))
        if args.bf16:
            variants.append(("bfloat16", True))

    print(f"\n=== dtype x memory layout, batch {B} for every architecture "
          + "=" * 12)
    for a in archs:
        best = None
        for d, cl in variants:
            r = run_isolated(spec(a, dtype=d, channels_last=cl), T)
            R.add(r)
            v = (f"{r.get('peak_vram_gb', 0):.1f} GB"
                 if r.get("ok") else r.get("error", "")[:44])
            print(f"  {a:16s} {d:9s} cl={str(cl):5s} {r['img_s']:8.1f} img/s   {v}")
            if r.get("ok") and (best is None or r["img_s"] > best["img_s"]):
                best = r
        if best:
            win[a] = {"dtype": best["dtype"],
                      "channels_last": best["channels_last"],
                      "batch": B, "img_s": best["img_s"],
                      "peak_vram_gb": best.get("peak_vram_gb", 0),
                      "params_m": best.get("params_m"),
                      "headroom": headroom(best.get("peak_vram_gb", 0), B,
                                           args.total_vram)}

    if args.data_dir:
        print("\n=== loader ceiling, NO model on the GPU " + "=" * 30)
        WK = [2, 6] if args.quick else [0, 2, 4, 6, 8]
        best = None
        for wk in [w for w in WK if w <= MAX_WORKERS]:
            r = run_isolated({"kind": "loader", "res": res, "n_cls": n_cls,
                              "batch": B, "workers": wk,
                              "data_dir": str(args.data_dir),
                              "steps": 30, "vram_frac": V}, T)
            R.add(r)
            extra = (f"   wait {r['wait_frac']*100:.0f}% / aug "
                     f"{r['augment_frac']*100:.0f}%" if "wait_frac" in r else "")
            print(f"  workers={wk:3d}  {r['img_s']:8.1f} img/s{extra}"
                  + ("" if r.get("ok") else f"   {r.get('error','')[:40]}"))
            if r.get("ok") and (best is None or r["img_s"] > best["img_s"]):
                best = r
        if best:
            win["_loader"] = {"workers": best["workers"],
                              "ceiling_img_s": best["img_s"]}
    else:
        print("\n=== loader ceiling: SKIPPED (no --data-dir) " + "=" * 26)
    return win


def plan(win, n_train=119_395, epochs=100) -> dict:
    rows, total = [], 0.0
    for a, w in sorted(win.items()):
        if a.startswith("_") or not w.get("img_s"):
            continue
        sec = n_train / w["img_s"]
        h = sec * epochs / 3600.0
        rows.append({"arch": a, "img_s": w["img_s"], "sec_per_epoch": sec,
                     "hours_per_run": h, "hours_3_seeds": h * 3})
        total += h * 3
    return {"per_arch": rows, "atlas_gpu_hours": total,
            "atlas_days": total / 24.0, "epochs": epochs}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data-dir", default=os.environ.get("MSC_IN100_DIR"))
    ap.add_argument("--no-loader", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="alias --no-loader")
    ap.add_argument("--batch", type=int, default=SAFE_BATCH)
    ap.add_argument("--vram-frac", type=float, default=VRAM_FRAC)
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    if a.child:
        return child(json.loads(a.child))

    sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("MSC_OFFLINE", "1")
    import torch
    import msc_lib as M

    if not torch.cuda.is_available():
        print("no CUDA -- nothing to measure.")
        return 1
    p = torch.cuda.get_device_properties(0)
    a.bf16 = torch.cuda.is_bf16_supported()
    a.total_vram = p.total_memory / 2**30
    archs = a.archs or M.zoo_for_dataset(DATASET)
    res, n_cls = M.native_res(DATASET), M.num_classes_for(DATASET)

    if a.batch > MAX_BATCH:
        print(f"note: --batch {a.batch} exceeds the {MAX_BATCH} ceiling; "
              f"using {MAX_BATCH}. Large batches are what took the machine "
              f"down, and headroom is predicted rather than probed.")
        a.batch = MAX_BATCH

    if a.synthetic or a.no_loader:
        if a.data_dir:
            print("note: --synthetic given, so the loader stage is skipped even "
                  "though\n      --data-dir was passed. Drop --synthetic to "
                  "measure it -- it uses\n      almost no VRAM and is the "
                  "stage that says whether the small\n      models are "
                  "data-bound.\n")
        a.data_dir = None
    if a.data_dir and not M.data_present(DATASET, a.data_dir)[0]:
        print(f"note: no pack at {a.data_dir}; loader stage skipped")
        a.data_dir = None

    n_var = 1 if a.quick else (2 + int(a.bf16))
    n_cfg = len(archs) * n_var + ((2 if a.quick else 5) if a.data_dir else 0)

    print("=" * 72)
    print(f"  {p.name}   {a.total_vram:.1f} GiB   sm_{p.major}{p.minor}")
    print(f"  torch {torch.__version__}   bf16 {'yes' if a.bf16 else 'no'}   "
          f"cpus {os.cpu_count()}")
    print()
    print("  REMOVED, because it was the dangerous stage and a useless one")
    print("    batch-size ladder   all eight architectures must share ONE batch")
    print("                        size or batch becomes a confounded variable.")
    print("                        A per-architecture winner was never usable.")
    print(f"                        Fixed at {a.batch}; larger batches are")
    print("                        PREDICTED from measured peak VRAM instead.")
    print("    torch.compile       most likely thing to hang on Windows")
    print()
    print("  SAFETY")
    print(f"    VRAM cap        {a.vram_frac:.0%}  ->  "
          f"{a.total_vram * a.vram_frac:.1f} of {a.total_vram:.1f} GiB")
    print(f"    batch           {a.batch} for every architecture "
          f"(hard ceiling {MAX_BATCH})")
    print(f"    isolation       every config in its OWN process")
    print(f"    timeout         {a.timeout}s, then the process TREE is killed")
    print(f"    results         written after EVERY config")
    print()
    print(f"  {len(archs)} architectures at {res}px  ->  {n_cfg} configurations")
    print(f"  worst case ~{n_cfg * a.timeout / 60:.0f} min if every single one "
          f"times out;")
    print(f"  realistically ~{n_cfg * 35 / 60:.0f} min")
    print("=" * 72)
    if a.plan_only:
        print("\n--plan-only: nothing was executed.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"bench_{platform.node()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    R = Results(fp, {
        "host": platform.node(), "platform": platform.platform(),
        "gpu": p.name, "vram_gb": a.total_vram, "sm": f"{p.major}{p.minor}",
        "cpus": os.cpu_count(), "torch": torch.__version__, "bf16": a.bf16,
        "vram_frac": a.vram_frac, "timeout_s": a.timeout, "batch": a.batch,
        "dataset": DATASET, "res": res, "epochs": a.epochs, "quick": a.quick,
        "loader_measured": bool(a.data_dir),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"\n  writing to {fp}\n")

    t0 = time.time()
    win = sweep(a, archs, res, n_cls, R)
    pl = plan(win, epochs=a.epochs)
    R.doc.update({"winner": win, "plan": pl,
                  "elapsed_min": (time.time() - t0) / 60})
    R.flush()

    print("\n" + "=" * 72)
    print(f"  RESULT  (batch {a.batch}, the same for every architecture)")
    print("=" * 72)
    print(f"  {'arch':16s} {'dtype':>9s} {'cl':>3s} {'img/s':>8s} {'VRAM':>7s} "
          f"{'s/ep':>7s} {'h x3':>7s}")
    for r in pl["per_arch"]:
        w = win[r["arch"]]
        print(f"  {r['arch']:16s} {w.get('dtype','?'):>9s} "
              f"{str(w.get('channels_last'))[0]:>3s} {r['img_s']:8.1f} "
              f"{w.get('peak_vram_gb',0):6.1f}G {r['sec_per_epoch']:7.0f} "
              f"{r['hours_3_seeds']:7.1f}")
    print(f"\n  atlas: {pl['atlas_gpu_hours']:.0f} GPU-hours "
          f"({pl['atlas_days']:.1f} days) at {a.epochs} epochs")
    if pl["per_arch"]:
        d = (pl["atlas_gpu_hours"] - 235) / 235 * 100
        print(f"  the plan estimated 235 -- it was "
              f"{'optimistic' if d > 0 else 'conservative'} by {abs(d):.0f}%")

    hp = [(a_, w["headroom"]) for a_, w in win.items()
          if not a_.startswith("_") and w.get("headroom")]
    if hp:
        worst = min(hp, key=lambda t: t[1]["largest_estimated_batch"])
        print(f"\n  HEADROOM, predicted from peak VRAM -- not probed")
        print(f"  tightest architecture: {worst[0]}  "
              f"{worst[1]['peak_gb_at_batch']:.1f} GB at batch {a.batch}")
        print(f"  estimated largest batch that fits in 60% of the card: "
              f"{worst[1]['largest_estimated_batch']}")
        print(f"  batches all eight could take: {worst[1]['safe_batches']}")
        print(f"  This is an ESTIMATE from a linear model. If you want a bigger")
        print(f"  batch, step ONE notch and re-run this, rather than jumping.")

    lo = win.get("_loader")
    if lo:
        print(f"\n  loader ceiling {lo['ceiling_img_s']:.0f} img/s at "
              f"{lo['workers']} workers")
        st = [r["arch"] for r in pl["per_arch"]
              if r["img_s"] > lo["ceiling_img_s"] * 0.9]
        print("  " + (f"DATA-BOUND: {st} -- the lever is workers, not the model"
                      if st else "all GPU-bound; worker count barely matters"))

    bad = [c for c in R.doc["configs"] if not c.get("ok")]
    if bad:
        print(f"\n  {len(bad)} configuration(s) failed. All contained in their "
              f"own process; none touched this one.")
        for c in bad[:6]:
            print(f"    {c.get('arch', 'loader'):16s} {str(c.get('error'))[:56]}")
    print(f"\n  wrote {fp}   --  send me this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
