#!/usr/bin/env python3
"""
bench_throughput.py -- find the fastest SAFE configuration for this machine.

READ THIS FIRST -- the previous version crashed a workstation
-------------------------------------------------------------
It swept batch sizes up to 256 at 224px on a 20 GB card, with no cap on how
much VRAM the process could take and no isolation between configurations. On a
GPU that is *also driving the display* that is a recipe for a Windows TDR
(Timeout Detection & Recovery) driver reset: the desktop compositor is starved,
the driver stops responding for more than two seconds, Windows resets it, and
the machine hangs.

Three things were wrong, and all three are now fixed:

  1. **No VRAM ceiling.** PyTorch was allowed to allocate everything. Now
     `set_per_process_memory_fraction` caps it (default 70%), so an oversized
     configuration raises a clean, catchable Python OOM *long before* the
     driver runs out of room for the desktop.

  2. **No isolation.** Every configuration ran in this process, so one bad
     allocation poisoned the CUDA context for everything after it, and leaked
     dataloader workers accumulated across the sweep. Now **every
     configuration runs in its own subprocess** with a hard timeout. A child
     that OOMs, hangs or dies takes nothing with it, and its CUDA context and
     workers are released by the OS when it exits.

  3. **No timeout.** A hung kernel hung the sweep, and the machine with it.
     Every child is killed (with its process tree) after `--timeout` seconds.

Also changed: `torch.compile` is OFF by default (Triton on Windows is
unreliable and compilation can hang), workers are capped at 8 rather than 16,
and results are written to disk **after every configuration** so a crash costs
one data point rather than the whole run.

If it still misbehaves, drop the ceiling further: `--vram-frac 0.5`.

Usage
-----
    python benchmark/bench_throughput.py --data-dir "D:\\msc_data\\in100"
    python benchmark/bench_throughput.py --quick            # ~8 min
    python benchmark/bench_throughput.py --no-loader        # model stages only
    python benchmark/bench_throughput.py --plan-only        # print, run nothing
    python benchmark/bench_throughput.py --vram-frac 0.5    # extra cautious

Writes `benchmark/results/bench_<host>_<stamp>.json` incrementally.
Send me that file.
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

# Conservative by construction. Every one of these can be raised from the
# command line; none of them is raised by default.
DEFAULT_VRAM_FRAC = 0.70      # leave >=30% for the display and everything else
DEFAULT_TIMEOUT = 240         # seconds per configuration, then the tree is killed
MAX_WORKERS = 8               # not 16: each worker maps the 24 GiB pack
STEPS, WARMUP = 20, 6         # short runs -> short kernels -> no TDR window


# ===========================================================================
# CHILD -- runs exactly ONE configuration, prints one JSON line, exits.
# ===========================================================================
def child(spec: dict) -> int:
    """Everything dangerous happens here, in a process that can be killed."""
    os.environ.setdefault("MSC_OFFLINE", "1")
    # One torch thread: the parent runs these serially, and oversubscribing
    # 24 cores across a process that is meant to be GPU-bound adds nothing but
    # scheduler noise.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(ROOT / "src"))

    out = dict(spec)
    try:
        # Imports are INSIDE the try so the child always emits a result line.
        # A child that dies before printing gives the parent nothing to report
        # except "no result", which is the least useful failure message
        # available and hides an ImportError behind a spawn problem.
        import torch
        import msc_lib as M

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA in child")

        # THE SAFETY VALVE. Caps this process's share of VRAM so an oversized
        # batch raises a Python OOM we catch, instead of starving the display
        # driver until Windows resets it.
        torch.cuda.set_per_process_memory_fraction(float(spec["vram_frac"]), 0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        res, n_cls = spec["res"], spec["n_cls"]
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[spec["dtype"]]

        if spec["kind"] == "loader":
            # No model at all -- this measures the loader's own ceiling.
            cfg = M.base_config("resnet18", DATASET, seed=1)
            cfg.update({"data_root": spec["data_dir"],
                        "batch_size": spec["batch"],
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

        # ---- model configuration --------------------------------------
        model = M.build_model(spec["arch"], n_cls, dataset=DATASET).cuda()
        if spec["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        if spec.get("compile"):
            model = torch.compile(model, mode=spec["compile"])
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
                    "ok": True})
    except Exception as e:                                       # noqa: BLE001
        name = type(e).__name__
        oom = ("OutOfMemory" in name or "out of memory" in str(e).lower())
        out.update({"ok": False, "img_s": 0.0,
                    "error": "OOM" if oom else f"{name}: {str(e)[:160]}"})
    return _emit(out)


def _emit(d: dict) -> int:
    print("@@RESULT@@" + json.dumps(d, default=str))
    sys.stdout.flush()
    return 0


# ===========================================================================
# PARENT -- orchestrates, never touches CUDA itself.
# ===========================================================================
def _kill_tree(proc):
    """Kill the child AND its dataloader workers. An orphaned worker holding a
    24 GiB memmap is how the next configuration runs out of RAM."""
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
    """One configuration, in its own process. Cannot take the parent down."""
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
                    "error": f"TIMEOUT after {timeout}s (killed)"}
    except Exception as e:                                       # noqa: BLE001
        return {**spec, "ok": False, "img_s": 0.0,
                "error": f"spawn failed: {type(e).__name__}: {e}"}

    for line in (so or "").splitlines():
        if line.startswith("@@RESULT@@"):
            r = json.loads(line[len("@@RESULT@@"):])
            r["wall_s"] = time.time() - t0
            return r
    tail = (se or "").strip().splitlines()[-3:]
    return {**spec, "ok": False, "img_s": 0.0,
            "error": "child produced no result: " + " | ".join(tail)[:200]}


class Results:
    """Written after every configuration. A crash costs one data point."""

    def __init__(self, path: Path, header: dict):
        self.path = path
        self.doc = {**header, "configs": [], "winner": {}}
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


def base(arch, res, n_cls, vram, **kw) -> dict:
    return {"kind": "model", "arch": arch, "res": res, "n_cls": n_cls,
            "batch": 64, "dtype": "float16", "channels_last": True,
            "compile": None, "steps": STEPS, "warmup": WARMUP,
            "vram_frac": vram, **kw}


def sweep(args, archs, res, n_cls, R: Results) -> dict:
    win, T, V = {}, args.timeout, args.vram_frac
    BS = ([32, 64, 128] if args.quick else [32, 64, 96, 128, 192])
    WK = ([2, 6] if args.quick else [0, 2, 4, 6, 8])
    WK = [w for w in WK if w <= MAX_WORKERS]

    print("\n=== A: dtype x memory layout (batch 64) " + "=" * 30)
    dts = ["float16"] + (["bfloat16"] if args.bf16 else [])
    for a in archs:
        best = None
        for d in dts:
            for cl in (True, False):
                r = run_isolated(base(a, res, n_cls, V, dtype=d,
                                      channels_last=cl), T)
                R.add(r)
                print(f"  {a:16s} {d:9s} cl={str(cl):5s} {r['img_s']:8.1f} img/s"
                      + ("" if r.get("ok") else f"   {r.get('error','')[:44]}"))
                if r.get("ok") and (best is None or r["img_s"] > best["img_s"]):
                    best = r
        if best:
            win[a] = {"dtype": best["dtype"],
                      "channels_last": best["channels_last"]}

    print("\n=== B: batch size ladder, stops at the first OOM " + "=" * 21)
    for a in archs:
        if a not in win:
            continue
        best = None
        for bs in BS:
            r = run_isolated(base(a, res, n_cls, V, batch=bs, **win[a]), T)
            R.add(r)
            v = (f"{r.get('peak_vram_gb',0):.1f} GB" if r.get("ok")
                 else r.get("error", "")[:44])
            print(f"  {a:16s} bs={bs:4d} {r['img_s']:8.1f} img/s   {v}")
            if r.get("ok"):
                if best is None or r["img_s"] > best["img_s"]:
                    best = r
            elif r.get("error") == "OOM":
                print(f"  {'':16s} -> stopping the ladder; larger will also OOM")
                break
        if best:
            win[a].update({"batch": best["batch"],
                           "peak_vram_gb": best.get("peak_vram_gb", 0),
                           "img_s": best["img_s"]})

    if args.data_dir:
        print("\n=== C: dataloader workers, NO model " + "=" * 34)
        best = None
        for wk in WK:
            spec = {"kind": "loader", "res": res, "n_cls": n_cls, "batch": 128,
                    "workers": wk, "data_dir": str(args.data_dir),
                    "dtype": "float16", "channels_last": True,
                    "steps": 30, "warmup": 0, "vram_frac": V}
            r = run_isolated(spec, T)
            R.add(r)
            extra = (f"  wait {r['wait_frac']*100:.0f}% / aug "
                     f"{r['augment_frac']*100:.0f}%" if "wait_frac" in r else "")
            print(f"  workers={wk:3d}  {r['img_s']:8.1f} img/s{extra}"
                  + ("" if r.get("ok") else f"   {r.get('error','')[:40]}"))
            if r.get("ok") and (best is None or r["img_s"] > best["img_s"]):
                best = r
        if best:
            win["_loader"] = {"workers": best["workers"],
                              "ceiling_img_s": best["img_s"]}
    else:
        print("\n=== C: skipped (no --data-dir) " + "=" * 39)

    if args.compile:
        print("\n=== D: torch.compile " + "=" * 49)
        print("  off by default; Triton on Windows is unreliable and compile "
              "can hang")
        for a in archs:
            if "batch" not in win.get(a, {}):
                continue
            w = {k: win[a][k] for k in ("dtype", "channels_last", "batch")}
            r = run_isolated(base(a, res, n_cls, V, compile="default",
                                  warmup=12, **w), max(T, 600))
            R.add(r)
            gain = ((r["img_s"] / win[a]["img_s"] - 1) * 100
                    if win[a].get("img_s") and r.get("ok") else 0.0)
            print(f"  {a:16s} {r['img_s']:8.1f} img/s  {gain:+5.1f}%"
                  + ("" if r.get("ok") else f"   {r.get('error','')[:40]}"))
            if r.get("ok") and r["img_s"] > win[a]["img_s"] * 1.03:
                win[a].update({"compile": "default", "img_s": r["img_s"]})

    print("\n=== E: confirmation from cold, longer run " + "=" * 28)
    for a in archs:
        if "batch" not in win.get(a, {}):
            continue
        w = {k: win[a][k] for k in ("dtype", "channels_last", "batch")}
        r = run_isolated(base(a, res, n_cls, V, steps=50, warmup=12,
                              compile=win[a].get("compile"), **w), T)
        R.add(r)
        if r.get("ok"):
            win[a]["img_s_confirmed"] = r["img_s"]
        print(f"  {a:16s} {r['img_s']:8.1f} img/s   "
              f"{r.get('peak_vram_gb',0):.1f} GB peak"
              + ("" if r.get("ok") else f"   {r.get('error','')[:40]}"))
    return win


def plan(win, n_train=119_395, epochs=100) -> dict:
    rows, total = [], 0.0
    for a, w in sorted(win.items()):
        s = w.get("img_s_confirmed") or w.get("img_s")
        if a.startswith("_") or not s:
            continue
        sec = n_train / s
        h = sec * epochs / 3600.0
        rows.append({"arch": a, "img_s": s, "sec_per_epoch": sec,
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
    ap.add_argument("--no-loader", action="store_true",
                    help="skip stage C even if --data-dir is given")
    ap.add_argument("--synthetic", action="store_true",
                    help="alias for --no-loader")
    ap.add_argument("--compile", action="store_true",
                    help="also try torch.compile (off by default: unreliable "
                         "on Windows and can hang)")
    ap.add_argument("--vram-frac", type=float, default=DEFAULT_VRAM_FRAC,
                    help="cap on this process's VRAM. Lower it if the display "
                         "stutters. 0.5 is very safe.")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--plan-only", action="store_true",
                    help="print what would run, execute nothing")
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
    archs = a.archs or M.zoo_for_dataset(DATASET)
    res, n_cls = M.native_res(DATASET), M.num_classes_for(DATASET)

    if a.synthetic or a.no_loader:
        if a.data_dir:
            print("note: --synthetic/--no-loader given, so stage C (the loader\n"
                  "      ceiling) is skipped even though --data-dir was passed.\n"
                  "      Drop --synthetic to measure it -- it is the stage that\n"
                  "      says whether the small models are data-bound.\n")
        a.data_dir = None
    if a.data_dir and not M.data_present(DATASET, a.data_dir)[0]:
        print(f"note: no pack at {a.data_dir}; stage C skipped")
        a.data_dir = None

    cap = p.total_memory * a.vram_frac / 2**30
    n_model = len(archs) * ((2 if a.quick else 2) * (1 + a.bf16)
                            + (3 if a.quick else 5) + 1) + \
        (len(archs) if a.compile else 0)
    n_loader = (2 if a.quick else 5) if a.data_dir else 0

    print("=" * 72)
    print(f"  {p.name}   {p.total_memory/2**30:.1f} GiB   sm_{p.major}{p.minor}")
    print(f"  torch {torch.__version__}   bf16 {'yes' if a.bf16 else 'no'}   "
          f"cpus {os.cpu_count()}")
    print()
    print("  SAFETY")
    print(f"    VRAM cap        {a.vram_frac:.0%}  ->  {cap:.1f} GiB of "
          f"{p.total_memory/2**30:.1f} GiB")
    print(f"                    the rest stays free for the display driver.")
    print(f"                    An oversized batch now raises a catchable OOM")
    print(f"                    instead of starving the compositor into a TDR.")
    print(f"    isolation       every configuration runs in its OWN process")
    print(f"    timeout         {a.timeout}s per config, then the process TREE "
          f"is killed")
    print(f"    batch ceiling   {192 if not a.quick else 128} (was 256)")
    print(f"    workers ceiling {MAX_WORKERS} (was 16)")
    print(f"    torch.compile   {'ON (you asked)' if a.compile else 'OFF'}")
    print(f"    results         written after EVERY config")
    print()
    print(f"  {len(archs)} architectures at {res}px  ->  ~{n_model + n_loader} "
          f"configurations")
    print(f"  worst case ~{(n_model + n_loader) * a.timeout / 60:.0f} min if "
          f"everything times out; typically far less")
    print("=" * 72)
    if a.plan_only:
        print("\n--plan-only: nothing executed.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"bench_{platform.node()}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    R = Results(fp, {
        "host": platform.node(), "platform": platform.platform(),
        "gpu": p.name, "vram_gb": p.total_memory / 2**30,
        "sm": f"{p.major}{p.minor}", "cpus": os.cpu_count(),
        "torch": torch.__version__, "bf16": a.bf16,
        "vram_frac": a.vram_frac, "timeout_s": a.timeout,
        "dataset": DATASET, "res": res, "epochs": a.epochs,
        "quick": a.quick, "loader_measured": bool(a.data_dir),
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
    print("  CHOSEN CONFIGURATION")
    print("=" * 72)
    print(f"  {'arch':16s} {'bs':>4s} {'dtype':>9s} {'cl':>3s} {'img/s':>8s} "
          f"{'VRAM':>6s} {'s/ep':>7s} {'h x3':>7s}")
    for r in pl["per_arch"]:
        w = win[r["arch"]]
        print(f"  {r['arch']:16s} {w.get('batch',0):4d} {w.get('dtype','?'):>9s} "
              f"{str(w.get('channels_last'))[0]:>3s} {r['img_s']:8.1f} "
              f"{w.get('peak_vram_gb',0):5.1f}G {r['sec_per_epoch']:7.0f} "
              f"{r['hours_3_seeds']:7.1f}")
    print(f"\n  atlas: {pl['atlas_gpu_hours']:.0f} GPU-hours "
          f"({pl['atlas_days']:.1f} days) at {a.epochs} epochs")
    if pl["per_arch"]:
        d = (pl["atlas_gpu_hours"] - 235) / 235 * 100
        print(f"  the plan estimated 235 -- it was "
              f"{'optimistic' if d > 0 else 'conservative'} by {abs(d):.0f}%")

    lo = win.get("_loader")
    if lo:
        print(f"\n  loader ceiling {lo['ceiling_img_s']:.0f} img/s at "
              f"{lo['workers']} workers")
        st = [r["arch"] for r in pl["per_arch"]
              if r["img_s"] > lo["ceiling_img_s"] * 0.9]
        print(f"  {'DATA-BOUND: ' + str(st) if st else 'all GPU-bound'}")

    failed = [c for c in R.doc["configs"] if not c.get("ok")]
    if failed:
        print(f"\n  {len(failed)} configuration(s) failed "
              f"({sum(1 for c in failed if c.get('error') == 'OOM')} OOM, "
              f"{sum(1 for c in failed if 'TIMEOUT' in str(c.get('error')))} "
              f"timeout). All contained; none touched this process.")
    print(f"\n  wrote {fp}   --  send me this file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
