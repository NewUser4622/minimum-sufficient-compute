#!/usr/bin/env python3
"""
verify_d55.py -- measure what the memory-format fix is actually worth.

WHY THIS EXISTS
---------------
ResNet-50 held a flat 80 img/s for 69 consecutive epochs on an RTX 4000 Ada.
Flat is the tell: thermal throttling drifts, disk contention spikes, a layout
conversion is a fixed tax. `GPUBatchLoader` emits `channels_last` activations
unconditionally and `base_config` declares `channels_last: True`, but of the
sixteen places this library builds a model, only `backbone_dry_run` applied
that format. So cuDNN converted a layout on every convolution of every batch,
forward and backward, and the run trained correctly to 80.6% val -- slowly.

D-43 is why this file measures instead of asserting. I previously wrote off an
82 img/s benchmark reading as "understated, re-measure pending". The real run
then produced 80. The benchmark had been right and the prose had been wrong,
so this reports numbers and the numbers decide.

SAFETY -- this machine has been crashed once by a benchmark
----------------------------------------------------------
Every measurement runs in its own subprocess, capped at 50% of VRAM via
`set_per_process_memory_fraction`, under a hard timeout that kills the process
tree. Batch size is FIXED at the training value; there is no batch ladder --
that is the stage that caused the Windows TDR reset, and it is not coming back.
A child process that dies takes nothing with it.

USAGE
-----
    python tools/verify_d55.py                    # resnet50, the atlas default
    python tools/verify_d55.py --arch vit_small_p16
    python tools/verify_d55.py --arch resnet50 --iters 40

Runs in about 60-90 seconds. Nothing is written outside --out.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

VRAM_FRAC = 0.50
TIMEOUT = 300


# ---------------------------------------------------------------------------
# child: one layout, one process
# ---------------------------------------------------------------------------
def _child(arch: str, layout: str, bs: int, res: int, iters: int,
           vram_frac: float) -> dict:
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(ROOT / "src"))
    import msc_lib as M

    dev = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(float(vram_frac), 0)

    # D-43: the same backend function the trainer uses, or this measures a
    # machine the pipeline never runs on.
    M.set_perf_flags(deterministic=False)

    n_cls = M.num_classes_for("imagenet100")
    model = M.build_model(arch, n_cls, dataset="imagenet100").to(dev)
    if layout == "channels_last":
        model = model.to(memory_format=torch.channels_last)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    crit = nn.CrossEntropyLoss()

    # The loader ALWAYS hands over channels_last. That is the fixed half of the
    # comparison: what changes between the two arms is only the model.
    x = torch.randn(bs, 3, res, res, device=dev).contiguous(
        memory_format=torch.channels_last)
    y = torch.randint(0, n_cls, (bs,), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(12):                      # warmup: cudnn.benchmark autotunes
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return {"arch": arch, "layout": layout, "batch_size": bs, "res": res,
            "iters": iters, "img_s": round(bs * iters / dt, 1),
            "ms_per_batch": round(1000 * dt / iters, 1),
            "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "gpu": torch.cuda.get_device_name(0)}


# ---------------------------------------------------------------------------
def run_isolated(arch, layout, bs, res, iters, timeout) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_child",
           "--arch", arch, "--layout", layout, "--bs", str(bs),
           "--res", str(res), "--iters", str(iters)]
    env = dict(os.environ, MSC_OFFLINE="1")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env)
    try:
        so, se = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return {"layout": layout, "error": f"TIMEOUT {timeout}s (killed)"}
    for line in so.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return {"layout": layout,
            "error": (se.strip().splitlines() or ["no result"])[-1][:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=TIMEOUT)
    ap.add_argument("--vram-frac", type=float, default=VRAM_FRAC)
    ap.add_argument("--out", default=str(ROOT / "benchmark"))
    ap.add_argument("--layout", default="channels_last")
    ap.add_argument("--_child", action="store_true")
    a = ap.parse_args()

    if a._child:
        print("__RESULT__" + json.dumps(
            _child(a.arch, a.layout, a.bs, a.res, a.iters, a.vram_frac)))
        return 0

    print(f"""
{'='*70}
D-55 verification -- what the memory-format fix is worth
{'='*70}
  arch {a.arch}   batch {a.bs}   {a.res}px   AMP on   {a.iters} timed iters
  each arm in its own subprocess, VRAM capped at {a.vram_frac:.0%}, {a.timeout}s timeout
  no batch ladder -- that is the stage that crashed this machine
{'='*70}""")

    res = {}
    for layout in ("contiguous", "channels_last"):
        label = ("contiguous  (NCHW model -- what ran for 3 days)"
                 if layout == "contiguous" else
                 "channels_last (NHWC model -- after the fix)")
        print(f"\n  {label}")
        r = run_isolated(a.arch, layout, a.bs, a.res, a.iters, a.timeout)
        res[layout] = r
        if "error" in r:
            print(f"    FAILED: {r['error']}")
        else:
            print(f"    {r['img_s']:>8.1f} img/s   {r['ms_per_batch']:>6.1f} "
                  f"ms/batch   peak {r['peak_mb']} MB")

    old, new = res.get("contiguous", {}), res.get("channels_last", {})
    print(f"\n{'='*70}")
    if "img_s" in old and "img_s" in new:
        sp = new["img_s"] / old["img_s"]
        ep_old = 119395 / old["img_s"] / 60
        ep_new = 119395 / new["img_s"] / 60
        print(f"  speedup {sp:.2f}x")
        print(f"  epoch (119,395 train imgs): {ep_old:.1f} min -> {ep_new:.1f} min")
        print(f"  100 epochs:                 {ep_old*100/60:.1f} h -> "
              f"{ep_new*100/60:.1f} h")
        print()
        if sp < 1.15:
            print("  NOT the bottleneck. Layout is not what is costing you.")
            print("  Send this output back -- the next suspect is the input")
            print("  pipeline, and epochs.csv already has dataload_frac and")
            print("  gpu0_util_mean_pct recorded for all 69 epochs.")
        else:
            print(f"  Your 69 logged epochs ran at 80 img/s. The measured")
            print(f"  channels_last figure above is the one to compare against.")
            print("  The fix is already in msc_lib -- rerun NB2 and it resumes")
            print("  from epoch 69; the checkpoint is unaffected by layout.")
    else:
        print("  incomplete -- send the FAILED lines above")
    print('='*70)

    outp = Path(a.out)
    outp.mkdir(parents=True, exist_ok=True)
    f = outp / f"d55_{a.arch}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    f.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
