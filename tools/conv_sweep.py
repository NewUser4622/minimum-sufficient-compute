#!/usr/bin/env python3
"""
conv_sweep.py -- why is ResNet-50 7.5x slower than ViT-S at the same FLOPs?

THE MEASUREMENT THAT NARROWED IT
--------------------------------
From C:\\msc_results, last 20 epochs, same card, same loader, same augmentation:

    ResNet-50      796 ms/batch    80 img/s    98.5% GPU util   110 W
    ViT-S/16       106 ms/batch   604 img/s    97.7% GPU util   120 W

    dataload_frac  0.000    <- the loader is NOT the bottleneck
    augment_frac   0.012    <- augmentation is NOT the bottleneck
    compute_time   99.6% of train time

ViT-S/16 has MORE FLOPs than ResNet-50 (4.6 vs 4.1 GFLOPs forward) and runs
7.5x faster. Whatever is wrong is specific to convolutions -- which is why the
loader fix (D-56) and the memory-format fix (D-55) both changed nothing.

Each row below changes exactly ONE thing from the configuration that is
running now. The row that recovers the time is the answer.

The `plain torchvision` row is the control I should have had first: it skips
StagedBackbone entirely. If plain ResNet-50 is fast and ours is slow, the
wrapper is implicated and nothing about cuDNN matters.

SAFETY
------
Isolated subprocess per row, VRAM capped, hard timeout killing the tree.
Batch rows go 64 -> 128 -> 256 and STOP on the first OOM rather than probing
upward. No ladder past a failure (D-41).

USAGE
    python tools/conv_sweep.py
    python tools/conv_sweep.py --arch vgg16
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

# name -> (channels_last, amp, cudnn_benchmark, batch, plain_torchvision)
ROWS = {
    "current  (chlast+amp+bench, bs64)": (True,  True,  True,  64,  False),
    "no channels_last":                  (False, True,  True,  64,  False),
    "no cudnn.benchmark":                (True,  True,  False, 64,  False),
    "no AMP (fp32)":                     (True,  False, True,  64,  False),
    "plain torchvision (no wrapper)":    (True,  True,  True,  64,  True),
    "batch 128":                         (True,  True,  True,  128, False),
    "batch 256":                         (True,  True,  True,  256, False),
}


def _child(arch, cl, amp, bench, bs, plain, res, iters, vram_frac):
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(ROOT / "src"))
    import msc_lib as M

    dev = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(float(vram_frac), 0)

    torch.backends.cudnn.benchmark = bool(bench)
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    n_cls = M.num_classes_for("imagenet100")
    if plain:
        import torchvision.models as tvm
        if not hasattr(tvm, arch):
            return {"error": f"torchvision has no {arch}"}
        model = getattr(tvm, arch)(weights=None, num_classes=n_cls)
    else:
        model = M.build_model(arch, n_cls, dataset="imagenet100")
    model = model.to(dev)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp))
    crit = nn.CrossEntropyLoss()

    x = torch.randn(bs, 3, res, res, device=dev)
    x = (x.contiguous(memory_format=torch.channels_last) if cl
         else x.contiguous())
    y = torch.randint(0, n_cls, (bs,), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=bool(amp)):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(12):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"img_s": bs * iters / dt, "ms_per_batch": 1000 * dt / iters,
            "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "batch": bs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--vram-frac", type=float, default=0.55)
    ap.add_argument("--_child", action="store_true")
    for k in ("cl", "amp", "bench", "plain"):
        ap.add_argument(f"--{k}", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()

    if a._child:
        try:
            r = _child(a.arch, a.cl, a.amp, a.bench, a.bs, a.plain,
                       a.res, a.iters, a.vram_frac)
        except RuntimeError as e:
            r = {"error": ("CUDA OOM" if "out of memory" in str(e).lower()
                           else f"{type(e).__name__}: {e}")[:200]}
        print("__RESULT__" + json.dumps(r))
        return 0

    print(f"""
{'='*78}
{a.arch} @ {a.res}px -- one change per row, isolated subprocess, VRAM {a.vram_frac:.0%}
  measured in training now: ResNet-50 796 ms/batch (80 img/s)
                            ViT-S/16  106 ms/batch (604 img/s), same loader
{'='*78}
  {'configuration':36s}{'img/s':>10}{'ms/batch':>11}{'peak MB':>10}
  {'-'*36}{'-'*31}""")

    out, base = {}, None
    for label, (cl, amp, bench, bs, plain) in ROWS.items():
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_child",
               "--arch", a.arch, "--res", str(a.res), "--iters", str(a.iters),
               "--vram-frac", str(a.vram_frac), "--bs", str(bs),
               "--cl", str(int(cl)), "--amp", str(int(amp)),
               "--bench", str(int(bench)), "--plain", str(int(plain))]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=dict(os.environ, MSC_OFFLINE="1"))
        try:
            so, se = p.communicate(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            p.kill(); p.communicate()
            so, se = "", f"TIMEOUT {a.timeout}s"
        r = {"error": (se.strip().splitlines() or ["no result"])[-1][:120]}
        for line in so.splitlines():
            if line.startswith("__RESULT__"):
                r = json.loads(line[len("__RESULT__"):])
        out[label] = r
        if "error" in r:
            print(f"  {label:36s}{'FAILED':>10}   {r['error'][:34]}")
            if "OOM" in r["error"] and label.startswith("batch"):
                print("      stopping the batch ladder here -- no probing upward")
                break
            continue
        if base is None:
            base = r["img_s"]
        mark = ""
        if base and r["img_s"] > base * 1.5:
            mark = f"   <-- {r['img_s']/base:.1f}x FASTER"
        print(f"  {label:36s}{r['img_s']:>10.1f}{r['ms_per_batch']:>11.1f}"
              f"{r['peak_mb']:>10}{mark}")

    ok = {k: v["img_s"] for k, v in out.items() if "img_s" in v}
    print(f"\n{'='*78}")
    if base and ok:
        best = max(ok, key=lambda k: ok[k])
        if ok[best] > base * 1.5:
            print(f"  ANSWER: '{best}' gives {ok[best]/base:.1f}x "
                  f"({base:.0f} -> {ok[best]:.0f} img/s).")
            print(f"  ResNet-50 3 seeds x 100 epochs: "
                  f"{119395*100*3/base/3600:.0f} h -> "
                  f"{119395*100*3/ok[best]/3600:.0f} h")
        else:
            print("  No single change recovers the time. The cost is intrinsic")
            print("  to convolutions on this card at this size -- which makes")
            print("  the lever architecture selection or resolution, not tuning.")
    print('='*78)

    d = ROOT / "benchmark"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"convsweep_{a.arch}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    f.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
