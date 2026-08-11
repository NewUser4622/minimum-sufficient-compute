#!/usr/bin/env python3
"""
bisect_speed.py -- stop theorising. Time each stage separately.

WHY THIS EXISTS
---------------
Two wrong diagnoses in a row, both argued from aggregate numbers:

  D-55  "the model is in the wrong memory format"  -> fixed it, 0% change.
  D-56  "the loader is reading from a slow disk"   -> fixed it, 0% change.
        The RAM cache then read the pack at 3.40 GiB/s, disproving the premise
        on the way past.

The number that was available the whole time and never checked: 0.045 kWh per
1491 s epoch is 108 W on a 130 W card. The GPU was at 83% of TDP. It was never
idle and it was never starved -- it was busy, drawing near-full power, and
producing 80 img/s. That is the signature of memory-bound work, not of a stall.

An aggregate throughput number cannot distinguish "waiting", "augmenting" and
"computing", so any argument built on it is a guess. This bisects.

  [1] compute only     synthetic tensor, resident on GPU. No loader, no
                       augmentation. The ceiling this model can possibly hit.
  [2] + augmentation   the real GPUBatchLoader path on a fixed input.
  [3] + data           the real loader end to end.
  [4] + instrumentation the metrics/dynamics the training loop adds per batch.

Each stage adds exactly one thing. Whichever step drops throughput is the
answer, and it is a measurement rather than an opinion.

SAFETY
------
Every stage runs in its own subprocess, VRAM capped at 50%, hard timeout that
kills the tree. Batch size fixed -- no ladder (D-41). Nothing is written except
the JSON summary.

USAGE
    python tools/bisect_speed.py --data-dir C:\\msc_data\\in100
    python tools/bisect_speed.py --data-dir C:\\msc_data\\in100 --arch vit_small_p16
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

STAGES = ("compute", "augment", "data", "full")


def _child(stage, arch, data_dir, bs, res, iters, vram_frac):
    import numpy as np
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(ROOT / "src"))
    import msc_lib as M

    dev = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(float(vram_frac), 0)
    M.set_perf_flags(deterministic=False)

    n_cls = M.num_classes_for("imagenet100")
    cfg = {"channels_last": True}
    model = M.place_model(M.build_model(arch, n_cls, dataset="imagenet100"),
                          dev, cfg)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    crit = nn.CrossEntropyLoss()

    def fwd_bwd(x, y):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return out, loss

    # ---- stage 1: compute only -------------------------------------------
    if stage == "compute":
        x = torch.randn(bs, 3, res, res, device=dev).contiguous(
            memory_format=torch.channels_last)
        y = torch.randint(0, n_cls, (bs,), device=dev)
        for _ in range(12):
            fwd_bwd(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fwd_bwd(x, y)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {"stage": stage, "img_s": bs * iters / dt,
                "ms_per_batch": 1000 * dt / iters}

    # everything below needs the real augmentation path
    spec = M.dataset_spec("imagenet100")
    ds = M.PackedImageDataset(Path(data_dir), "train")
    base = M.pack_root_of(ds)

    # ---- stage 2: augmentation on a FIXED resident batch ------------------
    if stage == "augment":
        raw = torch.randint(0, 255, (bs, base.stored_res, base.stored_res, 3),
                            dtype=torch.uint8)

        class _Fixed:
            batch_size = bs
            dataset = ds
            def __len__(self): return iters + 12
            def __iter__(self):
                for _ in range(iters + 12):
                    yield raw, torch.randint(0, n_cls, (bs,)), torch.arange(bs)

        gl = M.GPUBatchLoader(_Fixed(), dev, res, base.stored_res,
                              spec["mean"], spec["std"], train=True, seed=1)
        it = iter(gl)
        for _ in range(12):
            x, y, i = next(it)
            fwd_bwd(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 0
        for _ in range(iters):
            x, y, i = next(it)
            fwd_bwd(x, y)
            n += x.shape[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tm = gl.timing()
        return {"stage": stage, "img_s": n / dt, "ms_per_batch": 1000 * dt / iters,
                "augment_ms_per_batch": 1000 * tm["augment_s"] / max(1, tm["batches"])}

    # ---- stages 3 and 4: the real loader ----------------------------------
    arr = M.load_pack_to_ram(Path(data_dir), base.count, base.stored_res)
    if arr is None:
        return {"stage": stage, "error": "RAM cache declined"}
    rl = M.RAMBatchLoader(ds, arr, bs, shuffle=True, seed=1, pin=True)
    gl = M.GPUBatchLoader(rl, dev, res, base.stored_res, spec["mean"],
                          spec["std"], train=True, seed=1)

    dyn = None
    if stage == "full":
        dyn = M.TrainingDynamics(getattr(ds, "index_space", len(ds)),
                                 el2n_epoch=1)

    it = iter(gl)
    for _ in range(12):
        x, y, i = next(it)
        fwd_bwd(x, y)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        x, y, i = next(it)
        out, loss = fwd_bwd(x, y)
        if dyn is not None:
            # exactly what the training loop adds per batch
            import torch.nn.utils as _u
            _u.clip_grad_norm_(model.parameters(), float("inf"))
            dyn.observe_batch(i, out, y, 1)
            float(loss.item())
            int((out.argmax(1) == y).sum().item())
        n += x.shape[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tm = gl.timing()
    return {"stage": stage, "img_s": n / dt, "ms_per_batch": 1000 * dt / iters,
            "wait_ms_per_batch": 1000 * tm["wait_s"] / max(1, tm["batches"]),
            "augment_ms_per_batch": 1000 * tm["augment_s"] / max(1, tm["batches"])}


def run_isolated(stage, a):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--_child",
           "--stage", stage, "--arch", a.arch, "--data-dir", a.data_dir,
           "--batch-size", str(a.batch_size), "--res", str(a.res),
           "--iters", str(a.iters), "--vram-frac", str(a.vram_frac)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=dict(os.environ, MSC_OFFLINE="1"))
    try:
        so, se = p.communicate(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        p.kill(); p.communicate()
        return {"stage": stage, "error": f"TIMEOUT {a.timeout}s"}
    for line in so.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    tail = (se.strip().splitlines() or ["no result"])[-1]
    return {"stage": stage, "error": tail[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--vram-frac", type=float, default=0.50)
    ap.add_argument("--stage", default="compute")
    ap.add_argument("--_child", action="store_true")
    a = ap.parse_args()

    if a._child:
        print("__RESULT__" + json.dumps(
            _child(a.stage, a.arch, a.data_dir, a.batch_size, a.res,
                   a.iters, a.vram_frac)))
        return 0

    labels = {
        "compute":  "1. compute only        model fwd+bwd, synthetic resident tensor",
        "augment":  "2. + augmentation      GPUBatchLoader on a fixed batch",
        "data":     "3. + real data         RAM loader end to end",
        "full":     "4. + instrumentation   grad-norm, dynamics, .item() syncs",
    }
    print(f"""
{'='*76}
speed bisection   {a.arch}   batch {a.batch_size}   {a.res}px   AMP + channels_last
  each stage adds exactly ONE thing; the step that drops throughput is the cause
  isolated subprocesses, VRAM capped {a.vram_frac:.0%}, no batch ladder
{'='*76}""")

    res, prev = {}, None
    for st in STAGES:
        print(f"\n  {labels[st]}")
        r = run_isolated(st, a)
        res[st] = r
        if "error" in r:
            print(f"      FAILED: {r['error']}")
            continue
        line = f"      {r['img_s']:8.1f} img/s   {r['ms_per_batch']:7.1f} ms/batch"
        if prev:
            drop = prev - r["img_s"]
            line += f"   ({-100*drop/prev:+.0f}%)"
        print(line)
        for k, lbl in (("augment_ms_per_batch", "of which augment"),
                       ("wait_ms_per_batch", "of which waiting")):
            if k in r:
                print(f"          {lbl}: {r[k]:.1f} ms")
        prev = r["img_s"]

    print(f"\n{'='*76}")
    ok = {k: v["img_s"] for k, v in res.items() if "img_s" in v}
    if len(ok) >= 2:
        names = [s for s in STAGES if s in ok]
        worst, worst_drop = None, 0.0
        for i in range(1, len(names)):
            d = ok[names[i - 1]] - ok[names[i]]
            if d > worst_drop:
                worst, worst_drop = names[i], d
        base = ok[names[0]]
        print(f"  model ceiling on this card: {base:.0f} img/s")
        print(f"  observed in training:       80 img/s")
        if worst and worst_drop > 0.15 * base:
            print(f"\n  BIGGEST SINGLE LOSS: stage '{worst}' "
                  f"costs {worst_drop:.0f} img/s.")
            print(f"  That is the thing to fix. Nothing else is close.")
        elif base < 150:
            print("\n  The model ALONE cannot beat ~150 img/s here. Nothing in")
            print("  the input path is responsible -- this card at this batch")
            print("  size and resolution is the limit. The lever is batch size,")
            print("  resolution or architecture, not plumbing.")
        else:
            print("\n  No single stage dominates. Send this whole output.")
    print('='*76)

    out = ROOT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"bisect_{a.arch}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    f.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
