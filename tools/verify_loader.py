#!/usr/bin/env python3
"""
verify_loader.py -- is the input pipeline the bottleneck, and does RAM fix it?

WHY
---
ResNet-50 ran at a flat 80 img/s (0.84 s per batch of 64) while the model needs
roughly 0.07 s of that. Fixing the memory format (D-55) changed nothing, which
is itself the finding: a compute fix that buys nothing means compute was never
the limit.

The per-sample path did one random 192 KiB read from a 24 GiB file per image,
64 times a batch, then shipped 12.6 MiB through Windows IPC. That is ~15 MiB/s
effective -- spinning-disk numbers, not SSD.

This measures the loader ALONE. No model, no optimizer, no autograd. If the
loader on its own cannot beat the full training loop's 80 img/s, then the
loader is the ceiling and nothing done to the model matters.

SAFETY
------
Reads only. The RAM arm allocates the pack, so it first asks
`ram_budget_ok()` and refuses rather than risk paging the machine (D-41). Use
`--skip-ram` to measure only the current path. GPU is not touched unless
`--cuda` is passed, and then only for a host-to-device copy.

USAGE
    python tools/verify_loader.py --data-dir C:\\msc_data\\in100
    python tools/verify_loader.py --data-dir C:\\msc_data\\in100 --batches 60
    python tools/verify_loader.py --data-dir C:\\msc_data\\in100 --skip-ram
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-ram", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import msc_lib as M

    root = Path(a.data_dir)
    bs, nb = a.batch_size, a.batches

    print(f"""
{'='*72}
loader-only throughput   batch {bs}   {nb} batches   no model, no autograd
  data {root}
{'='*72}""")

    ds = M.PackedImageDataset(root, "train")
    n_img = len(ds)
    base = M.pack_root_of(ds)
    nbytes = base.count * base.stored_res * base.stored_res * 3
    print(f"  split holds {n_img:,} images; pack is {nbytes/2**30:.1f} GiB "
          f"at {base.stored_res}px\n")

    results = {}

    # ---- arm 1: the current path -----------------------------------------
    print(f"  [1] memmap + {a.workers} workers  (what is running now)")
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=a.workers,
                    pin_memory=True, persistent_workers=bool(a.workers),
                    prefetch_factor=(4 if a.workers else None))
    it = iter(dl)
    for _ in range(3):                                     # warm the workers
        next(it)
    t0 = time.perf_counter()
    got = 0
    for _ in range(nb):
        x, y, i = next(it)
        if a.cuda:
            x = x.cuda(non_blocking=True)
        got += x.shape[0]
    if a.cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    results["memmap"] = got / dt
    print(f"      {got/dt:8.1f} img/s   {1000*dt/nb:6.1f} ms/batch   "
          f"{got*base.stored_res**2*3/dt/2**20:.0f} MiB/s\n")
    del it, dl

    # ---- arm 2: resident ---------------------------------------------------
    if not a.skip_ram:
        ok, why = M.ram_budget_ok(nbytes)
        print(f"  [2] RAM-resident  ({why})")
        if not ok:
            print("      DECLINED -- not enough free RAM. Close things, or run")
            print("      with --skip-ram and we solve this a different way.\n")
        else:
            arr = M.load_pack_to_ram(root, base.count, base.stored_res)
            if arr is None:
                print("      load declined\n")
            else:
                rl = M.RAMBatchLoader(ds, arr, bs, shuffle=True, seed=1,
                                      pin=torch.cuda.is_available())
                it = iter(rl)
                for _ in range(3):
                    next(it)
                t0 = time.perf_counter()
                got = 0
                for _ in range(nb):
                    x, y, i = next(it)
                    if a.cuda:
                        x = x.cuda(non_blocking=True)
                    got += x.shape[0]
                if a.cuda:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                results["ram"] = got / dt
                print(f"      {got/dt:8.1f} img/s   {1000*dt/nb:6.1f} ms/batch   "
                      f"{got*base.stored_res**2*3/dt/2**20:.0f} MiB/s\n")

    # ---- verdict -----------------------------------------------------------
    print('='*72)
    mm = results.get("memmap")
    rm = results.get("ram")
    OBSERVED = 80.0
    if mm:
        print(f"  Your training loop achieves {OBSERVED:.0f} img/s end to end.")
        print(f"  The loader alone delivers   {mm:7.1f} img/s.")
        if mm < OBSERVED * 1.6:
            print("\n  The loader cannot go much faster than the whole training")
            print("  loop does. It IS the ceiling -- the GPU is idle waiting.")
        else:
            print("\n  The loader has headroom over the observed rate, so the")
            print("  input path is not the whole story. Send this output.")
    if mm and rm:
        print(f"\n  RAM-resident is {rm/mm:.1f}x the memmap path "
              f"({mm:.0f} -> {rm:.0f} img/s).")
        ceiling = min(rm, 100000)
        print(f"  With the loader no longer limiting, the run should be bound")
        print(f"  by the model instead. Expected epoch time then depends on")
        print(f"  compute, not disk -- rerun tools/verify_d55.py for that half.")
        if rm < 300:
            print("\n  NOTE: even resident, this is under 300 img/s. That would")
            print("  point at the gather or the pinning rather than the disk.")
    print('='*72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
