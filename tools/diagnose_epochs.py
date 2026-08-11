#!/usr/bin/env python3
"""
diagnose_epochs.py -- where did the epoch time actually go?

Reads `metrics/epochs.csv` from a finished or in-flight run. Pure file read:
no GPU, no model, no allocation. Safe to run while training is going.

The 69 epochs already logged carry the answer to "is this GPU-bound or
input-bound", because D-40 split the timing into `dataload_time_sec`,
`augment_time_sec`, `compute_time_sec` and `backward_time_sec`, and NVML
recorded `gpu0_util_mean_pct` and `gpu0_power_mean_w` alongside. Guessing at a
bottleneck that has been sitting in a CSV for three days is not an approach.

How to read the verdict
-----------------------
  gpu0_util high, power near board limit, dataload_frac low
      -> GPU-bound. The work per image is too expensive. D-55 (memory format)
         is the first suspect; batch size is the second.
  gpu0_util low, dataload_frac high
      -> input-bound. Workers, memmap paging, or host RAM pressure.
  gpu0_util high but throughput still poor
      -> the GPU is busy doing the WRONG work. Layout conversion looks exactly
         like this, which is why it hid for 69 epochs.

USAGE
    python tools/diagnose_epochs.py                      # newest run found
    python tools/diagnose_epochs.py --run p0-resnet50-imagenet100-base-s1
    python tools/diagnose_epochs.py --results C:\\msc_results
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

DEFAULT_RESULTS = [Path(r"C:\msc_results"), Path(r"D:\msc_results"),
                   Path.cwd() / "msc_results"]


def _f(row, key, default=None):
    v = (row.get(key) or "").strip()
    if v in ("", "nan", "None"):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def summarise(csv_path: Path, run_id: str) -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        print(f"  {run_id}: epochs.csv is empty")
        return

    def col(k):
        return [v for v in (_f(r, k) for r in rows) if v is not None]

    print(f"\n{'='*74}\n  {run_id}   ({len(rows)} epochs logged)\n{'='*74}")

    ep = col("epoch_time_sec")
    tr = col("train_time_sec")
    va = col("val_time_sec")
    thr = col("throughput_train_img_s")

    if thr:
        print(f"  throughput   mean {st.mean(thr):7.1f} img/s   "
              f"min {min(thr):.1f}   max {max(thr):.1f}   "
              f"spread {max(thr)-min(thr):.1f}")
        if len(thr) > 3 and (max(thr) - min(thr)) < 0.05 * st.mean(thr):
            print("               FLAT across every epoch -- a fixed tax, not")
            print("               contention or thermals. Layout, batch size,")
            print("               or a per-batch serial cost.")
    if ep:
        print(f"  epoch time   mean {st.mean(ep):7.1f} s "
              f"({st.mean(ep)/60:.1f} min)")
    if tr and va:
        t, v = st.mean(tr), st.mean(va)
        print(f"    train      {t:7.1f} s  ({100*t/(t+v):.1f}%)")
        print(f"    val        {v:7.1f} s  ({100*v/(t+v):.1f}%)")

    print("\n  -- where the train time went " + "-"*44)
    parts = [("dataload_time_sec", "waiting for data"),
             ("augment_time_sec", "GPU augmentation"),
             ("compute_time_sec", "forward"),
             ("backward_time_sec", "backward"),
             ("optimizer_time_sec", "optimizer")]
    base = st.mean(tr) if tr else None
    any_part = False
    for k, label in parts:
        c = col(k)
        if not c:
            continue
        any_part = True
        m = st.mean(c)
        pct = f"{100*m/base:5.1f}%" if base else "    ?"
        print(f"    {label:22s} {m:8.1f} s   {pct}")
    if not any_part:
        print("    (no timing columns populated in this run)")

    for k, label, unit in [("dataload_frac", "dataload fraction", ""),
                           ("augment_frac", "augment fraction", ""),
                           ("step_time_mean_ms", "step time", " ms"),
                           ("step_time_p99_ms", "step p99", " ms")]:
        c = col(k)
        if c:
            print(f"    {label:22s} {st.mean(c):8.3f}{unit}")

    print("\n  -- GPU " + "-"*64)
    gu, gp, gm, gt = (col("gpu0_util_mean_pct"), col("gpu0_power_mean_w"),
                      col("gpu0_mem_used_mb"), col("gpu0_mem_total_mb"))
    if gu:
        print(f"    utilisation          {st.mean(gu):8.1f} %")
    if gp:
        print(f"    power                {st.mean(gp):8.1f} W")
    if gm and gt:
        print(f"    memory used          {st.mean(gm):8.0f} MB "
              f"of {st.mean(gt):.0f}  ({100*st.mean(gm)/st.mean(gt):.1f}%)")
    if not (gu or gp or gm):
        print("    (no NVML samples in this run)")

    print("\n  -- reading " + "-"*60)
    u = st.mean(gu) if gu else None
    d = st.mean(col("dataload_frac")) if col("dataload_frac") else None
    if u is not None and u >= 85 and (d is None or d < 0.15):
        print("    GPU-BOUND. The card is busy; the question is whether it is")
        print("    busy on useful work. A model in a different memory format")
        print("    from its input keeps the GPU at ~100% doing layout")
        print("    conversions. Run: python tools/verify_d55.py")
    elif d is not None and d >= 0.25:
        print(f"    INPUT-BOUND. {100*d:.0f}% of train time is waiting for")
        print("    batches. Look at num_workers, memmap paging and host RAM")
        print("    before touching the model.")
    elif u is not None and u < 60:
        print(f"    GPU idle {100-u:.0f}% of the time but dataload does not")
        print("    account for it -- suspect per-batch host-side serial work")
        print("    or a synchronisation in the metrics path.")
    else:
        print("    Mixed. Send this output back with the verify_d55.py result.")
    if gm and gt and st.mean(gm) < 0.45 * st.mean(gt):
        print(f"\n    Note: only {100*st.mean(gm)/st.mean(gt):.0f}% of VRAM is in")
        print("    use. Batch size has headroom -- but changing it changes the")
        print("    recipe and therefore comparability with finished runs.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()

    roots = ([Path(a.results)] if a.results else
             [p for p in DEFAULT_RESULTS if p.exists()])
    if not roots:
        print("no results root found. Pass --results C:\\msc_results")
        return 1

    found = []
    for root in roots:
        for csvp in sorted((root / "runs").glob("*/metrics/epochs.csv")):
            rid = csvp.parent.parent.name
            if a.run and rid != a.run:
                continue
            found.append((csvp, rid))

    if not found:
        print(f"no epochs.csv under {[str(r) for r in roots]}")
        return 1
    for csvp, rid in found:
        summarise(csvp, rid)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
