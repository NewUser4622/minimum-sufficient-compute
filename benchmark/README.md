# Throughput benchmark

**~17 minutes. Send me the JSON it writes.**

---

## The batch-size ladder is gone

It was the dangerous stage **and** a useless one, and the second part is why it
was removed rather than made safer.

> All eight architectures must train at the **same batch size**, or batch size
> joins accuracy and family as a confounded variable. Learning rate is scaled
> linearly from a reference batch, so a per-architecture batch "winner" would
> mean eight different recipes — and the seed-reliability comparison would be
> measuring the recipe as much as the architecture.

There was never anything to *do* with a per-architecture optimum. The sweep
risked the machine to produce a number that could not have been used.

What is actually needed: how fast does each architecture train at the batch size
we are going to use, and does it fit with room to spare. Both are answerable at
a **fixed batch of 64**, and headroom is now **predicted from measured peak
VRAM** rather than probed:

```
peak 4.2 GB at batch 64  ->  largest estimated batch 182   safe: [32, 64, 96, 128]
peak 9.1 GB at batch 64  ->  largest estimated batch  84   safe: [32, 64]
```

Activation memory is close to linear in batch size, so measuring 64 tells you
what 128 would cost **without ever allocating it** — and allocating it is what
took the machine down. *Predict, don't probe.*

`torch.compile` is also removed: it is the thing most likely to hang on Windows.

| | before | now |
|---|---|---|
| batch sizes tried | up to **256** | **64 only** (hard ceiling 96) |
| configurations | ~80 | **~29** |
| wall time | ~45 min | **~17 min** |
| VRAM cap | none | **50%** — 10 GiB always free |
| timeout | none | **120 s**, kills the process tree |
| `torch.compile` | tried | **removed** |
| workers ceiling | 16 | **8** |

---

## Run it, in this order

```
python benchmark/bench_throughput.py --plan-only
```

Executes **nothing**. Prints the VRAM cap in GiB, the batch size, the timeout,
the configuration count and the worst case. Read it before anything runs.

```
python benchmark/bench_throughput.py --quick --vram-frac 0.4
```

~6 minutes, 40% of the card — the most cautious setting available. **If the
display stutters at all, stop and tell me.**

```
python benchmark/bench_throughput.py --data-dir "D:\msc_data\in100"
```

The full run, once you trust it.

| flag | effect |
|---|---|
| `--plan-only` | print the plan, run nothing |
| `--quick` | one variant per architecture (~6 min) |
| `--vram-frac 0.4` | more headroom for the display |
| `--timeout 60` | kill a config sooner |
| `--archs swin_tiny` | one architecture only |
| `--synthetic` | skip the loader stage |
| `--batch N` | capped at 96 whatever you pass |

---

## What remains, and why each is safe

**dtype × memory layout at batch 64** — 1–3 short runs per architecture. Real
10–40% effects, and at batch 64 even `vgg16` sits at a few GB. Your card is Ada
(sm_89) so **bf16 is native**: often the same speed as fp16, with better
numerics and no loss-scale overflow.

**the loader ceiling** — measured with **no model on the GPU at all**, so it is
the least risky part of the run. It is also the only way to learn whether
`resnet18` and `shufflenetv2` are data-bound, which no amount of model tuning
would reveal. Worth keeping unless you have a reason not to.

---

## Safety mechanisms, all exercised

| mechanism | what it prevents |
|---|---|
| `set_per_process_memory_fraction(0.50)` | an oversized allocation raises a **catchable Python OOM** instead of starving the display driver into a TDR reset |
| every config in **its own subprocess** | a child that OOMs, hangs or dies takes nothing with it; the OS reclaims its CUDA context and workers on exit |
| **timeout kills the process *tree*** | an orphaned worker holding the 24 GiB memmap is how the *next* config runs out of RAM |
| results written **after every config** | an interruption costs one data point, not the run |

Verified rather than asserted: a failing child returns a structured result with
the parent alive; a deliberately hung child is killed at the timeout; the
results file is valid after every append with no `.tmp` left behind.

---

## Why bother at all

The atlas is **~235 GPU-hours** by estimate. A 20% throughput win is 47 hours.

More to the point, every number in `20_IN100_PORT_PLAN.md` §6 is an estimate
anchored on one guessed figure for `resnet50`. The CIFAR programme's first cost
table was **40% low** and only found out by running (D-10). This one ends by
printing how far the plan was off.

---

## What I expect to see

Rough priors for an RTX 4000 Ada at 224px, batch 64. Wildly different numbers
suggest something is misconfigured rather than a property of the hardware.

| arch | expected img/s | if much lower |
|---|---|---|
| `resnet18` | 700–1100 | probably loader-bound — check the loader stage |
| `shufflenetv2_in` | 600–1000 | depthwise convs are bandwidth-bound; low is plausible |
| `resnet50` | 350–500 | check `channels_last` won |
| `vit_small_p16` | 300–450 | check bf16 is being used |
| `deit_small` | 300–450 | should be within a few % of `vit_small_p16` |
| `convnext_tiny` | 250–400 | 7×7 depthwise; sensitive to `channels_last` |
| `swin_tiny` | 200–350 | window attention has poor kernel efficiency |
| `vgg16` | 120–200 | very compute-dense |

**`deit_small` and `vit_small_p16` far apart is the one result that would
indicate a defect** rather than hardware. They are built by one function with
one argument set and differ only in training recipe, which this does not touch.

---

## Two things it will not tell you

- **Whether the models train well.** It measures speed on noise at a fixed step
  count. A configuration can be fast and wrong — bf16 changes numerics. Accuracy
  is settled by Phase 0, not here.
- **Whether to raise the batch size.** The headroom figure is an estimate from a
  linear model. If you want a larger batch, step **one notch** and re-run this
  — do not jump to the largest number it prints. And it would have to be applied
  to all eight architectures uniformly, for the reason at the top of this file.

---

## What went wrong the first time

The original swept to batch 256 at 224px on a 20 GB card that also drives the
display, in-process, with no VRAM cap and no timeout. That starves the desktop
compositor; when the driver stops responding for more than ~2 s Windows fires a
TDR reset and the machine hangs.

The reasoning error, not just the coding one: I wrote a tool whose purpose is to
find the limits of the hardware and did not treat approaching those limits as
dangerous. Catching `OutOfMemoryError` felt like sufficient handling — but a
caught OOM is the *benign* case. The damaging case is the allocation that
**succeeds** and leaves the display driver with nothing, which no exception
handler ever sees.

Full write-up, including what it cost: **D-41** in
[`22_IN100_LAB_NOTEBOOK.md`](../docs/imagenet100/22_IN100_LAB_NOTEBOOK.md).
