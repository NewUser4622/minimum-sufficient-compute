# Throughput benchmark — run this before the atlas

**One command. ~45 minutes. Send me the JSON it writes.**

```
python benchmark/bench_throughput.py --data-dir "D:\msc_data\in100"
```

No packed dataset yet? Run it anyway — everything except the loader stage works
on synthetic data:

```
python benchmark/bench_throughput.py --synthetic
```

Faster, coarser (~10 min): add `--quick`.
Single architecture: `--archs swin_tiny`.

---

## Why bother

The atlas is **~235 GPU-hours** by estimate. A 20% throughput win is 47 hours; a
40% win is 94. That is worth 45 minutes of measurement.

And it is worth *measuring* rather than assuming, because every number in
`20_IN100_PORT_PLAN.md` §6 is an estimate anchored on one guessed figure for
`resnet50`. The CIFAR programme's first cost table was **40% low**, and it only
found out by running (D-10). The benchmark ends by printing how far the plan was
off.

It also answers a question the plan cannot: **is this pipeline GPU-bound or
loader-bound, and does the answer differ by architecture?** If `resnet18` is
sitting at the loader ceiling, tuning the model buys nothing and the lever is
workers or the pack format. If everything is GPU-bound, worker count barely
matters and the lever is batch size and dtype.

---

## What it sweeps, and what it deliberately doesn't

**Not a full grid.** 8 architectures × 5 batch sizes × 5 worker counts × 2
dtypes × 2 layouts × 3 compile modes is 2,400 configurations and would take
longer than it saves. Instead a **staged coordinate descent**, each stage fixing
the winner of the last:

| stage | sweeps | why here |
|---|---|---|
| **A** | fp16 vs bf16 × channels_last on/off | 4 configs, biggest per-config effect, cheapest to test. Your card is Ada (sm_89) so **bf16 is native** — often the same speed as fp16 with better numerics and no loss-scale overflow |
| **B** | batch size 32 → 256, stopping at OOM | finds both the largest that fits and the fastest, which are not always the same |
| **C** | 0 → 16 dataloader workers, **with no model at all** | the loader's own ceiling. This is the number that says whether stages A/B/D matter |
| **D** | `torch.compile` at each winner | expensive to compile (30–90 s), so only at one config per architecture. Compile time is charged to warmup, not throughput |
| **E** | confirmation, longer run, from cold | the stage bests are each a local maximum found under slightly different conditions. A plan built by adding them up would be a *sum of measurements* rather than a measurement |

Coordinate descent can miss an interaction a grid would find. It is used anyway
because the interactions here are weak — batch size and worker count are close
to separable once neither is starving — and because a sweep that takes six hours
will not be run.

---

## Measurement details that change the answer

- **`torch.cuda.synchronize()` around every timed region.** CUDA is
  asynchronous; timing without a sync measures how fast Python can *submit* work
  to the queue, which is a number about Python.
- **Warmup before timing** — 6 to 15 steps. Covers CUDA context creation, cuDNN
  autotuning, and for `torch.compile` the entire graph capture. Without it a
  compiled model looks catastrophically slow.
- **Peak VRAM is reported per configuration**, so the chosen batch size has
  headroom rather than sitting one allocation away from an OOM at epoch 60.
- **OOM is caught and recorded, not raised.** The ladder stops climbing rather
  than crashing the sweep.
- **Stages A, B and D use GPU-resident synthetic batches**, so they measure the
  *model* with no loader in the way. Stage C measures the loader with no model
  in the way. Mixing them would give one number that explains nothing.

---

## What it writes

`benchmark/results/bench_<host>_<timestamp>.json`, containing:

- every configuration tried, with `img_s`, `peak_vram_gb`, and any error
- the chosen configuration per architecture
- the loader ceiling and the wait-vs-augment split
- a **re-planned run matrix**: seconds per epoch, hours per run, hours × 3
  seeds, and the atlas total — against measured throughput rather than a guess

Plus a printed table ending with a line comparing the measurement to the plan's
235-hour estimate.

**Send me that JSON.** I will rewrite `20_IN100_PORT_PLAN.md` §6, re-anchor
`ARCH_COST_HINT`, and set the per-architecture batch size and dtype in
`base_config` from it.

---

## What I expect to see, so you can tell if something is off

Rough priors for an RTX 4000 Ada at 224px. If the measurement is wildly
different from these, something is misconfigured and worth looking at before
trusting the sweep:

| arch | expected img/s | if much lower |
|---|---|---|
| `resnet18` | 700–1100 | probably loader-bound — check stage C |
| `shufflenetv2_in` | 600–1000 | depthwise convs are bandwidth-bound; low is plausible |
| `resnet50` | 350–500 | check `channels_last` won stage A |
| `vit_small_p16` | 300–450 | check bf16 is being used |
| `deit_small` | 300–450 | should be within a few % of `vit_small_p16` — **they are the same network**, so a large gap means a bug |
| `convnext_tiny` | 250–400 | 7×7 depthwise; sensitive to `channels_last` |
| `swin_tiny` | 200–350 | window attention has poor kernel efficiency; low is expected |
| `vgg16` | 120–200 | very compute-dense |

**`deit_small` and `vit_small_p16` being far apart is the one result that would
indicate a defect rather than a property of the hardware** — they are built by
one function with one argument set and differ only in recipe, which the
benchmark does not exercise.

---

## Two things the benchmark will *not* tell you

- **Whether the models train well.** It measures speed at a fixed number of
  steps on noise. A configuration can be fast and wrong — bf16 in particular
  changes numerics. The accuracy question is settled by Phase 0, not here.
- **Whether the chosen batch size is scientifically right.** Learning rate is
  scaled linearly from a reference batch, so changing batch size changes the
  recipe. If the benchmark picks a much larger batch than the plan's 128, that
  is a *recipe* change and needs to be applied uniformly across all eight
  architectures — otherwise batch size joins accuracy and family as a
  confounded variable, which is exactly the mistake the equal-epoch decision
  was made to avoid.
