# Study 4 — plan

**Nothing here has been run.** This closes the gap between "a correct result"
and "a result a Q1 journal will accept", identified in
[`../PAPER_CLAIM.md`](../PAPER_CLAIM.md).

Studies 1–3 are complete. Study 3's findings: [`../study3/04_FINDINGS.md`](../study3/04_FINDINGS.md).

---

## The claim being defended

> **Oracle upper bounds for early-exit inference exceed the accuracy of the
> network they bound, and the excess grows as the exits improve.**

Measured on CIFAR-100: **+6.86 pt** over full-compute accuracy in **90/90**
runs, rising to **8.55–10.64 pt** when exits are trained jointly. A learned
gate on the deployable signal recovers **1.7 %** of it.

**Three objections stand between that and an archival venue**, and each maps to
one phase below.

| objection | phase | cost |
|---|---|---|
| *"only one baseline — what about entropy, patience, learned policies?"* | **P1** | **free** (partly) |
| *"one dataset, 32px, no transformer"* | **P2** | ~20 GPU-h |
| *"your architecture is not the one the claim is about"* | **P3** | **~5 GPU-h** |
| *"no error bars, and your headroom curve is a table"* | **P0** | **free** |

---

## Phases, cheapest-decisive-first

```
P0  figures + bootstrap intervals        free, CPU minutes   ── submit A ──▶
P1  extra baselines (margin, patience)   free, CPU minutes
P2  ImageNet-100 + a transformer         ~20 GPU-h           ── gate ──▶
P3  MSDNet: a real early-exit network    ~5 GPU-h  (BUILT)
```

**P0 and P1 cost nothing and make Paper A submittable.** Do them, submit to
TMLR, then run P2 and P3 while it is under review.

### P0 — the free half of the paper · CPU minutes

Two things reviewers will ask for that we already have the data for.

**The ρ-sweep as Figure 1.** `s2_headroom_sweep.csv` already spans
ρ = 0.40–0.95. It is currently a table in the log. Plotted — honest headroom
negative across the entire range, with the in-seed bound drawn above it — one
figure carries the whole argument.

**Bootstrap intervals.** Study 3's Q1 used one seed per architecture, which is
correct (the quantity is a per-run identity) but leaves the joint numbers
without error bars. Bootstrap over the **10,000 test samples** is a legitimate
interval for a per-run quantity and costs nothing.

### P1 — the baselines we can actually build · CPU minutes

**Checked against the parquet before promising it:**

| baseline | needs | available? |
|---|---|---|
| **margin threshold** (`top1p − top2p` at exit *k*) | per-exit top-2 | **yes — free** |
| **patience / PABEE** (exit when *n* consecutive exits agree) | `pred_dk` | **yes — free** |
| learned per-exit gate | — | **already done** (Study 3 Q2) |
| **entropy threshold** | full per-exit softmax | **no** — only top-2 was stored |

Entropy is the one reviewers name most often and we cannot compute it honestly
from `top1p`/`top2p`. Two options, and the protocol picks the second:

1. re-run measurement storing full logits (~2 GPU-h, and 100× the parquet size);
2. **report margin and patience, and state plainly that entropy needs logits we
   did not store.** Margin is the closer relative of entropy anyway, and three
   baselines that all behave identically is a stronger argument than four.

### P2 — scale and a transformer · ~20 GPU-h

The standard objection, and the pipeline already exists (`notebooks_in100/`).

Joint-exit training on **ImageNet-100 at 224px**: `resnet50` and
`vit_small_p16`, one seed each. **A transformer matters** — every number we have
is convolutional, and early-exit work is now largely transformer work.

> **Cost warning, from the measured record.** `p0-resnet50-imagenet100-base-s1`
> took **41.5 GPU-h**. That is the D-59 `channels_last` run — the docs put the
> correct figure at ~6 h. `vit_small_p16` took 5.7 h. **The 41.5 h number must
> not be used for planning**, and P2 carries a throughput gate in epoch 1 that
> aborts if the layout regression has returned.

### P3 — a real early-exit architecture · ~5 GPU-h · BUILT, not yet run

Our exits are heads on a staged backbone. The field means **MSDNet** and
**BranchyNet**: multi-scale dense connections, exits designed in from the start.

**If the identity holds on MSDNet, the claim stops being "our setup" and becomes
"early-exit networks".** That is the single biggest scientific win available.

**It is also the riskiest item**, because it needs a new architecture in the zoo
rather than a new config flag — and this environment has no torch, so it ships
unexecuted. The dry-run and shape canaries exist for exactly that.

> **BUILT 2026-09-01 — run `notebooks_study4/S4_NB4_MSDNet.ipynb`.**
>
> 3 scales × 20 multi-scale dense layers, base 16, growth 6; exits at layers
> 4/8/12/16/20 with widths 160/256/352/448/544; trained **jointly**, because
> MSDNet trains all its classifiers jointly by design. The comparator is
> therefore Study 3's **joint** runs, not the frozen ones.
>
> **Cost revised: ~5 GPU-h, not ~15.** The old figure predated the
> architecture; at ≈ 0.24 GFLOPs it is ~2.5 h/seed. The notebook times epoch 1
> and extrapolates rather than trusting that.
>
> **The unexecuted-code risk is handled in cell 4, before any GPU time.** It
> verifies probed dims against the torch-free spec, that all five exits sit on
> the **coarsest** scale (8×8, not 32×32 — if they read the finest, MSDNet
> degenerates into the attached head it exists to be contrasted with), that
> `forward_prefix` matches `forward_features`, and — via forward hooks — that
> `forward_prefix(x, 0)` runs **4 of 20 layers** instead of computing
> everything and slicing. `tools/s4_msdnet_canaries.py` covers the arithmetic
> with 47 checks, each proven able to fail against a corrupted spec.
>
> **`msdnet` is a probe, not an atlas entry.** `atlas=False` keeps it out of
> `zoo_for_dataset` and out of every notebook's `measured_runs`, so the study
> population stays at 15 architectures. Without that guard, training it would
> have silently moved the published P0 intervals — D-90 in `03_LOG.md`.

---

## What this buys

| after | Paper A is |
|---|---|
| P0 + P1 | **submittable to TMLR** — correct, well-verified, honestly scoped |
| + P2 | credible at a JCR-Q1 journal (*Pattern Recognition*, *Neural Networks*) |
| + P3 | architecture-independent; TPAMI/IJCV becomes arguable |

**Total ~25 GPU-h**, of which the first two phases are free.

---

## Files

| file | what it is |
|---|---|
| [`01_PROTOCOL.md`](01_PROTOCOL.md) | pre-registration — predictions and thresholds before any run |
| [`02_RISKS.md`](02_RISKS.md) | how this fails, with detectors and responses |
| [`03_LOG.md`](03_LOG.md) | **live log** — machine facts, status board, what to do next |

---

## What Study 4 inherits

Study 3 cost ~16 GPU-h and produced two clean results and one honest null. It
also produced **six defects**, every one of the same family: *code written
against an assumed state*. D-87 (one flag, two defaults), D-88 (`stage` labels
but `fn` selects), a guard that grepped instead of testing, run ids assembled
instead of looked up, an embedding dump for checkpoints that were never
downloaded, and twice reading a truncated API response as evidence of absence.

**The rule Study 4 adds: open the artifact before writing code that needs it.**
Every phase below starts with a preflight cell that lists what it requires and
whether each item exists — and refuses, naming the gap, rather than failing five
cells later.
