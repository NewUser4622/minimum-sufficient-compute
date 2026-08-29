# What we can claim, and what it takes to publish it

**Honest assessment, written from the measured results in
[`RESULTS.md`](RESULTS.md).** Every number here is in a CSV on
[HuggingFace](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100).

Two papers are available. **Paper A is close.** Paper B needs one more
experiment. The gap analysis at the end is the part worth acting on.

---

## PAPER A — the main claim

> ### Oracle upper bounds for early-exit inference exceed the accuracy of the network they bound, and the excess grows as the exits improve.

### The one-paragraph version

Early-exit papers motivate adaptive inference by reporting an oracle bound: the
accuracy obtainable if each sample exited at the ideal layer. That oracle is
computed from the network being routed. We show it **exceeds that network's own
full-compute accuracy** — by 6.9 points on CIFAR-100, in 100 % of 90 runs — and
that the excess is exactly the fraction of samples some early exit gets right
while the final layer gets them wrong. Counter-intuitively the excess **grows**
when exits are trained jointly rather than post-hoc (8.6–10.6 pt), because
rescuing a sample the final layer fails requires a *competent* early exit. Using
a second training seed as an instrument, we show the excess does not survive a
change of seed (optimism bias +22.4 pt), and that a learned per-exit gate reading
the deployable signal recovers **1.7 %** of it. The headroom that motivates
per-sample adaptive inference is therefore largely an artifact of scoring a
network against itself.

### Why this is a real contribution, not a straw man

The target is named and current. A survey defines the oracle as *"the shallowest
internal classifier that provides a correct label prediction"* — precisely what
we implement. DE3-BERT draws the inference we challenge, explicitly:

> the oracle outperforms the backbone model and existing exiting strategies by a
> large margin… which indicates **significant room for improving the estimation
> of prediction correctness**

Oracle-above-backbone → headroom. That is the reading this paper argues is
unsafe.

### The four measurements, and how strong each is

| # | claim | evidence | strength |
|---|---|---|---|
| **1** | the bound exceeds the network's own full accuracy | +6.86 pt, **90/90 runs**, exact identity with the early-right/final-wrong pool | **very strong** — arithmetic, needs no seeds |
| **2** | it **grows** with better exits | frozen 6.4–8.0 → joint 8.6–10.6 pt, 3/3 architectures; survives conditioning on accuracy (+2.49 median) | **strong** — paired, one variable, mechanism explained |
| **3** | it does not survive a change of seed | optimism bias **+22.4 pt**, 90 pairs | **medium** — see the caveat below |
| **4** | a deployable gate recovers almost none of it | **1.7 %** cross-seed; in-seed no higher, so not noise-memorisation | **strong**, and it is a lower bound |

**Claim 1 is the paper's spine.** It needs one trained network and its own
per-exit predictions. No seeds, no instrument, no assumptions. A bound above
full compute cannot be reached by any router — that is not an empirical finding,
it is what the number means.

**Claim 2 is what makes it publishable rather than pedantic.** The obvious
rebuttal to Claim 1 is *"your exits were weak"*. We tested it and the effect went
the **other** way, with a mechanism: weak exits are right on almost nothing, so
they cannot rescue anything; rescues require competence. **Pre-registered before
the runs**, and a free extrapolation from existing data predicted the direction
and magnitude (+10.98 predicted at exit_quality 0.86; 8.55–10.64 measured at
0.75–0.92).

**Claim 3 is the weakest and must be presented as decomposed.** +22.4 splits
into +6.9 (the exact identity) and +15.3 (cross-seed non-transfer). The second
part conflates "the signal does not transfer" with "acting on a non-transferring
signal is worse than not acting". A reviewer will press this. Report the
decomposition, lead with the +6.9.

### Verification, which is a selling point

Reviewers of a limits paper ask *"how do you know your measurement is right?"* We
have an unusually good answer:

- **96 canaries** across three suites, each required to be able to *fail*
- the load-bearing one: the statistic must **detect** an effect in a world where
  the effect certainly exists — because "no headroom" and "cannot see headroom"
  produce identical output
- identical seeds → bias **exactly 0.0000**; independent seeds → +46.4 pt
- the in-seed oracle is asserted ≥ baseline on every run and holds **90/90**
- **461-check library selftest**
- the routing measurement was implemented wrong **three times** (+5.165, −10, −8
  pt), each producing a plausible number, and each was caught by a canary rather
  than by reading

That history is worth a short "Threats to validity" subsection. It converts an
embarrassment into evidence that the final number was hard to get wrong.

### Where to send it

| venue | fit | realistic? |
|---|---|---|
| **TMLR** | limits/critique work is explicitly in scope; no novelty bar | **best first target** — submit close to as-is |
| **Pattern Recognition** / **Neural Networks** (Q1, Elsevier) | empirical-analysis papers welcome | yes, **after** the scale gap is closed |
| **IEEE TPAMI / IJCV** (Q1) | needs breadth we do not yet have | not without ImageNet + a transformer |
| **NeurIPS/ICML D&B or main** | possible, but reviewers want a method | riskier than TMLR |

**Recommendation: TMLR first.** It is the natural home for "the field's
measurement is wrong and here is the correction", acceptance is on correctness
rather than novelty, and it is indexed and respected. If a JCR-Q1 journal is
specifically required, target *Pattern Recognition* **after** the ImageNet
experiment below.

---

## The gap analysis — what stands between this and a Q1 journal

Ordered by how much each buys per GPU-hour.

### G1 · A real early-exit architecture · ~15 GPU-h · **the biggest single win**

Our exits are heads on a staged backbone, trained jointly. The field uses
**MSDNet** and **BranchyNet**, which are architecturally different (dense
multi-scale connections, exits designed in from the start).

**Reviewer question:** *"does your identity hold on the architectures the claim
is about?"* Right now: unknown.

**Experiment.** Implement MSDNet on CIFAR-100, 2 seeds, measure the same
identity. It is a per-run quantity so even one run answers it.
**If the excess persists, Claim 1 becomes architecture-independent** and the
paper's scope widens from "our setup" to "early-exit networks".

### G2 · Scale · ~25 GPU-h · **closes the standard objection**

CIFAR-100 at 32px only. You already have an ImageNet-100 pipeline and 2
architectures measured.

**Experiment.** Joint-exit training on ImageNet-100, `resnet50` +
`vit_small_p16`, 1 seed each. **A transformer matters** — every claim we have is
convolutional, and early-exit work is now largely transformer work (DE3-BERT is
BERT). Reuse `notebooks_in100/`.

### G3 · Baselines beyond confidence thresholding · ~2 GPU-h · **cheapest credibility**

We compare against one baseline. Reviewers will name **entropy thresholding**,
**patience-based exiting (PABEE)**, and **learned exit policies**.

**Experiment.** Entropy and patience baselines are pure re-analysis of the
existing parquets — **no training at all**. A learned policy is Q2's gate, done.
This is nearly free and removes an easy criticism.

### G4 · The operating-point curve as a figure · **free**

`s2_headroom_sweep.csv` already has ρ = 0.40–0.95. It is currently a table.
**Make it Figure 1**: honest headroom versus compute budget, negative across the
whole range, with the in-seed bound plotted above it. One figure carries the
argument.

### G5 · Confidence intervals · **free**

Q1's joint runs are one seed per architecture. The identity does not need seeds,
but reviewers expect error bars. Bootstrap over the **10,000 test samples** —
that is a legitimate interval for a per-run quantity and it costs nothing.

**Total to Q1-ready: ~42 GPU-h**, of which G3–G5 are free or nearly so.

---

## PAPER B — the independent second paper

> ### Softmax difficulty scores are unmeasurable on data the network has memorised.

`ce_loss` seed-reliability falls from **0.647 on held-out data to 0.108** on
training data the network has fit (`mixer_nano`; `convnext_femto` 0.709 → 0.150).
The collapse is predicted by **softmax saturation (ρ = +0.832)** and by train
accuracy (+0.746) — and **not** by test accuracy (**−0.114**), so it is not
"weaker models are noisier". Training-dynamics scores (`el2n`,
`forget_events`) are unaffected.

**Why it matters outside our setting.** Dataset pruning, coreset selection and
curriculum learning compute exactly these scores, on exactly this data, usually
from a single seed — and for the high-capacity models where the collapse is
worst.

**Strength: high.** A large effect, a confirmed mechanism, and a control that
rules out the obvious alternative.

**What it still needs.** A demonstration that the collapse causes *downstream*
harm. Our attempt (Study 3 Q3) is **confounded**: it used "keep the hardest
samples" at 30 % retention, where that rule is known to be poor, so a *more
reliable* score keeps a harder, noisier subset and trains worse. Both guided arms
lost to random — the tell.

**Fix: ~6 GPU-h.** Rerun with the rule inverted (keep easiest) or at mild
retention (70–90 %), reporting both directions. Subsets are index lists, so only
retraining costs.

**With that, Paper B is a standalone Q1 submission** and does not depend on
Paper A being accepted.

---

## What we must not claim

Rule 12 applies to our own manuscript.

- **Not** "adaptive inference does not work." We measured one bound at one
  scale, and Q2's gate is a lower bound.
- **Not** "the excess cannot be reached by any router." A second seed cannot,
  and our gate on exit-local confidence cannot. A gate with pooled embeddings is
  untested.
- **Not** "reliable scores prune worse." Q3 is confounded; the ordering is
  suggestive at best.
- **Not** the +12.45 pt extrapolated figure from P0 — it sits outside the
  observed `exit_quality` range.
- **Not** a magnitude claim across scales. Everything is CIFAR-100.

---

## Recommended sequence

1. **G4 + G5 + G3** (free / ~2 GPU-h) → **submit Paper A to TMLR.**
2. **Q3 rerun** (~6 GPU-h) → **submit Paper B.**
3. **G1 MSDNet** (~15 GPU-h) → the architecture-independence result.
4. **G2 ImageNet + transformer** (~25 GPU-h) → resubmit A to a JCR-Q1 journal.

Steps 1 and 2 need almost no compute and produce two submissions. Steps 3 and 4
are what turn "correct and careful" into "broad enough for TPAMI".

**The honest summary: you have a publishable result now (TMLR), and ~42 GPU-h
between you and a credible Q1 journal submission.**
