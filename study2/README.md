# Study 2 — complete

**Run, verified, and written up.** The result is in
[`PAPER.md`](PAPER.md); every number traces to a CSV in `analysis/`.

Study 1 (`docs/cifar100/`, `docs/imagenet100/`) stays as it is and is cited.

---

## The result in one line

> **Oracle upper bounds for early-exit routing are inflated by +22.41 accuracy
> points — more than the entire headroom they appear to show — and the excess is
> per-exit noise that does not survive a change of training seed.**

Across 15 CIFAR-100 architectures × 3 seeds (90 ordered seed pairs), at ρ = 0.80:

| | median |
|---|---|
| confidence baseline (deployable) | 62.39 % |
| full compute, final exit | 71.21 % |
| oracle scored from the **same** seed it routes | **78.30 %** (+12.20 pt) |
| oracle scored from a **different** seed | **54.50 %** (−7.90 pt) |
| **optimism bias** | **+22.41 pt** |

**0 of 15** architectures retain positive honest headroom, at any budget from
ρ = 0.40 to 0.95.

### The mechanism, exactly

The in-seed oracle beats the network's **own full-compute accuracy** in 100 % of
runs, by a median of **+6.86 pt**. The fraction of samples where an early exit is
right while the final layer is wrong is **6.86 %**. Identical, because at ρ = 0.80
the budget never binds and the oracle is simply *P(correct at any exit)*.

An oracle bound above full compute is not finding headroom — it is harvesting
samples the network gets right early and wrong at the end. That pool is noise:
unreachable by any router, and gone when the seed changes.

### Hypotheses as pre-registered

| | | outcome |
|---|---|---|
| **H1** | ρ_seed varies by ≥ 0.15 | **SUPPORTED** — range 0.667 |
| **H3** | in-seed oracle optimistic by ≥ 0.5 pt | **SUPPORTED for the true oracle** (+22.41 pt); **not supported** for per-sample difficulty scores |
| **H4** | corr(unreliability, bias) ≥ 0.5 | **NOT SUPPORTED** — ρ = +0.346, p = 0.002, right sign, below threshold |
| **H5** | nothing clears +1.0 pt honest headroom | **SUPPORTED** — every score negative at every budget |

**This confirms Study 1 rather than overturning it.** B11 measured +0.00007 and
was read as a possible MSC artifact. It was not — the debiased headroom is
negative across five scores and seven budgets, and Study 2 supplies the reason.

---

## Secondary findings

- **The four softmax scores are one score.** ρ = 0.997–1.000 between `msp`,
  `margin`, `entropy`, `ce_loss`. Eight candidates are **three families**; the
  grid's effective n is 3 × 15, not 8 × 15.
- **Difficulty scores collapse on memorised training data.** ce_loss ρ_seed
  falls 0.647 → **0.108** (`mixer_nano`), 0.673 → **0.116** (`vit_tiny`) between
  the test split and `train_holdout` — which is a slice *of* train. Low-capacity
  nets barely move (`mobilenetv2` 0.874 → 0.849). `forget_events` stays at 0.852.
  Dataset pruning computes exactly these scores, on exactly this data, from one
  seed. *(Saturation is the proposed mechanism; `S2_NB1` now tests it.)*

---

## What it cost

**No GPU beyond re-measuring FLOP budgets. No training. No new runs.** Every
per-sample parquet already carried per-exit predictions, so the whole study is
CPU re-analysis of files Study 1 had already published.

---

## Read in this order

| file | what it is |
|---|---|
| [`01_POSTMORTEM.md`](01_POSTMORTEM.md) | why Study 1 fell short — five diagnoses, each with evidence |
| [`02_PROTOCOL.md`](02_PROTOCOL.md) | the pre-registration: questions, hypotheses, gates, stopping rules |
| [`03_INVENTORY.md`](03_INVENTORY.md) | exactly what data exists and what it can answer without new runs |
| [`04_DESIGN.md`](04_DESIGN.md) | the plan, phase by phase, with costs |
| [`05_OPEN_DECISIONS.md`](05_OPEN_DECISIONS.md) | **what needs your call before anything starts** |
| [`06_RISK_REGISTER.md`](06_RISK_REGISTER.md) | **how this fails, and what stops it** — seven risks, each with a detector, trigger and response |
| [`PAPER.md`](PAPER.md) | **the write-up — read this first** |
| [`07_PROGRESS.md`](07_PROGRESS.md) | live log — newest first, updated every session |
| [`08_RELATED_WORK.md`](08_RELATED_WORK.md) | **what already exists and where the gap is** — the literature check, done |

---

## The rule this study is built around

> **Measure the ceiling before building the method.**

Study 1 spent 79 GPU-hours training 18 MSC-KD students to close a gap that a
two-hour oracle measurement showed did not exist. The ceiling was computed
*after* the method, and only because the method's failure forced the question.

In Study 2, every claim of the form "signal X could improve Y" is preceded by
"what is the best X could possibly do", and that measurement is a gate: if the
ceiling is flat, the method is not built.

---

## What actually went wrong

Kept here because the postmortem is the point of this folder.

**The measurement was implemented wrong three times**, each time producing a
plausible number that had to be attacked rather than believed:

1. **+5.165 pt** — the "oracle" was `pred_depth` and the baseline thresholded
   the *final* exit's confidence. Both need a full forward pass, so it compared
   two oracles.
2. **−10 pt** — the baseline was fixed, but it could choose any exit histogram
   while difficulty scores were forced through a rigid quantile spread. That
   measured the mechanism, not the signal.
3. **−8 pt** — histograms matched, but a per-*sample* score was being compared
   against a per-*exit* baseline. The baseline knew "am I right at exit k"; the
   score only knew "is this sample hard". Better-informed side won.

The fourth implementation is a from-scratch Lagrangian maximum over every
assignment meeting the budget, which dominates any router by construction. It
passes 18 canaries including *identical seeds → bias exactly 0.0000* and
*independent seeds → +46.4 pt*.

**Two contamination bugs were found by checking artifacts rather than plans:**
`p0`/`p1` pilot pairs were the *same seed* run twice (49 → 45 runs, 118 → 90
pairs), and P0a's correlation matrix was computed on a run that deduplication
had already dropped.

**A statistic reported a conclusion from zero samples.** P0a printed `(n=0)`, an
all-NaN matrix, and "the scores carry distinct information", because a listwise
NaN mask dropped every row. Every statistic now carries a canary proving it can
report a wrong answer.

**The lesson that generalises:** the study's headline is "headroom ≈ 0", and a
broken instrument reports ≈ 0 too. The canary that mattered most builds a world
where headroom certainly exists and requires the measurement to find it
(+49.3 pt). Without it, three of the four wrong implementations above would have
produced a publishable-looking null.
