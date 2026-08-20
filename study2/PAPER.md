# Oracle upper bounds for early-exit routing are inflated by per-exit noise

**Status: draft. Every number below is measured and traceable to a CSV in
`analysis/`. One claim (§4, the saturation mechanism) is a hypothesis with a
test attached that has not been run yet, and is marked as such.**

---

## Abstract

Early-exit papers routinely report an oracle upper bound: the accuracy
obtainable if each sample exited at the ideal layer. That oracle is computed
from the very network being routed, so it can exploit the network's own
per-exit noise. Using a second training seed as an independent instrument, we
measure how much of the bound is noise. Across 15 CIFAR-100 architectures ×
3 seeds (90 ordered seed pairs), the in-seed oracle sits **+12.20 points**
above a deployable confidence baseline, but a cross-seed oracle sits
**−7.90 points** below it. The optimism is **+22.41 accuracy points** — larger
than the entire apparent headroom, in **0 of 15** architectures leaving any
positive honest ceiling, and negative at every compute budget from ρ = 0.40 to
0.95. We identify the mechanism exactly: the in-seed oracle's excess over the
network's own full-compute accuracy equals, to numerical precision, the
fraction of samples where some early exit is correct while the final layer is
wrong. That pool is per-exit noise; it does not survive a change of seed, and
no deployable router can reach it.

We also report a reliability atlas for five per-sample difficulty scores across
the same grid, and two secondary findings: the four softmax-derived scores are
**one signal** (ρ = 0.997–1.000), and score reliability **collapses on training
data the network has fit** (ce_loss ρ_seed 0.647 → 0.108 for `mixer_nano`),
which is precisely where the dataset-pruning literature computes them.

---

## 1. What is being claimed

The oracle early-exit bound in common use is:

> exit at the first layer whose prediction matches that of the last layer —
> an ideal upper bound for how much computation could be saved.

Every quantity in it comes from one trained network. Our claim is not that this
is arithmetically wrong; it is that **it is an upper bound on the wrong thing**.
It bounds what a router could achieve *if it had access to this network's own
per-exit correctness*, and no deployable router does.

The instrument is a second training seed. Same architecture, same recipe,
different initialisation:

```
in-seed   oracle : exits chosen from seed i's correctness, evaluated on seed i
cross-seed oracle: exits chosen from seed j's correctness, evaluated on seed i
optimism bias    = in-seed − cross-seed
```

The bias is the part of the bound that depends on *which* network was trained,
rather than on which samples are hard.

## 2. Method

**Data.** Study 1's 45 measured CIFAR-100 runs — 15 architectures × 3 seeds,
5 depth exits each. Every run's `per_sample/test.parquet` carries `pred_d1..d5`,
`top1p_d*`, `top2p_d*` and `label` for all 10,000 test samples, so per-exit
correctness is `pred_dk == label` and costs come from `budgets/{arch}.json`.
**The entire study is CPU re-analysis; no model is loaded and nothing is
retrained.**

Two architectures (`resnet32x4`, `wrn_40_2`) additionally had pilot runs at
seeds 1–2 with byte-identical configs. Those are *replicates* of an existing
seed, not extra seeds, and pooling them would have let 2 of 15 architectures
supply 40 of 118 ordered pairs. They are deduplicated on `(arch, seed)`, leaving
**45 runs and 90 ordered seed pairs**.

**Baseline.** Confidence thresholding on the **early** exit: a sample leaves at
the first exit whose top-1 probability clears a threshold, bisected to meet the
budget. This is deployable — it reads confidence at the exit it is deciding at.

**Oracle.** The maximum over *every* assignment meeting the budget, by
Lagrangian relaxation: choose `k` per sample maximising
`correct[i,k] − λ·ρ[k]`, bisect λ to the budget, then spend any residual budget
greedily. Being a maximum over all assignments, it dominates any particular
router — which is what makes it a bound. `correct_choose` selects the exits and
`correct_eval` scores them, so the same routine yields both the in-seed and the
cross-seed oracle.

**Verification.** Every statistic carries a canary that must be able to fail
(`tools/s2_routing_canaries.py`, 18/18; `tools/s2_canaries.py`, 3/3). The
load-bearing ones:

| canary | why it exists |
|---|---|
| in-seed oracle never loses to the baseline, 200 adversarial draws | a maximum cannot be beaten; if it is, the harness is broken |
| identical seeds → bias **exactly** 0.0000 | the statistic must not manufacture bias from a model compared with itself |
| independent seeds → **+46.4 pt** | and it must be able to see bias when it is there |
| detects real headroom when it exists (**+49.3 pt**) | "headroom ≈ 0" and "cannot see headroom" produce the same number |

The notebook additionally asserts `in-seed oracle ≥ baseline` on every
(architecture, seed) pair and raises if violated. It holds on **90/90**.

## 3. Result

All figures are medians over 90 ordered seed pairs, CIFAR-100, ρ = 0.80,
from `analysis/s2_true_oracle.csv`.

| quantity | median | IQR |
|---|---|---|
| confidence baseline | 62.39 % | [60.20, 68.84] |
| full compute, final exit | 71.21 % | [67.57, 73.00] |
| **oracle, in-seed** | **78.30 %** | [76.11, 79.69] |
| **oracle, cross-seed** | **54.50 %** | [52.13, 60.07] |

| difference (per-run medians) | value |
|---|---|
| in-seed − baseline (the published-style bound) | **+12.20 pt** |
| cross-seed − baseline (**honest**) | **−7.90 pt** |
| **optimism bias** | **+22.41 pt** |

*(Levels and differences are both medians, so they do not subtract: a median of
differences is not a difference of medians.)*

**The bias exceeds the entire apparent headroom.** The honest ceiling is not
merely smaller — it is negative, in **0 of 15** architectures positive, ranging
from +15.93 pt (`resnet8x4`) to +32.18 pt (`mixer_nano`) of bias.

**It is not an artifact of the operating point.** Sweeping the budget
(`s2_headroom_sweep.csv`), honest headroom is negative everywhere:

| ρ | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| best deployable score | −5.68 | −8.48 | −9.99 | −9.09 | −6.30 | −3.35 | −1.34 |
| `pred_depth` (oracle-only) | −4.00 | −4.96 | −4.60 | −3.91 | −3.06 | −1.72 | −0.81 |

### 3.1 The mechanism, exactly

The in-seed oracle exceeds the network's **own full-compute accuracy** in
**100 %** of runs, by a median of **+6.86 pt**. The fraction of samples where
some early exit is correct while the final layer is wrong is **6.86 %**.

These are the same number because, at ρ = 0.80, **the budget never binds**:
`oracle_in == acc_full + frac_early_saves` exactly, on all 90 rows. The in-seed
oracle is simply *P(correct at any exit)*.

That identity is the whole result. An oracle bound above full compute is not
finding headroom — it is collecting samples the network gets right early **and
wrong at the end**. That is per-exit noise. It cannot be predicted by any
router, and it does not transfer across seeds, which is exactly why the
cross-seed instrument removes it.

**A consequence for how this must be described:** because the budget is
inactive, the in-seed oracle *spends less than the baseline while scoring
higher*, so this is **not** a matched-FLOPs comparison and we do not call it
one. It remains a valid upper bound, and a conservative one — the baseline at
the oracle's lower cost would be worse still.

### 3.2 Relation to Study 1

Study 1's B11 baseline gave a router the student's own true post-hoc MSC and
measured **+0.00007** over confidence thresholding — no headroom. That was a
single metric at a single operating point, and was read as a possible artifact
of MSC. It was not: with the true per-exit oracle, across five scores and seven
budgets, the debiased headroom is negative everywhere. **The two studies
agree**, and Study 2 supplies the reason Study 1's number was near zero.

## 4. Secondary findings

### 4.1 The four softmax scores are one score

From `s2_reliability_grid.csv` and the P0a matrix:

| | msp | margin | entropy | ce_loss |
|---|---|---|---|---|
| **msp** | 1.00 | 0.997 | 0.999 | 1.000 |
| **margin** | | 1.00 | 0.992 | 0.997 |
| **entropy** | | | 1.00 | 0.999 |

Eight candidate scores are **three families**: `{msp, margin, entropy,
ce_loss}`, `{el2n, forget_events}` (ρ = 0.58), and `{pred_depth}`. Any "N
scores" claim over this battery — ours included — should be read as N families,
and the grid's effective n is 3 × 15, not 8 × 15. A curriculum-learning survey
reports difficulty functions agreeing ">70 % in all but one case"; measured
here, it is 99.7–100 %.

### 4.2 Reliability varies by 0.667 across the grid

ρ_seed on the test split spans **0.207** (`mixer_nano`, entropy) to **0.874**
(`mobilenetv2`, ce_loss). `ce_loss` is the most reliable score (mean 0.779) and
the most architecture-stable of the softmax family (range 0.228); `entropy` is
the least stable (range 0.624). The two non-convolutional architectures are the
least reliable on every softmax score, replicating Study 1's ViT/Mixer finding
across five scores rather than one.

### 4.3 Difficulty scores collapse on memorised training data

`train_holdout` is **a slice of train, not withheld from it**, so the network
has fit those samples. ce_loss ρ_seed, test versus train_holdout:

| arch | test | train_holdout | drop |
|---|---|---|---|
| `mixer_nano` | 0.647 | **0.108** | 0.539 |
| `vit_tiny` | 0.673 | **0.116** | 0.557 |
| `convnext_femto` | 0.709 | **0.150** | 0.558 |
| `resnet32x4` | 0.709 | 0.257 | 0.452 |
| … | | | |
| `mobilenetv2` | 0.874 | 0.849 | 0.026 |
| `resnet8x4` | 0.870 | 0.799 | 0.071 |

Meanwhile the training-dynamics scores, which integrate over the trajectory
rather than reading the endpoint, are **stable**: `forget_events` averages
0.852 and `el2n` 0.614 on the same data.

**This matters outside our setting.** Dataset pruning, coreset selection and
curriculum learning compute exactly these softmax scores, on exactly this data
— the training set — and usually from a single seed. For the high-capacity
models those methods target, a single-seed softmax difficulty score on training
data carries almost no reproducible signal, while EL2N and forgetting events do.

> **Hypothesis, not yet confirmed.** The proposed mechanism is softmax
> saturation: on memorised data the network is confident on nearly everything,
> the ranking degenerates, and seed agreement collapses. `S2_NB1` now contains a
> cell that tests it — the drop must track train accuracy and the fraction of
> samples with top-1 probability > 0.99. **If those correlations come back weak
> or negative, this subsection is withdrawn, not reworded.**

## 5. Limitations

- **One dataset, one scale.** CIFAR-100 at 32px, 15 architectures. ImageNet-100
  (2 architectures × 2 seeds) shows the same direction on the reliability atlas
  but cannot support a cross-seed bias estimate — one pair per architecture, no
  spread. It is a consistency note, not an appendix.
- **The cross-seed oracle is one honest counterfactual, not the only one.** It
  maximises seed *j*'s accuracy, which at a fixed budget forces early exits that
  seed *i* may not survive. A reviewer can reasonably read the −7.90 pt as
  "a mis-specified router does badly" rather than "the ceiling is negative". The
  claim we defend is the **bias** (+22.41 pt), which is a difference between two
  oracles built identically and differing only in whose noise they see.
- **R-03 fired.** Within-pair asymmetry (0.719 pt) exceeds the *per-score* bias
  (−0.57 pt), so the sign of that secondary quantity is not interpretable until
  seed accuracy is regressed out. It does not touch the oracle bias, which is
  30× larger than the asymmetry.
- **`pred_depth` is not deployable.** `prediction_depth()` runs a kNN probe over
  every layer's features and targets the network's own final answer, so it costs
  a full forward pass. We report it as an oracle-only signal and never as a
  method.
- **Five scores, not eight,** on the test split: `msc` is not persisted, and
  `el2n`/`forget_events` are undefined for samples with no training history.

## 6. What we are not claiming

That early-exit routing cannot work. That confidence thresholding is optimal.
That these numbers transfer to ImageNet or to transformers at scale. The claim
is narrow and, we think, solid: **an oracle bound computed from the network it
routes overstates achievable gains by more than the gains themselves, and the
excess is identifiable, mechanistic, and removable with one extra seed.**

---

## Reproduction

```
S2_NB0_Fetch.ipynb        pull the 45 CIFAR-100 runs from HuggingFace
S2_NB1_Reliability.ipynb  P-1 inventory · P0a collinearity · atlas · memorisation
S2_NB2_Ceiling.ipynb      oracle ceiling · optimism bias · budget sweep
tools/s2_routing_canaries.py   18/18 — the statistics can report wrong answers
tools/s2_canaries.py            3/3
```

Outputs: `analysis/s2_true_oracle.csv`, `s2_reliability_grid.csv`,
`s2_headroom_sweep.csv`, `s2_optimism_bias.csv`, `s2_memorisation.csv`.
