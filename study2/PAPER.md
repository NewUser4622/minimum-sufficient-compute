# Oracle upper bounds for early-exit routing are inflated by per-exit noise

**Status: draft, all measurements complete.** Every number is traceable to a CSV
in `analysis/`. The saturation mechanism (§4.3) was a hypothesis in the first
draft and is now tested and confirmed. §3.1 carries a correction: the first
draft attributed the whole optimism bias to noise harvesting, which the data
does not support — it is decomposed there instead.

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
0.95. The bias decomposes into two parts of unequal evidential weight. **+6.86 pt is
an exact identity**: the in-seed oracle exceeds the network's *own full-compute
accuracy* in 100 % of runs, by precisely the fraction of samples where some
early exit is correct while the final layer is wrong. That component needs no
second seed, cannot be reached by any router, and is the paper's core claim.
The remaining +15.33 pt measures failure to transfer across seeds and is weaker
evidence, since it conflates non-transfer with the cost of acting on a
non-transferring signal; we report it separately rather than folded in.

We also report a reliability atlas for five per-sample difficulty scores across
the same grid, and two secondary findings: the four softmax-derived scores are
**one signal** (ρ = 0.997–1.000), and score reliability **collapses on training
data the network has fit** (ce_loss ρ_seed 0.647 → 0.108 for `mixer_nano`) —
which is precisely where the dataset-pruning literature computes them. The
collapse is predicted by softmax saturation (ρ = **+0.832**) and by train
accuracy (**+0.746**), but **not** by test accuracy (−0.114), so it is not a
generic "weaker model" effect.

---

## 1. What is being claimed

The oracle early-exit bound in common use is:

> exit at the first layer whose prediction matches that of the last layer —
> an ideal upper bound for how much computation could be saved.

**The definition we target is the standard one.** A survey of early-exit
networks states it directly: *"the oracle is an ideal model that can always
enable each sample to exit at the shallowest internal classifier that provides a
correct label prediction"* — a **label** oracle, which is exactly
`pred_dk == label` as implemented here (not the weaker "matches the final
prediction" variant, which is capped at full accuracy by construction).

And the inference we are challenging is made explicitly. DE3-BERT observes that
*"the oracle outperforms the backbone model and existing exiting strategies by a
large margin… which indicates significant room for improving the estimation of
prediction correctness."* That reading — oracle above backbone, therefore
headroom — is the one this paper argues is unsafe.

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

The single most important comparison is **oracle in-seed (78.30 %) against full
compute (71.21 %)**: the bound sits above the accuracy you get by simply running
the whole network.

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

### 3.1 Decomposing the bias — and what is actually unarguable

**Correction applied after checking.** An earlier draft attributed the whole
+22.41 pt to per-exit noise harvesting. That is wrong: across architectures the
bias does **not** correlate with the size of the noise pool
(Spearman = **+0.011**). The bias has two components with very different
evidential strength, and they must be separated.

```
bias = (in-seed − full compute) + (full compute − cross-seed)
     =        A                 +           B
```

| | median | range | share |
|---|---|---|---|
| **A** in-seed oracle **above** the network's own full accuracy | **+6.86 pt** | [4.81, 11.56] | 31 % |
| **B** cross-seed oracle **below** full accuracy | **+15.33 pt** | [7.64, 27.81] | 69 % |
| A + B | +22.19 pt | | |
| measured bias (median of per-run values) | **+22.41 pt** | | |

Both are positive in **100 %** of runs.

**A is unarguable and is the paper's core claim.** It is an exact identity:
`oracle_in − acc_full == frac_early_saves` to numerical precision on all 90
rows (max deviation 0.000000000 pt), because at ρ = 0.80 the budget never binds
and the in-seed oracle is simply *P(correct at any exit)*.

So the published-style bound sits **6.86 points above the accuracy the network
achieves when you simply run all of it**. No router can reach that, whatever its
signal, because the excess consists entirely of samples the network gets right
at some early exit and **wrong at the final layer**. It is per-exit noise, it is
not a property of the samples, and it does not survive a change of seed.

*A needs no second seed to establish.* It follows from one trained network and
its own per-exit predictions. That makes it the most robust result here.

**B is weaker and we flag it as such.** It measures how badly seed *j*'s
correctness transfers to seed *i* — but it conflates *"the signal does not
transfer"* with *"acting on a non-transferring signal is worse than not
acting at all"*. A cross-seed oracle maximises seed *j*'s accuracy, which at a
fixed budget forces early exits that seed *i* may not survive. A reviewer may
fairly read B as evidence about a mis-specified router rather than about a
ceiling. We report it, decomposed, rather than folding it into a single headline
number.

**Consequence for how this is described.** Because the budget is inactive, the
in-seed oracle *spends less than the baseline while scoring higher*, so this is
**not** a matched-FLOPs comparison and we do not call it one. It remains a valid
upper bound, and a conservative one — the baseline at the oracle's lower cost
would be worse still.

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

**Mechanism: confirmed.** Saturation was a hypothesis in the first draft; the
test is now run (`analysis/s2_memorisation.csv`, 15 architectures):

| predictor of the reliability drop | Spearman |
|---|---|
| fraction of train samples with top-1 probability > 0.99 | **+0.832** |
| train accuracy | **+0.746** |
| train − test accuracy gap | +0.693 |
| **test accuracy** (the control) | **−0.114** |

`convnext_femto` fits 99.99 % of the training set with 71 % of samples above
0.99 confidence and drops 0.558; `mobilenetv2` fits 83.1 % with 16 % saturated
and drops 0.026. The near-zero correlation with *test* accuracy is the control
that matters: this is not "worse models are noisier". It is that a network which
has memorised its training data produces a degenerate ranking on it, and the
ranking is what every one of these scores depends on.

**This is independent of the oracle result.** The optimism bias does not
correlate with saturation (−0.214) or train accuracy (−0.221). Two separate
findings, not one mechanism seen twice.

## 5. Limitations

- **The exits are post-hoc heads on a frozen backbone, not a trained
  early-exit network.** This is the most serious limitation and the first thing
  a reviewer will raise. Study 1 trained exit heads with the backbone **frozen**
  (`msc_lib.py`, "exit heads: backbone frozen"), whereas MSDNet, BranchyNet and
  DE3-BERT-style networks train exits *jointly*, producing stronger and
  better-calibrated early classifiers. Weaker exits plausibly **enlarge** the
  early-right/final-wrong pool, so the **+6.86 pt magnitude is likely an
  overestimate** for a properly trained early-exit model. The *direction* is
  structural — a cheapest-correct-exit oracle can never fall below full accuracy
  and will exceed it whenever any early exit is right where the final layer is
  wrong — but the magnitude must be re-measured on a jointly-trained network
  before it is quoted as a general number. **This is the single highest-value
  follow-up experiment.**
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
- **H4 is not supported on the oracle bias either.** Testing
  corr(1 − ρ_seed, oracle bias) across architectures gives ρ = +0.232 to +0.457
  depending on which score supplies the reliability estimate — the right sign,
  consistently, but below the pre-registered 0.5 on every one.
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
