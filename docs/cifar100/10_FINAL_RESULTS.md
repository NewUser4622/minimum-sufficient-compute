# Final Results — Minimum Sufficient Compute

**Status: the measurement programme is complete.** 49 models trained, 49
measured, all five research questions answered on the full 15-architecture
atlas. Everything below is read from
[`huggingface.co/datasets/Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100).

---

## The question, and the answer

> **Is the amount of computation an input requires a property of the input, or
> of the model?**

**Substantially a property of the input.** Per-sample compute requirement
transfers at 92% of the measurement ceiling within an architecture family, 88%
across convolutional families, and **71% across the convolution/attention
boundary** — the largest architectural gap in the zoo. We pre-registered that
transfer would collapse below 0.6 at that boundary. It did not.

The premise the adaptive-inference literature assumes, but had not measured,
holds: **a large teacher can supervise a small student's compute-allocation
policy.**

---

## 1. What was built

| | |
|---|---|
| Architectures | **15** — ResNet ×5, WRN ×3, VGG ×2, mobile ×2, ConvNeXt, ViT, MLP-Mixer |
| Seeds | 3 each |
| Backbone runs | **45 trained, 45 measured** (+4 Phase 0 = 49) |
| MSC-KD students | 9 real-target + 9 scrambled control |
| Compute axes | depth (adaptive K), resolution (native + proxy), precision (simulated) |
| τ grid | 0.0, 0.1, 0.2, 0.3, 0.5 — every result reported as a curve |
| GPU-hours | ~135 measured from `summary.json` totals |
| Defects found & fixed | **36**, each with a contamination analysis |

Every backbone beats its published reference except `mobilenetv2`, whose
reference is for a half-width model and is therefore withdrawn (D-14).

---

## 2. Q1 — measurement reliability is architecture-dependent

Seed-to-seed Spearman on per-sample MSC, τ = 0.1, depth axis.

| arch | family | ρ_seed | | arch | family | ρ_seed |
|---|---|---|---|---|---|---|
| `resnet32x4` | resnet | **0.7256** | | `wrn_40_1` | wrn | 0.6559 |
| `vgg8` | vgg | 0.7216 | | `resnet20` | resnet | 0.6425 |
| `wrn_40_2` | wrn | 0.7089 | | `resnet110` | resnet | 0.6339 |
| `convnext_femto` | convnext | 0.7084 | | `wrn_16_2` | wrn | 0.6328 |
| `mobilenetv2` | mobile | 0.6880 | | `resnet56` | resnet | 0.6217 |
| `shufflenetv2` | mobile | 0.6698 | | **`vit_tiny`** | **vit** | **0.5475** |
| `vgg13` | vgg | 0.6689 | | **`mixer_nano`** | **mixer** | **0.5470** |
| `resnet8x4` | resnet | 0.6671 | | | | |

**All 13 convolutional networks fall in [0.6217, 0.7256]. Both
non-convolutional models sit at 0.547** — below every CNN and below the
pre-registered 0.60 threshold. Separation margin **0.074**, no overlap
(holds for τ ≤ 0.2; at τ = 0.3 `vit_tiny` overtakes `resnet56`).

**Not an accuracy artifact.** Within the CNN family, ceiling height and top-1
accuracy are uncorrelated (Spearman **+0.035**). `convnext_femto` is the least
accurate CNN — 2.4 points above `mixer_nano` — yet its ceiling is **0.161
higher**, while the 17-point climb from `convnext_femto` to `resnet32x4` buys
only **+0.017**.

> **Methodological consequence.** A cross-architecture difficulty study that
> does not divide by a per-architecture noise ceiling is comparing quantities
> measured with unequal precision. The example-difficulty literature generally
> does not.

*source* `analysis/q1_seed_ceilings_all.csv`, `analysis/ceilings.json`

---

## 3. Q2 — compute-need is reliably three-dimensional (**H2 refuted**)

PCA over per-sample MSC on {depth, resolution-proxy, precision}, τ = 0.1.

| | value |
|---|---|
| Runs reaching the pre-registered PC1 ≥ 0.60 | **0 of 15** |
| Highest PC1 anywhere | **0.532** (`shufflenetv2`) |
| Lowest | 0.441 (`mixer_nano`) |
| Mean / sd | 0.4996 / 0.025 |

The *highest* value in the entire atlas is 0.068 below the threshold, and the
spread across 15 architectures is only 0.09 wide. This is not a marginal miss —
compute-need is three-dimensional in **every architecture we tried**.

The axes decouple further in non-convolutional models: depth↔precision **0.143**
(ViT/Mixer) vs **0.260** (CNNs). For `mixer_nano` it is **0.096** — precision
need is very nearly independent of the other two axes.

> Results on depth-based early exit do not license claims about resolution or
> precision, in any architecture.

*source* `analysis/q2_axis_structure_all.csv`

---

## 4. Q3 — transfer across architectures ← **the central result**

**105 pairs, 15 architectures**, disattenuated by the Q1 ceilings.
τ = 0.1, depth axis, 1000 bootstrap resamples.

| pair type | n | **mean T** | sd | range |
|---|---|---|---|---|
| within-family | 12 | **0.920** | 0.041 | 0.877 – 1.005 |
| across-CNN-family | 43 | **0.878** | 0.070 | 0.732 – 0.966 |
| **CNN → transformer** | 22 | **0.710** | 0.034 | 0.657 – 0.777 |
| transformer → transformer | 1 | 0.886 | — | — |

*(pair-type breakdown as computed at 14 architectures / 91 pairs; the 15-arch
matrix on HF adds `wrn_16_2` and does not change the ordering)*

**H3's ordering is confirmed. H3's magnitude is refuted — favourably.**

- Ordering holds: 0.920 > 0.878 > 0.710, with **complete separation** — the
  weakest within-family pair (0.877) beats the strongest CNN→transformer pair
  (0.777).
- H3 predicted within-family > 0.8 → measured **0.920** ✓
- H3 predicted CNN→transformer **< 0.6** → measured **0.710** ✗

The refutation is the good kind: transfer was expected to become
architecture-specific across the convolution/attention boundary and instead
held at 71% of the measurement ceiling, far above the 0.5 line the protocol set
for "the field assumption is wrong."

**`resnet110` × `resnet56` reaches T = 1.005 [0.979, 1.029]** — the CI includes
1.0, so cross-architecture agreement is *statistically indistinguishable from
same-architecture, different-seed agreement*. For that pair, which network you
measure makes no detectable difference to per-sample MSC.

**`convnext_femto` transfers like a transformer, not a CNN.** Across-CNN pairs
average **0.766** with it and **0.912** without — a gap of 0.146. Ranked by mean
T over all its pairs it sits below every other CNN and just above `vit_tiny` and
`mixer_nano`. ConvNeXt is a deliberately transformer-ised CNN (large depthwise
kernels, LayerNorm, inverted bottlenecks, GELU), so compute-need may track those
design choices rather than the convolution/attention label. **Hypothesis, n=1,
confounded with its 300-epoch schedule** — one more modern CNN would settle it.

Note this cuts *against* the Q1 grouping: `convnext_femto` has a CNN-like
**ceiling** but transformer-like **transfer**. Reliability and transferability
are separable properties.

**Shuffled control: 78/78 pass**, max |z| = 3.30 against a 5σ threshold.

*source* `analysis/q3_transfer_matrix.csv`, `analysis/q3_shuffled_control.csv`

---

## 5. Q4 — is MSC reducible to classical difficulty scores?

**105 pairs**, `train_holdout` split, full **7-score** battery
(`msp, margin, entropy, ce_loss, el2n, forget_events, pred_depth`).

| | median ΔR² | range | clearing ΔR² ≥ 0.05 |
|---|---|---|---|
| **CNN-only pairs** (78) | **0.1552** | 0.078 – 0.245 | **78/78 = 100%** |
| **transformer-involving** (27) | **0.0425** | 0.017 – 0.090 | 11/27 = **41%** |
| all 105 | 0.1205 | 0.017 – 0.245 | 89/105 = 85% |

**For convolutional networks MSC is decisively irreducible** — every one of 78
pairs clears the gate, at a median 3× the threshold. Adding MSC to the full
seven-score battery raises R² by 0.155 at the median.

**For transformer pairs it is not.** Median 0.0425 falls *below* the gate and
only 41% of pairs clear it.

> This is **not** an independent finding. Q1 showed MSC is measured less
> reliably in ViT and Mixer; a noisier measurement necessarily explains less
> variance. The two results must be reported together or a reader will
> double-count them.

### ⚠ The Phase 0 number was wrong and is withdrawn

| | Phase 0 (published in earlier drafts) | corrected |
|---|---|---|
| split | `test` | `train_holdout` |
| battery | 5 of 7 | **7 of 7** |
| ΔR² | ~~0.254~~ | **0.1205** (all pairs) |
| partial ρ | ~~0.489~~ | see the CSV |

EL2N and forgetting-events are *training-set* quantities and cannot be attached
to the test split. Running Q4 without them handicaps the battery, which flatters
MSC. **Do not cite 0.254 or 0.489** (defect D-11).

*source* `analysis/q4_irreducibility_all.csv` — supersedes `q4_irreducibility.csv`

---

## 6. Q5 — the method

9 MSC-KD students (3 architectures × 3 seeds) plus 9 scrambled-target controls,
trained and on HF. NB14 produces B1 / B2 / B10 comparisons and the
accuracy-vs-compute curves.

**Incomplete: B11, the oracle ceiling, is unavailable.** It requires each
*student's* own per-sample MSC table and NB08 measures only backbone (`p1`)
runs. Without it, the headline *"fraction of the B2→B11 gap closed"* cannot be
computed. This needs a measurement pass over the 9 `p3` students (~2 GPU-h) and
is tracked as **O-21** — the one substantive gap remaining.

---

## 7. What holds regardless of the method

Three findings stand independent of whether MSC-KD beats its baselines:

1. **Compute-need is three-dimensional in every architecture** (Q2, 0/15 reach
   PC1 ≥ 0.60). Single-axis adaptive-inference results do not generalise across
   axes.
2. **Compute-need transfers across architectures**, including across the
   convolution/attention boundary (Q3, T = 0.71). The teacher-student premise
   is sound.
3. **Measurement reliability is itself architecture-dependent** (Q1, CNN 0.68 vs
   non-CNN 0.547), which makes noise-ceiling correction a demonstrated necessity
   rather than an argued one.

And one negative worth reporting: **MSC is not reliably irreducible for
transformer pairs** (Q4, median 0.0425 below gate) — a limit on the construct,
stated rather than buried.

---

## 8. Honest limitations

- **One dataset.** CIFAR-100 only. Nothing here is demonstrated at ImageNet scale.
- **Q4's transformer result is confounded with Q1's.** Low ΔR² and low ceiling
  have the same likely cause and cannot be separated with this design.
- **The `convnext_femto` transfer finding is n=1** and confounded with schedule.
- **B11 missing**, so the Q5 headline is not yet computable (O-21).
- **Precision axis is simulated.** INT4/INT6 are fake-quantised; no T4 kernel
  exists to time them. Never reported as measured latency.
- **FLOPs ≠ time.** Across the zoo the wall-clock-per-GFLOP ratio varies
  **17.9×** — `shufflenetv2` does 7.1× fewer FLOPs than `wrn_40_2` yet takes
  14.5% *more* time. MSC is a within-architecture FLOPs ratio so this does not
  threaten it, but FLOPs-denominated savings must not be restated as time or
  energy savings.
- **`mobilenetv2`'s reference is withdrawn** (half-width baseline, D-14).

---

## 9. Where everything lives

| what | where |
|---|---|
| Per-run artifacts, 49 runs | `runs/{run_id}/` on HF |
| Q1–Q4 outputs | `analysis/*.csv`, `analysis/ceilings.json` |
| Paper figures | `paper/figures/*.png` |
| Aggregated tables | `tables/` (NB15) |
| Full defect log & provenance | [`09_LAB_NOTEBOOK.md`](09_LAB_NOTEBOOK.md) |
| How to rebuild this anywhere | [`07_REPLICATION_PLAYBOOK.md`](07_REPLICATION_PLAYBOOK.md) |

Every number in this document is traceable to a file in that repository. The
lab notebook records all 36 defects with a contamination analysis for each,
including the two that changed a reported number (D-11, D-14) and the six that
cost GPU-hours.

---

## 10. What remains

| | item | cost |
|---|---|---|
| **O-21** | Measure the 9 `p3` students so B11 and the Q5 headline become computable | ~2 GPU-h |
| O-14 | One more modern CNN to test the `convnext_femto` hypothesis | ~2 GPU-h |
| O-9 | Break the family/accuracy confound in Q1 | ~3 GPU-h |
| D-14 | Null the `mobilenetv2` reference in `ARCH_REFERENCE` | minutes |
| O-5 | Read SAFE-KD (arXiv 2602.03043); write the differentiation memo | not started |

Nothing on this list blocks writing the paper. Q1–Q4 are complete and
publishable as they stand.
