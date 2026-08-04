# Lab Notebook

**Running record of every result, every defect, and every decision that changed.**

Append-only. Newest entries at the top of each log. The purpose is that when the
paper is written, three questions can be answered from one file:

1. **What did we measure, and exactly where does that number live?**
2. **Was any reported number produced by code that later turned out to be wrong?**
3. **What changed in the design after the plan was frozen, and why?**

§1 is the results ledger. §2 is the defect log with a contamination analysis for
each. §3 is decisions changed. §4 maps all of it onto paper sections.

---

## 0. Status board

| | |
|---|---|
| **Phase** | **All five research questions answered.** NB13 (MSC-KD) running |
| **Verdict** | `FULL-PROGRAM` (2026-08-02) |
| **Runs trained** | Phase 0 **4/4** · Phase 1 **44/45** — only `p1-wrn_16_2-…-s1` missing |
| **Runs measured** | Phase 0 **4/4** · Phase 1 **39/45** — see D-15 |
| **Analysis on the atlas** | Q1 ✅ 14 archs · Q2 ✅ 13 · Q3 ✅ **78 pairs** · Q4 ⚠ 15 biased pairs (D-18) |
| **Q3 — the central result** | within-family **0.920** > across-CNN **0.878** > CNN→transformer **0.710**, no overlap |
| **Hypotheses** | H2 **refuted** 15/15 · H3 ordering ✅ but magnitude **refuted favourably** · H4 ΔR² ✅, partial ρ marginal |
| **GPU-hours spent** | ~115 (9.5 Phase 0 + ~82 atlas + ~20 measurement + 2.9 wasted to D-12) |
| **GPU-hours remaining** | ~45 (gap-fill ~8 · NB13 MSC-KD ~30 · NB14 ~5) — analysis re-runs are CPU-only |
| **Library version** | `msc_lib` 1.0.0 · **207** offline + **3 torch-gated** self-checks |
| **Defects found** | 23 · 20 fixed · **3 open** (D-14, D-15, D-18 re-run) · **O-19 dry run now shipped — D-21/D-22/D-23 would all have been caught in <1 s** |
| **Artifacts** | `huggingface.co/datasets/Shanmuk4622/msc-cifar100` @ `9b18d2b`, 2026-08-04T06:1x Z |

> **⚠ Two numbers in older documents are now known to be wrong.**
> **ΔR² = 0.254** (README, `08_PHASE0_RESULTS.md`) was measured on the wrong
> split with 5 of 7 difficulty scores; the corrected value is **~0.10** (§1.5).
> **`mobilenetv2` +5.50** is against a half-width baseline (D-14). Neither may
> be published.

**Audit basis for this board:** HF repo-info API at revision `a4aef3ac`, plus
direct `tree/` listings for every run whose state the truncated file list left
ambiguous. Counts below are from files that actually exist on HF, not from the
run ledger.

---

## 1. Results ledger

Every number we have measured, with its provenance. `source` is the file in the
HF repo; `code` is the library version that produced it.

### 1.1 Phase 0 backbone training

Standard CRD/DKD recipe, CIFAR-100, 240 epochs, SGD 0.05, ×0.1 at 150/180/210,
batch 64, wd 5e-4, random crop + horizontal flip. AMP on. Kaggle T4.

| run_id | Top-1 | Top-5 | Published | Δ | GPU-h | kWh | epochs |
|---|---|---|---|---|---|---|---|
| `p0-resnet32x4-cifar100-base-s1` | 79.59 | 94.28 | 79.42 | **+0.17** | 2.89 | 0.216 | 240/240 |
| `p0-resnet32x4-cifar100-base-s2` | 79.63 | — | 79.42 | **+0.21** | 2.89 | 0.216 | 240/240 |
| `p0-wrn_40_2-cifar100-base-s1` | 76.89 | 93.81 | 75.61 | **+1.28** | 1.88 | 0.140 | 240/240 |
| `p0-wrn_40_2-cifar100-base-s2` | 76.72 | — | 75.61 | **+1.11** | 1.88 | 0.140 | 240/240 |

*source* `runs/{run_id}/summary.json` · *code* `msc_lib` 1.0.0 · *reference* DKD paper / mdistiller

**All four exceed published values.** This is the acceptance test for the whole
pipeline: MSC computed from an under-trained model is meaningless, and an
under-trained model is otherwise easy to miss. `recipe_ok: true` on all four.

Seed-pair spread: resnet32x4 0.04 pts, wrn-40-2 0.17 pts.

`sample_order_hash = 80031c23f8300724…` identical across all four → per-sample
tables are index-aligned and may be correlated.

### 1.1b Phase 1 atlas — ResNet family (NB04)

Same recipe. Four Kaggle accounts, workers 0–3, cost-balanced split.
**15 of 15 complete.** All beat their published references.

| arch | s1 | s2 | s3 | mean | published | Δ |
|---|---|---|---|---|---|---|
| `resnet20` | 70.25 | 70.36 | 69.78 | **70.13** | 69.06 | **+1.07** |
| `resnet56` | 73.88 | 73.35 | 73.85 | **73.69** | 72.34 | **+1.35** |
| `resnet110` | 74.31 | 74.57 | 74.26 | **74.38** | 74.31 | **+0.07** |
| `resnet8x4` | 73.35 | 73.39 | 73.04 | **73.26** | 72.50 | **+0.76** |
| `resnet32x4` | 79.72 | 80.03 | 79.46 | **79.74** | 79.42 | **+0.32** |

`p1-resnet32x4-cifar100-base-s3` — the run D-12 abandoned at epoch 79/240 — was
resumed and **completed 240/240 on 2026-08-03T21:10Z at 79.46%**. This is the
first end-to-end proof that the resume path works on a real interrupted run
rather than on the synthetic test of D-06. Open item O-1b is closed.

Seed spread: resnet20 0.58 pts, resnet56 0.53, resnet110 0.31, resnet8x4 0.35,
resnet32x4 0.57. All share `sample_order_hash = 80031c23…`, matching Phase 0 —
the atlas and pilot tables are mutually correlatable.

Wall-clock: 11:51 → 20:31 (~8.7 h) across four accounts for ~32 GPU-hours of
work, of which 2.9 h were wasted re-training a run another account had already
done (D-12).

*source* `runs/p1-*/summary.json`, `registry/events/*.jsonl`

### 1.1c Phase 1 atlas — remaining ten architectures (NB05–NB07)

Completed 2026-08-03 → 2026-08-04. CNNs 240 epochs on the CRD recipe; the three
modern architectures 300 epochs. One representative seed shown per architecture —
these are the values read directly off HF, not seed means, because the combined
table (`tables/`) is produced by NB15 and NB15 has not run.

| arch | family | seed | top-1 | top-5 | published | Δ | params | GFLOPs | GPU-h | kWh | ep |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `wrn_40_2` | wrn | 1 | 76.06 | 93.46 | 75.61 | **+0.45** | 2.26 M | 0.658 | 1.94 | 0.146 | 240 |
| `wrn_40_1` | wrn | 1 | 72.41 | 92.41 | 71.98 | **+0.43** | 0.57 M | 0.168 | 1.64 | 0.107 | 240 |
| `wrn_16_2` | wrn | 2 | 73.79 | 93.56 | 73.26 | **+0.53** | 0.70 M | 0.203 | 0.94 | 0.071 | 240 |
| `vgg13` | vgg | 1 | 75.70 | 92.85 | 74.64 | **+1.06** | 9.46 M | 0.458 | 1.12 | 0.082 | 240 |
| `vgg8` | vgg | 2 | 71.73 | 91.26 | 70.36 | **+1.37** | 3.96 M | 0.136 | 0.74 | 0.055 | 240 |
| `mobilenetv2` | mobile | 1 | 70.10 | 92.33 | 64.60 | **+5.50** ⚠ | 2.35 M | 0.183 | 2.20 | 0.166 | 240 |
| `shufflenetv2` | mobile | 1 | 71.93 | 92.64 | 70.50 | **+1.43** | 1.36 M | 0.092 | 2.22 | 0.130 | 240 |
| `convnext_femto` | convnext | 1 | 62.67 | 79.51 | — | — | 4.87 M | 0.126 | 1.99 | 0.148 | 300 |
| `vit_tiny` | vit | 1 | 59.33 | 79.49 | — | — | 5.38 M | 0.694 | 2.65 | 0.199 | 300 |
| `mixer_nano` | mixer | 1 | 60.23 | 79.96 | — | — | 2.50 M | 0.343 | 1.59 | 0.118 | 300 |

⚠ **The `mobilenetv2` +5.50 is not a win — see D-14.** The 64.60 reference is
for the *half-width* MobileNetV2 used in the CRD/mdistiller student tables;
our model has 2.35 M parameters against roughly 0.81 M for that baseline. Do
not report this row as beating a published number.

The three modern architectures correctly carry `reference_accuracy: null` — no
established from-scratch CIFAR-100 number exists for them, so no Δ is claimed.
Their 59–63% top-1 is the expected range for attention/MLP models trained from
scratch on 50 k images without heavy augmentation, and is **not** evidence of an
under-trained run. It does, however, constrain what the atlas can claim about
them — see §1.2.

All ten share `sample_order_hash = 80031c23…`. `recipe_ok: true` on every CNN.

*source* `runs/p1-{arch}-cifar100-base-s{n}/summary.json`

### 1.1d Measurement coverage (NB08)

**39 of 45 Phase 1 runs have `per_sample/test.parquet`.** The six that do not:

| run | trained? | measured? | consequence |
|---|---|---|---|
| `p1-wrn_16_2-…-s1` | ❌ no | ❌ | — |
| `p1-wrn_16_2-…-s2` | ✅ | ❌ | |
| `p1-wrn_16_2-…-s3` | ✅ | ❌ | **`wrn_16_2` has 0 usable seeds — absent from Q1 entirely** |
| `p1-wrn_40_1-…-s3` | ✅ | ❌ | `wrn_40_1` ceiling rests on 2 seeds |
| `p1-vgg8-…-s1` | ✅ | ❌ | `vgg8` ceiling uses s2/s3 |
| `p1-mixer_nano-…-s3` | ✅ | ❌ | `mixer_nano` ceiling rests on 2 seeds |

See **D-15**. `wrn_16_2` is the material gap: it is one of the 15 pre-registered
architectures and currently contributes nothing to any analysis.

---

### 1.2 Q1 — noise ceiling (ρ_seed) — **now atlas-wide, 14 architectures**

Spearman between MSC of two seeds of the *same* architecture, depth axis,
irreducible samples masked. This is the denominator every transfer number gets
divided by, so it is the single most important quantity in the project.

| arch | family | τ=0.0 | **τ=0.1** | τ=0.2 | τ=0.3 | τ=0.5 |
|---|---|---|---|---|---|---|
| `resnet32x4` | resnet | 0.6927 | **0.7256** | 0.7242 | 0.7202 | 0.6946 |
| `vgg8` | vgg | 0.6576 | **0.7248** | 0.7285 | 0.7443 | 0.7295 |
| `wrn_40_2` | wrn | 0.6671 | **0.7089** | 0.7032 | 0.7059 | 0.6788 |
| `convnext_femto` | convnext | 0.6188 | **0.7084** | 0.7187 | 0.7198 | 0.7037 |
| `mobilenetv2` | mobile | 0.6248 | **0.6880** | 0.7007 | 0.6951 | 0.6675 |
| `shufflenetv2` | mobile | 0.6124 | **0.6698** | 0.6871 | 0.6894 | 0.6623 |
| `vgg13` | vgg | 0.6019 | **0.6689** | 0.6821 | 0.6825 | 0.6666 |
| `resnet8x4` | resnet | 0.6134 | **0.6671** | 0.6820 | 0.6795 | 0.6734 |
| `wrn_40_1` | wrn | 0.6172 | **0.6559** | 0.6768 | 0.6765 | 0.6462 |
| `resnet20` | resnet | 0.5784 | **0.6425** | 0.6512 | 0.6482 | 0.6268 |
| `resnet110` | resnet | 0.5996 | **0.6339** | 0.6234 | 0.6099 | 0.5819 |
| `resnet56` | resnet | 0.6104 | **0.6217** | 0.6156 | 0.5944 | 0.6013 |
| `vit_tiny` | **vit** | 0.4775 | **0.5475** | 0.5831 | 0.6026 | 0.6056 |
| `mixer_nano` | **mixer** | 0.5080 | **0.5470** | 0.5322 | 0.5034 | 0.4733 |

`wrn_16_2` is absent — it has no measured seed pair (D-15).

*source* `analysis/q1_seed_ceilings_all.csv`, `analysis/ceilings.json` · gate ≥ 0.60

#### The headline: MSC is measurably less seed-stable in non-convolutional models

At the primary operating point τ=0.1 the split is **clean and total**:

| | n | range | mean |
|---|---|---|---|
| convolutional | 12 | 0.6217 – 0.7256 | 0.6763 |
| attention / MLP-mixing | 2 | 0.5470 – 0.5475 | 0.5473 |

Separation margin **0.0742** — every CNN is above every non-CNN, with no overlap.
Both non-CNNs **fail** the pre-registered ρ_seed ≥ 0.60 gate; all 12 CNNs pass.

**This is not an accuracy artifact.** The obvious objection is that `vit_tiny`
(59.33%) and `mixer_nano` (60.23%) are also the two least accurate models, so
maybe low ceilings just mean under-converged models. Three things rule that out:

1. **Within the 12 CNNs, ρ_seed and top-1 are uncorrelated** — Spearman **+0.035**,
   Pearson **−0.007**. Accuracy carries essentially no information about ceiling
   height once you are inside the convolutional family.
2. **`convnext_femto` breaks the confound directly.** At 62.67% it is the least
   accurate CNN — only **2.4 points** above `mixer_nano` — yet its ceiling is
   **0.7084**, fourth-highest of all 14 and **0.161 above** `mixer_nano`. Moving
   *17 points* of accuracy from `convnext_femto` up to `resnet32x4` buys only
   **+0.017** of ceiling. The 2.4-point step across the architectural boundary is
   ~10× larger than the 17-point step within it.
3. `convnext_femto` and the two non-CNNs were trained on the *same* 300-epoch
   schedule, so schedule length is not the difference either.

**Honest caveat:** CNN accuracies (62.67–79.72) and non-CNN accuracies
(59.33–60.23) do not overlap, so family and accuracy remain partly confounded at
the level of the atlas as a whole. The `convnext_femto` comparison is the
strongest available evidence and it rests on **one** architecture. A stronger
test would train one non-CNN to CNN-level accuracy or one CNN down to ~60%.
Logged as **O-9**.

#### τ-dependence differs by family

Every CNN is non-monotone in τ — rising to a peak near τ=0.2–0.3, then falling as
the τ=0.5 mask removes a third of the sample. `vit_tiny` is **the only
architecture that rises monotonically** (0.4775 → 0.6056), and consequently the
clean CNN/non-CNN separation holds at τ ∈ {0, 0.1, 0.2} but **breaks at τ=0.3**,
where `vit_tiny` (0.6026) overtakes `resnet56` (0.5944). `mixer_nano` is below
every CNN at every τ without exception.

So the correct claim is *"at τ ≤ 0.2"*, not *"at every τ"*. Any sentence in the
paper asserting the separation must carry that qualifier.

#### Rank agreement and top-decile agreement are not the same measurement

J₁₀ (Jaccard overlap of the top-10% highest-MSC samples) at τ=0.1 ranges from
**0.180** (`vit_tiny`) to **0.664** (`wrn_40_1`) — and it is **almost unrelated
to ρ_seed**, Spearman **+0.130**. `vgg8` has the second-highest ρ_seed (0.7248)
and nearly the lowest J₁₀ (0.2431); `wrn_40_1` is ninth on ρ_seed but first on J₁₀.

What J₁₀ *does* track is mean MSC — Spearman **+0.780**, Pearson **+0.819**.
Architectures whose samples sit near ρ=1.0 on average (`wrn_40_1` 0.859,
`resnet110` 0.845) have a large tied group at the top of the distribution and
therefore a stable top decile; architectures with low mean MSC (`vit_tiny` 0.519,
`convnext_femto` 0.646) have a genuinely sparse hard tail that reshuffles easily.

**Consequence for the paper:** reporting only one of these two statistics would
mislead, in either direction. Both must appear in the Q1 and Q3 tables, and the
mean-MSC confound on J₁₀ must be stated — otherwise a reviewer will read a low
J₁₀ as a weak result when it is partly a property of where the architecture sits
on the compute scale. Logged as **O-10**.

#### What Phase 0 could not have told us

Phase 0 measured exactly two architectures — `resnet32x4` (0.7256) and `wrn_40_2`
(0.7089) — which turn out to be **the 1st and 3rd most seed-stable of all 14**.
The Phase 0 ceiling of ~0.715 was therefore an optimistic sample, not a typical
one. The atlas mean is 0.6763 for CNNs and 0.6579 across all 14. Every
disattenuated transfer number computed against a Phase 0 ceiling is
correspondingly conservative; every one computed against a `vit_tiny` or
`mixer_nano` ceiling will be inflated by a *smaller* denominator and needs the
wider CI that the low ceiling implies.

### 1.3 Q2 — axis structure (PC1) — **atlas-wide, 13 architectures**

**H2 is refuted across the entire atlas, not just one architecture.** PC1 at
τ=0.1, PCA over per-sample MSC on {depth, res_proxy, precision}:

| arch | PC1 | | arch | PC1 |
|---|---|---|---|---|
| `shufflenetv2` | 0.5316 | | `resnet110` | 0.5015 |
| `resnet8x4` | 0.5264 | | `resnet32x4` | 0.4989 |
| `mobilenetv2` | 0.5245 | | `wrn_40_2` | 0.4950 |
| `vgg13` | 0.5139 | | `vit_tiny` | 0.4800 |
| `resnet56` | 0.5109 | | `convnext_femto` | 0.4536 |
| `resnet20` | 0.5106 | | `mixer_nano` | **0.4407** |
| `wrn_40_1` | 0.5077 | | | |

**0 of 15 runs clear the pre-registered PC1 ≥ 0.60.** Range 0.4407–0.5316, mean
0.4996, sd 0.0252. The *highest* PC1 anywhere in the atlas is 0.068 below the
threshold — this is not a marginal miss, and the Phase 0 value of 0.503 was
typical rather than a fluke. **O-4 closed.**

That the spread is only 0.09 wide across 13 architectures spanning ResNets,
VGG, depthwise-separable nets, a modern CNN, a ViT and an MLP-Mixer is itself
the point: compute-need is *reliably* three-dimensional, in every architecture
we tried.

#### Sub-finding: the axes decouple further in non-convolutional models

| | depth↔precision | res↔precision | PC1 loading on precision |
|---|---|---|---|
| ViT + Mixer | **0.143** | **0.141** | 0.385 |
| the 11 CNNs | 0.260 | 0.248 | 0.483 |

`mixer_nano` is the extreme: depth↔precision **0.096**, res↔precision **0.085** —
precision-need is very nearly *independent* of the other two axes, and its PC1
loading on precision falls to 0.308. So the one-dimensionality assumption is
worst exactly where the field is now moving.

*source* `analysis/q2_axis_structure_all.csv` · H2 predicted ≥ 0.60 · **REFUTED 15/15**

#### The original Phase 0 measurement, for reference

PCA over per-sample MSC on {depth, res_proxy, precision}, resnet32x4 seed 1.

| τ | PC1 | PC2 | PC3 | depth↔res | depth↔prec | res↔prec | n |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.5155 | 0.2739 | 0.2106 | 0.381 | 0.278 | 0.278 | 10000 |
| **0.1** | **0.5032** | 0.2860 | 0.2108 | 0.377 | 0.232 | 0.240 | 9041 |
| 0.2 | 0.4940 | 0.2938 | 0.2122 | 0.374 | 0.202 | 0.210 | 8548 |
| 0.3 | 0.4955 | 0.2911 | 0.2135 | 0.370 | 0.204 | 0.211 | 8192 |
| 0.5 | 0.4984 | 0.2894 | 0.2123 | 0.370 | 0.201 | 0.199 | 7419 |

PC1 loadings at τ=0.1: depth 0.616, resolution 0.633, precision 0.469.

*source* `analysis/q2_axis_structure_phase0.csv` · H2 predicted ≥ 0.60 · **REFUTED**

### 1.4 Q3 — transfer (T) — **THE CENTRAL RESULT**, 78 pairs, 13 architectures

This is the number the project exists to produce. τ=0.1, depth axis, 1000
bootstrap resamples, disattenuated by the per-architecture ceilings in §1.2.

| pair type | n | **mean T** | sd | range | mean J₁₀ | mean ρ_raw |
|---|---|---|---|---|---|---|
| within-family | 12 | **0.920** | 0.041 | 0.877 – 1.005 | 0.480 | 0.608 |
| across-CNN-family | 43 | **0.878** | 0.070 | 0.732 – 0.966 | 0.374 | 0.592 |
| CNN → transformer | 22 | **0.710** | 0.034 | 0.657 – 0.777 | 0.267 | 0.430 |
| transformer → transformer | 1 | 0.886 | — | — | 0.147 | 0.485 |

*source* `analysis/q3_transfer_matrix.csv` · shuffled control: **78/78 pass**

#### H3: the ordering is confirmed, the magnitude is refuted — favourably

H3 pre-registered *within-family > across-CNN-family > CNN→transformer*.

- **Ordering holds**: 0.920 > 0.878 > 0.710, with **complete separation** —
  the weakest within-family pair (0.877) still beats the strongest
  CNN→transformer pair (0.777). No overlap at all.
- **H3 predicted within-family > 0.8** → measured **0.920**. ✓
- **H3 predicted CNN→transformer < 0.6** → measured **0.710**. ✗ **REFUTED.**

The refutation is the good kind. H3 expected compute-need to become
*architecture-specific* across the CNN/Transformer boundary; instead it stays
at 71% of the measurement ceiling — far above the 0.5 line the protocol set for
"architecture-specific, the field assumption is wrong."

> **The project's central question has a positive answer.** How much computation
> an input needs is substantially a property of the *input*, and it survives the
> largest architectural gap in the zoo. A large teacher genuinely can supervise
> a small student's compute-allocation policy — the premise the adaptive-inference
> literature assumed, now measured, corrected for measurement noise, across 78
> architecture pairs.

#### `resnet110` × `resnet56` reaches T = 1.005

CI [0.979, 1.029], which **includes 1.0**. For that pair, cross-architecture
agreement is statistically indistinguishable from same-architecture,
different-seed agreement: *which* of the two networks you measure makes no
detectable difference to per-sample MSC. That is the strongest possible form of
the result, and it is a single sentence in the paper.

T slightly above 1.0 is not an error — disattenuation divides by an *estimated*
ceiling, so T can exceed 1 when that estimate is a little low. It should be
reported with the CI, never as a bare 1.005.

#### `convnext_femto` transfers like a transformer, not like a CNN

| across-CNN-family pairs | n | mean T | range |
|---|---|---|---|
| **including** `convnext_femto` | 10 | **0.766** | 0.732 – 0.808 |
| **excluding** `convnext_femto` | 33 | **0.912** | 0.824 – 0.966 |

A gap of **0.146**. Ranked by mean T over all its pairs, `convnext_femto`
(0.766) sits below every other CNN and just above `vit_tiny` (0.725) and
`mixer_nano` (0.724) — the bottom three of thirteen.

The candidate explanation is that ConvNeXt is a deliberately *transformer-ised*
CNN: large depthwise kernels, LayerNorm, inverted bottlenecks, GELU, few
activations. If per-sample compute-need tracks those design choices rather than
the convolution/attention label, this is what it would look like. **That is a
hypothesis, not a finding** — n=1 architecture, and it is confounded with the
300-epoch schedule and 62.67% accuracy. It is also cheap to test: add one more
modern CNN. Logged as **O-14**.

Note this cuts *against* the §1.2 grouping: `convnext_femto` has a high, very
CNN-like *ceiling* (0.7084) but a low, transformer-like *transfer*. Reliability
and transferability are separate properties, and this architecture separates
them. Worth a sentence, because it stops the two findings being read as one.

#### The original Phase 0 pair, for reference

resnet32x4-s1 → wrn_40_2-s1, depth axis, 1000 bootstrap resamples.

| τ | ρ_S raw | **T** | 95% CI | J₁₀ | n |
|---|---|---|---|---|---|
| 0.0 | 0.6441 | 0.8997 | [0.882, 0.921] | 0.430 | 10000 |
| **0.1** | 0.6772 | **0.9459** | [0.927, 0.966] | 0.440 | 8549 |
| 0.2 | 0.6675 | 0.9324 | [0.913, 0.952] | 0.456 | 7766 |
| 0.3 | 0.6499 | 0.9078 | [0.887, 0.927] | 0.465 | 7194 |
| 0.5 | 0.5966 | 0.8333 | [0.813, 0.854] | 0.474 | 6267 |

*source* `analysis/q3_transfer.csv` · gate ≥ 0.70 · **pass at every τ** · H3 predicted within-family > 0.8

### 1.5 Q4 — irreducibility (ΔR²) — **corrected; the headline number was 2.5× too high**

NB12 has now run on `train_holdout` with the **full 7-score battery**
(`msp, margin, entropy, ce_loss, el2n, forget_events, pred_depth`). **D-11 is
confirmed, and it mattered:**

| | Phase 0 (reported) | corrected | |
|---|---|---|---|
| split | `test` | `train_holdout` | |
| battery | 5 of 7 | **7 of 7** | |
| **ΔR²** | **0.254** | **0.1009** (median) | **−60%** |
| partial ρ | 0.489 | **0.2954** (median) | −40% |

**The 0.254 in `08_PHASE0_RESULTS.md` and the README overstates irreducibility
by a factor of 2.5 and must not be published.** Running Q4 without `el2n` and
`forget_events` handicapped the battery, which is exactly the direction that
flatters MSC. The prediction in D-11 was right.

The gate still passes, but the margin is now 2× not 5×:

| | value | gate | |
|---|---|---|---|
| median ΔR² | **0.1009** | ≥ 0.05 | ✓ 2× |
| median partial ρ | **0.2954** | ≥ 0.30 | ✗ **marginally under** |
| pairs clearing ΔR² ≥ 0.05 | 13/15 | — | |
| pairs clearing partial ≥ 0.30 | 7/15 | — | |

**H4's partial-correlation arm now fails by 0.0046.** That is too close to call
either way and must be reported as such rather than rounded into a pass.

#### The split that explains most of it

| pairs | n | median ΔR² | median partial ρ |
|---|---|---|---|
| CNN-only | 10 | **0.1211** | **0.3165** |
| involving `vit_tiny` or `mixer_nano` | 5 | **0.0504** | **0.1752** |

Transformer pairs carry less than half the unique information. This is **not a
separate finding** — §1.2 showed MSC is measured less reliably in those two
architectures, and a noisier measurement necessarily explains less variance.
The two results have to be reported together or a reader will double-count them.

#### ⚠ These 15 pairs are not a sample of the atlas — see D-18

`pairs[:15]` over an alphabetically sorted list gave **12 `convnext_femto` pairs
+ 3 `mixer_nano` pairs**. Every number in this section is therefore dominated by
the two most atypical architectures in the zoo: `convnext_femto` is the
transfer outlier (§1.4) and `mixer_nano` has the lowest ceiling (§1.2) *and* the
lowest PC1 (§1.3).

Both atypicalities push ΔR² **down**, so the corrected figures above are
probably a **lower bound** — the true atlas median is likely higher than 0.1009,
and the partial correlation likely clears 0.30. **Do not quote these numbers
until NB12 has been re-run on all 78 pairs.** The fix is committed; the re-run
is O-15 and costs minutes on CPU.

*source* `analysis/q4_irreducibility_all.csv` · **supersedes** `q4_irreducibility.csv`

#### The superseded Phase 0 measurement

| τ | R² battery | R² +MSC | **ΔR²** | 95% CI | partial ρ |
|---|---|---|---|---|---|
| 0.0 | 0.288 | 0.485 | 0.197 | [0.180, 0.214] | 0.434 |
| **0.1** | 0.321 | 0.575 | **0.254** | [0.234, 0.273] | 0.489 |
| 0.2 | 0.332 | 0.608 | 0.276 | [0.255, 0.297] | 0.491 |
| 0.3 | 0.338 | 0.620 | 0.282 | [0.260, 0.303] | 0.485 |
| 0.5 | 0.331 | 0.622 | 0.291 | [0.267, 0.314] | 0.447 |

Battery used: `msp, margin, entropy, ce_loss, pred_depth`.
Absent: `el2n, forget_events` — training-set scores, undefined on the test split.

*source* `analysis/q4_irreducibility.csv` · gate ≥ 0.05 · **pass**, but **recompute on `train_holdout` before publication**

### 1.6 Shuffled control

| | value |
|---|---|
| T (shuffled) | **0.0072** |
| ρ_S raw (shuffled) | 0.0052 |
| expected | ≈ 0 |

Confirms the per-sample tables are genuinely row-aligned and every correlation
above is computed on paired images rather than a coincidental index match.

### 1.7 Irreducible subpopulation |U_τ| / N

| τ | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| resnet32x4 | 0.0% | 8.6% | 13.7% | 17.6% | 25.5% |
| wrn-40-2 | 0.0% | 8.1% | 14.1% | 18.9% | 27.5% |

Excluded from all correlations. The close agreement in *size* is suggestive but
not evidence about *membership*; testable in the atlas.

### 1.8 Measured cost model — all 13 measured architectures

Used to calibrate the scheduler. See D-10. Cost unit = (s/epoch) ÷ 8.32.
**These are observations, not scheduler inputs** — per D-12, `ARCH_COST_HINT`
must stay static or ownership drifts between sessions.

| arch | s/epoch | GPU-h | kWh | cost unit | hint | error |
|---|---|---|---|---|---|---|
| `resnet32x4` | 43.26 | 2.88 | 0.216 | 5.20 | 5.20 | anchor |
| `mobilenetv2` | 32.97 | 2.20 | 0.166 | 3.96 | — | — |
| `shufflenetv2` | 33.29 | 2.22 | 0.130 | 4.00 | — | — |
| `vit_tiny` | 31.85 | 2.65 | 0.199 | 3.83 | — | — |
| `wrn_40_2` | 29.06 | 1.94 | 0.146 | 3.49 | 3.38 | +3.3% |
| `wrn_40_1` | 24.66 | 1.64 | 0.107 | 2.96 | — | — |
| `convnext_femto` | 23.88 | 1.99 | 0.148 | 2.87 | — | — |
| `mixer_nano` | 19.05 | 1.59 | 0.118 | 2.29 | — | — |
| `vgg13` | 16.78 | 1.12 | 0.082 | 2.02 | — | — |
| `wrn_16_2` | 14.16 | 0.94 | 0.071 | 1.70 | — | — |
| `vgg8` | 11.16 | 0.74 | 0.055 | 1.34 | — | — |

`SECONDS_PER_COST_UNIT = 8.32`. The `wrn_40_2` hint is 3.3% low against the
Phase 1 run — within tolerance, no change warranted.

### 1.9 FLOPs are a poor predictor of time and energy across architectures

Falls out of §1.8 for free, and it bears on how MSC may be described.

| arch | GFLOPs | s/epoch | **s per GFLOP** |
|---|---|---|---|
| `shufflenetv2` | 0.0925 | 33.29 | **360.0** |
| `convnext_femto` | 0.1264 | 23.88 | 188.9 |
| `mobilenetv2` | 0.1828 | 32.97 | 180.4 |
| `wrn_40_1` | 0.1680 | 24.66 | 146.7 |
| `vgg8` | 0.1363 | 11.16 | 81.9 |
| `wrn_16_2` | 0.2032 | 14.16 | 69.7 |
| `mixer_nano` | 0.3430 | 19.05 | 55.5 |
| `vit_tiny` | 0.6944 | 31.85 | 45.9 |
| `wrn_40_2` | 0.6581 | 29.06 | 44.2 |
| `vgg13` | 0.4576 | 16.78 | 36.7 |
| `resnet32x4` | 2.1494 | 43.26 | **20.1** |

**A 17.9× spread.** The extremes are stark: `shufflenetv2` does **7.1× fewer
FLOPs** than `wrn_40_2` yet takes **14.5% more wall-clock time** and 89% of the
energy. Depthwise and grouped convolutions are memory-bandwidth-bound on a T4,
not arithmetic-bound.

**Does this threaten MSC?** No — ρ(c) = FLOPs(f,c)/FLOPs(f,c_full) is a ratio
taken *within a single architecture*, so an architecture-level FLOPs/time
mismatch cancels. But it constrains the language:

- Never write "MSC of 0.65 means 35% less time" or "35% less energy." It means
  35% fewer FLOPs. Those are different quantities and this table proves it.
- Cross-architecture statements about *absolute* savings must be in FLOPs, or
  must be re-measured in the unit actually claimed.
- The open empirical question is whether the ratio holds *within* an
  architecture across exits — i.e. does an early exit at ρ=0.5 on `mobilenetv2`
  actually take half the time? Per-budget latency may already be in
  `telemetry/`; if so this is a free extra result. Logged as **O-11**.

This is a limitations paragraph the paper needs regardless, and we now have the
measurement to write it rather than hedge it.

*source* `runs/*/summary.json` — `total_time_sec`, `total_energy_kwh`, `full_flops`

---

## 2. Defect log

Every bug found, with a **contamination analysis** — the question a reviewer
would ask, and the one we need to have answered before writing.

### D-23 · The teacher's exit heads were retrained on every MSC-KD run — D-16 was not cosmetic

**Severity:** ~20 wasted epochs **per run**, nine times over · **Status:** **fixed**
**Found:** 2026-08-04, from a user log showing `exits ep 19/20` on run 1 of 9

```
[MSCKD] teacher exit heads missing -- training them now (backbone frozen)
exits ep 19/20:  51%
```

`run_oracle` **writes** the trained exit heads to `runs/{run}/exit_heads.pt`.
`train_msc_kd` **read** `runs/{run}/checkpoints/exit_heads.pt`. The file was
never found, so every MSC-KD run retrained the teacher's heads from scratch —
for a file that has been sitting on HuggingFace since NB08. Confirmed directly:
`runs/p1-resnet32x4-cifar100-base-s1/exit_heads.pt`, **360,048 bytes**, at the
run root.

**This is a repeat of a defect I closed as harmless.** D-16 recorded the same
path split and concluded:

> *"Contamination: none. Nothing reads the path by convention; the loader is
> handed the same value the writer used. Purely a documentation/tidiness
> defect."*

**Three call sites read it by convention**, and one of them was in the hot path
of the entire method. The claim was not checked — I read the writer, saw it was
self-consistent, and never grepped for readers. Classifying a defect as
cosmetic is a *finding*, and it needed the same evidence as any other.

**Contamination analysis.**

- **No number is wrong.** Retrained exit heads are *correct* exit heads — the
  procedure is deterministic given the frozen backbone and the seed. This cost
  time, not validity.
- The waste is ~20 epochs of exit-head training per MSC-KD run. At 9 runs that
  is a substantial fraction of the notebook's total budget, spent recomputing
  one 360 KB file.
- It also **masked D-21 and D-22**: both defects sit *after* the exit-head
  block, so every debugging cycle paid the retraining cost before reaching the
  actual bug. Three defects, three ~1-hour round trips, one of them entirely
  self-inflicted.

**Fix.**

- `exit_heads_path(work, run_id)` — one canonical accessor, used by the writer
  and every reader. There was no such function, which is *why* they diverged.
- `find_exit_heads(work, run_id)` — returns the canonical path, or the legacy
  `checkpoints/` one if that is what exists, so anything written before this fix
  still loads instead of retraining.
- `train_msc_kd` now pulls the teacher's run directory from HF before concluding
  the heads are absent, and logs **where it looked** on both branches.
- The two status reporters (`Session.run_report`, `preflight`) checked the wrong
  path too, so they have been reporting `exit_heads: False` for every run in the
  project. Both now accept either location.

**Guard added:** 5 self-checks asserting the writer's path is what the reader
finds, that the canonical location is the run root, that the legacy path is
still honoured, and that canonical wins when both exist. Self-checks 202 → **207**.

**Also shipped: O-19, the dry run.** `msckd_dry_run()` now executes the entire
MSC-KD step — `MSCStudent` under `autocast`, `MSCLoss`, `backward`, and one
`msckd_history_row` through `append_history_row` — on a **2-image synthetic
batch** before any teacher work. Under a second, no dataset, no sweep. **It
would have caught D-21 and D-22 immediately**, and it now fails the run with
`no GPU time has been spent` rather than an hour in. This should have been added
when O-19 was first opened rather than filed for later; filing it cost two more
hour-long cycles.

### D-22 · Five wrong column names killed every MSC-KD run at the end of epoch 0

**Severity:** blocked NB13 · **Status:** **fixed** · **Found:** 2026-08-04, immediately after D-21

```
File "msc_lib.py", line 6781, in train_msc_kd
    w.writerow(row)
ValueError: dict contains fields not in fieldnames:
    'recall', 'f1_score', 'throughput_img_s', 'grad_norm', 'precision'
```

The MSC-KD history row used five names that are not columns. The schema has
`f1_macro`, `precision_macro`, `recall_macro`, `grad_norm_mean` and
`throughput_train_img_s`; the row wrote `f1_score`, `precision`, `recall`,
`grad_norm`, `throughput_img_s`. `csv.DictWriter` raises on any key it does not
recognise — **at the end of the first epoch**, after the training is done and
the time is unrecoverable.

**The two training paths disagreed about what an unknown column means, and both
answers were wrong.**

| | behaviour | consequence |
|---|---|---|
| `train_msc_kd` | `DictWriter` default → **raises** | every run dies at epoch 0 |
| `train_backbone` | `extrasaction="ignore"` → **silently drops** | a typo becomes a column of blanks in a 171-column table nobody reads by eye |

The second is the more dangerous of the two on a project whose standing
instruction is *"we only train once — collect every detail."* A crash gets
fixed; a silently missing column gets discovered while writing the paper.

**Contamination analysis.** No MSC-KD run ever wrote a history row, so nothing
is lost there. For the atlas, `train_backbone` built its row from explicit
literals plus `**g, **sysagg, **pw` and then `row.setdefault(_c, NA)` over every
field — so the 171 columns were all present and only genuinely-extra
machine-specific keys were dropped. **The 44 atlas runs are unaffected**, but
that was luck rather than a guarantee, because nothing checked.

**Fix.**

- `append_history_row(path, row, strict=True)` — one writer for both paths.
  `strict=True` raises **and names the column you probably meant**
  (`Did you mean: {'f1_score': ['f1_macro', ...]}`). `strict=False` still writes,
  because `train_backbone`'s GPU/power dicts legitimately vary by machine, but
  **logs what it dropped**, once per key. Silent loss becomes visible loss.
- `msckd_history_row(...)` — the row extracted into a pure function so its key
  set can be validated offline in microseconds instead of after an hour of real
  training on a real teacher.

**And a gap worth more than the bug.** The old row recorded no identity columns
and — for a *method* notebook — **threw away the three-term loss decomposition
entirely.** `L_CE`, `L_KD` and `L_MSC` were computed every epoch, printed to
stdout, and never written down. The whole argument of the paper is how those
three trade off against each other; the curve was being discarded. The new row
records all three plus `alpha`, `beta`, `temperature`, `run_id`, `arch`, `seed`,
`phase`, `method`, `config_hash`, `is_best` and `samples_seen` — 42 columns
where there were 23.

**Guard added:** 12 self-checks. They build a representative row and assert
every key is in `HISTORY_FIELDS`, that each of the five bad names is gone, that
the three loss components **sum to the total**, that `is_best` compares against
the *previous* best rather than the one just written, that the header is written
exactly once, that strict mode raises with a usable suggestion, and that
non-strict mode still writes. Self-checks 190 → **202**.

**The pattern across D-21 and D-22.** Both were in `train_msc_kd`, both were
trivial, and both cost a full hour of GPU setup to discover — because
`train_msc_kd` does teacher loading, exit-head training and a 50,000-image sweep
*before* the first student batch, and the history write is at the *end* of the
first epoch after that. Two bugs, two hours, both findable in milliseconds.
**O-19 (dry-run one batch early) is now the highest-value engineering item in
the project** — it would have caught both.

### D-21 · The MSC-KD loss cannot run under AMP — the method's own training step was never tested

**Severity:** **blocked NB13 entirely** — the method could not train at all
**Status:** **fixed** · **Found:** 2026-08-04, ~1 h into a real multi-account run

```
File "msc_lib.py", line 5652, in forward
    bce = F.binary_cross_entropy(suff_pred.clamp(1e-6, 1 - 1e-6), ...)
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss
are unsafe to autocast. ... combine the two layers using
torch.nn.functional.binary_cross_entropy_with_logits
```

`F.binary_cross_entropy` is on PyTorch's **banned list** under AMP autocast, and
every training path in this project runs under `torch.amp.autocast`. So
`MSCLoss` — the loss the entire method rests on — **could never have executed**.
Not "was slow", not "was wrong": it raised on the first batch.

**The fix is the one torch's own error message gives.** The head already
computes `sigmoid(theta_k - u(x))`, so a clean logit was sitting right there:

- `OrdinalSufficiencyHead.logits(feat)` returns `theta_k - u(x)`;
  `forward()` is now `sigmoid(logits())` and is bit-identical to before.
- `MSCStudent.forward(x, suff_logits=False)` — the training loop opts in, and
  routing/inference keep getting probabilities, so no other call site moves.
- `MSCLoss` uses `binary_cross_entropy_with_logits`.

This is strictly better than a workaround. The `.clamp(1e-6, 1 - 1e-6)` the old
code needed was papering over the `log(0)` that the fused logit kernel avoids by
construction, so the fix removes a numerical hack as well as the crash.
**Monotonicity is untouched** — `thresholds()` is increasing and sigmoid is
monotone, so `s_k` is non-decreasing in `k` whether or not the sigmoid is
applied. The architectural guarantee in §3 still holds. Same change applied to
the reference implementation in `msc_torch.py`.

**Contamination analysis.** None. No MSC-KD run ever completed, so no number
anywhere derives from this code path. Q1–Q4 do not touch `MSCLoss`.

**Why it survived — and this is the part that matters.** The NB00 preflight
builds every architecture, runs `forward_features`, `forward_prefix`, a
backward pass, and checks the budget tables. It is a genuinely good preflight.
**It never constructs `MSCStudent`, never calls `MSCLoss`, and never enters an
autocast block.** So the one component that is the project's actual
contribution — the three-term loss and the ordinal head — had **zero** test
coverage, while the fifteen backbones we did not write had thorough coverage.

That inversion is the lesson. Testing effort had gone where the code was
unfamiliar, not where it was load-bearing. The failure surfaced only after a
teacher checkpoint, exit heads, and a full multi-exit sweep over 50,000
training images had been paid for — about an hour of GPU time before the first
student batch is even attempted, which is why it took a real run to find.

**Guard added:** the torch-gated self-test now builds an `MSCStudent`, runs a
full forward → `MSCLoss` → `backward` **inside `torch.amp.autocast`**, and
asserts the loss is finite. CPU autocast enforces the same ban as CUDA, so this
reproduces the failure with no GPU and runs in NB00 on every account. Two more
checks pin that `forward()` is exactly `sigmoid(logits())` and that the
sufficiency curve is still monotone in `k`, so the refactor cannot have changed
what the head computes.

**Related open item.** `train_msc_kd` does roughly an hour of expensive setup —
teacher checkpoint load, exit-head training, full training-set sweep — *before*
the first student step. A one-batch dry run of the student+loss immediately
after the teacher is ready would have failed in seconds instead of an hour.
Logged as **O-19**.

### D-20 · The D-19 verification cell raised a false alarm on healthy runs

**Severity:** no data lost, but it told the user their work was at risk when it
was not · **Status:** **fixed** · **Found:** 2026-08-04, minutes after shipping D-19

The verification cell added to fix D-19 printed:

```
[VERIFY] 0/9 run(s) confirmed on HuggingFace
    NOT ON HF: p3-resnet8x4-cifar100-mscKDshuffromresnet32x4-s1
    ... (9 of 9)
[ALARM] 9 run(s) are NOT on HuggingFace. DO NOT close this session.
        Closing now means retraining them.
```

**Every word after "0/9" was false.** The same output block reported
`uploaded=1283 commits=8 pending=0` — the queue had drained completely, and all
nine runs had `checkpoints/ckpt_last.pt` on HF. They would have resumed from
their exact epoch, losing nothing.

**The bug.** `confirm_on_hf` defaulted to `require=("summary.json",)`, i.e. it
asked *"did this run finish?"* — then phrased the answer as *"is this run
safe?"*. Those are different questions. The session had been running 1.08 h
against a job needing ~30 GPU-h, so of course nothing had finished. **Being
unfinished is the normal state of a paused run, not a failure.**

Safety has three states, and the check collapsed them into two:

| state | evidence | safe to close? |
|---|---|---|
| finished | `summary.json` | yes, nothing left |
| **resumable** | `checkpoints/ckpt_last.pt` | **yes — resumes at its epoch** |
| at risk | neither | **no** |

**Contamination analysis.** No data lost, no number affected. The cost was a
false alarm on a live session and the user having to ask whether to trust it.

**But that cost is not small, and it is the same cost as D-17.** A check that
fires on healthy data teaches you to ignore it, and the next alarm is the real
one. D-19 was itself a silent-failure defect; shipping its fix with a false
alarm attached is close to the worst outcome, because it makes the mechanism
that was supposed to restore trust into another thing to second-guess.

**Why it happened.** `require=("summary.json",)` was chosen while writing the
guard for *completed* runs — D-19's story was about nine runs that had finished.
The in-progress case was never considered, and the self-tests I added for D-19
covered `already_finished` thoroughly and `confirm_on_hf` **not at all**. I
tested the function whose logic I had reasoned about and skipped the one I had
merely written.

**Fix.** Three-way classification, `RESUMABLE` reported as safe with its epoch
from the ledger, and the ALARM reserved for runs with neither artifact. The
summary line now reads `9 run(s): 0 finished, 9 resumable, 0 at risk` followed
by *"Nothing is at risk … Safe to close the session."*

**Guard added:** 6 self-checks pinning all three states, including that a
checkpoint alone is `resumable` (the exact false-alarm case) and that a
`config.yaml`/`STATUS.json` pair is **not** reassurance — those are written
before any real work exists. Also pins the `make_run_id` hyphen-stripping that
produces the odd-looking `mscKDshuffromresnet32x4`, since the report prints
those ids and they read like corruption: stripping is deliberate, it is what
keeps the `-` split into exactly five fields unambiguous. Self-checks 184 → **190**.

### D-19 · NB13 restarted completed MSC-KD runs from epoch 0 — resume was never wired into the method notebooks

**Severity:** **the worst defect in the project — silently destroys GPU-hours**
**Status:** fixed in code, HF verification pending · **Found:** 2026-08-04, reported by the user

Nine MSC-KD runs were completed across five accounts. The sessions were closed
without checking the output. On re-opening NB13 with one worker, **it began
training from epoch 0.**

**Three independent failures had to line up, and all three did.**

**(1) `run_all` only recognised one entry point.**

```python
if done_fn is None and fn is getattr(self, "oracle", None):
    done_fn, stage = self.measured, "measure"
```

That names a single function by identity. NB13 passes a *closure* over
`train_msc_kd`, so it falls straight through with `done_fn = None`, and
`plan_work` then defines "done" as **the ledger and nothing else**:

```python
done = {r for r in universe if latest.get(r, {}).get("state") in done_states}
```

`Session.trained()` already existed and does the right thing — ledger **or**
`summary.json` — and no notebook ever passed it. The D-09 fix was applied to the
one code path that had failed and never generalised. This is the same defect
class, one notebook over.

**(2) Nothing pulled the run's own checkpoint back.** `load_checkpoint` returns
`start_epoch = 0` when the file is absent. Kaggle wipes `/kaggle/temp` between
sessions, so on a fresh session *every* run's checkpoint is absent. `run_oracle`
pulled its own run directory before deciding; **neither training entry point
did.** They relied entirely on the notebook having called `sync_state` with the
right scope near the top — an invisible coupling between a cell on page one and
a decision taken deep in the library.

**(3) `can_claim` is ledger-only too.** So a lost completion event meant
`can_claim` → True, no checkpoint → `start_epoch = 0`, and the run trains again.
No error, no warning. **It looks exactly like normal work.**

**And the alarm that should have caught it was dead code:**

```python
unfinished = [r for r in plan.mine if done_fn is not None and not done_fn(r)]
```

With `done_fn = None`, this list is *always* empty, so the D-09 zero-work ALARM
could never fire in NB13 — the one notebook where it was most needed.

**Contamination analysis.**

- **No published number is affected.** This destroys compute, not correctness.
  A re-trained run is a *valid* run; it is just paid for twice.
- **Nothing is silently corrupted.** If the original nine runs did reach HF,
  their `summary.json` files are intact and the fix will find them.
- **The exposure is up to ~30 GPU-hours** (9 MSC-KD runs) if the completed work
  is not recoverable.
- **`train_backbone` had the same two holes** and was only saved by NB01 and the
  atlas notebooks calling `sync_state()` with checkpoints in scope. That is luck,
  not design — NB13 called `sync_state` too, and it was not enough.

**Why it survived every previous audit.** Five defects in this project have been
about resume (D-05, D-06, D-09, D-12, and now D-19). The acceptance test from
D-06 exercises `train_backbone` *within a single session*, where the checkpoint
is still on local disk. **The failure mode is specifically cross-session**, and
no test crosses a session boundary. D-06's lesson was "a test that validated
nothing"; this is its sequel.

**Fix — four changes, all committed.**

1. **`run_all` defaults to `self.trained`** for any non-oracle entry point,
   instead of falling through to the raw ledger. A lost ledger event alone can
   no longer cause a re-run, and the zero-work ALARM is live again because
   `done_fn` is never `None`.
2. **`ensure_run_local(hub, work, run_id)`** — pulls the run's own artifacts
   from HF before `load_checkpoint` reads an absent file as "never started".
   Called from **both** `train_backbone` and `train_msc_kd`. Free when the
   checkpoint is already local.
3. **`already_finished(hub, work, run_id, cfg, registry)`** — checks
   `summary.json` for `num_epochs_run >= num_epochs` and returns the cached
   result. In `train_msc_kd` this runs **before the teacher sweep**, which is
   the expensive part; discovering "already done" after paying for a full
   multi-exit pass over 50,000 images would be no use. It also **repairs the
   ledger** when the artifact and the ledger disagree, so the next worker
   inherits the answer.
4. **`Session.confirm_on_hf(run_ids)`**, now the last cell of every training
   notebook. `finish()` drains the upload queue and prints `[SESSION] done`,
   which *reads* like confirmation and is not one — draining says the queue
   emptied, not that the files landed. The new cell asks the repository and
   prints `NOT ON HF` per run with an explicit "do not close this session".
   **This is the fix that addresses what the user actually did**: closing a tab
   should not be able to lose work, and if it can, the notebook must say so
   before the tab is closed.

**Guard added:** 8 self-checks, including that a **79/240 partial run is not**
mistaken for finished (the guard must not skip resumable work), that
`force_rerun` still overrides, and that a corrupt `summary.json` does not crash
it. Self-checks 176 → **184**.

**Open:** confirm on HF whether the nine `p3-*-mscKD*` runs are present. If they
are, the fix makes NB13 skip them and nothing is lost. If they are not, they
must be retrained — and no code change can recover them, only prevent a repeat.

### D-18 · Analysis sampled alphabetically, not representatively

**Severity:** **affects reported numbers (Q4)** · **Status:** fixed in code,
**re-run pending** · **Found:** 2026-08-04 auditing the NB10–NB12 outputs

Two separate bugs with the same root cause: *code that silently selects a
subset, and a subset that is not a sample.*

**(a) The truncation caps.** NB12 ran Q4 on `pairs[:15]` and NB11 ran its τ-curve
check on `pairs[:8]`, both over `sorted(seed1)`. Alphabetically the first
architecture in our zoo is `convnext_femto`, so:

- Q4's "atlas" result = **12 `convnext_femto` pairs + 3 `mixer_nano` pairs**.
- The τ-stability check = **8 `convnext_femto` pairs**, i.e. one architecture,
  and no CNN→transformer pair at all — the check could not have detected a
  τ-dependence in the pair type the paper cares most about.

This is worse than a small sample: it is a **biased** one, and biased toward the
two most atypical members of the zoo. `convnext_femto` is the transfer outlier
(§1.4, mean T 0.766 vs 0.912 for other across-CNN pairs) and `mixer_nano` has
the lowest ceiling (§1.2) and lowest PC1 (§1.3). Both push ΔR² down.

**(b) `seed1` dropped an architecture silently.** All three analysis notebooks
selected runs with

```python
seed1 = {m['arch']: r for r, m in runs.items() if m['seed'] == 1}
```

`vgg8` has two measured seeds and the **second-highest noise ceiling in the
atlas** (0.7248) — but its seed 1 was never measured (D-15). So `vgg8` was
excluded from Q2, Q3 and Q4 for a bookkeeping reason rather than a data reason,
and a dict comprehension cannot announce what it skipped. The atlas analysis
covered **13 architectures while reporting itself as the atlas**.

**Contamination analysis.**

- **§1.5 (Q4) is affected and is flagged in place.** The median ΔR² of 0.1009
  and median partial ρ of 0.2954 are computed on the biased 15. Because both
  atypicalities depress ΔR², these are most likely a **lower bound** — which
  matters, because the partial correlation currently misses its gate by 0.0046.
  A biased-low estimate sitting 0.005 under a threshold is not a result.
- **§1.4 (Q3) is NOT affected.** The transfer matrix ran on all 78 pairs; only
  the τ-curve *robustness check* was truncated. The headline T values stand.
- **§1.3 (Q2) is NOT affected.** Q2 is per-run, not per-pair, so no cap applied.
- **§1.2 (Q1) is NOT affected** — same reason.
- `vgg8`'s absence costs one architecture from all three analyses. Nothing
  computed is wrong; the coverage is smaller than claimed.

**Why it survived review.** Both caps look like deliberate cost control, and
they were — Q4 with 500 bootstraps over 78 pairs is slower than over 15. The
error was pairing a defensible *budget* with an indefensible *selection rule*.
`[:15]` is only a sample if the list order is unrelated to the quantity being
measured, and `sorted()` guarantees it is not.

**Fix.**

- `representative_runs(runs, require=ceilings)` — picks the lowest **usable**
  seed per architecture instead of insisting on seed 1, and takes an explicit
  membership test for "measured". NB11 now also *prints which architectures are
  missing and why*, so the next omission is loud instead of silent.
- `stratified_pairs(pairs, kind_fn, per_kind=3)` — samples up to N pairs from
  **each** pair type. The τ check now covers all four pair types by construction.
- The Q4 cap is removed entirely; it now runs all pairs and additionally reports
  the CNN-only vs transformer-involving split, so the §1.2 reliability effect and
  the §1.5 irreducibility effect cannot be silently confounded.
- 8 self-checks pin all of this, including an assertion that the *old* idiom
  would have dropped `vgg8` and that plain truncation yields a single kind.
  Self-checks 168 → **176**.

**Still to do:** re-run NB10, NB11 and NB12 with the fixed selection (O-15).
All three are CPU-only and take minutes. Until then §1.5's numbers are
provisional and are marked as such.

### D-17 · The shuffled control cried wolf — a sanity check that was itself unsound

**Severity:** blocked NB11 · **Status:** **fixed** · **Found:** 2026-08-04 by NB11 halting

NB11 stopped at the scrambled-target control:

```
[ALARM] SHUFFLED CONTROL FAILED: T=-0.0506 (expected ~0). This is a BUG, not a
finding -- check sample_idx alignment between p1-convnext_femto-...-s1 and
p1-resnet20-...-s1.
24/25 pairs pass
AssertionError: Scrambled control FAILED -- tables are misaligned. Bug.
```

**There was no misalignment.** The check was miscalibrated, and it announced a
bug in the data when the bug was in itself.

**The arithmetic.** T is the *disattenuated* statistic, so the underlying raw
rank correlation was `T × √(c_a·c_b) = −0.0506 × 0.6747 = −0.0341`. Under a
random permutation the correlation of two rank vectors has mean 0 and variance
exactly `1/(n−1)`; at n ≈ 5,872 that is SD **0.0131**. So the observed value was
**z = −2.61**, a two-tailed p of 0.009. Across 25 pairs the chance of seeing at
least one such draw is **20%**; across the full 78 pairs it is **50%**. The
control was a coin-flip away from firing on a perfectly healthy pipeline.

**Three independent calibration errors.**

1. **Sample-size blind.** `abs(T) < 0.05` is a constant. At n = 6,000 it is 2.6σ;
   at n = 25,000 it is 5σ. The same threshold means completely different
   strictness depending on how many samples survived the τ mask — and the mask
   changes n by architecture.
2. **Ceiling-dependent, in the worst possible direction.** T divides by
   `√(c_a·c_b)`, so a *low-ceiling* pair reaches the same T at a *smaller* raw
   correlation. Measured on our own ceilings:

   | pair | denominator | trips at \|ρ\| | in σ | false-alarm rate |
   |---|---|---|---|---|
   | `vit_tiny` × `mixer_nano` | 0.5472 | 0.0274 | 2.10σ | **3.6%** |
   | `convnext_femto` × `resnet20` | 0.6746 | 0.0337 | 2.58σ | 1.0% |
   | `resnet32x4` × `vgg8` | 0.7252 | 0.0363 | 2.78σ | 0.5% |

   The control was **~7× more likely to false-alarm on the low-ceiling pairs** —
   which are exactly the ViT and Mixer pairs carrying the headline finding of
   §1.2. A sanity check biased against your own most important result is worse
   than no check.
3. **Multiplicity blind.** ~1% per pair × 78 pairs. Not *whether*, only *when*.

**A fourth problem, and the one that settles it.** The test was two-sided
against a **one-sided failure mode**. Index leakage makes a shuffle behave like
a non-shuffle — it drives the correlation *up*, toward the true transfer of
~0.6. There is no mechanism by which misaligned tables produce a *small negative*
correlation. The observed −0.0341 could not have been the thing being tested for,
whatever threshold was used.

**Contamination analysis.**

- **No published number is affected.** The control gates Q3; it never enters one.
  The Phase 0 shuffled control (T = 0.0072, §1.6) passes under old and new rules
  alike, and its raw ρ of 0.0052 is z ≈ 0.5 — genuinely null.
- **No result was wrongly accepted.** The failure mode was false-alarm, not
  false-pass. Nothing got through that should not have.
- **The cost was a stop, and nearly worse.** The real risk is the *other*
  branch: a check that fires on healthy data trains you to dismiss it, and the
  next alarm is the real one. That is the damage this defect could have done.

**Fix.** `shuffled_control_verdict(rho, n, z_max=5.0, rho_floor=0.10)`:

- operates on the **raw** correlation, so ceilings never enter the decision;
- compares against the **exact** permutation null `1/√(n−1)` — exact rather than
  asymptotic, which matters because MSC is heavily tied (it takes only K
  distinct budget values, so a normal approximation would be the wrong tool);
- requires **both** `|z| > 5` **and** `|ρ| > 0.10`. Both terms carry weight:
  without the z term the rule is sample-size blind again; without the ρ floor,
  n = 10⁶ makes ρ = 0.02 a 20σ "failure" that is true and meaningless.
- Takes the **worst of 3 permutations**, so one lucky draw cannot certify a
  broken pipeline.
- Calls **`assert_aligned` directly.** This is the point. `analyse_q3_transfer`
  has always called it (line 5978); the control was an indirect proxy for a
  hash comparison that was already available and is definitive. The proxy now
  sits behind the real check rather than in front of it.

At 5σ with a 0.10 floor, a genuine leak lands at **z ≈ 46** and fails by a wide
margin, while the family-wise false-alarm rate over ~100 pairs is ~10⁻⁴.

**Also fixed in NB11 while here:**

- the control ran on `pairs[:25]` of 78 — **53 pairs were never tested.** Now all.
- `save_analysis` ran *after* the assert, so a genuine failure would have thrown
  away the evidence needed to diagnose it. Now saves first, asserts second.
- the cell printed a bare dict; it now reports ρ, z, n and the null SD, sorted
  worst-first, so the reader can see how far from the threshold they are.
- `disattenuated_transfer` gained an `n_boot <= 0` short-circuit — the control
  never used the CI it was paying 200 resamples per pair for.

**Guard added:** 11 self-checks pin the rule offline, including the exact
(−0.0341, n=5872) case as a regression, a synthetic leak at ρ=0.60, and both
degenerate corners (huge-n/tiny-ρ, tiny-n/moderate-ρ). Self-checks 157 → **168**.
The decision rule was deliberately split into a pure function so it is testable
without measured parquet files on disk — **the old rule was unreachable by any
offline test, which is why it shipped wrong.**

### D-16 · `exit_heads.pt` is written outside the documented layout

**Severity:** cosmetic · **Status:** open, trivial · **Found:** 2026-08-04 audit

`06_DATA_SCHEMA.md` documents trained exit heads at
`runs/{run_id}/checkpoints/exit_heads.pt`. They are actually on HF at
`runs/{run_id}/exit_heads.pt` — one level up. `run_oracle` passes `L["base"]`
where it should pass `L["checkpoints"]`.

**Contamination:** none. Nothing reads the path by convention; the loader is
handed the same value the writer used. Purely a documentation/tidiness defect.

**Fix:** either move the write into `checkpoints/` or correct the schema doc.
Moving it is the better option but invalidates 43 already-pushed paths, so the
cheaper correct action is to **document the real path** and leave the data alone.
Deferred to the schema-doc pass, tracked as part of O-12.

### D-15 · Six atlas runs trained but never measured; one architecture lost entirely

**Severity:** medium — costs one of 15 pre-registered architectures
**Status:** **open** · **Found:** 2026-08-04 audit

NB08 measured 39 of the 45 Phase 1 runs. The gaps are not random — they cluster
at the end of the work queue:

| run | trained | measured |
|---|---|---|
| `p1-wrn_16_2-…-s1` | ❌ | ❌ |
| `p1-wrn_16_2-…-s2` | ✅ 2026-08-04T04:32Z | ❌ |
| `p1-wrn_16_2-…-s3` | ✅ | ❌ |
| `p1-wrn_40_1-…-s3` | ✅ | ❌ |
| `p1-vgg8-…-s1` | ✅ | ❌ |
| `p1-mixer_nano-…-s3` | ✅ | ❌ |

**Why it happened.** These are the *cheapest* runs in the zoo — `wrn_16_2` at
cost 1.70 and `vgg8` at 1.34 are the two least expensive architectures. The LPT
bin-packer places the smallest items last, so they land at the tail of every
worker's queue and are the first casualties when a Kaggle session hits its
time limit. `wrn_16_2-s2` finished training at 04:32Z and the repo's last write
was 05:28Z — under an hour later. The session ended before measurement reached it.
This is expected scheduler behaviour, not a scheduler bug.

**Contamination analysis.**

- **`wrn_16_2` contributes nothing to any analysis.** With zero measured seeds it
  has no ceiling, appears in no Q1 row, and can enter no Q3 pair. The atlas is
  effectively **14 architectures, not 15**, and every "15 architectures" claim in
  the protocol, README and Q1 tables is currently false. This is the material
  consequence.
- `wrn_40_1`, `vgg8` and `mixer_nano` each still have the 2 seeds a ceiling
  requires, so their §1.2 numbers are **valid but minimum-power** — a 2-seed
  ceiling has a wider CI than a 3-seed one, and `mixer_nano` is one of the two
  architectures carrying the headline low-ceiling finding. Strengthening it is
  worth the 0.5 GPU-h.
- No number already recorded is *wrong*. This is missing data, not bad data.

**Fix:** re-run NB08. It is idempotent — `plan_work(..., done_fn=, stage=)` skips
the 39 already-measured runs, so the marginal cost is training `wrn_16_2-s1`
(~0.94 GPU-h) plus measuring six runs (~3 GPU-h). Under 4 GPU-hours to close.

**Guard to add:** NB08 should end by printing measured-vs-trained coverage per
architecture and **ALARM when any architecture has fewer than 2 measured seeds**,
by exact analogy with the zero-work ALARM added for D-09. A silent 39/45 that
looks like success is precisely the failure mode D-09 was about. Tracked as O-12.

### D-14 · `mobilenetv2` is compared against a half-width baseline

**Severity:** **affects a reported number** · **Status:** open · **Found:** 2026-08-04

`ARCH_REFERENCE` records `reference_accuracy = 64.60` for `mobilenetv2`, and the
trained model reaches **70.10%**, producing an apparent **+5.50** — by far the
largest margin over a published reference in the whole atlas, and about 4×
the next largest (`vgg8`, +1.37).

That margin is an artifact. The 64.60 figure comes from the CRD/mdistiller
student tables, where the entry labelled "MobileNetV2" is `mobile_half` —
**MobileNetV2 at width multiplier 0.5**, roughly 0.81 M parameters. Our model has
**2.35 M parameters**. We are comparing a full-width network against a
half-width baseline and calling the difference a win.

**How it was caught.** Not by a test — by noticing that +5.50 was implausible
next to every other Δ in the table and checking the parameter count against the
baseline's. Anomalously *good* results get less scrutiny than bad ones, which is
exactly why this survived.

**Contamination analysis.**

- The **training is fine** and the **MSC measurement is fine.** `mobilenetv2`'s
  ceiling (0.6880) and every per-sample quantity are unaffected — none of them
  reference the published number.
- What is wrong is one **claim**: the sentence "every architecture beats its
  published reference." With `mobilenetv2` excluded that sentence is still true
  of the other 9 CNNs with references, at margins of +0.32 to +1.43.
- `08_PHASE0_RESULTS.md` and `README.md` are unaffected — neither mentions
  `mobilenetv2`.
- **`shufflenetv2` was checked and is sound**: 1.36 M parameters against the
  ~1.36 M of the ShuffleNetV2 ×1.0 baseline, +1.43 over 70.50.

**Fix, two options.** Either (a) set `reference_accuracy = null` for
`mobilenetv2`, as already done for the three modern architectures, and claim no
Δ; or (b) switch the zoo to `mobile_half` so the reference applies. **(a) is
correct** — the wider model is the more useful atlas member, and an honest null
beats a flattering comparison. Do not report the +5.50 anywhere.

**Guard to add:** `recipe_ok` should also assert that the parameter count is
within a tolerance of the reference model's, not just that accuracy cleared the
reference. A reference number without a parameter count attached is unfalsifiable.
Tracked as O-12.

### D-13 · NB08 crashed on `int(None)` — run identity read from the ledger
**Found** user ran NB08: `TypeError: int() argument must be ... not 'NoneType'`
at `sess.config(r['arch'], seed=int(r['seed']))`.
**Cause** NB08 built its work list from ledger events and read `arch`/`seed` out
of them. Not every event carries those fields: `repair_ledger` reconstructs a
completion from `history.csv` and knows only the run_id, so it writes
`{run_id, state, best_accuracy, num_epochs_run, repaired: true}` and nothing
else. Exactly one such event existed — `p1-resnet8x4-cifar100-base-s1`, written
by acct4 at 18:06 while recovering from D-12 — and it was enough to stop the
notebook.
**The deeper mistake** is treating the ledger as the source of run *metadata*.
The run_id format `{phase}-{arch}-{dataset}-{method}-s{seed}` exists precisely so
identity never needs a lookup. It was sitting in plain text in the key.
**Fix** `parse_run_id()` and `run_meta()` derive identity from the id, which is
authoritative by construction; ledger fields only enrich. New
`Session.completed_runs(phase=)` returns fully-resolved rows and is now what
NB04–NB08, NB13 and NB14 use. `repair_ledger` also writes arch/seed now, but
nothing depends on that any more.
**Contamination** none — a crash, and it happened before NB08 measured anything.
**Lesson** if an identifier encodes the facts, parse the identifier. A lookup
that *usually* has the field will fail on the one record that does not.

---

### D-12 · Work assignment drifted between sessions — one run abandoned, one duplicated
**Found** auditing NB04 on HF: `p1-resnet32x4-cifar100-base-s3` stopped at epoch
79/240 with no completion event, while `p1-resnet32x4-cifar100-base-s1` was
trained **twice** — by acct2 (79.72%) and again by acct4 (79.54%).
**Cause** *My own feature, working exactly as written and being wrong anyway.*
`Session.plan()` called `estimate_costs_from_history()` and merged measured
per-epoch times into the cost table **used for the assignment**. The whole
sharding guarantee is "identical code + identical input → identical ownership,
with no communication". Measured timings are not identical input: they change as
runs finish. acct4's first session (11:51, nothing finished) computed a different
packing than its second (18:06, twelve runs finished), so it stopped owning
resnet32x4-s3 and started owning acct2's resnet32x4-s1.
**Cost** 2.9 GPU-hours wasted, one run left at epoch 79, and the "no overlap, no
gaps" property silently violated. A self-test now shows **4 of 15 runs move**
when the cost table drifts.
**Fix** ownership uses `ARCH_COST_HINT` **only**, always. Measured timings still
refine *displayed* estimates via `estimate_phase()`, which has no effect on who
owns what. `assign_workers` documents that `costs` must be a stable table.
Self-test asserts a worker's slice is identical before and after 12 runs finish,
and that all slices still partition the universe.
**Contamination** none — every *completed* run is valid and trained under the
correct config. Two independent trainings of resnet32x4-s1 agreeing to 0.18 pts
is incidental evidence that training is reproducible. Only wasted time and one
unfinished run.
**Lesson** an optimisation that improves an *estimate* must never be allowed to
change a *decision* that had to be deterministic. Feeding measurement back into
allocation broke the one property the design existed to provide.

---

### D-11 · Q4 ran with 5 of 7 difficulty scores
**Found** reading the Phase 0 output (`battery` column listed five names).
**Cause** EL2N and forgetting events are *training-set* quantities indexed by
training images. The test split's `sample_idx` refers to entirely different
images, so attaching them there would be meaningless — the code correctly wrote
NaN, but `analyse_q4_irreducibility` defaulted to the test split anyway.
**Fix** default changed to `train_holdout` (5,000 training images, augmentation
off, carries all seven). Test split retained as a robustness check.
`n_battery_scores` now recorded in the output.
**⚠ CONTAMINATION — YES, one reported number.** ΔR² = 0.254 is a *test-split,
five-score* result. A smaller battery is an **easier** test for MSC, so this
figure **overstates** irreducibility. The margin over the 0.05 gate is large
enough that the Phase 0 *decision* is unaffected, but §1.5 must be recomputed on
`train_holdout` before the number appears in the paper.
**Action** rerun NB03 Step 7 after re-uploading. Tracked in §5.

---

### D-10 · Cost estimates ~40% low
**Found** comparing predicted runtimes against Phase 0 actuals.
**Cause** `SECONDS_PER_COST_UNIT` was guessed at 5.0; measured 8.32. The
resnet32x4/wrn ratio was also wrong (1.18 hinted vs 1.54 measured).
**Fix** anchored both scale and ratio on the two measured architectures.
Predictions now match actuals to within 0.01 h. `MEASURED_ARCHS` records which
entries are real.
**Contamination** none — scheduling estimates only, no scientific quantity.

### D-09 · NB02 planned zero work and exited in 30 s
**Found** user reported the notebook finishing suspiciously fast; NB03 then
reported no per-sample tables.
**Cause** *Modelling error, not a typo.* A run passes through several stages
(train → measure → method) but the ledger carries **one** `state` per run.
`plan_work` treated `state == "completed"` as done — but that was set by
*training*. The measurement stage therefore saw all four runs as finished.
**Blast radius** would have done the same to the 45-run atlas in NB08.
**Fix** `plan_work(done_fn=, stage=)`; `Session.measured()` decides measurement
completion from **artifact existence**, not the ledger. Zero planned work is now
an `[ALARM]` when the worker owns unfinished runs. NB02/NB08 print per-run
measured status before and after.
**Contamination** none — nothing ran, so nothing was produced. All Phase 0
measurements were produced after the fix.
**Lesson** the worst failure shape is no error, plausible output, and a
consequence that surfaces two notebooks later.

### D-08 · Analysis raised `FileNotFoundError` six frames deep
**Found** NB03 run before NB02 completed.
**Cause** no prerequisite check; a missing input surfaced from inside a
statistic.
**Fix** `MissingInputs` exception type, `check_inputs()` per-run table,
`require_inputs()` hard stop naming the notebook to run next. `load_per_sample`
distinguishes *never trained* from *trained but not measured*.
**Contamination** none — crash, not corruption.

### D-07 · Twelve analysis cells assumed a non-empty DataFrame
**Found** user hit `KeyError: 'split'` in NB02; swept for the same pattern.
**Cause** `pd.DataFrame([])` has no columns; `.groupby('split')` fails.
**Fix** guarded all thirteen sites; each now explains what is missing. Added an
offline harness that executes every code cell against a stubbed empty project
and asserts no `KeyError`/`IndexError` — it caught an NB12 case the manual pass
missed, **and** a `\n` escaping bug I introduced while fixing NB02.
**Contamination** none.

### D-06 · A test that validated nothing
**Found** the resume acceptance test "failed" — but had never exercised resume.
**Cause** it simulated a kill by training a *shorter run*, which is a clean
completion followed by an extension. That path never touches the interrupt
handler, emergency flush, or resume logic. It was also correctly blocked by the
claim protocol, which refuses to restart a completed run.
**Fix** a debug hook fires a real `KeyboardInterrupt` mid-run (excluded from
`config_hash`), then resumes with an identical config. The test now compares
**per-epoch training loss after the seam**, which is where a lost RNG state
shows up — final accuracy can match by luck, the loss curve cannot.
**Contamination** none, but this is the most alarming entry in the log: for a
period we believed resume was verified when it was not.
**Lesson** a test that cannot fail for the right reason manufactures confidence.

### D-05 · Claim protocol blocked self-resume
**Found** the corrected resume test.
**Cause** the 2-hour staleness window was applied without checking *ownership*.
A session that paused at the 8.5 h limit could not be resumed by its own account
for two hours — defeating the resumability contract at exactly the moment it
exists for.
**Fix** ownership checked before freshness. Same account → always allowed, with
a note. Other accounts → unchanged.
**Contamination** none — would have blocked work, not corrupted it.

### D-04 · Shared ledger silently lost writes
**Found** auditing the live repo: two runs training, ledger listing one.
**Cause** HuggingFace has no append operation. Every worker rewrote
`registry/runs.jsonl` and pushed it; the last push destroyed the others' lines.
No error.
**Blast radius** `plan_work` reads *completion* from the ledger, so a lost
`completed` entry makes a finished 3-hour run look unfinished and it gets
retrained.
**Fix** one event shard per `(account, worker, session)`, merged on read.
Terminal states are sticky so a late heartbeat cannot resurrect finished work.
Events carry a float clock because second granularity sorts ambiguously across
shards.
**Contamination** none — occurred in the *previous* repo (`msc-kd`/`msc-kd-data`).
`msc-cifar100` was created after the fix and has per-worker shards from its first
commit. Even had it occurred, retraining is deterministic under a fixed seed.

### D-03 · Rate limiter counted per repository
**Found** design review while consolidating repos.
**Cause** HF meters writes **per user**; the limiter lived on the uploader. Two
repos × 20 commits/hour × 6 accounts = 240/hour against a ceiling near 128.
**Fix** one shared token bucket keyed by token, process-wide. Also motivated
consolidating to a single repo (one commit per cycle instead of two).
**Contamination** none — would have throttled uploads, not altered data.

### D-02 · MLP-Mixer cannot run at any other resolution
**Found** NB00 preflight: `mat1 and mat2 shapes cannot be multiplied (192x16 and 64x96)`.
**Cause** the token-mixing block is `Linear(n_tokens → hidden)` — the weight
matrix's input dimension *is* the patch count. Unlike a ViT's positional
embedding, there is no principled resampling. **This is a property of the
architecture, not a code defect.**
**Fix** `supports_native_resolution = False`; resolution axis uses the
downsample-upsample proxy and records the limitation.
**Decision consequence** the proxy became the **primary** resolution measurement
for *all 15* architectures — see DC-3.
**Contamination** none; Phase 0 Q2 already used `res_proxy`.

### D-01a · ViT could not run below 32 px
**Found** NB00 preflight: `size of tensor a (17) must match tensor b (65)`.
**Cause** the learned positional embedding is sized for the 8×8 patch grid
(64 + CLS = 65). A 16 px input gives 4×4 + CLS = 17 tokens.
**Fix** bicubic resampling of the grid portion onto whatever grid the input
needs — the standard ViT/DeiT resolution-transfer procedure. Verified all five
resolutions divide the patch size and yield square grids (16/25/36/49/64 tokens).
**Contamination** none — ViT is not in Phase 0. Would have blocked the entire
resolution axis for the architecture that makes Q3 interesting.

### D-01b · `resnet8x4` produced duplicate compute budgets
**Found** NB00 preflight — and it **passed**, which was the real problem.
**Cause** requesting 5 depth exits from a 3-block network produced cuts
`(1,2,3,3,3)` and `rho = [0.295, 0.648, 1.0, 1.0, 1.0]`. The preflight checked
monotonicity with `<=`, so duplicates slipped through.
**Blast radius** `msc_core.compute_msc` requires *strictly* ascending costs —
"the smallest sufficient budget" is undefined when two budgets cost the same.
Would have crashed ~3 hours into Phase 1b, or worse, returned an MSC depending
on which identical entry `argmax` happened to pick.
**Fix** adaptive K — take as many distinct cuts as the depth allows, record the
fractions achieved. Preflight now asserts strict ascent, distinctness, and
termination at 1.0. Cross-architecture comparison is unaffected: MSC is a cost
*fraction*, not an exit index.
**Contamination** none — resnet8x4 is not in Phase 0.

---

## 3. Design decisions changed after the plan was frozen

| | Decision | Was | Now | Why |
|---|---|---|---|---|
| DC-1 | Repository layout | Two repos (`msc-kd`, `msc-kd-data`) | **One** dataset repo `msc-cifar100`, one folder per run | HF meters per user, so two repos doubled commit consumption for no benefit (D-03). A run's artifacts belong together. Dataset repos render CSV/Parquet previews. |
| DC-2 | Work allocation | Optimistic claim protocol | **Cost-balanced deterministic split**, claims demoted to recovery | Hash sharding gave 4.91× imbalance on 45 uneven jobs; LPT bin packing gives 1.02×. A phase ends when the slowest worker ends. |
| DC-3 | Resolution axis | Native primary, proxy alongside | **Proxy primary** (all 15), native as robustness check (14/15) | MLP-Mixer cannot run natively (D-02). 14 measured one way and 1 another would make cross-architecture claims on this axis compare different quantities. `01_PHASE0_GO_NOGO.md` §3 asks us to decide and stay consistent. |
| DC-4 | Depth exits K | Fixed at 5 | **Adaptive** per architecture | A 3-block network cannot carry 5 distinct budgets (D-01b). |
| DC-5 | Notebook granularity | 5 (one per phase) | **16** | No single notebook should be too long to finish in a Kaggle session; a crash costs less. |
| DC-6 | Staging location | `/kaggle/working` (20 GB) | **`/kaggle/temp`** (~1 TB) | 240 epochs × 10 Hz power sampling + step traces is never disk-constrained there. |
| DC-7 | `NUM_WORKERS` default | 6 | **1** | Splitting across accounts is an optimisation, not a prerequisite. Defaulting to it made the simple case look complicated. |
| DC-8 | Q4 split | test | **train_holdout** | Only that split carries all seven battery scores (D-11). |
| DC-9 | Deleted loss terms | Omitted | **Columns present, value `NA`** | Uniform schema; enabling a term later needs no migration. Fabricating a number for a loss the model never computed would be worse. Protocol's 3-term objective intact. |
| DC-11 | Cost model in assignment | Refined by measured timings | **Static table only** | Measured costs change as runs finish, so ownership stopped being stable (D-12). Estimates still use measurements; allocation must not. |
| DC-10 | Telemetry breadth | 22 columns/epoch | **171** + 91 final | "We only train once." Re-running the atlas to recover a forgotten metric is unrecoverable time. |

---

## 4. Paper crib — what feeds which section

| Paper section | Source | Notes |
|---|---|---|
| §3 MSC definition | `00_RESEARCH_PROTOCOL.md` §2 | Unchanged since freeze |
| §3.1 Cost normalisation | `budgets/{arch}.json`, §1.8 here | Profiler name + version recorded per architecture |
| §3.2 Stable sufficiency | protocol §2.2 + `msc_core.compute_msc` | The suffix-closure argument |
| §3.3 Irreducible subpopulation | **§1.7** | Report |U_τ|/N per model per τ |
| §4.1 Protocol & noise ceiling | **§1.2** | ρ_seed ≈ 0.72; this is the denominator |
| §4.2 [Q2] Axis structure | **§1.3** | **H2 refuted.** PC1 = 0.503 vs predicted ≥ 0.60 |
| §4.3 [Q3] Transfer | **§1.4** | T = 0.946 within-family; awaiting CNN→token |
| §4.4 [Q4] Irreducibility | **§1.5** | ⚠ recompute on `train_holdout` first (D-11) |
| §6.1 Setup | §1.1 + `env/environment.json` | Recipe validated against published numbers |
| §6.4 Efficiency | `metrics/final.csv` | FLOPs primary; energy as methodology only |
| §7 Limitations | **D-02, D-11, §1.7**, batching caveat | See below |
| Reproducibility statement | **§2 entire**, `paper/provenance.csv` | Every artifact → run_id |

### Limitations paragraph — assembled

Draft, from the record:

> Per-sample dynamic routing yields no wall-clock speedup under batched
> inference unless the batch is split by route; our deployment claims are scoped
> to the batch-1, edge and streaming regimes and measured there. The precision
> axis is simulated by fake quantisation — no INT4/INT6 kernel exists for the T4
> — and is priced by an analytic bit-operation model, never reported as measured
> latency. The resolution axis uses a downsample-upsample proxy uniformly across
> all architectures because MLP-Mixer's token-mixing layer is dimensioned to the
> patch count and cannot be evaluated at another resolution; native-resolution
> measurements are reported alongside for the fourteen architectures that
> support it. Risk control is calibrated at ε = 0.03 rather than 0.01, because a
> Hoeffding bound at δ = 0.05 requires ≈ 14,979 calibration samples and the
> CIFAR-100 test set contains 10,000; we calibrate on a held-out training slice
> and note that the calibration distribution is train-like. All experiments are
> on a single hardware class (Kaggle T4) at CIFAR-100 scale.

### Sentences the record already supports

- *"All backbones matched or exceeded published accuracies for the standard
  recipe (Table X), confirming that measurements derive from correctly trained
  models."*
- *"A shuffled-target control yielded T = 0.007, confirming per-sample tables are
  correctly aligned across architectures."*
- *"Every statistic is reported as a curve over τ ∈ {0, 0.1, 0.2, 0.3, 0.5}; no
  conclusion in this paper depends on the choice of τ."*
- *"Contrary to our pre-registered hypothesis H2, a single component explains
  only 50.3% of the variance in per-sample compute requirements across reduction
  axes."* ← **the sentence a reviewer will remember**

**The abstract sentence, now that Q3 is in** (§1.4):

- *"Across 78 architecture pairs spanning ResNets, VGG, depthwise-separable
  networks, a modern CNN, a vision transformer and an MLP-Mixer, per-sample
  minimum sufficient compute transfers at 92% of the measurement ceiling within
  an architecture family, 88% across convolutional families, and **71% across
  the convolution/attention boundary** — with no overlap between the three
  distributions. How much computation an input requires is substantially a
  property of the input."* ← **the result the project exists to produce**
- *"Contrary to our pre-registered prediction that transfer would collapse below
  0.6 across the convolution/attention boundary, it remained at 0.71. We record
  this as a refuted hypothesis in the favourable direction."* ← **pre-registration
  paying off; report it exactly this way**
- *"For `resnet110` and `resnet56`, disattenuated transfer reaches 1.005
  [0.979, 1.029]: cross-architecture agreement is statistically
  indistinguishable from same-architecture, different-seed agreement."*
- *"Compute-need is reliably three-dimensional: no architecture in our atlas
  reaches a first principal component above 0.532, against a pre-registered
  threshold of 0.60, and the spread across thirteen architectures is only 0.09
  wide."* ← **H2 refuted 15/15, far stronger than the pilot's single value**

**Sentences the record now forbids** — both were true-looking and both are wrong:

- ~~*"MSC explains 25% of variance beyond classical difficulty scores."*~~
  Measured on the wrong split with 5 of 7 scores. The corrected figure is ~0.10
  and the partial correlation is marginal (§1.5, D-11).
- ~~*"Every architecture in the atlas beats its published reference."*~~
  `mobilenetv2`'s reference is a half-width model (D-14).

**Added by the 2026-08-04 atlas audit** (all supported by §1.2 and §1.9):

- *"Across fourteen architectures, the seed-to-seed reliability of per-sample
  minimum sufficient compute separates cleanly by architecture family: all twelve
  convolutional networks fall in [0.622, 0.726], while both non-convolutional
  models sit at 0.547 — below every CNN, and below our pre-registered
  reliability threshold."* (τ ≤ 0.2; the separation does not hold at τ = 0.3.)
- *"This is not explained by accuracy: within the convolutional family, ceiling
  height and top-1 accuracy are uncorrelated (Spearman +0.035), and
  `convnext_femto` — the least accurate CNN, only 2.4 points above `mixer_nano` —
  attains a ceiling 0.161 higher."*
- *"Measurement reliability is itself architecture-dependent, which means any
  cross-architecture transfer study that does not disattenuate is comparing
  quantities measured with unequal precision."* ← **the methodological
  contribution, now demonstrated rather than argued**
- *"Rank agreement and top-decile agreement are near-independent measures of the
  same construct (Spearman +0.130 across architectures); top-decile overlap is
  instead largely determined by an architecture's mean compute requirement
  (+0.780). We therefore report both."*
- *"Our pilot happened to select the two most reliably-measured architectures in
  the zoo; the atlas mean ceiling is 0.676 against the pilot's 0.715, so pilot
  transfer estimates should be read as conservative."* ← **pre-empts the
  cherry-picking objection by raising it first**
- *"Minimum sufficient compute is defined in floating-point operations. Across our
  zoo the ratio of wall-clock time to FLOPs varies by a factor of eighteen, so
  FLOPs-denominated savings should not be restated as time or energy savings."*

---

## 5. Open items

| | Item | Blocks | Cost | Priority |
|---|---|---|---|---|
| ~~O-17~~ | ~~Confirm the nine `p3-*-mscKD*` runs survived~~ — **CLOSED.** All nine are checkpointed on HF and resumable; nothing was lost. The alarm that said otherwise was D-20 | — | — | done |
| ~~O-19~~ | ~~Dry-run one student batch before the teacher sweep~~ — **DONE.** `msckd_dry_run()` runs the full step on a 2-image batch in <1 s, before any teacher work | — | — | shipped |
| O-18 | **Add a cross-session resume test.** Five defects have now been about resume (D-05, D-06, D-09, D-12, D-19) and every test we have runs inside one session — which is precisely the case that works. The acceptance test must delete the local run directory to simulate a fresh Kaggle session | the next D-19 | ~1 h | **high** |
| O-15 | **Re-run NB10 → NB11 → NB12 with the D-18 fix.** Recovers `vgg8`, un-biases Q4, and stratifies the τ check. §1.5's numbers are provisional until this lands | the Q4 headline | **~20 min, CPU** | **highest** |
| O-12 | **Close D-14/D-15:** null the `mobilenetv2` reference; measure the 6 unmeasured runs + train `wrn_16_2-s1`; add the NB08 coverage ALARM and the parameter-count assertion | "15 architectures" being true | ~4 GPU-h | **high** |
| O-5 | Read SAFE-KD (2602.03043); write the differentiation memo | Related work | ~1 day | **high, not started** |
| O-9 | **Break the family/accuracy confound in §1.2** — train one non-CNN to CNN-level accuracy, or one CNN down to ~60%, so the low-ceiling finding does not rest on `convnext_femto` alone | the Q1 headline | ~3 GPU-h | **high** |
| O-14 | **Test the `convnext_femto` explanation (§1.4)** — is low transfer a property of transformer-ised CNNs, or of that one model? One more modern CNN settles it | a quotable sub-finding | ~2 GPU-h | medium |
| O-10 | Report ρ_seed *and* J₁₀ in every table, and state the mean-MSC confound on J₁₀ | Q1/Q3 presentation | writing | high |
| O-16 | Correct **ΔR² = 0.254** in `README.md` and `08_PHASE0_RESULTS.md` to the §1.5 value once O-15 fixes the sample | published numbers | ~15 min | **high** |
| O-11 | Check whether `telemetry/` already holds per-budget latency; if so, test whether MSC-in-FLOPs predicts MSC-in-time within an architecture (§1.9) | free extra result | ~30 min | medium |
| O-7 | Test whether architectures agree on \|U_τ\| *membership*, not just size — 13 architectures now available | free sub-finding | ~20 min | medium |
| O-13 | Run NB15 to build `tables/` so per-seed means stop needing manual assembly | writing efficiency | ~10 min | medium |
| O-6 | Verify the 2026 arXiv IDs cited in the protocol | Bibliography | ~1 h | medium |
| O-8 | Width axis (slimmable) — deferred | Q2 completeness | — | low |

**Closed this audit:** O-1 (Q4 re-run on `train_holdout` with all 7 scores —
and the D-11 prediction was correct, see §1.5), O-2 (Q3 across 78 pairs — **the
central result is in**), O-4 (H2 refuted 15/15, not a `resnet32x4` quirk),
O-3/O-1b/O-1c earlier.

**The expensive half of the project is finished.** 44 models trained, 39
measured, all five questions answered. Everything on this list except O-9,
O-12 and O-14 is now writing or minutes of CPU.

---

## 6. Timeline

| Date | Event |
|---|---|
| 2026-08-04 | D-23 — **the teacher's exit heads were retrained on every MSC-KD run.** `run_oracle` writes to the run root, `train_msc_kd` read `checkpoints/`. D-16 closed this as "cosmetic — nothing reads the path by convention"; three things did. **O-19 dry run shipped at last** |
| 2026-08-04 | D-22 — **five wrong column names killed every MSC-KD run at the end of epoch 0.** The two training paths disagreed about unknown columns: one raised, one silently dropped. Also recovered the three-term loss decomposition, which was being computed and thrown away |
| 2026-08-04 | **D-21 — the MSC-KD loss cannot run under AMP.** `F.binary_cross_entropy` is banned under autocast, so the method's training step could never execute. The NB00 preflight covers all 15 backbones and neither `MSCStudent` nor `MSCLoss` |
| 2026-08-04 | D-20 — **the D-19 verification cell cried wolf**: nine healthy checkpointed runs reported as "NOT ON HF ... closing means retraining". Safety has three states, not two |
| 2026-08-04 | **D-19 — NB13 restarted completed MSC-KD runs from epoch 0.** Resume was never wired into the method notebooks: `run_all` recognised only one entry point, neither training function pulled its own checkpoint, and the zero-work ALARM was dead code. The worst defect so far — it destroys GPU-hours silently |
| 2026-08-04 | D-18 — **analysis sampled alphabetically**; Q4's 15 pairs were 12 convnext + 3 mixer, and `vgg8` was silently dropped from all three analyses |
| 2026-08-04 | **Q4 corrected on `train_holdout` with 7 scores: ΔR² 0.254 → ~0.10.** D-11's prediction confirmed; the published number overstated irreducibility 2.5× |
| 2026-08-04 | **Q3 across 78 pairs — THE CENTRAL RESULT.** within-family 0.920 > across-CNN 0.878 > CNN→transformer 0.710, complete separation. H3's ordering confirmed, its "< 0.6" magnitude refuted **favourably**: compute-need transfers across the CNN/Transformer boundary |
| 2026-08-04 | **H2 refuted across the whole atlas** — 0 of 15 runs reach PC1 ≥ 0.60; highest is 0.532. Axes decouple further in ViT/Mixer |
| 2026-08-04 | NB11 re-run after the D-17 fix: **78/78 shuffled controls pass**, max \|z\| = 3.30 against a 5σ threshold |
| 2026-08-04 | **D-17 — the shuffled control was miscalibrated and halted NB11 on healthy data.** Rule rebuilt against the exact permutation null; 11 regression checks added; the control now tests all 78 pairs, not 25 |
| 2026-08-04 | D-16 — `exit_heads.pt` written outside the documented layout |
| 2026-08-04 | D-15 — **6 of 45 runs unmeasured; `wrn_16_2` has zero usable seeds.** The atlas is currently 14 architectures, not 15 |
| 2026-08-04 | D-14 — **`mobilenetv2` +5.50 is against a half-width baseline.** Not a win; reference to be nulled |
| 2026-08-04 | **Q1 re-run atlas-wide: MSC is measurably less seed-stable in ViT/Mixer than in every CNN** (0.547 vs 0.622–0.726 at τ=0.1, clean separation). `convnext_femto` shows this is not an accuracy artifact |
| 2026-08-04 | §1.9 — FLOPs predict wall-clock to within only 18× across the zoo; MSC language constrained accordingly |
| 2026-08-04 | **Audit: atlas 44/45 trained, 39/45 measured.** NB09 re-run; NB10/NB11/NB12 still Phase-0-only |
| 2026-08-04 | **`p1-resnet32x4-s3` resumed from epoch 79 and completed 240/240 at 79.46%** — first proof of resume on a genuinely interrupted run (O-1b closed) |
| 2026-08-03 → 04 | NB05–NB07 completed: WRN, VGG, Mobile and the three modern architectures trained; NB08 measurement run |
| 2026-08-02 | ~~**Audit: only NB04 runs are on HF.**~~ **Retracted 2026-08-04** — the HF tree API served a stale cached response; the runs were present. See the note below |
| 2026-08-02 | D-13 found — NB08 crashed on a ledger event lacking arch/seed; identity now parsed from the run_id |
| 2026-08-02 | D-12 found auditing NB04 — assignment drift; ownership now uses a static cost table |
| 2026-08-02 | **NB04 atlas: 14/15 ResNet runs complete**, all beating published references |
| 2026-08-02 | **Phase 0 verdict `FULL-PROGRAM`.** All gates cleared at every τ. H2 refuted. |
| 2026-08-02 | D-11 found reading the Q4 output; Q4 default moved to `train_holdout` |
| 2026-08-02 | Phase 0 measurement (NB02) completed for all four runs |
| 2026-08-02 | D-09 found — NB02 was a no-op; stage-aware completion added |
| 2026-08-02 | D-10 found — cost model recalibrated against real timings |
| 2026-08-02 | Phase 0 training completed, all four beating published references |
| 2026-08-02 | D-07, D-08 — empty-state guards and prerequisite checks |
| 2026-08-02 | D-04, D-05, D-06 — ledger sharding, self-resume, the resume test itself |
| 2026-08-02 | Repos consolidated to `msc-cifar100` (DC-1); D-03 rate limiter |
| 2026-08-02 | D-01a/b, D-02 found by the NB00 preflight before any GPU-hour was spent |
| 2026-08-02 | Pipeline built: `msc_lib` 1.0.0, 16 notebooks |

### A note on the retracted 2026-08-02 audit

On 2026-08-02 an audit concluded that only the NB04 ResNet runs existed on HF and
that NB05/NB06 had not pushed. **That conclusion was wrong**, and the way it was
wrong is worth recording because it will recur.

The audit used `GET /api/datasets/{repo}/tree/main/runs`. That endpoint is served
from a cache, and it returned a response with **byte-identical `oid` values**
across two audits taken hours apart — which was read as "nothing changed" when it
actually meant "you were served the same cached page twice." The runs had been
pushed the whole time.

A second failure compounded it: the full repo-info endpoint returns ~69 KB for
this repo and **was silently truncated mid-JSON**. Parsing the truncated body
yielded a plausible-looking but short file list, alphabetically cut off just past
`vgg8` — which is precisely where `vit_tiny` and `wrn_*` would have appeared.
Two independent methods therefore agreed on the same wrong answer.

**What to do instead**, and the rule this project now follows:

1. Treat the recursive/aggregate listing endpoints as a *hint*, never as proof of
   absence. Confirm any negative finding with a narrow `tree/main/runs/{run_id}`
   call, which is small enough not to truncate.
2. Check `lastModified` on the repo before concluding nothing has changed. It
   read `2026-08-04T05:28:37Z` here — which alone would have falsified the
   "nothing was pushed" conclusion.
3. If a response is parsed programmatically, **assert it parses** rather than
   working from a regex over a possibly-truncated body. The truncation announced
   itself as a `JSONDecodeError`; that error was the signal, not a nuisance.
4. Identical `oid`s across audits mean *identical content served*, which is
   consistent with both "unchanged" and "cached." It is not evidence.

Cost of the mistake: no GPU-time, but one incorrect entry stood in this notebook
for two days and O-1c was tracked as a high-priority open item that had never
been a real problem. Negative findings deserve the same verification standard as
positive ones.

---

*Append new entries at the top of each log. Every claim here should be traceable
to a file in the HuggingFace repository or a commit in this one.*
