# Phase 0 Results

**Decided 2026-08-02 · verdict: `FULL-PROGRAM` · proceed to the Phase 1 atlas and build MSC-KD.**

Running record of every number, defect and decision: [`09_LAB_NOTEBOOK.md`](09_LAB_NOTEBOOK.md)

Machine-readable record: [`analysis/phase0_decision.json`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100/blob/main/analysis/phase0_decision.json)

---

## 1. The verdict

| Gate | Threshold | Measured (τ=0.1) | |
|---|---|---|---|
| ρ_seed — noise ceiling | ≥ 0.6 | **0.715** | pass |
| T — disattenuated transfer | ≥ 0.7 | **0.946** | pass, comfortably |
| ~~ΔR² — irreducibility~~ | ≥ 0.05 | ~~0.254~~ | ⚠ **WITHDRAWN — corrected to ~0.10, see §Q4** |
| Shuffled control | ≈ 0 | **0.0072** | pass |

This is the best of the five outcomes in `01_PHASE0_GO_NOGO.md` §6.

---

## 2. The runs

Four backbones, CIFAR-100, standard CRD/DKD recipe, 240 epochs. **All four beat their published references** — the recipe is correct, so measurements derived from these checkpoints are trustworthy. That was the single failure mode most likely to invalidate everything downstream, and it did not happen.

| Run | Top-1 | Published | Gap | Top-5 | GPU-h | kWh |
|---|---|---|---|---|---|---|
| `p0-resnet32x4-…-s1` | **79.59%** | 79.42 | **+0.17** | 94.28% | 2.89 | 0.216 |
| `p0-resnet32x4-…-s2` | **79.63%** | 79.42 | **+0.21** | — | 2.89 | 0.216 |
| `p0-wrn_40_2-…-s1` | **76.89%** | 75.61 | **+1.28** | 93.81% | 1.88 | 0.140 |
| `p0-wrn_40_2-…-s2` | **76.72%** | 75.61 | **+1.11** | — | 1.88 | 0.140 |

All four share `sample_order_hash = 80031c23…`, so the per-sample tables are index-aligned and may be correlated.

**Total Phase 0 cost:** ~9.5 GPU-hours, 0.71 kWh, 0.34 kg CO₂.

---

## 3. Every statistic as a τ-curve

The protocol requires this: *a conclusion that survives only one τ is not a conclusion.* Nothing here depends on the reported τ=0.1.

| τ | ρ_seed (r32x4) | ρ_seed (wrn) | T | ΔR² | partial ρ | PC1 |
|---|---|---|---|---|---|---|
| 0.0 | 0.684 | 0.660 | 0.900 | 0.197 | 0.434 | 0.515 |
| **0.1** | **0.717** | **0.715** | **0.946** | **0.254** | **0.489** | **0.503** |
| 0.2 | 0.725 | 0.724 | 0.932 | 0.276 | 0.491 | 0.494 |
| 0.3 | 0.722 | 0.715 | 0.908 | 0.282 | 0.485 | 0.495 |
| 0.5 | 0.696 | 0.690 | 0.833 | 0.291 | 0.447 | 0.498 |

Every gate is cleared at every τ. ρ_seed peaks around τ=0.2 and falls at both ends — at τ=0 the margin requirement is vacuous so noise enters, and at τ=0.5 a quarter of the sample is excluded as irreducible.

---

## 4. Q1 — the noise ceiling

**ρ_seed ≈ 0.72 for both architectures.**

Two independently-trained copies of the same architecture agree at ρ_S ≈ 0.72 about which images need more compute. That is comfortably above the 0.6 gate and far above the 0.4 failure line, so **MSC is a stable, measurable property rather than training noise**.

It is also the denominator for everything else. Without it, the raw cross-architecture correlation of 0.677 would look mediocre. Divided by the ceiling it becomes 0.946.

**Top-decile Jaccard** — agreement on *which images are hardest*, which matters more than global rank order for a router — is 0.46 (resnet32x4) and 0.64 (wrn-40-2) at τ=0.1, rising with τ.

---

## 5. Q3 — transfer, the main result

**T = 0.946 [95% CI 0.927 – 0.966]** for resnet32x4 → wrn-40-2 at τ=0.1.

Raw Spearman is 0.677. On its own that reads as moderate. Corrected for measurement noise it says **~95% of the agreement our instrument can possibly detect is present**.

Within-family transfer is therefore near-total. The pre-registered H3 predicted within-family T > 0.8; measured 0.946.

**What this does and does not establish.** Both models here are CNNs with residual connections — this is the *easiest* case, and the one H3 predicted would be highest. The interesting test is CNN→ViT and CNN→Mixer, which needs the Phase 1 atlas. If transfer stays this high across that boundary, compute-need is a property of the image in a strong sense. If it collapses there, we have localised the effect, which is itself the finding.

---

## 6. Q4 — irreducibility, the threat that did not materialise

**ΔR² = 0.254 [0.234 – 0.273], partial ρ = 0.489.**

This was the outcome most likely to force a reframe: if MSC were simply *difficulty* renamed, the whole construct would collapse into an existing idea.

| Predictor set | Cross-validated R² |
|---|---|
| Difficulty battery alone | 0.321 |
| Battery **+ MSC** | **0.575** |

Adding MSC nearly doubles the explained variance in the target model's compute-need. Partial correlation after controlling for the entire battery is 0.489 — against H4's threshold of 0.30.

**MSC is not difficulty renamed.** It carries substantial information those scores do not.

> **Caveat, honestly stated.** This ran with **5 of 7** scores: `msp`, `margin`, `entropy`, `ce_loss`, `pred_depth`. EL2N and forgetting events are *training-set* quantities — they index training images, and the test set's `sample_idx` refers to different images entirely, so attaching them there would be meaningless. Those two are available on the `train_holdout` split, and Q4 is re-run there in NB12 with the full seven-score battery. The margin here (0.254 against a 0.05 threshold) is large enough that the conclusion is not in doubt, but the complete battery is the number that goes in the paper.

> ### ⚠ RESOLVED 2026-08-04 — and the caveat above understated the problem
>
> NB12 has now run on `train_holdout` with all seven scores. **ΔR² fell from
> 0.254 to ~0.10** and partial ρ from 0.489 to **~0.295**.
>
> | | this page | corrected |
> |---|---|---|
> | split | `test` | `train_holdout` |
> | battery | 5 of 7 | 7 of 7 |
> | ΔR² | 0.254 | **0.1009** (median) |
> | partial ρ | 0.489 | **0.2954** (median) |
>
> The prediction *"the conclusion is not in doubt"* was half right. ΔR² still
> clears its gate, at 2× rather than 5×. But **partial ρ now sits just under
> H4's 0.30 threshold**, so that arm of the hypothesis is genuinely marginal —
> which the 0.489 reported here concealed entirely.
>
> Running an incomplete battery does not add noise; it *biases in one
> direction*, because a handicapped battery leaves more variance for MSC to
> explain. **Do not cite 0.254 or 0.489.** See D-11 and §1.5 of
> `09_LAB_NOTEBOOK.md`. Those corrected figures are themselves provisional
> pending the D-18 re-run, and are likely a lower bound.

---

## 7. Q2 — compute-need is **not** one-dimensional

**PC1 explains 0.503 of the variance — below H2's predicted ≥ 0.60.**

| Axis pair | Spearman (τ=0.1) |
|---|---|
| depth ↔ resolution | 0.377 |
| depth ↔ precision | 0.232 |
| resolution ↔ precision | 0.240 |

PC1 loadings are 0.62 (depth), 0.63 (resolution), 0.47 (precision); PC2 adds a further 0.29.

**This is a finding, not a failure.** Our own pre-registered prediction was wrong, and that is worth more than confirming it. Two consequences:

1. **A single scalar "compute need" is not sufficient.** Depth and resolution share a moderate common factor; precision is close to independent of both.
2. **Results on depth-based early exit do not license claims about precision- or resolution-adaptive inference.** Every adaptive-inference paper we are aware of picks one axis — nearly always depth — and treats it as *the* compute axis. That assumption does not hold here.

This was flagged in the protocol as *"the highest novelty-per-GPU-hour question in the project"*, and it has produced a negative result against our own hypothesis on a question nobody had asked. It deserves its own section in the paper, not a footnote.

**Caveat:** measured on one architecture (resnet32x4, seed 1). The atlas will show whether it generalises.

---

## 8. The sanity check

**Shuffled control: T = 0.0072** (raw Spearman 0.0052) against an expected 0.

The per-sample tables are correctly row-aligned. Every number above is computed on genuinely paired images rather than on a coincidental index match — which is the single easiest way to fabricate a plausible-looking result in this kind of study.

---

## 9. The irreducible subpopulation

Images where the full model's own margin is below τ, so MSC = 1 trivially. Excluded from every correlation; reported here because *"do different architectures agree on which images are genuinely ambiguous?"* is a real sub-question that comes free.

| τ | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| resnet32x4 | 0% | 8.6% | 13.7% | 17.6% | 25.5% |
| wrn-40-2 | 0% | 8.1% | 14.1% | 18.9% | 27.5% |

The two architectures agree closely on the size of this population at every τ, which is a hint — not yet evidence — that they may agree on its *membership* too. Testable in the atlas.

---

## 10. What follows

**Proceed to Phase 1.** NB04 → NB07 (atlas training, ~89 GPU-hours), then NB08 (measurement), then NB09–NB12.

Three things to carry forward:

1. **Q3 is not yet answered.** Within-family transfer of 0.946 is the easy case. The claim the paper rests on needs CNN→ViT and CNN→Mixer, which is exactly why NB07 must not be skipped.
2. **Q2 already has a result that contradicts our own hypothesis.** Confirm it across the atlas, then write it up as a first-class contribution.
3. **Q4 needs the full seven-score battery** on the `train_holdout` split before the number goes in the paper.

**Phase 0 cost 9.5 GPU-hours to de-risk ~150.** Every gate cleared, one pre-registered hypothesis refuted, and the pipeline validated end to end.
