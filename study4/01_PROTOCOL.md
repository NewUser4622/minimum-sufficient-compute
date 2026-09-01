# Study 4 — pre-registration

**Written before any run.** Predictions and thresholds fixed here so results
cannot be re-cut. Amendments get a date and a reason, above the original text.

The claim under test is Study 2 + Study 3's:

```
oracle_in − acc_full  >  0        (the excess)
```

measured at **+6.86 pt** on 90/90 CIFAR-100 runs with frozen exits, and
**8.55 / 9.15 / 10.64 pt** with jointly trained exits.

---

## P0 — Figures and intervals · free · no hypothesis

Not a test; a presentation task with one honest requirement.

**Figure 1: honest headroom vs compute budget.** From
`s2_headroom_sweep.csv` (ρ = 0.40–0.95, already measured). Plot the cross-seed
headroom for every score, with the in-seed bound above it and zero marked.

**Bootstrap intervals.** Study 3's Q1 has one seed per architecture. The
quantity is a per-run identity, so seeds are not required — but an interval is.
Resample the **10,000 test samples** with replacement, 1,000 draws, and report
the 95 % interval on `oracle_in − acc_full` per run.

> **The interval must not be presented as a seed interval.** It captures
> sampling noise over the test set, not training variation. Saying so is the
> whole point of computing it.

**Refusal condition.** If any bootstrap interval for a joint run includes zero,
that architecture's excess is not established at n = 10,000 and must be reported
as such, regardless of the point estimate.

---

## P1 — Does the conclusion survive better baselines?

**Background.** Study 2 and Study 3 compare against **one** baseline:
confidence thresholding on `top1p_dk`. A reviewer will name entropy
thresholding, patience-based exiting (PABEE), and learned policies.

**What we can build, checked against the parquet:**

| baseline | rule | available |
|---|---|---|
| confidence (existing) | exit when `top1p_dk ≥ τ` | yes |
| **margin** | exit when `top1p_dk − top2p_dk ≥ τ` | **yes** |
| **patience / PABEE** | exit when *n* consecutive exits agree on the label | **yes** |
| learned gate | Study 3 Q2 | done |
| entropy | needs full per-exit softmax | **no — logits not stored** |

**Amendment, stated now rather than later:** entropy is omitted because
`per_sample/test.parquet` carries only top-2 probabilities. Approximating
entropy from `top1p`/`top2p` would be a fabricated baseline and is not done.
Margin is entropy's closest available relative.

**H6 (pre-registered).** The conclusion is baseline-independent: with margin
and patience substituted for confidence, the **honest (cross-seed) headroom
remains negative at every operating point** ρ = 0.40–0.95.

| measured | verdict |
|---|---|
| honest headroom negative for all 3 baselines at all 7 budgets | **H6 supported** — the conclusion does not depend on the baseline |
| any baseline shows positive honest headroom | **H6 falsified** — that baseline is the weak one, and the paper must lead with the strongest |

> **RESULT 2026-08-20 — H6 SPLITS.**
> **(a) baseline-independence: SUPPORTED.** Confidence and margin — the two
> rules that hit the budget exactly — agree to within **1.78 pt** at every
> operating point. At matched cost: confidence 58.03 %, margin 58.18 %,
> patience 55.62 %. Confidence was not a weak comparator; R-04 did not fire.
> **(b) "negative at every operating point": FALSIFIED.** Honest headroom is
> **positive at ρ ≤ 0.60** (+7.74, +7.29, +3.74) and negative from ρ = 0.70
> up, changing sign near ρ ≈ 0.65.
>
> This does not contradict Study 2, which swept per-sample *score* routing, a
> different quantity. Where they overlap (ρ = 0.80) they agree to 0.40 pt
> (−7.90 vs −8.30) — an independent cross-check across two notebooks.
>
> *(The first run of this phase was void under D-89: `route_threshold`'s
> bisection was inverted, so accuracy fell as the budget rose. Fixed and
> re-run; accuracy is now monotone.)*

**The strongest baseline must be the comparator.** If patience beats confidence,
the headroom numbers are recomputed against patience and the paper says so. A
limits paper that picks a weak baseline is worthless.

---

## P2 — Does the identity hold at scale, and on a transformer?

**Design.** Joint-exit training on ImageNet-100 at 224px, reusing
`notebooks_in100/` unchanged apart from `joint_exits=True`.

| arch | why | measured baseline time |
|---|---|---|
| `resnet50` | convolutional, the reference at scale | **see the warning below** |
| `vit_small_p16` | **a transformer** — every result so far is convolutional | 5.7 GPU-h / 100 epochs |

One seed each; the quantity is a per-run identity.

> **The 41.5 GPU-h figure for `resnet50` is the D-59 `channels_last` run and
> must not be used for planning.** The documented correct figure is ~6 h. P2
> therefore carries a **throughput gate**: measure img/s during epoch 1 and
> abort if it is more than 2× below the contiguous benchmark
> (`MEASURED_THROUGHPUT` in `msc_lib`). Better to lose one epoch than 35 hours.

**H4 (pre-registered).** The excess holds at ImageNet-100 scale:

```
oracle_in − acc_full  ≥  2.0 accuracy points,  in 2 of 2 architectures
```

**H4b.** It holds on the **transformer** specifically — `vit_small_p16` alone
must clear 2.0 pt. This is separated because a convolution-only result would
leave the claim exactly where a reviewer will attack it.

| measured | consequence |
|---|---|
| both ≥ 2.0 pt | scale objection closed; JCR-Q1 submission viable |
| conv yes, transformer no | **the claim becomes architecture-conditional** and the paper must say so in the title |
| neither | the effect is CIFAR-specific; Paper A retreats to a small-scale finding |

---

## P3 — Does the identity hold on a real early-exit architecture?

**Background.** Our exits are heads attached to a staged backbone. MSDNet
maintains multiple scales throughout and places classifiers on them by design.
If the excess is an artifact of *attaching* exits rather than *designing* them,
MSDNet is where that shows.

**Design.** MSDNet on CIFAR-100, 2 seeds. Measured with the same code as
everything else — that invariance is the point of the paired design.

> **BUILT 2026-09-01, not yet run.** `notebooks_study4/S4_NB4_MSDNet.ipynb`.
> Configuration as registered in `msc_lib.ZOO`: 3 scales × 20 multi-scale dense
> layers, base 16, growth 6, exits at layers 4/8/12/16/20 (exactly
> `DEPTH_FRACTIONS`), widths 160/256/352/448/544. Trained **jointly**, because
> MSDNet trains all its classifiers jointly by design — so the comparator is
> Study 3's **joint** runs (8.55 / 9.15 / 10.64 pt), not the frozen ones.
>
> **Cost revised down: ~5 GPU-h, not ~15.** The original figure predated the
> architecture. At ≈ 0.24 GFLOPs against `resnet32x4`'s ≈ 1.09 it is ~2.5 h per
> seed. Still an estimate; the notebook times epoch 1 and extrapolates.
>
> **`msdnet` is a probe, not an atlas entry** (`atlas=False`). It is excluded
> from `zoo_for_dataset` and from every notebook's `measured_runs`, so the
> study population stays at the 15 architectures Studies 1–3 measured. Adding
> it without that guard would have moved the published P0 intervals — see D-90
> in `03_LOG.md`.

**H5 (pre-registered).**

```
oracle_in − acc_full  ≥  2.0 accuracy points,  in 2 of 2 seeds
```

| measured | consequence |
|---|---|
| ≥ 2.0 pt | **the claim is architecture-independent.** This is the strongest available outcome and changes the paper's scope |
| 0.5 – 2.0 pt | present but attenuated; report both and attribute the difference to architecture |
| < 0.5 pt | **the excess is a property of attached exits.** Still publishable, and arguably more useful: *"oracle bounds are inflated for post-hoc and jointly-trained attached exits, and sound for architectures with designed exits"* — directly actionable |

**Falsifier stated now:** the third row is not a failure. It is a sharper claim
than the one we currently have, and it must be reported with equal prominence.

---

## Stopping rules

1. **A null is reported as a null.** No re-cutting. Study 3's H3 is the model:
   falsified, confounded, recorded, not quietly redesigned.
2. **The free phase gates the paid one.** P0 and P1 cost nothing and must be
   complete before P2 starts.
3. **No statistic without a canary that can fail** — including one requiring it
   to *detect* the effect in a world where the effect certainly exists.
4. **Favourable results are attacked first.** Every Study 3 headline was
   scrutinised before being written up, and two were corrected as a result.
5. **Open the artifact before writing code that needs it.** Six of Study 3's
   defects were code written against an assumed state.

## What would make Study 4 wrong

- **Our MSDNet is not the real MSDNet.** A re-implementation can differ from the
  published architecture in ways that matter. Record the configuration and cite
  the source; if H5 lands near a threshold, that ambiguity is load-bearing.
  **Deviations, recorded in `msc_lib.msdnet_channel_spec`:** no bottleneck (1×1)
  convolutions, no channel-reduction transitions, and the project's standard
  linear `ExitHead` in place of MSDNet's two-conv classifier. The last is
  deliberate — holding the head fixed across every architecture is what makes
  P3 a statement about the backbone — and it means a weak H5 cannot be blamed
  on a changed head.
- **MSDNet has no `REFERENCE_ACC` entry, so there is no recipe check.** Every
  number in that table is a published figure for a specific architecture; ours
  is a re-implementation, and inventing a reference would be worse than having
  none. The notebook substitutes an explicit **floor** (55 %, below
  `mobilenetv2`'s 64.60) and labels it as weaker than `recipe_ok`. An
  undertrained MSDNet would make H5 a statement about training, not about
  designed exits.
- **One seed at ImageNet scale.** Valid for the identity, but no interval.
- **Entropy remains untested** as a baseline (logits not stored).
- **CIFAR-100 and ImageNet-100 are both small.** Neither is ImageNet-1k.
- **The bootstrap interval is over samples, not seeds** — it must never be
  described as the latter.
