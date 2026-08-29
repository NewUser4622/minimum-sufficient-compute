# Study 3 — findings

**All three questions have answers.** Two are clean; the third is a null whose
design flaw is visible and stated. Every number traces to a CSV in `analysis/`.

Pre-registration: [`01_PROTOCOL.md`](01_PROTOCOL.md) · narrative and defects:
[`03_LOG.md`](03_LOG.md)

---

## Scoreboard

| | question | threshold | measured | verdict |
|---|---|---|---|---|
| **H1** | oracle excess survives jointly trained exits | ≥ 2.0 pt, 3 of 3 | **8.55 / 9.15 / 10.64 pt** | **SUPPORTED** |
| **H2** | a learned router captures little of the gap | < 25 % cross-seed | **1.7 %** | **SUPPORTED** |
| **H3** | saturated-source pruning is worse | ≥ 1.0 pt at 30 % keep | **−8.54 pt (reversed)** | **FALSIFIED — confounded** |
| **H3b** | saturated source ≈ random pruning | ± 0.5 pt at 30 % keep | **−2.70 pt** | **FALSIFIED** |

**Cost: ~16 GPU-hours over about a day.** Study 1 spent 215.

---

## Q1 — The oracle excess is not an artifact. It is *larger* when exits are trained properly.

This was the blocker on `study2/PAPER.md`. Study 1 trained exit heads on a
**frozen** backbone; MSDNet, BranchyNet and DE3-BERT train them **jointly**. If
the +6.86 pt excess were a property of weak post-hoc exits, Study 2's central
claim would collapse.

`analysis/s3_q1_comparison.csv` — 3 architectures × 1 seed, because the quantity
is a per-run identity and seeds are not needed for it:

| arch | frozen | **joint** | change | Δ acc_full | exit_quality |
|---|---|---|---|---|---|
| `resnet20` | 6.69 | **10.64** | +3.95 | −1.53 | 0.413 → 0.756 |
| `resnet32x4` | 6.42 | **8.55** | +2.13 | +1.39 | 0.535 → 0.919 |
| `vgg8` | 7.95 | **9.15** | +1.20 | +1.08 | 0.634 → 0.753 |

**Supported by more than four times the threshold, and the excess grew in 3 of 3.**

**The mechanism, now clear.** A weak early exit is right on almost nothing, so
it cannot rescue a sample the final layer gets wrong. **Rescues require
competent exits**, so the pool grows with exit quality. The frozen-backbone
figure was a *lower* bound, not an inflated one.

**It was a real test.** `exit_quality` rose in all three (+0.119 to +0.384,
`resnet32x4` reaching 0.919), so joint training changed exactly what it was
meant to change and H1 was genuinely at risk.

**R-02 fired, and conditioning strengthens the result.** Backbone accuracy moved
by more than 1 pt in every architecture, so the conditioned number is primary.
Using the excess-on-`acc_full` slope measured across the 45 frozen runs
(−0.2559 pt per pt):

| arch | raw Δ | **conditioned Δ** |
|---|---|---|
| `resnet20` | +3.95 | +3.56 |
| `resnet32x4` | +2.13 | +2.49 |
| `vgg8` | +1.20 | +1.48 |
| **median** | +2.13 | **+2.49** |

### The free gate predicted this

**P0 cost nothing** — 45 existing runs, no GPU — and predicted from
`exit_quality` alone that the excess would **grow**, reaching ~+10.98 pt at
`exit_quality` 0.86. Measured: **8.55–10.64 pt at 0.75–0.92.** Direction right,
magnitude within a couple of points, written down *before* the runs.

That is the strongest evidence in this project for doing the cheap decisive
check first. (P0's +12.45 pt figure at `exit_quality` = 1.0 was outside the
observed range and is not quoted.)

**P0 also nearly misled us.** `exit_quality = mean(acc_k / acc_full)` has
`acc_full` as its own denominator, and `acc_full` correlates with the excess at
−0.672 — so the raw +0.669 could have been circularity. The partial correlation
holding accuracy fixed is **+0.617**. The relationship survives; only the
partial is quotable.

---

## Q2 — A deployable gate captures essentially none of the gap

Study 2 claimed the excess "cannot be reached by any router". What it had shown
was narrower: a second **seed** cannot reach it. Q2 asks whether a *learned
gate* can.

`analysis/s3_router_capture.csv`, ρ = 0.80, exit-local confidence features:

| arch | baseline | router | oracle | gap | **capture (cross-seed)** |
|---|---|---|---|---|---|
| `resnet20` | 60.58 | 60.85 | 76.22 | 15.64 | **1.7 %** |
| `resnet32x4` | 66.68 | 67.21 | 78.30 | 11.62 | **4.6 %** |
| `vgg8` | 70.07 | 70.05 | 79.25 | 9.18 | **−0.2 %** |

**Supported: 1.7 % median against a 25 % threshold.**

**And the gate is not overfitting**, which makes this stronger than expected.
In-seed capture (3.8 / 2.4 / −0.4 %) is barely distinguishable from cross-seed
(1.7 / 4.6 / −0.2 %). A gate memorising seed noise would show high in-seed and
near-zero cross-seed. Both near zero says: **there is nothing in exit-local
confidence to capture.** The gap is not merely non-transferable — it is not
expressible in the signal a deployed gate reads.

This tightens Study 2 rather than contradicting it: a second *seed* cannot reach
the gap, and neither can a *learned gate on the deployable signal*.

**Stated limitation.** Features are exit-local confidence (`top1p_dk`,
`top2p_dk`, margin, two ratios), because `S2_NB0_Fetch` deliberately excluded
checkpoints and the 45 base runs carry no weights. A gate with pooled embeddings
might capture more. **H2 is a lower bound**, and the notebook prints that itself.

---

## Q3 — Falsified in reverse, and the design is confounded

`analysis/s3_pruning.csv` — fixed 5,000-sample pool, `resnet20` target, 2 seeds:

| keep | saturated | unsaturated | random | full |
|---|---|---|---|---|
| 100 % | — | — | — | **70.24** |
| 50 % | 25.05 | **14.76** | 25.58 | |
| 30 % | 16.46 | **7.92** | 19.16 | |

H3 predicted the saturated source **worse** by ≥ 1.0 pt. It is **better by
8.54**. H3b predicted saturated within ±0.5 of random; it is 2.70 below.

### Why neither verdict tests the hypothesis

**Both guided arms lose to random pruning at 30 %** (saturated −2.70,
unsaturated −11.24). That is the tell.

The rule is *"keep the hardest samples"*, and at aggressive retention that is a
known-poor strategy — hard subsets are dominated by atypical and mislabelled
examples. Sorscher et al. (*Beyond neural scaling laws*, 2022) is explicit: keep
hard examples when data is abundant, keep **easy** ones when pruning hard.

So the experiment inverts itself. A **more reliable** score is **better** at
finding genuinely hard samples, therefore keeps a harder and noisier training
set, therefore trains worse. It measures *"how well does this score find hard
samples"* through a rule that punishes finding them.

Read that way the ordering is **consistent with Study 2** — `mobilenetv2`'s
`ce_loss` (ρ_seed 0.849) finds hard samples more effectively than
`convnext_femto`'s (ρ_seed 0.150) and is penalised for it. Weak supporting
evidence, arriving through a lens that reverses its sign.

**The regime compounds it.** 30 % of a 5,000-sample pool is 1,500 images across
100 classes — **15 per class**. Keeping atypical examples is maximally harmful
there, and absolute accuracy falls to 8–25 % against 70 % on full data.

### What would answer H3

1. **Invert the rule** — keep the *easiest* samples, as the pruning literature
   does at these rates; or
2. **use mild retention** (70–90 %), where keeping-hard is sensible; and
3. **report both directions**, since "which rule is right" is itself the
   confound.

Subsets are index lists, so rebuilding is free; only retraining costs (~6 GPU-h).

**Recorded as falsified-and-confounded, not re-cut.** The stopping rule says a
null is reported as a null.

**The pre-registration error was mine.** H3 assumed "better score → better
pruning", which holds only if the selection rule suits the regime. I did not
consider the regime when writing it — underspecified, not merely wrong.

---

## What Study 3 changes about Study 2

| Study 2 claim | after Study 3 |
|---|---|
| oracle excess +6.86 pt (frozen exits) | **confirmed and enlarged** — 8.55–10.64 pt with jointly trained exits |
| "the excess cannot be reached by any router" | **narrowed and re-supported** — a learned gate on the deployable signal captures 1.7 % |
| the memorisation collapse damages downstream use | **not yet demonstrated** — Q3's design cannot test it |

`study2/PAPER.md`'s blocking limitation is removed. The paper should report the
joint numbers as primary evidence and the frozen ones as the conservative case.

---

## Published artifacts

**HuggingFace: [`Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100)**
— 107 run directories, verified by scoped `tree/main/runs` query:

| | count |
|---|---|
| `p4-*-jointexit-s1` (Q1) | 3 |
| `p5-*-prune*` (Q3) | 14 |
| p0 / p1 / p3 (Studies 1–2) | 90 |

Each Study 3 run carries `config.yaml`, `config_hash.txt`, `summary.json`,
`STATUS.json`, `exit_heads.pt`, and `checkpoints/ env/ metrics/ per_sample/
telemetry/`.

**`analysis/` is complete on HuggingFace too** — 51 files, including all seven
`s2_*` and all five `s3_*` CSVs, verified **byte-identical** to the local copies
(12/12). Consolidated in [`../RESULTS.md`](../RESULTS.md).

---

## Limitations, all of them

- **Three architectures, one dataset, one scale.** Same as Study 2.
- **One joint-training recipe** (uniform deep supervision). Loss weighting,
  gradient rescaling and gradient equilibrium are untested variants (R-03).
- **Q2 is a lower bound** — exit-local confidence features, not embeddings.
- **Q3 is a 5,000-sample pool**, not CIFAR-100's 50,000, because `ce_loss` is
  written only to `train_holdout.parquet`.
- **Q1 used one seed per architecture.** Valid, since the quantity is a per-run
  identity — but it means no interval on the joint numbers.

## Reproduction

```
S3_NB0_Extrapolate   free gate: exit quality -> excess     ~15 min CPU
S3_NB1_JointTrain    Q1 joint exit training + measurement  ~10 GPU-h
S3_NB2_Compare       Q1 verdict, paired and conditioned    ~10 min CPU
S3_NB3_Router        Q2 learned gate, cross-seed control   minutes CPU
S3_NB4_Pruning       Q3 pruning demonstration              ~6 GPU-h
S3_NB5_Publish       upload everything, once, at the end   minutes
```

Verification: `tools/s3_canaries.py` (75), `tools/s3_nb0_harness.py`,
`tools/s3_nb3_harness.py`, `tools/s2_routing_canaries.py` (18),
`tools/s2_canaries.py` (3), `msc_lib --selftest` (461).
