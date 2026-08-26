# Study 3 — pre-registration

**Written before any run.** Predictions and thresholds are fixed here so the
result cannot be re-cut afterwards. Amendments get a date and a reason.

---

## Q1 — Does the oracle excess survive jointly trained exits?

**Background.** Study 2 measured `oracle_in − acc_full = +6.86 pt` (median over
90 runs, range 4.81–11.56, positive in 100 %). Study 1's exit heads were trained
on a **frozen backbone**. Jointly trained exits are stronger and better
calibrated, so the pool of "early exit right, final layer wrong" samples may
shrink.

**Design.** Paired. One variable changed.

| held constant | varied |
|---|---|
| architecture, dataset, epochs, optimiser, augmentation | exit-head training: **frozen** vs **joint** |
| exit positions, FLOP budgets, measurement code | |

3 architectures × 1 seed. **No seeds needed** — the quantity is a per-run
identity. Architectures chosen to span the range and stay cheap:

| arch | Study 1 time | why |
|---|---|---|
| `resnet20` | 1.3 h | cheapest; low capacity, small existing pool |
| `resnet32x4` | 2.9 h | mid; the reference architecture throughout both studies |
| `vgg8` or `mobilenetv2` | ~2.2 h | different family, guards against a ResNet-only result |

Joint training adds a forward/backward through K heads: budget **+30–40 %**, so
≈ 9–10 GPU-h total.

> **These timings may be pessimistic (D-87).** Study 1's CIFAR runs predate
> `assert_layout_match` and were most likely paying a layout-conversion tax
> on every convolution. Correctness is unaffected — it is a throughput bug —
> so the results stand, but the wall-clock figures they are derived from may
> be too high. Re-measure from Study 3's own runs rather than trusting these.

**H1 (pre-registered).** With jointly trained exits, the oracle still exceeds
full-compute accuracy:

```
oracle_in − acc_full  ≥  2.0 accuracy points,  in 3 of 3 architectures
```

**Thresholds and what each outcome means.**

| measured | verdict | consequence |
|---|---|---|
| **≥ 2.0 pt** in 3/3 | **H1 supported** | Study 2's claim is not a frozen-backbone artifact. Report both numbers; the archival paper is unblocked. |
| **0.5 – 2.0 pt** | partial | the effect is real but much smaller than +6.86. The paper leads with the joint number and reports frozen as an upper bound. |
| **< 0.5 pt** or negative | **H1 falsified** | the effect IS an artifact of post-hoc exits. Study 2 is withdrawn to a methodological note, and **Q3 becomes the main paper**. |

**The comparison must be at matched final accuracy.** Joint training changes
backbone accuracy, so a raw pool comparison confounds two things. Report the
pool both raw and conditioned on `acc_full`; if joint training moves final
accuracy by more than 1 pt, the conditioned number is primary.

**Falsifier, stated now.** If the pool vanishes under joint training, the honest
reading is that Study 2 measured a property of *weak post-hoc exits*, not of
oracle bounds in general. That is a real finding too — it says the literature's
bounds are fine when exits are trained properly — and it must be reported with
the same prominence, not buried.

---

## Q2 — How much of the gap can a learned router capture?

**Background.** Study 2 says the excess "cannot be reached by any router". What
was shown is narrower: *a second seed* cannot reach it. A learned router seeing
the input may do better.

**Design.** For each run, dump per-exit pooled features (10k test + 5k holdout ×
K exits × ~256 dims ≈ 50 MB/run, one forward pass). Train a small gate at each
exit *k* using **only features available at exit k** — this is the deployability
constraint and it is the whole point. Evaluate at matched cost against:

- confidence thresholding (Study 2's baseline)
- the in-seed oracle (the ceiling)

```
capture fraction = (router − baseline) / (oracle_in − baseline)
```

**H2 (pre-registered).** A learned router captures **less than 25 %** of the
oracle gap. Rationale: Study 2's cross-seed result implies most of the gap is
seed-specific, and a router trained on one seed's idiosyncrasies is learning
noise that will not generalise — which the held-out split will expose.

**Critical control.** Train the gate on seed *i* and evaluate on seed *j*'s
network. If capture collapses across seeds, the router memorised noise; if it
holds, the signal is real and transferable and **H2 is falsified in the most
interesting way** — that would be a method, not a bound.

**This is the experiment most likely to produce something publishable in its own
right**, because all three outcomes say something and two of them are positive.

---

## Q3 — Does the memorisation collapse damage downstream pruning?

**Background.** Study 2 §4.3: `ce_loss` ρ_seed falls 0.647 → 0.108 on data the
network has fit; predicted by saturation (+0.832), not by test accuracy
(−0.114). Dataset pruning computes these scores on training data, from one seed,
for exactly the high-capacity models where the collapse is worst.

**Design.** Score CIFAR-100's training set with a single seed of:

- **saturated source**: `convnext_femto` (99.99 % train acc, 71 % of samples
  above 0.99 confidence, ρ_seed drop 0.558)
- **unsaturated source**: `mobilenetv2` (83.1 % train acc, 16 % saturated, drop
  0.026)

Prune to 50 % and 30 %, keeping the hardest samples. Retrain a **fixed target**
(`resnet20`, 1.3 h) on each subset, 2 seeds. Baselines: random pruning at each
rate, and full data.

| arm | runs |
|---|---|
| saturated source × 2 rates × 2 seeds | 4 |
| unsaturated source × 2 rates × 2 seeds | 4 |
| random × 2 rates × 2 seeds | 4 |
| full data × 2 seeds | 2 |
| **total** | **14 ≈ 18 GPU-h** |

**H3 (pre-registered).** Pruning guided by the saturated source performs worse
than the unsaturated source by **≥ 1.0 accuracy point** at 30 % retention, and
the gap **widens** as retention falls.

**H3b.** The saturated source is **not distinguishable from random pruning**
(within ±0.5 pt) at 30 % retention. If true, that is the headline: *a
single-seed difficulty score from a memorising model is worth no more than
chance.*

**Cheap pre-check before spending 18 GPU-h.** Compute the overlap between the
kept-sets. If the saturated and unsaturated sources select ≥ 90 % of the same
samples, no downstream difference is possible and the experiment is cancelled.
**This gate costs minutes and runs first.**

---

## Stopping rules

1. **A null is reported as a null.** No re-cutting, no post-hoc subgroup search.
   Study 2 followed this for H3 and H4 and was better for it.
2. **The cheap gate decides the expensive run.** P0 before P1; the kept-set
   overlap check before P3. If a gate says stop, stop.
3. **Any statistic without a failing canary is not reported.** Study 2's P0a
   printed a conclusion from zero samples.
4. **Favourable results get attacked first.** If a number matches the
   prediction, the next action is an attempt to break it, not a write-up.
5. **Amendments are dated and recorded here**, with the reason, above the
   original text.

## What would make Study 3 wrong

- **Our "joint training" is not the field's.** Deep supervision has many
  variants (loss weighting, gradient rescaling, gradient equilibrium). We use
  one standard recipe and must say which, and that sensitivity to the variant is
  untested.
- **Three architectures, one dataset, one scale.** Same limitation as Study 2.
- **The learned router is a lower bound on what routers can do.** A better
  architecture or more features could capture more. H2 bounds *our* router, and
  the paper must say so.
- **The pruning target is one small model.** A different target could respond
  differently to the same subset.
