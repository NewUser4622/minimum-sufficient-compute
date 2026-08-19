# Study 2 — pre-registration

**Status: DRAFT. Not yet committed to.** Once P0 runs, this file is frozen and
every change to it is recorded with a date and a reason.

Study 1's mistake was inheriting a pre-registration across a design change
(`01_POSTMORTEM.md` §5). This one is written for *this* experiment, and every
threshold below has a stated justification rather than being carried over.

---

## The claim

> Per-sample difficulty scores are measured with architecture-dependent noise
> that the literature does not report, and correcting for it removes the
> headroom that motivates per-sample adaptive inference.

---

## Scores under test

Eight, all already computed per sample in every measured run:

| score | source | what it is |
|---|---|---|
| `msp` | max softmax probability | the confidence baseline everything is compared to |
| `margin` | top1 − top2 probability | |
| `entropy` | predictive entropy | |
| `ce_loss` | per-sample cross-entropy | |
| `el2n` | Paul et al. 2021 | early-training gradient-norm proxy |
| `forget_events` | Toneva et al. 2019 | count of correct→incorrect transitions |
| `pred_depth` | Baldock et al. 2021 | earliest layer a k-NN probe is right |
| `msc` | Study 1 | cost-normalised stable-sufficiency depth |

MSC is **one column among eight**, not the subject. That is deliberate.

---

## R1 — How reliable is each difficulty score?

**ρ_seed(score, arch)** = Spearman between the score computed from two seeds of
the same architecture, over the same samples.

**H1.** ρ_seed varies by **both** score and architecture, with a range of at
least **0.15** across the score × architecture grid.

*Justification for 0.15:* Study 1 measured a 0.275 range for MSC alone
(0.547–0.822). A threshold at roughly half that is a conservative bar for "this
variation is real and worth reporting", chosen before seeing the other seven
scores.

**Falsified if** every score sits within 0.15 of every other on every
architecture — i.e. reliability is essentially constant, and the literature's
practice of not reporting it is harmless.

**This is the outcome that would kill Half 1, and it is a real possibility.**

## R2 — How much does unmeasured noise distort published-style comparisons?

For each score, compute cross-architecture agreement **raw** and
**disattenuated** by √(ρ_a·ρ_b).

**H2.** The median raw-to-disattenuated correction exceeds **0.10** for at
least one score, and the *ranking* of architecture pairs by agreement changes
under correction.

*Justification:* a correction smaller than 0.10 that preserves rank order would
mean the omission is cosmetic. A rank change means published orderings could be
artifacts of unequal measurement precision — which is the claim with teeth.

## R3 — What is the oracle ceiling for routing on each score? (the gate)

For each score *s*, route each sample to the cheapest exit using the **true**
value of *s* for that sample, and measure accuracy at matched average FLOPs
against `msp` thresholding.

**H3.** No score's oracle exceeds `msp` by more than **1.0 accuracy point** at
matched FLOPs.

*Justification:* 1.0 point is the effect size Study 1's H5 pre-registered for
MSC-KD, so using it keeps the two studies commensurable. Study 1 measured
+0.00007 for MSC's oracle — four orders of magnitude below it.

**This hypothesis predicts a null, on purpose.** It is a bound, and it is the
**gate for P2**:

> If **no** score clears +1.0 point, no routing method is built. The result is
> the ceiling itself.
>
> If **some** score clears it, that score becomes the subject and the study
> pivots from a bound to a method — a better outcome, and one this design
> detects for ~6 GPU-hours instead of 79.

## R4 — Does reliability explain the ceiling?

**H4.** Across scores, ρ_seed correlates with oracle-ceiling headroom at
Spearman **≥ 0.5**.

If true, "difficulty routing does not help" and "difficulty is measured
noisily" are the same fact, which is a tighter story than two separate
observations. If false, they are independent limitations and both must be
reported.

*Justification for 0.5:* with eight scores, |ρ| ≥ 0.5 is roughly the point at
which a monotone relationship is visible above sampling noise at n=8. It is
deliberately modest — with n=8 this is suggestive, never conclusive, and will
be reported as such.

---

## Stopping rules

Written down now so they are not negotiated later against a result.

1. **After P0**, if H1 is falsified, Half 1 is reported as a null and the study
   continues on Half 2 alone. It does not get re-cut to find a positive.
2. **After P1**, if H3 holds (no score clears +1.0), **P2 is not run.** The
   paper is the ceiling.
3. **After P1**, if a score clears +1.0, P2 is run *for that score only*, and
   the pre-registration is amended in writing with the date and the reason.
4. **n = 3 seeds minimum** for any ρ_seed reported as a headline number. Study 1's
   ImageNet arm has 2, so ImageNet ρ_seed values are reported as
   **point estimates without error bars** and are never used for a family-level
   claim.

## What would make this study wrong

- ρ_seed is high and flat for all eight scores → H1 falsified, Half 1 is a null.
- Some oracle clears +1.0 comfortably → H3 falsified, the framing inverts.
- The eight scores are so correlated that they are one score → R1–R4 collapse to
  a statement about one quantity. **Checked in P0 before anything else**, by
  reporting the score × score correlation matrix.

## Deliberately not claimed

- **Nothing about scale.** CIFAR-100 and ImageNet-100 share no architecture
  (`01_POSTMORTEM.md` §4). Any cross-dataset statement is confounded and is out
  of scope unless P2 adds a shared architecture.
- **Nothing about which score is "best"** for anything other than routing at
  the operating points measured.
- **No claim that MSC is or is not a good metric.** It is one column.
