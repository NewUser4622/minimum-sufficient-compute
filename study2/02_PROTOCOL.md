# Study 2 — pre-registration

**Status: FROZEN 2026-08-19, SCORED 2026-08-20.** The hypotheses below are as
pre-registered. The outcomes are in the scoreboard immediately following; the
text of each R-section is unchanged from before the data was seen.

## Scoreboard — as pre-registered, scored against the data

| | hypothesis | threshold | measured | verdict |
|---|---|---|---|---|
| **H1** | ρ_seed varies across the grid | ≥ 0.15 | **0.667** | **SUPPORTED** |
| **H2** | disattenuation changes published-style comparisons | — | see `s2_reliability_grid.csv` | reported, not gated |
| **H3** | in-seed oracle is optimistic | ≥ 0.5 pt | **+22.41 pt** (A +6.86, B +15.33) | **SUPPORTED** for the true per-exit oracle; **NOT** supported for per-sample difficulty scores (−0.57 pt) |
| **H4** | bias tracks unreliability | ρ ≥ 0.5 | **+0.232 … +0.457** | **NOT SUPPORTED** — right sign, below threshold on every score |
| **H5** | nothing clears +1.0 pt honest headroom | +1.0 pt | negative at all 7 budgets, 0 of 15 architectures positive | **SUPPORTED** |

**Amendments made after freezing**, each with a reason:

1. **The oracle was re-specified** (2026-08-20). The pre-registration said
   "oracle" without saying whether it was per-sample or per-exit. Three
   implementations were tried before the ambiguity was noticed; the final one is
   the per-exit Lagrangian maximum, which is what the early-exit literature
   means. Both readings are reported: the per-exit oracle in §R3, the
   per-sample-score variant alongside it.
2. **"At matched FLOPs" withdrawn as a description** (2026-08-20). At ρ = 0.80
   the budget does not bind for the in-seed oracle, which therefore spends *less*
   than the baseline. Still a valid and conservative upper bound; the phrase was
   simply wrong.
3. **H3's headline decomposed** (2026-08-20). The bias does not correlate with
   the noise pool across architectures (ρ = +0.011), so attributing all of it to
   noise harvesting was unsupported. Split into A (exact identity) and B
   (cross-seed non-transfer).

---

### original pre-registration follows, unedited

Once P0 runs this file is frozen and
every change is recorded with a date and a reason.

**v2 changed the centrepiece.** v1 had two halves that could both land as nulls
— R1 might come back flat, and R3 *predicted* a null by design. That is a bad
bet, and it was my design error. See `06_RISK_REGISTER.md` §R-01.

---

## The claim

> **Oracle upper bounds for per-sample adaptive inference are optimistically
> biased, because they are computed from the same model they route.** The bias
> is measurable, it scales with how unreliably the routing signal is measured,
> and correcting it removes most of the headroom the field reports.

This is a **positive, quantitative** result that holds whether or not the
corrected ceiling turns out flat. That property is the whole point of v2.

---

## The mechanism, stated plainly

An "oracle" ceiling asks: *if we knew each sample's true difficulty, how well
could we route?* In practice the difficulty is computed **from the same trained
model being routed**. So the oracle partly routes on that model's own noise —
its seed-specific idiosyncrasies — which no deployable router could ever have.

With ≥2 seeds of the same architecture you can separate the two:

| | routing signal | what it measures |
|---|---|---|
| **in-seed oracle** | score from seed *i* | routes seed *i*'s model — **optimistic** |
| **cross-seed oracle** | score from seed *j ≠ i* | routes seed *i*'s model — **honest** |

```
optimism bias  =  in-seed accuracy  −  cross-seed accuracy       (at matched FLOPs)
```

Study 1 already reported the in-seed number: B11 for MSC was **+0.00007** over
confidence. Nobody has reported the cross-seed one, for any score.

---

## Scores under test

Eight, all already computed per sample in every measured run:

`msp` · `margin` · `entropy` · `ce_loss` · `el2n` · `forget_events` ·
`pred_depth` · `msc`

MSC is **one column among eight**. That is deliberate — `01_POSTMORTEM.md` §2.

---

## R1 — How reliable is each difficulty score?

ρ_seed(score, arch) = Spearman between the score from two seeds of the same
architecture, over the same samples.

**H1.** ρ_seed varies across the score × architecture grid by at least **0.15**.

*Justification:* Study 1 measured a 0.275 range for MSC alone (0.547–0.822).
Half that is a conservative bar for "this variation is real", set before seeing
the other seven scores.

*If falsified:* reliability is essentially constant. R1 is reported as a null;
**R3 is unaffected** — the optimism bias can be large even when reliability is
uniformly high.

## R2 — Does ignoring reliability distort published-style comparisons?

Cross-architecture agreement per score, raw and disattenuated by √(ρ_a·ρ_b).

**H2.** The median correction exceeds **0.10** for at least one score, **and**
the ranking of architecture pairs changes under correction.

*Justification:* a correction that preserves rank order is cosmetic. A rank
change means published orderings may be artifacts of unequal precision.

## R3 — How optimistic is the in-seed oracle? **(the centrepiece)**

For every architecture, every score, every ordered seed pair (i, j), at matched
average FLOPs.

**H3.** The median optimism bias across scores and architectures is
**≥ 0.5 accuracy points**, and is **> 0 for at least 6 of the 8 scores**.

*Justification for 0.5:* half the +1.0 point effect size that Study 1
pre-registered for a *method*. If oracle bounds are inflated by half of what a
method is expected to gain, every such bound in the literature is materially
wrong. The 6-of-8 clause stops one outlier score carrying the claim.

*If falsified* (bias ≈ 0 everywhere): in-seed oracles are honest, the field's
upper bounds stand, and **that is itself a useful, reportable methods result** —
it licenses a practice nobody had validated.

**This hypothesis cannot land as "nothing to report" in either direction.**

## R4 — Does the bias follow reliability?

**H4.** Across the score × architecture grid, optimism bias correlates with
(1 − ρ_seed) at Spearman **≥ 0.5**.

*Justification:* the mechanism predicts it — a less reliably measured score has
more seed-specific noise for an in-seed oracle to exploit. Confirming it turns
two observations into one mechanism, and gives a **cheap predictor**: measure
ρ_seed (minutes, no model) and you can estimate how inflated a published oracle
bound is without recomputing it.

*Caveat, stated now:* n = 8 scores. This is suggestive, never conclusive, and
will be reported as such with the full scatter, not just the coefficient.

## R5 — After correction, is there any headroom left?

**H5.** No score's **cross-seed** oracle exceeds `msp` thresholding by more than
**1.0 accuracy point** at matched FLOPs.

*Justification:* 1.0 point is Study 1's H5 bar, kept so the two studies are
commensurable.

**This is the gate for any method work.** If some score clears it, that score
becomes the subject and the protocol is amended in writing. If none does, the
bound is the result and no method is built.

---

## Stopping rules

Written before any result exists.

1. **After P0**: if H1 is falsified, R1 is reported as a null. It is not re-cut
   looking for a positive, and R3 continues regardless.
2. **After P1**: if H5 holds, **no method is built.** The corrected ceiling is
   the paper.
3. **If H5 fails** for some score: amend this file with the date and reason,
   then scope any method to that score at that operating point.
4. **≥ 3 seeds** for any headline ρ_seed or bias. CIFAR-100 has 3.
   ImageNet-100 has 2, so ImageNet numbers are **point estimates without error
   bars** and never carry a family-level claim.
5. Any ratio whose denominator is within 2 SE of zero is reported as
   **"within noise"**, never as a ratio. (Study 1 published `26.0`, `−47.9` and
   `83.6` by dividing by 0.00007 — D-80.)

## What would make this study wrong

- **The eight scores are near-collinear** → R1–R5 collapse to a statement about
  one quantity. **Checked first, in P0, before anything else is computed.**
- **Bias ≈ 0 everywhere** → H3 falsified; reported as a positive validation of
  existing practice.
- **ρ_seed flat AND bias ≈ 0** → both halves null. This is the genuine failure
  case; `06_RISK_REGISTER.md` §R-01 gives the fallback.

## Deliberately not claimed

- **Nothing about scale.** CIFAR-100 and ImageNet-100 share no architecture
  (`01_POSTMORTEM.md` §4).
- **Nothing about which score is best** outside routing at the measured
  operating points.
- **No claim that MSC is or is not a good metric.** It is one column.
