# Related work — what exists, and where the gap is

I said I could not search from this environment. That was wrong — I had web
search available and asked you to do it anyway. Done now, ~30 minutes, and the
result changes two things in the plan.

**Verdict: the core idea appears unclaimed, but it is closer to existing work
than I implied, and one risk got worse rather than better.**

---

## 1. Oracle upper bounds in early exit — standard practice, same-model by construction

Early-exit papers routinely report an oracle bound. The usual definition:

> An Oracle EE strategy **exits at the first layer whose prediction matches that
> of the last layer**, which provides an ideal upper bound for how much
> computation could be saved.
> — [Early Exit Is a Natural Capability in Transformer-based Models](https://arxiv.org/html/2412.01455v1)

Note what that is: the oracle is computed **from the very network being routed**.
Every quantity in it — the per-layer predictions, the final prediction — comes
from one trained model. That is precisely the construction Study 2 argues is
optimistically biased. So the practice we are targeting is real, current, and
widespread rather than a straw man.

A related limitation is already acknowledged in that literature:

> Similarity between an exit layer and the final layer does not directly
> translate to end-to-end task accuracy when early-exit is applied.

That is a *different* criticism (agreement ≠ accuracy). Ours is about **whose
noise the oracle is exploiting**.

## 2. Optimism of selection — known in general, not applied here

The statistical principle is established:

> Oracle selection takes a maximum over noisy performance estimates; such
> procedures are **positively biased, systematically overstating achievable
> gains**.

and there is a formal literature on exactly this — e.g.
[Excess Optimism: How Biased is the Apparent Error of an Estimator Tuned by
SURE?](https://arxiv.org/pdf/1612.09415).

**This is the honest antecedent of our claim.** The general phenomenon is not
new. What I did not find is anyone **quantifying it for per-sample early-exit
routing**, or using a **second training seed** as the debiasing instrument.

## 3. Noise ceilings and disattenuation — standard in neuroscience, absent here

This was the most useful find. Computational neuroscience has used noise
ceilings as routine practice for a decade:

- [Methods for computing the maximum performance of computational models of
  fMRI responses](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6426260/)
- [Noise ceiling on the crossvalidated performance of reweighted models of
  representational dissimilarity](https://www.biorxiv.org/content/10.1101/2020.03.23.003046v1)
  (Khaligh-Razavi & Kriegeskorte addendum)

> The noise ceiling of goodness of fit is identical to the output of the
> **Spearman-Brown prophecy formula**.

and noise-ceiling-**corrected** Spearman correlation is the normal way to
compare models there.

**The example-difficulty literature does not do this.** That is a clean framing:
*import a standard correction from a field that solved this, into a field that
has not adopted it.* It is citable, uncontroversial as a method, and the
contribution is the application plus the consequence — not the statistic.

## 4. Seed instability of difficulty scores — known qualitatively, not mapped

> The outcomes of snapshot methods like **EL2N differ significantly from run to
> run**, making it difficult to obtain a reliable score in a single run. Methods
> using training dynamics offer more reliability.
> — [Lightweight Dataset Pruning without Full Training](https://arxiv.org/html/2502.06905v2)

Forgetting-events work has looked at multi-seed stability too
([Toneva et al.](https://arxiv.org/pdf/1812.05159) estimate forgetting across 5
seeds).

So **"these scores are seed-noisy" is not a new observation.** What I did not
find is a *systematic grid* — reliability per score × per architecture — or the
disattenuation consequence for published cross-architecture comparisons.

**This weakens R1 as a headline.** It should be positioned as the *instrument*
for the optimism-bias result, not as the finding.

**MEASURED 2026-08-20 — and this section understated what was available.** The
seed instability of these scores is known qualitatively, but nobody appears to
have reported that it is **conditional on whether the model has fit the data**:
ce_loss ρ_seed falls from 0.647 to **0.108** for `mixer_nano` between held-out
and memorised samples, predicted by softmax saturation (ρ = +0.832) and not by
test accuracy (−0.114). Dataset-pruning methods compute these scores on training
data, from one seed, for exactly the high-capacity models where the collapse is
worst. See [`PAPER.md`](PAPER.md) §4.3.

## 5. Cross-architecture difficulty agreement — reported, uncorrected

[Deep Learning Through the Lens of Example Difficulty](https://arxiv.org/pdf/2106.09647)
(Baldock et al., the `pred_depth` source) reports cross-architecture consistency
with Spearman coefficients — **and no noise ceiling**. That is exactly the class
of number R2 says is attenuated by an unmeasured amount.

## 6. The finding that hurts — scores may be near-collinear

> Almost all scoring functions share a **very similar notion of difficulty**,
> with all approaches **agreeing to more than 70%** in all but one case.
> — [Does the Definition of Difficulty Matter?](https://arxiv.org/html/2411.00973)

This is the one result that makes the plan *worse*, and it must not be buried.
**MEASURED 2026-08-20: this risk was real and worse than the survey suggested.**
The four softmax scores correlate at **ρ = 0.997–1.000**, not 70 %. Eight
candidates are three families. See [`PAPER.md`](PAPER.md) §4.1.

If eight scores are largely one score, then "8 scores × 15 architectures" is not
120 independent cells, and R1/R4's effective n is much smaller than it looks.

**Consequence:** `06_RISK_REGISTER.md` §R-02 moves from MEDIUM to **HIGH**, and
P0a (the collinearity matrix) stops being a sanity check and becomes a
**decision point**. Run it first, and if the softmax-derived four collapse into
one, report four families rather than eight scores and recompute the effective n.

---

## Where the gap actually is

| ingredient | exists? | where |
|---|---|---|
| oracle upper bounds for early exit | **yes**, standard | early-exit literature |
| oracle bounds are computed from the routed model | **yes**, by construction | same |
| "oracle selection is positively biased" as a principle | **yes** | SURE / excess-optimism |
| noise ceilings + disattenuation as routine method | **yes** | computational neuroscience |
| difficulty scores are seed-noisy | **yes**, qualitatively | EL2N, forgetting-events work |
| cross-architecture difficulty agreement, uncorrected | **yes** | Baldock et al. |
| **a second seed used to debias a per-sample routing oracle** | **not found** | — |
| **optimism bias quantified per score and per architecture** | **not found** | — |
| **bias related to measured score reliability** | **not found** | — |

The novelty is the **combination and the measurement**, not any single
ingredient. That is a normal contribution and it is stronger for saying so
plainly — every component has a citation, so the method is not in question, only
the result.

**Revised framing, one sentence:**

> Oracle upper bounds for early-exit routing are computed from the model being
> routed and are therefore optimistically biased; using a second training seed
> as an unbiased instrument, we measure that bias across 8 difficulty scores and
> 15 architectures, relate it to each score's measured reliability, and report
> the corrected headroom.

## Caveats on this search

Four queries, US-indexed web search, no access to paywalled venues or a proper
citation graph. **Absence of evidence over four searches is weak evidence of
absence.** Before submission this needs a real pass — Semantic Scholar forward
citations from Baldock et al. and from the early-exit surveys, and a look at
[Early-Exit Deep Neural Network — A Comprehensive Survey](https://dl.acm.org/doi/10.1145/3698767),
which is the obvious place a prior version of this idea would be catalogued.

**Not a blocker for P0/P1.** Those cost a day, produce the artifact either way,
and the artifact is what makes the literature question answerable with numbers
rather than adjectives.
