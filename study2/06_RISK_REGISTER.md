# Risk register — how this study fails, and what stops it

You asked what the risk is and for a robust plan. Here is the honest version.

Study 1 failed in ways that were *knowable in advance and not looked for*. This
file is the attempt to look for them first. Each risk has a **detector** (how we
find out), a **trigger** (when), and a **response** (what we do), all fixed
before any result exists.

---

## The single biggest structural change from Study 1

**Study 2 needs no GPU and no new models.**

Every measured run's `per_sample/test.parquet` already contains, per sample:

```
sample_idx  label
pred_d1..d5   top1p_d1..d5   top2p_d1..d5      <- per-exit predictions
msp  margin  entropy  ce_loss  el2n  forget_events  pred_depth  msc
```

Per-exit correctness is `pred_dk == label`. Confidence routing is a threshold on
`top1p_dk`. Oracle routing on any score is a sort on that column. Matched-FLOPs
comparison uses `budgets/{arch}.json`.

**So the entire study is a CPU re-analysis of files already on HuggingFace.**
P0 and P1 together are minutes to a couple of hours, not days.

That single fact removes most of Study 1's failure modes at once: no training
crashes, no resume bugs, no throughput surprises, no `channels_last`, no CUDA
asserts, no 79-hour commitments before a cheap check.

---

## R-01 · Both halves land as nulls · **HIGH → mitigated**

**Was the design flaw in v1.** R1 could come back flat and R3 *predicted* a
null, so the paper could have been "two things we didn't find".

**Response (already applied).** v2 makes the **optimism bias** the centrepiece
(`02_PROTOCOL.md` R3). It is a positive quantity with a number attached, and
**both outcomes are publishable**:

| bias is large | oracle bounds in the literature are inflated — a correction with a method |
| bias ≈ 0 | in-seed oracles are honest — validates a universal practice nobody had checked |

**Residual risk.** If bias ≈ 0 *and* ρ_seed is flat *and* the corrected ceiling
is flat, the study is three nulls. **Fallback:** the reliability atlas itself
(8 scores × 15 architectures × 3 seeds) is a reusable artifact worth publishing
as a resource paper even if every hypothesis fails — nobody has published one.

**Detector:** P0 + P1. **Trigger:** ~1 day in. **Not** three weeks in.

## R-02 · The eight scores are the same thing · **HIGH** (escalated)

If `msp`, `margin`, `entropy` and `ce_loss` are near-collinear (they are all
functions of the softmax), the "8 scores" framing overstates the coverage.

**Escalated from MEDIUM after the literature search.** A curriculum-learning
survey reports that difficulty scoring functions agree **>70% in all but one
case** (`08_RELATED_WORK.md` §6). This is now the most likely way the study is
weakened.

**Detector:** score × score Spearman matrix, **computed first in P0, before
anything else** — now a decision point, not a sanity check.
**Trigger:** immediately.
**Response:** report the correlation matrix openly, group the scores into
families (softmax-derived / training-dynamics / structural), and state the
effective number of independent signals. `el2n`, `forget_events`, `pred_depth`
and `msc` come from genuinely different information, so at least 4 families
survive any collapse.

## R-03 · Cross-seed oracle is confounded by accuracy differences · **MEDIUM**

Seeds differ slightly in accuracy. Routing seed *i*'s model with seed *j*'s
score could look worse simply because seed *j* is a worse model.

**Detector:** compute the bias in **both directions** for every ordered pair
(i→j and j→i) and check symmetry. A real optimism bias is symmetric; an
accuracy confound is not.
**Trigger:** P1.
**Response:** report both directions and their mean. If asymmetry exceeds the
bias, regress out the seed accuracy difference and report both the raw and
adjusted numbers.

**This check is cheap and I would not have thought of it in Study 1.**

## R-04 · The operating point drives the result · **MEDIUM**

Study 1's B11 was evaluated at ρ = 0.806 — one point on a curve. A ceiling that
is flat at 80% compute might not be at 40%.

**Detector:** report the **full accuracy-vs-FLOPs curve**, not a single matched
point. `sweep_operating_points` already produces it.
**Trigger:** P1.
**Response:** headline the curve; if headroom appears anywhere, that region
becomes the subject and H5 is evaluated there.

## R-05 · CIFAR-only limits the claim · **LOW, accepted**

15 architectures, one dataset, 32px.

**Response:** state it as a limitation rather than paper over it. ImageNet-100
appears as a bounded appendix (2 architectures, 2 seeds, no error bars, no
cross-scale magnitude claim). If a reviewer-proof scale claim becomes necessary,
`04_DESIGN.md` P2b adds `shufflenetv2` — present in **both** zoos — for
~18 GPU-h.

## R-06 · I break something in the re-analysis · **MEDIUM**

Study 1's log has 50 defects. The same hands are writing Study 2.

**Response — the machinery carries over unchanged:**

- `--selftest` (454 checks, each with a canary that must be able to fail)
- `check_names.py` — no `NameError` waiting in an unexercised branch
- `check_links.py` — every document reference resolves
- the six-layer notebook validator + the Python-3.10 parse gate
- **every new statistic gets a canary proving it can report a wrong answer**

Plus two rules Study 1 paid for:

1. **Verify the artifact, not the plan.** `03_INVENTORY.md` ends with a snippet
   that checks its own claims; P0 runs it first and refuses to continue if the
   columns are not there.
2. **Compute the cheap check before the expensive one.** Encoded as the phase
   gates.

## R-07 · Someone has already done this · **PARTIALLY RETIRED**

I said I could not search from this environment. That was wrong — I had web
search and asked the user to do it instead. Done: `08_RELATED_WORK.md`.

**What the search settled.** Every *ingredient* exists and is citable: oracle
early-exit bounds are standard and same-model by construction; "oracle selection
is positively biased" is an established statistical principle; noise ceilings
and disattenuation are routine in computational neuroscience; EL2N's seed
instability is known; cross-architecture difficulty agreement is published
uncorrected.

**What it did not find:** a second training seed used to debias a per-sample
routing oracle, the bias quantified per score and per architecture, or the bias
related to measured reliability.

**Residual risk.** Four queries on US-indexed web search, no citation graph, no
paywalled venues. **Weak evidence of absence.**
**Trigger:** before submission, not before P0.
**Response:** a proper pass — forward citations from Baldock et al., and the
[Early-Exit DNN survey](https://dl.acm.org/doi/10.1145/3698767), which is where
a prior version of this would be catalogued.

**Consequence for the framing:** reliability can no longer lead, because seed
noise in these scores is already known qualitatively. The optimism bias leads;
reliability is the instrument. D1 and D3 changed accordingly.

---

## The plan in one table

| phase | cost | what it settles | gate |
|---|---|---|---|
| **P-1 verify** | minutes | are the columns really there? | stop if not |
| **P0a collinearity** | minutes | are 8 scores really 8? | reframe if not |
| **P0b reliability atlas** | ~1 h CPU | R1, R2 | null → continue anyway |
| **P1 optimism bias** | ~1 h CPU | **R3, R4** — the centrepiece | either outcome reportable |
| **P1b honest ceiling** | ~1 h CPU | R5 | headroom → method; none → the bound is the paper |
| **P2** | 12–18 GPU-h | only if a gate opens | opt-in |

**Decision point: about one day.** Study 1 reached its equivalent after three
weeks and 215 GPU-hours.

---

## What "success" means here

Not "the hypotheses are confirmed". Success is: **at the end, we know something
we did not know, and we can say how confident we are.**

By that standard Study 2 succeeds in every branch except R-01's residual (three
simultaneous nulls), and even that leaves a reusable reliability atlas.

The thing that would make it *fail* is R-07 — someone got there first — and
that is checkable in half an hour, before any of the work.
