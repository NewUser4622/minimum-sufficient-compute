# Study 2 — proposal

**Nothing here has been run.** This folder is a design, written after Study 1
finished, so that the decision to start is made on paper rather than after
another 80 GPU-hours.

Study 1 (`docs/cifar100/`, `docs/imagenet100/`) is complete and stays as it is.

---

## The one-sentence version

> **Oracle upper bounds for per-sample adaptive inference are optimistically
> biased, because they are computed from the same model they route.** The bias
> is measurable, it scales with how unreliably the routing signal is measured,
> and correcting it removes most of the headroom the field reports.

An oracle ceiling asks *"if we knew each sample's true difficulty, how well
could we route?"* — and then computes that difficulty **from the model being
routed**. So it partly routes on that model's own noise, which no deployable
router could have. With ≥2 seeds you can separate the two:

```
in-seed   oracle : score from seed i  ->  routes seed i's model   (optimistic)
cross-seed oracle: score from seed j  ->  routes seed i's model   (honest)
optimism bias    = in-seed - cross-seed              at matched FLOPs
```

**Both outcomes are publishable**, which is the point: a large bias means the
field's upper bounds are inflated and here is the correction; a bias of zero
validates a practice nobody had checked. Neither depends on MSC being a good
metric — the flaw that sank Study 1.

*(v1 of this proposal had two halves that could both land as nulls. That was my
design error; see `06_RISK_REGISTER.md` §R-01.)*

---

## Why this is worth doing

**Half 1 — reliability, as the instrument.**
*(Reframed after the literature check: seed noise in these scores is already
known qualitatively, so it cannot lead. It is what makes Half 2 measurable.)*
Study 1 measured ρ_seed (agreement between two seeds of the same architecture)
for MSC and found it ranges **0.547 → 0.822** depending on architecture. The
example-difficulty literature — EL2N, forgetting events, C-score, prediction
depth, memorisation — routinely reports cross-architecture and cross-score
correlations, and routinely does **not** report a noise ceiling. Every such
correlation is attenuated by an unmeasured, architecture-dependent amount.

We can compute the ceiling for **eight** scores across **15 architectures**
from data already on HuggingFace. No new training.

**Half 2 — the routing ceiling.**
Study 1's B11 baseline gave a router the student's own *true* post-hoc MSC and
measured accuracy at matched FLOPs. Result: **+0.00007 over confidence
thresholding** — no headroom at all. If that generalises to the other seven
scores, then the premise behind per-sample adaptive routing is empty at these
operating points, regardless of method. Early-exit papers almost never compute
this ceiling; they compare a learned router against confidence and win by small
margins.

**Together:** *the signal is noisier than reported, and the ceiling is lower
than assumed.*

---

## What it costs

| phase | new training | compute | what it settles |
|---|---|---|---|
| **P0** reliability atlas | **none** | ~1 h **CPU** | ρ_seed for 8 scores × 15 architectures |
| **P1** optimism bias + honest ceiling | **none** | ~2 h **CPU** | R3, R4, R5 |
| **P2** confirmation | optional | 12–18 GPU-h | **only if a gate opens** |

**No GPU. No models loaded at all.** Every measured run's
`per_sample/test.parquet` already carries `pred_d1..dK`, `top1p_d*`, `top2p_d*`
and `label` — per-exit predictions for every sample. Correctness is
`pred_dk == label`; confidence routing is a threshold on `top1p_dk`; oracle
routing on any score is a sort; cost comes from `budgets/{arch}.json`.

That single fact removes most of Study 1's failure surface: no training, no
resume, no throughput, no `channels_last`, no CUDA asserts, no 79-hour
commitment before a cheap check.

---

## Read in this order

| file | what it is |
|---|---|
| [`01_POSTMORTEM.md`](01_POSTMORTEM.md) | why Study 1 fell short — five diagnoses, each with evidence |
| [`02_PROTOCOL.md`](02_PROTOCOL.md) | the pre-registration: questions, hypotheses, gates, stopping rules |
| [`03_INVENTORY.md`](03_INVENTORY.md) | exactly what data exists and what it can answer without new runs |
| [`04_DESIGN.md`](04_DESIGN.md) | the plan, phase by phase, with costs |
| [`05_OPEN_DECISIONS.md`](05_OPEN_DECISIONS.md) | **what needs your call before anything starts** |
| [`06_RISK_REGISTER.md`](06_RISK_REGISTER.md) | **how this fails, and what stops it** — seven risks, each with a detector, trigger and response |
| [`07_PROGRESS.md`](07_PROGRESS.md) | live log — newest first, updated every session |
| [`08_RELATED_WORK.md`](08_RELATED_WORK.md) | **what already exists and where the gap is** — the literature check, done |

---

## The rule this study is built around

> **Measure the ceiling before building the method.**

Study 1 spent 79 GPU-hours training 18 MSC-KD students to close a gap that a
two-hour oracle measurement showed did not exist. The ceiling was computed
*after* the method, and only because the method's failure forced the question.

In Study 2, every claim of the form "signal X could improve Y" is preceded by
"what is the best X could possibly do", and that measurement is a gate: if the
ceiling is flat, the method is not built.

---

## Honest risk

Full register in [`06_RISK_REGISTER.md`](06_RISK_REGISTER.md). The short form:

- **The genuine failure case** is three simultaneous nulls — flat reliability,
  zero bias, flat ceiling. Fallback: the reliability atlas itself (8 scores ×
  15 architectures × 3 seeds) is a reusable artifact nobody has published.
- **The literature check is done** (`08_RELATED_WORK.md`). Every *ingredient*
  exists and is citable — oracle early-exit bounds are standard and same-model
  by construction, "oracle selection is positively biased" is established,
  noise ceilings are routine in computational neuroscience, EL2N's seed noise is
  known. **What was not found**: a second seed used to debias a per-sample
  routing oracle. The novelty is the combination and the measurement.
- **The search made one risk worse.** Difficulty scoring functions are reported
  to agree **>70% with each other**, so the eight scores may be three or four
  families. R-02 is escalated to HIGH and P0a is now a decision point, not a
  sanity check.

Everything else is detected within about a day of CPU work, before any
commitment. Study 1 reached its equivalent decision point after three weeks and
215 GPU-hours.
