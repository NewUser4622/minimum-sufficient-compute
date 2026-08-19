# Study 2 — proposal

**Nothing here has been run.** This folder is a design, written after Study 1
finished, so that the decision to start is made on paper rather than after
another 80 GPU-hours.

Study 1 (`docs/cifar100/`, `docs/imagenet100/`) is complete and stays as it is.

---

## The one-sentence version

> Per-sample difficulty scores are measured with **architecture-dependent
> noise that essentially nobody reports**, and once you grant a router *oracle*
> access to the true score, the headroom that motivates adaptive inference
> **is not there**.

Two halves. The first is a measurement the literature is missing. The second is
a ceiling that bounds a whole family of proposed methods. Neither depends on
MSC being a good metric — which is precisely the flaw that sank Study 1.

---

## Why this is worth doing

**Half 1 — the noise ceiling nobody reports.**
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
| **P0** reliability atlas | **none** | ~2 CPU-h | ρ_seed for 8 scores × 15 architectures |
| **P1** routing ceiling | **none** | ~6 GPU-h | oracle ceiling per score, from saved checkpoints |
| **P2** confirmation | optional | ~12 GPU-h | a 3rd seed where Study 1 has 2 |

**P0 and P1 need no new models.** That is the point of the design: the
expensive part of Study 1 produced artifacts that answer a better question than
the one they were collected for.

---

## Read in this order

| file | what it is |
|---|---|
| [`01_POSTMORTEM.md`](01_POSTMORTEM.md) | why Study 1 fell short — five diagnoses, each with evidence |
| [`02_PROTOCOL.md`](02_PROTOCOL.md) | the pre-registration: questions, hypotheses, gates, stopping rules |
| [`03_INVENTORY.md`](03_INVENTORY.md) | exactly what data exists and what it can answer without new runs |
| [`04_DESIGN.md`](04_DESIGN.md) | the plan, phase by phase, with costs |
| [`05_OPEN_DECISIONS.md`](05_OPEN_DECISIONS.md) | **what needs your call before anything starts** |

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

This study can also fail, in two specific ways, and both are worth naming now:

1. **If ρ_seed turns out uniformly high** (say all scores > 0.85 across all
   architectures), Half 1 collapses to "the literature was fine". That is still
   publishable as a null, but it is a much smaller paper.
2. **If some score's oracle ceiling is substantially above confidence**, Half 2
   inverts: it becomes "here is the signal worth routing on", which is a
   *better* outcome scientifically but means the negative framing is wrong.

Both are decided by P0 and P1, which cost about a day between them. Neither
requires committing to a method first. That is the design.
