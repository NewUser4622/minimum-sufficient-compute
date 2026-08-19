# Study 2 — progress log

Newest first. Every entry: what changed, what it cost, what it settled, what is
next. Updated every session.

**Current state: notebooks built, nothing run.**

---

## Phase board

| phase | what | cost | status |
|---|---|---|---|
| **P-1** | verify the inventory against the artifacts | minutes | ready — `S2_NB1` cell 3 |
| **P0a** | collinearity — are 8 scores really 8? | minutes | ready — `S2_NB1` |
| **P0b** | reliability atlas, 8 scores × 15 archs | ~1 h CPU | ready — `S2_NB1` |
| **P1** | optimism bias (**centrepiece**) | ~1 h CPU | ready — `S2_NB2` |
| **P1b** | honest ceiling → the gate | ~1 h CPU | ready — `S2_NB2` |
| **P2** | confirmation runs | 12–18 GPU-h | **opt-in**, only if a gate opens |

**Blocking:** nothing technical. Two things need you —
`05_OPEN_DECISIONS.md` (five calls) and `06_RISK_REGISTER.md` §R-07 (30 minutes
of literature search, the one risk I cannot retire).

---

## 2026-08-19 · design v2 + notebooks built

**Found a hole in my own v1 design.** Both halves could have landed as nulls:
R1 might come back flat, and R3 *predicted* a null by construction. That is a
bad bet and it was my error.

**Also wrong in v1:** I assumed measurement noise caps routing gain. It does
not — Study 1's B11 used the model's own **true** post-hoc MSC, so it was never
noise-limited, and it still did not beat confidence.

**New centrepiece: the optimism bias.** An oracle ceiling computed from the same
seed it routes partly routes on that model's own noise. With ≥2 seeds you can
separate it:

```
in-seed   oracle : score from seed i  -> routes seed i   (optimistic)
cross-seed oracle: score from seed j  -> routes seed i   (honest)
bias = in-seed - cross-seed
```

**Both outcomes are publishable.** Large bias → the field's oracle bounds are
inflated, with a correction. Zero bias → validates a universal practice nobody
had checked. That property is why v2 exists.

### The discovery that changed the cost

Every measured run's `per_sample/test.parquet` already contains
`pred_d1..dK`, `top1p_d*`, `top2p_d*` and `label` — **per-exit predictions for
every sample.**

Per-exit correctness is `pred_dk == label`. Confidence routing is a threshold on
`top1p_dk`. Oracle routing on any score is a sort. Cost comes from
`budgets/{arch}.json`.

> **Study 2 needs no GPU and loads no model. It is CPU re-analysis of parquet
> files.** P0 and P1 are hours, not days.

That removes most of Study 1's failure surface at a stroke: no training, no
resume, no throughput, no `channels_last`, no CUDA asserts, no 79-hour
commitments before a cheap check.

### Built

| | |
|---|---|
| `build_notebooks_study2.py` | generator, reusing the Study 1 bootstrap and every validation layer |
| `S2_NB0_Fetch` | pull CIFAR-100 runs from HF — **checkpoints excluded** (~95% of bytes, never opened) |
| `S2_NB1_Reliability` | P-1 verify · P0a collinearity · P0b atlas · R1 verdict |
| `S2_NB2_Ceiling` | P1 bias both directions · R3 · R5 gate · R4 mechanism |

All three pass: undefined-name check across cells, Python-3.10 parse gate.

### Written

`02_PROTOCOL.md` rewritten to v2 (R1–R5, thresholds justified, stopping rules
fixed before results). `06_RISK_REGISTER.md` added — seven risks, each with a
detector, a trigger and a response.

### Next

1. **You:** the five decisions in `05_OPEN_DECISIONS.md`.
2. **You:** 30 minutes on R-07 — has someone already done cross-seed oracle
   debiasing? This is the only risk that decides novelty and I cannot check it
   from here.
3. **Then:** `python build_notebooks_study2.py`, run `S2_NB0_Fetch`, then
   `S2_NB1` as far as the P0a collinearity output — and stop there. If the eight
   scores are near-collinear the framing changes and nothing further should be
   written first.

---

## 2026-08-18 · Study 2 proposed

`study2/` created after Study 1 finished: README, postmortem, protocol v1,
inventory, design, open decisions.

Five diagnoses of Study 1, with one through-line — **the expensive thing was
always done before the cheap thing that would have said whether to do it.**
79 GPU-h of MSC-KD students before a 2-hour oracle measurement showing no gap;
45 backbone runs before checking the two zoos shared an architecture (they share
none, so the headline question was unanswerable by construction).

---

## How to read this file

Each entry states what was **settled**, not what was attempted. A phase is
`done` only when its artifact exists on disk and has been opened and checked —
Study 1's D-79 was 18 runs that all reported success while the number they
existed to produce was never computed.
