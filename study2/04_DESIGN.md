# Study 2 — the plan

> **SUPERSEDED by the results. Kept as the plan that was actually followed, so
> the plan and the outcome can be compared.** Three things below are now known
> to be wrong, all corrected in [`PAPER.md`](PAPER.md):
>
> 1. **"8 scores" is really 3 families.** `msp`/`margin`/`entropy`/`ce_loss`
>    correlate at ρ = 0.997–1.000. Read every "× 8 scores" as "× 3 families";
>    the grid's effective n is 3 × 15.
> 2. **Only 5 scores exist on the test split.** `msc` is never persisted, and
>    `el2n`/`forget_events` are NaN for samples with no training history.
> 3. **P1 needed no GPU.** The "~6 GPU-h" below was wrong: per-exit predictions
>    were already in the parquets, so the whole study ran on CPU.

Three phases, ordered so the cheap measurements decide whether the expensive
ones happen. This is the inversion of Study 1
(`01_POSTMORTEM.md` §"The through-line").

```
P0  reliability atlas     ~2 CPU-h    no new models   ── gate ──▶
P1  routing ceiling       ~6 GPU-h    no new models   ── gate ──▶
P2  confirmation          ~12 GPU-h   only if a gate opens
```

---

## P0 — the reliability atlas

**Cost: ~2 CPU-hours. No GPU. No training.**

For all 15 CIFAR architectures × 8 scores, from `per_sample/test.parquet`:

1. **Score × score correlation, within a run.** *Run this first.* If the eight
   scores are near-collinear, R1–R4 are statements about one quantity and the
   study needs rethinking before anything else happens
   (`02_PROTOCOL.md` "What would make this study wrong").
2. **ρ_seed(score, arch)** — Spearman between seed pairs, all 3 pairs per
   architecture, mean and spread.
3. **The grid**: 15 architectures × 8 scores, with CNN / non-CNN marked.
4. **Cross-architecture agreement**, raw and disattenuated, per score.

**Answers R1, R2, and the collinearity check.**

**Deliverables**
```
analysis/s2_reliability_grid.csv      arch × score × rho_seed, sd, n_pairs
analysis/s2_score_correlations.csv    score × score, per architecture
analysis/s2_disattenuation.csv        raw vs corrected, and rank changes
```

**Gate.** If ρ_seed is flat across the grid (range < 0.15), H1 is falsified;
Half 1 is reported as a null and does not get re-cut looking for a positive.

---

## P1 — the routing ceiling

**Cost: ~6 GPU-hours. No training** — one forward pass per run over saved
checkpoints.

For each measured run and each of the eight scores: route each sample to the
cheapest exit using its **true** score value, and measure accuracy at matched
average FLOPs against `msp` thresholding.

This is `evaluate_routing_methods` with `oracle_from_self` generalised from MSC
to an arbitrary column — the machinery already exists and already computes
matched-FLOPs comparisons and operating-point curves.

**Answers R3 (the gate) and R4.**

**Deliverables**
```
analysis/s2_routing_ceiling.csv       run × score × oracle acc, msp acc, delta, avg_rho
analysis/s2_ceiling_vs_reliability.csv joins P0 and P1 for R4
paper/figures/s2_fig1_ceiling.png     oracle headroom per score, with the noise floor drawn
```

**Report the noise floor on the same axes.** Study 1 produced
`fraction_of_gap_closed = 26.0` by dividing by a gap of 0.00007 (D-80). Any
headroom below 2 SE is reported as *"within noise"*, never as a ratio.

**Gate.**

| outcome | consequence |
|---|---|
| no score exceeds `msp` by > 1.0 pt | **P2 is not run.** The ceiling is the paper. |
| some score exceeds it | that score becomes the subject; amend the protocol in writing; P2 tests it |

---

## P2 — confirmation (conditional)

**Cost: ~12 GPU-hours. Only if a gate opens, or to close a stated limitation.**

Three variants, and which one runs depends on what P0/P1 found:

**P2a — the third ImageNet seed.** 2 runs, ~12 GPU-h. Turns ImageNet ρ_seed
from a point estimate into an interval. Needed only if ImageNet appears in a
headline claim.

**P2b — a shared architecture.** `shufflenetv2` and `convnext` exist in both
zoos. Running one on ImageNet-100 (3 seeds, ~18 GPU-h with the `channels_last`
fix) is the **only** way to make any cross-scale statement — see
`01_POSTMORTEM.md` §4. Run this only if the paper needs a scale claim; the
Study 2 thesis does not.

**P2c — the method**, if and only if P1 found real headroom. Scope it to the
one score that cleared the gate, at the operating point where it cleared.

**Default: none of them.** P2 is opt-in, per stopping rule 2.

---

## What gets built

Minimal. Two generalisations and one notebook.

| change | where | size |
|---|---|---|
| `analyse_reliability_grid(session, scores, phase)` | `msc_lib` | ~40 lines, generalises `analyse_q1_all` from `msc` to any column |
| `oracle_from_column` in `evaluate_routing_methods` | `msc_lib` | ~10 lines, generalises the existing `oracle_from_self` |
| `S2_NB1_Reliability.ipynb` | `notebooks_study2/` | P0 |
| `S2_NB2_Ceiling.ipynb` | `notebooks_study2/` | P1 |

Everything else is reused: the runner, resume, hashing, the six validation
layers, the disattenuation, the shuffled-target control machinery.

**The notebook generator, the validators and the self-test carry over
unchanged.** They are the part of Study 1 that unambiguously worked, and they
now encode 50 defects' worth of hard-won rules.

---

## Order of operations

1. Verify the inventory (`03_INVENTORY.md` — run the snippet, do not trust the file).
2. P0 collinearity check → decide the study is still coherent.
3. P0 full grid → R1, R2.
4. **Gate.**
5. P1 ceiling → R3, R4.
6. **Gate.**
7. Write. P2 only if a gate opened.

Steps 1–6 are about **one day of wall clock**. Study 1 reached its equivalent
decision point after roughly three weeks.

---

## Why this is a better bet than fixing Study 1

Fixing Study 1 means running the p1 atlas — 8 architectures × 3 seeds,
~334 GPU-h, ~14 days — to strengthen a claim about MSC specifically, when the
B11 result already suggests the method built on it cannot work.

Study 2 costs about a day to reach a decision point, produces a result whether
the answer is positive or negative, and does not depend on MSC being special.

**The 45 CIFAR runs and 22 ImageNet runs are not wasted.** They are the input.
