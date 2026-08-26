# Study 3 — plan

**Nothing here has been run.** This is a design, written before any compute is
spent, so the decision to start is made on paper.

Study 2 (`study2/PAPER.md`) is complete. It ends with one blocking limitation
and one unanswered question, and this folder is about both.

---

## Where Study 2 left off

Study 2's core claim is an exact identity:

> The oracle early-exit bound sits **+6.86 accuracy points above the network's
> own full-compute accuracy**, in 100 % of runs, and that excess is exactly the
> fraction of samples some early exit gets right while the final layer gets them
> wrong.

The target is real and named — a survey defines the oracle as "the shallowest
internal classifier that provides a correct label prediction", and DE3-BERT
reads the oracle-above-backbone gap as *"significant room for improving the
estimation of prediction correctness"*. That reading is what Study 2 challenges.

**But Study 1's exit heads were trained with the backbone frozen.** MSDNet,
BranchyNet and DE3-BERT train exits *jointly*. Weaker exits plausibly enlarge
the early-right/final-wrong pool, so +6.86 pt may be an overestimate. Until that
is settled the result is a workshop paper, not an archival one.

There is also a claim Study 2 makes too strongly. It says the excess "cannot be
reached by any router". What was actually shown is that **a second seed cannot
reach it**. A *learned* router with access to the input might do better. That is
an empirical question and nobody has measured it.

---

## The three questions

| | question | why | cost |
|---|---|---|---|
| **Q1** | Does the +6.86 pt survive **jointly trained** exits? | the blocker — decides workshop vs archival | **~10 GPU-h** |
| **Q2** | How much of the gap can a **learned router** actually capture? | turns a bound into a constructive result | ~5 GPU-h |
| **Q3** | Does the memorisation collapse **damage downstream pruning**? | a second, independent paper | ~18 GPU-h |

**Q1 needs no seeds.** Finding A is a per-run identity — `oracle_in − acc_full ==
frac_early_saves` — computed from one network's own per-exit predictions. Seeds
were needed for the *bias*, not for this. So the blocking experiment is
**3 architectures × 1 seed**, not 3 × 3. That is the single most useful thing
this plan discovered.

---

## Staging, cheapest decisive check first

```
P0  extrapolate from existing data      ~2 CPU-h    free   ── gate ──▶
P1  Q1: joint-exit training             ~10 GPU-h          ── gate ──▶
P2  Q2: learned router                  ~5 GPU-h
P3  Q3: pruning demonstration           ~18 GPU-h   (independent of P1/P2)
```

**Worst case ≈ 35 GPU-h, about two days.** Study 1 spent 215.

### P0 — the free check that might settle Q1 without training

Across the 15 existing architectures, exit-head quality already varies a lot.
If the early-saves pool **shrinks as the early exits get stronger**, then better
(jointly-trained) exits would shrink it further, and we can estimate by how much
before spending a GPU-hour. If the pool is **insensitive** to exit quality, joint
training is unlikely to remove it and Q1 is largely pre-answered.

Either way P0 costs a couple of CPU-hours on data already on disk, and it
converts Q1 from "unknown" into "expected value, with an interval".

**It is a gate, not evidence.** It is observational and cross-architectural, so
it cannot replace P1 — it only decides how much P1 is worth.

### P1 — the blocker

Three architectures trained with **deep supervision**: exits trained jointly
with the backbone, standard multi-exit loss, everything else held identical to
Study 1 so the comparison is paired.

The pairing is the whole design. Same architectures, same data, same budgets,
**same measurement code**, one variable changed: frozen vs joint exits.

### P2 — from a bound to a method

Study 2 says a second seed cannot reach the gap. It does not say a learned
router cannot. P2 dumps per-exit features (~50 MB per run, one forward pass),
trains a small gate that sees only what is available *at* each exit, and
measures what fraction of the +6.86 pt it captures.

Three outcomes, all publishable:

| a learned router captures | reading |
|---|---|
| most of it | the field is right, the gap is real headroom, and here is a router |
| a little | the bound is mostly noise, quantified — Study 2 strengthened |
| none | the bound is entirely noise; the strongest version of Study 2's claim |

### P3 — the second paper

Study 2 §4.3: softmax difficulty scores lose seed reliability on data the model
has memorised (ρ_seed 0.647 → 0.108), predicted by saturation (+0.832) and not by
test accuracy (−0.114). That is a warning to dataset pruning, which computes
exactly these scores, on exactly this data, from a single seed.

**A warning is not a demonstration.** P3 prunes with a score from a saturated
model and from a non-saturated one, retrains a fixed target, and measures the
accuracy difference. Independent of P1 and P2 — it can run first if preferred.

---

## Files

| file | what it is |
|---|---|
| [`01_PROTOCOL.md`](01_PROTOCOL.md) | pre-registration — predictions and thresholds fixed before any run |
| [`02_RISKS.md`](02_RISKS.md) | how this fails, with detectors and responses |
| [`03_LOG.md`](03_LOG.md) | **live log** — status board, pre-registered predictions, what to do next |

---

## What Study 3 inherits from Study 2's mistakes

Study 2's routing measurement was implemented **wrong three times**, each
producing a plausible number. The rules that came out of it apply here from the
first line of code:

1. **Every statistic gets a canary that must be able to fail** — including one
   that requires it to *detect* an effect in a world where the effect certainly
   exists. "No effect" and "cannot see the effect" produce identical output.
2. **Check the artifact, not the plan.** Study 2 found same-seed replicates
   pooled as distinct seeds, and a correlation matrix computed on a run that had
   been dropped, only by reading the files.
3. **Compare like with like.** Two of the three wrong implementations were
   mechanism mismatches, not arithmetic errors.
4. **Scrutinise favourable results hardest.** The +22.41 pt was exactly what the
   study predicted, and attacking it found that the mechanism story was wrong
   (ρ = +0.011) and had to be decomposed.
5. **Do the cheap decisive check first.** P0 exists because of this.
