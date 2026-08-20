# Study 2 — progress log

Newest first. Every entry: what changed, what it cost, what it settled, what is
next. Updated every session.

**Current state: P0 and P1 both run. The pre-registered centrepiece (H3, the
optimism bias) is NOT SUPPORTED. Two unplanned findings are stronger than the
plan was. NB2 must be re-run — its baseline was mis-specified.**

---

## 2026-08-20 · P0 + P1 complete — the centrepiece failed, something better fell out

### Scoreboard, against what was pre-registered

| | hypothesis | result |
|---|---|---|
| **H1** | ρ_seed varies by ≥ 0.15 | **SUPPORTED** — range **0.667** |
| **H3** | in-seed oracle optimistic by ≥ 0.5 pt, ≥ 6 of 8 scores | **NOT SUPPORTED** — bias **−0.200 pt**, 0 of 5 positive |
| **H4** | corr(unreliability, bias) ≥ 0.5 | **NOT SUPPORTED** — ρ = **−0.301** (p = 0.009), wrong sign |
| **H5** | nothing clears +1.0 pt honest headroom | **ambiguous** — see below, the baseline was wrong |

**The centrepiece is dead as stated.** `02_PROTOCOL.md` R3 predicted a positive
optimism bias; the measurement says it is small and **negative** — routing seed
*i*'s model with seed *j*'s score is *better* than using seed *i*'s own. And
R-03 has fired: within-pair asymmetry (0.443 pt) **exceeds** the bias (0.200 pt),
so the accuracy confound is larger than the effect. Under the register's own
rule the sign cannot be interpreted until seed accuracy is regressed out.

Per the stopping rule, this is reported as a null and not re-cut for a positive.

### Finding A — the four softmax scores are one score

P0a, the decision point, now works, and it fires at maximum:

```
                msp  margin  entropy  ce_loss   el2n  forget_events  pred_depth
msp            1.00    1.00     1.00     1.00   0.17           0.19        0.17
margin         1.00    1.00     0.99     1.00   0.19           0.21        0.18
entropy        1.00    0.99     1.00     1.00   0.16           0.17        0.16
ce_loss        1.00    1.00     1.00     1.00   0.17           0.19        0.17
el2n           0.17    0.19     0.16     0.17   1.00           0.58        0.43
forget_events  0.19    0.21     0.17     0.19   0.58           1.00        0.36
pred_depth     0.17    0.18     0.16     0.17   0.43           0.36        1.00
```

**Eight scores are three families**, not eight: `{msp, margin, entropy,
ce_loss}` at ρ ≈ 1.00, `{el2n, forget_events}` at 0.58, and `pred_depth` alone.
R-02 was escalated to HIGH on the strength of a survey reporting >70% agreement;
the measured number here is **99.7–100%**. Every "N scores" claim in this repo
must become "N families", and the effective n of the grid is 3 × 15, not 8 × 15.

### Finding B — softmax difficulty scores are unmeasurable on memorised data

This was not planned and it is the strongest thing here. ρ_seed for the same
architecture and the same score, on the test split versus `train_holdout`:

| arch | ce_loss (test) | ce_loss (train_holdout) | el2n (train_holdout) |
|---|---|---|---|
| convnext_femto | 0.709 | **0.150** | 0.684 |
| mixer_nano | 0.647 | **0.108** | 0.645 |
| vit_tiny | 0.673 | **0.116** | 0.620 |
| resnet32x4 | 0.742 | **0.284** | 0.667 |
| mobilenetv2 | 0.874 | 0.849 | 0.568 |
| resnet20 | 0.832 | 0.793 | 0.589 |

`train_holdout` is **a slice of train, not withheld from it** (`msc_lib.py`
§`_in100_loaders`) — the model has fit these samples. On data a high-capacity
network has memorised, its softmax is saturated, the per-sample ranking is
degenerate, and **seed reliability collapses to ~0.11–0.15**. Low-capacity nets
that do not saturate (mobilenetv2, resnet20) are barely affected. The
training-dynamics scores, which integrate over the whole trajectory rather than
reading the endpoint, hold steady at 0.57–0.68.

**Why this matters beyond us.** Dataset pruning, coreset selection and
curriculum learning compute exactly these scores on exactly this data — the
training set — and typically from **one seed**. This says that for the
high-capacity models those methods target, a single-seed softmax difficulty
score on training data carries almost no reproducible signal, while EL2N and
forgetting events do. That is a concrete, mechanistic, testable warning, and it
is squarely in the gap `08_RELATED_WORK.md` §4 identified.

### Finding C — the oracle ceiling is ~5 points, not ~0 — but re-measure it

`H5 FALSIFIED: best honest headroom +5.165 pt (pred_depth)`, every other score
between −0.16 and +0.23. Two defects sit under that number:

1. **`pred_depth` is not a routing signal.** `prediction_depth()` runs a kNN
   probe over the features of *every* layer and targets the network's own final
   answer. Obtaining it costs a full forward pass, so a router using it saves
   nothing. It is the textbook Oracle-EE rule (`08_RELATED_WORK.md` §1) in a
   score's clothing. Its number is a **ceiling**, never a method.
2. **The baseline was an oracle too.** It thresholded `conf[:, -1]` — the
   *final* exit's confidence — which also needs the full network. So the
   headline compared two oracles. Confidence routing has been rewritten to
   threshold the **early** exit's confidence, which is what is deployable and
   what Study 1's B2 did.

`route_by` also reached only exits 0 and K−1, a two-exit binary split, never the
intermediate ones. Both are fixed; `tools/s2_routing_canaries.py` (5/5) checks
that the budget is hit, that all K exits are reachable, that a perfect oracle
beats a random router, and that an *uninformative* confidence signal does **not**.

**Even so, the direction is interesting and contradicts Study 1.** B11 reported
**+0.00007** and concluded there was no headroom anywhere. B11 routed on MSC — a
cost-normalised aggregate — where this routes on raw per-sample sufficient
depth. If a corrected measurement keeps any large part of +5.2, Study 1's
central negative was an artifact of the metric, not a property of the problem.
That discrepancy is now the most valuable open question in the study.

### Also worth a second look

`forget_events` ρ_seed is 0.68–0.92 and almost flat across architectures, which
is suspiciously stable. Most likely a tie-mass artifact — if most samples are
never forgotten, Spearman is dominated by one easy/hard split. **Report the tie
fraction before using this number.**

---

---

## Phase board

| phase | what | cost | status |
|---|---|---|---|
| **P-1** | verify the inventory against the artifacts | minutes | **done — the inventory was wrong** |
| **P0a** | collinearity — are 8 scores really 8? | minutes | **fixed after a false pass; re-run needed** |
| **P0b** | reliability atlas | ~1 h CPU | **result obtained; re-run for the CIFAR-only cut** |
| **P1** | optimism bias (**centrepiece**) | ~1 h CPU | ready — `S2_NB2`, crash fixed |
| **P1b** | honest ceiling → the gate | ~1 h CPU | ready — `S2_NB2` |
| **P2** | confirmation runs | 12–18 GPU-h | **opt-in**, only if a gate opens |

**Next action: re-run `S2_NB1`, then `S2_NB2`.**

---

## 2026-08-19 (evening) · first run — one result, three defects

`S2_NB0_Fetch` succeeded. `S2_NB1` ran after you removed `msc` from `SCORES`
yourself. `S2_NB2` crashed. All three problems trace to the same root: **I wrote
what the data should contain instead of asking it.**

### R1 — the reliability atlas has an answer, and it is a strong one

```
grid range  0.667      (H1 threshold 0.15)          H1: SUPPORTED
```

ρ_seed spans **0.207 → 0.874**. The threshold was 0.15; the observed range is
more than four times it. The structure is not noise:

| arch | ce_loss | entropy | margin | msp | pred_depth |
|---|---|---|---|---|---|
| **mixer_nano** | 0.647 | **0.207** | 0.245 | 0.217 | 0.432 |
| **vit_tiny** | 0.673 | 0.283 | 0.310 | 0.289 | 0.464 |
| mobilenetv2 | 0.874 | 0.830 | 0.752 | 0.791 | 0.614 |
| resnet8x4 | 0.870 | 0.829 | 0.770 | 0.797 | 0.599 |

Two readings, both useful:

1. **The two non-CNNs are the least reliable on every softmax-derived score**,
   and by a wide margin — entropy on `mixer_nano` is 0.207 against 0.83 for the
   CNNs. This *replicates Study 1's ViT/Mixer finding across five scores rather
   than one*, which is much harder to dismiss as an MSC artifact.
2. **The effect is score-dependent.** `ce_loss` stays high everywhere (0.647
   even on `mixer_nano`) while `entropy` collapses; `pred_depth` is the most
   architecture-stable (range 0.308). So "difficulty is noisy" is too coarse —
   *which* difficulty, on *which* architecture.

**Provisional.** That grid mixed the 2 ImageNet architectures into the CIFAR
result, against decision D3, and used 5 scores because two silently vanished.
The direction will not change; the exact range must be recomputed.

### Defect 1 — `S2_NB2` crashed: budgets asked the wrong zoo

```
ValueError: 'convnext_femto' belongs to the 'cifar' zoo but dataset
'imagenet100' needs the 'imagenet' zoo.
```

`S2_NB1`/`S2_NB2` build their Session from `paths_cell(phase="p1")`, reused from
the ImageNet-100 generator, which hardcodes `dataset='imagenet100'`. The corpus
is **mixed** — 15 CIFAR architectures and 2 ImageNet ones — so a budget belongs
to the **run**, not to the session. `rho_for(arch)` is now `rho_for(arch,
dataset)`, keyed on the pair, and every grouping carries `dataset` through.

This is rule 4 charging interest: one accessor was reused across two studies
that do not share a dataset.

### Defect 2 — P0a was a false pass, and that is the serious one

```
|Spearman| between scores, run p0-resnet32x4-...  (n=0)
[all-NaN 7x7 matrix]
no pair exceeds |rho| = 0.9 -- the scores carry distinct information
```

It printed **n=0**, an all-NaN matrix, and then a reassuring conclusion. The mask
`ok = ~np.isnan(X).any(axis=1)` is *listwise*: one all-NaN column drops every
row. `el2n` and `forget_events` are all-NaN on the test split, so nothing
survived — and the cell reported success from zero samples.

This is the D-37 shape, the exact failure `06_RISK_REGISTER.md` §R-06 says every
new statistic must be canaried against, and I shipped it in the cell the risk
register calls the study's first decision point.

Now: pairwise-complete correlation, per-score coverage printed first, and an
explicit **refusal** when there is nothing to conclude from.
`tools/s2_canaries.py` proves all three checks can fail — canary A caught a
second bug in my own fix (a `KeyError` instead of a clean refusal when only one
score survives).

### Defect 3 — two scores vanished without a word

The atlas reported 5 scores. It should have said it was dropping 2. Skipped
`(split, arch, score)` cells are now counted and listed.

### What the data actually contains — `03_INVENTORY.md` was wrong

| score | `test.parquet` | `train_holdout.parquet` |
|---|---|---|
| `msp`, `margin`, `entropy`, `ce_loss`, `pred_depth` | populated | populated |
| `el2n`, `forget_events` | column exists, **all NaN** | **populated** |
| `msc` | **absent** | **absent** |

I sourced "eight scores" from a summary CSV instead of a parquet. You hit it
first and fixed it by hand.

**But the recovery is better than the original plan.** `el2n` and
`forget_events` are not lost — they are real on `train_holdout`, so the atlas now
runs on **both splits**: 5 scores on `test`, **7** on `train_holdout`. And the
constraint is itself a finding worth stating in the paper: *two of the eight
candidate difficulty signals cannot route unseen samples at all*, because a test
sample has no training history. That is a limit on the whole method family, not
a limitation of ours.

### Verification, given that I cannot read a parquet here

My sandbox has no `pyarrow` and no `scipy`, so parquet code has always shipped
unexecuted — the structural gap behind D-70, D-76, D-77 and D-79c. What I did
instead:

- read the parquet **footer bytes directly** to confirm `msc` is absent and the
  other seven are present in all three files;
- `tools/s2_cell_harness.py` — executes `S2_NB1`'s **real cells** against
  synthetic frames reproducing the on-disk shape (all-NaN train scores on test);
- `tools/s2_canaries.py` — 3/3 pass, and canary A found a bug in the fix.

---

## 2026-08-19 (later) · literature checked, decisions made

**I had web search the whole time and asked the user to do it instead.** Wrong
call; corrected. Four searches, ~30 minutes, written up in `08_RELATED_WORK.md`.

**What it settled.** Every ingredient of the idea exists and is citable:

- oracle early-exit bounds are **standard practice and same-model by
  construction** — so the target is real, not a straw man;
- "oracle selection over noisy estimates is positively biased" is an
  **established statistical principle**;
- noise ceilings and disattenuation are **routine in computational
  neuroscience** (Spearman-Brown, noise-ceiling-corrected Spearman) and absent
  from the difficulty literature — a clean, citable framing;
- EL2N's run-to-run instability is **already known qualitatively**;
- cross-architecture difficulty agreement is **published uncorrected**
  (Baldock et al.).

**Not found:** a second training seed used to debias a per-sample routing
oracle, the bias quantified per score and architecture, or the bias tied to
measured reliability. The novelty is the **combination and the measurement**.

**Two consequences, one of them bad.**

1. **Reliability can no longer lead.** Seed noise is known; the optimism bias
   leads and reliability becomes the instrument. D1 and D3 changed.
2. **R-02 escalated to HIGH.** One survey reports difficulty scoring functions
   agree **>70% with each other in all but one case**. Our eight scores may be
   three or four families. **P0a is now a decision point, not a sanity check.**

**Decisions settled** (all five, in `05_OPEN_DECISIONS.md`): optimism-bias
framing · MSC as one of eight · **CIFAR-only main result** (ImageNet's 2 seeds
cannot support a cross-seed estimate — one paragraph, not an appendix) ·
keep +1.0 with the noise floor always reported · keep Study 1 and cite it.

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
