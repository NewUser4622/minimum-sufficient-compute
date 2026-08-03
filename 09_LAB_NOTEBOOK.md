# Lab Notebook

**Running record of every result, every defect, and every decision that changed.**

Append-only. Newest entries at the top of each log. The purpose is that when the
paper is written, three questions can be answered from one file:

1. **What did we measure, and exactly where does that number live?**
2. **Was any reported number produced by code that later turned out to be wrong?**
3. **What changed in the design after the plan was frozen, and why?**

§1 is the results ledger. §2 is the defect log with a contamination analysis for
each. §3 is decisions changed. §4 maps all of it onto paper sections.

---

## 0. Status board

| | |
|---|---|
| **Phase** | 0 complete · Phase 1 **NB04 14/15** (one run at epoch 79/240) |
| **Verdict** | `FULL-PROGRAM` (2026-08-02) |
| **Runs completed** | Phase 0: 4/4 trained, 4/4 measured · NB04: **14/15** trained, 0/15 measured |
| **GPU-hours spent** | ~42 (9.5 Phase 0 + ~32 atlas, incl. 2.9 wasted to D-12) |
| **GPU-hours remaining** | ~120 |
| **Library version** | `msc_lib` 1.0.0 · 143 offline self-checks |
| **Defects found** | 12 · all fixed · 1 affects a reported number (D-11) · **1 cost GPU-time (D-12)** |
| **Artifacts** | `huggingface.co/datasets/Shanmuk4622/msc-cifar100` |

---

## 1. Results ledger

Every number we have measured, with its provenance. `source` is the file in the
HF repo; `code` is the library version that produced it.

### 1.1 Phase 0 backbone training

Standard CRD/DKD recipe, CIFAR-100, 240 epochs, SGD 0.05, ×0.1 at 150/180/210,
batch 64, wd 5e-4, random crop + horizontal flip. AMP on. Kaggle T4.

| run_id | Top-1 | Top-5 | Published | Δ | GPU-h | kWh | epochs |
|---|---|---|---|---|---|---|---|
| `p0-resnet32x4-cifar100-base-s1` | 79.59 | 94.28 | 79.42 | **+0.17** | 2.89 | 0.216 | 240/240 |
| `p0-resnet32x4-cifar100-base-s2` | 79.63 | — | 79.42 | **+0.21** | 2.89 | 0.216 | 240/240 |
| `p0-wrn_40_2-cifar100-base-s1` | 76.89 | 93.81 | 75.61 | **+1.28** | 1.88 | 0.140 | 240/240 |
| `p0-wrn_40_2-cifar100-base-s2` | 76.72 | — | 75.61 | **+1.11** | 1.88 | 0.140 | 240/240 |

*source* `runs/{run_id}/summary.json` · *code* `msc_lib` 1.0.0 · *reference* DKD paper / mdistiller

**All four exceed published values.** This is the acceptance test for the whole
pipeline: MSC computed from an under-trained model is meaningless, and an
under-trained model is otherwise easy to miss. `recipe_ok: true` on all four.

Seed-pair spread: resnet32x4 0.04 pts, wrn-40-2 0.17 pts.

`sample_order_hash = 80031c23f8300724…` identical across all four → per-sample
tables are index-aligned and may be correlated.

### 1.1b Phase 1 atlas — ResNet family (NB04)

Same recipe. Four Kaggle accounts, workers 0–3, cost-balanced split.
**14 of 15 complete.** All completed runs beat their published references.

| arch | s1 | s2 | s3 | mean | published | Δ |
|---|---|---|---|---|---|---|
| `resnet20` | 70.25 | 70.36 | 69.78 | **70.13** | 69.06 | **+1.07** |
| `resnet56` | 73.88 | 73.35 | 73.85 | **73.69** | 72.34 | **+1.35** |
| `resnet110` | 74.31 | 74.57 | 74.26 | **74.38** | 74.31 | **+0.07** |
| `resnet8x4` | 73.35 | 73.39 | 73.04 | **73.26** | 72.50 | **+0.76** |
| `resnet32x4` | 79.72 | 80.03 | *incomplete* | **79.88**\* | 79.42 | **+0.46** |

\* two seeds only. `p1-resnet32x4-cifar100-base-s3` stopped at **epoch 79/240**
(best 64.47% so far), resumable from its checkpoint. See D-12.

Seed spread: resnet20 0.58 pts, resnet56 0.53, resnet110 0.31, resnet8x4 0.35.
All four share `sample_order_hash = 80031c23…`, matching Phase 0 — the atlas and
pilot tables are mutually correlatable.

Wall-clock: 11:51 → 20:31 (~8.7 h) across four accounts for ~32 GPU-hours of
work, of which 2.9 h were wasted re-training a run another account had already
done (D-12).

*source* `runs/p1-*/summary.json`, `registry/events/*.jsonl`

---

### 1.2 Q1 — noise ceiling (ρ_seed)

Spearman between MSC of seed 1 and seed 2, same architecture, depth axis,
irreducible samples masked.

| τ | resnet32x4 | wrn-40-2 | J₁₀ (r32x4) | J₁₀ (wrn) | n (r32x4) |
|---|---|---|---|---|---|
| 0.0 | 0.6844 | 0.6604 | 0.4212 | 0.6221 | 10000 |
| **0.1** | **0.7172** | **0.7147** | 0.4599 | 0.6425 | 8549 |
| 0.2 | 0.7248 | 0.7241 | 0.4783 | 0.6553 | 7766 |
| 0.3 | 0.7217 | 0.7154 | 0.4894 | 0.6633 | 7194 |
| 0.5 | 0.6962 | 0.6900 | 0.5044 | 0.6789 | 6267 |

*source* `analysis/q1_seed_ceiling.csv` · gate ≥ 0.60 · **pass at every τ**

Non-monotone in τ, peaking near 0.2. At τ=0 the margin requirement is vacuous so
low-confidence noise enters; at τ=0.5 a quarter of the sample is excluded.

### 1.3 Q2 — axis structure (PC1)

PCA over per-sample MSC on {depth, res_proxy, precision}, resnet32x4 seed 1.

| τ | PC1 | PC2 | PC3 | depth↔res | depth↔prec | res↔prec | n |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.5155 | 0.2739 | 0.2106 | 0.381 | 0.278 | 0.278 | 10000 |
| **0.1** | **0.5032** | 0.2860 | 0.2108 | 0.377 | 0.232 | 0.240 | 9041 |
| 0.2 | 0.4940 | 0.2938 | 0.2122 | 0.374 | 0.202 | 0.210 | 8548 |
| 0.3 | 0.4955 | 0.2911 | 0.2135 | 0.370 | 0.204 | 0.211 | 8192 |
| 0.5 | 0.4984 | 0.2894 | 0.2123 | 0.370 | 0.201 | 0.199 | 7419 |

PC1 loadings at τ=0.1: depth 0.616, resolution 0.633, precision 0.469.

*source* `analysis/q2_axis_structure_phase0.csv` · H2 predicted ≥ 0.60 · **REFUTED**

### 1.4 Q3 — transfer (T)

resnet32x4-s1 → wrn_40_2-s1, depth axis, 1000 bootstrap resamples.

| τ | ρ_S raw | **T** | 95% CI | J₁₀ | n |
|---|---|---|---|---|---|
| 0.0 | 0.6441 | 0.8997 | [0.882, 0.921] | 0.430 | 10000 |
| **0.1** | 0.6772 | **0.9459** | [0.927, 0.966] | 0.440 | 8549 |
| 0.2 | 0.6675 | 0.9324 | [0.913, 0.952] | 0.456 | 7766 |
| 0.3 | 0.6499 | 0.9078 | [0.887, 0.927] | 0.465 | 7194 |
| 0.5 | 0.5966 | 0.8333 | [0.813, 0.854] | 0.474 | 6267 |

*source* `analysis/q3_transfer.csv` · gate ≥ 0.70 · **pass at every τ** · H3 predicted within-family > 0.8

### 1.5 Q4 — irreducibility (ΔR²)

⚠ **See D-11.** These ran on the `test` split with **5 of 7** battery scores.

| τ | R² battery | R² +MSC | **ΔR²** | 95% CI | partial ρ |
|---|---|---|---|---|---|
| 0.0 | 0.288 | 0.485 | 0.197 | [0.180, 0.214] | 0.434 |
| **0.1** | 0.321 | 0.575 | **0.254** | [0.234, 0.273] | 0.489 |
| 0.2 | 0.332 | 0.608 | 0.276 | [0.255, 0.297] | 0.491 |
| 0.3 | 0.338 | 0.620 | 0.282 | [0.260, 0.303] | 0.485 |
| 0.5 | 0.331 | 0.622 | 0.291 | [0.267, 0.314] | 0.447 |

Battery used: `msp, margin, entropy, ce_loss, pred_depth`.
Absent: `el2n, forget_events` — training-set scores, undefined on the test split.

*source* `analysis/q4_irreducibility.csv` · gate ≥ 0.05 · **pass**, but **recompute on `train_holdout` before publication**

### 1.6 Shuffled control

| | value |
|---|---|
| T (shuffled) | **0.0072** |
| ρ_S raw (shuffled) | 0.0052 |
| expected | ≈ 0 |

Confirms the per-sample tables are genuinely row-aligned and every correlation
above is computed on paired images rather than a coincidental index match.

### 1.7 Irreducible subpopulation |U_τ| / N

| τ | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| resnet32x4 | 0.0% | 8.6% | 13.7% | 17.6% | 25.5% |
| wrn-40-2 | 0.0% | 8.1% | 14.1% | 18.9% | 27.5% |

Excluded from all correlations. The close agreement in *size* is suggestive but
not evidence about *membership*; testable in the atlas.

### 1.8 Measured cost model

Used to calibrate the scheduler. See D-10.

| arch | s/epoch | 240 ep | cost unit |
|---|---|---|---|
| resnet32x4 | 43.29 | 2.89 h | 5.2 (anchor) |
| wrn_40_2 | 28.16 | 1.88 h | 3.38 |

`SECONDS_PER_COST_UNIT = 8.32`

---

## 2. Defect log

Every bug found, with a **contamination analysis** — the question a reviewer
would ask, and the one we need to have answered before writing.

### D-12 · Work assignment drifted between sessions — one run abandoned, one duplicated
**Found** auditing NB04 on HF: `p1-resnet32x4-cifar100-base-s3` stopped at epoch
79/240 with no completion event, while `p1-resnet32x4-cifar100-base-s1` was
trained **twice** — by acct2 (79.72%) and again by acct4 (79.54%).
**Cause** *My own feature, working exactly as written and being wrong anyway.*
`Session.plan()` called `estimate_costs_from_history()` and merged measured
per-epoch times into the cost table **used for the assignment**. The whole
sharding guarantee is "identical code + identical input → identical ownership,
with no communication". Measured timings are not identical input: they change as
runs finish. acct4's first session (11:51, nothing finished) computed a different
packing than its second (18:06, twelve runs finished), so it stopped owning
resnet32x4-s3 and started owning acct2's resnet32x4-s1.
**Cost** 2.9 GPU-hours wasted, one run left at epoch 79, and the "no overlap, no
gaps" property silently violated. A self-test now shows **4 of 15 runs move**
when the cost table drifts.
**Fix** ownership uses `ARCH_COST_HINT` **only**, always. Measured timings still
refine *displayed* estimates via `estimate_phase()`, which has no effect on who
owns what. `assign_workers` documents that `costs` must be a stable table.
Self-test asserts a worker's slice is identical before and after 12 runs finish,
and that all slices still partition the universe.
**Contamination** none — every *completed* run is valid and trained under the
correct config. Two independent trainings of resnet32x4-s1 agreeing to 0.18 pts
is incidental evidence that training is reproducible. Only wasted time and one
unfinished run.
**Lesson** an optimisation that improves an *estimate* must never be allowed to
change a *decision* that had to be deterministic. Feeding measurement back into
allocation broke the one property the design existed to provide.

---

### D-11 · Q4 ran with 5 of 7 difficulty scores
**Found** reading the Phase 0 output (`battery` column listed five names).
**Cause** EL2N and forgetting events are *training-set* quantities indexed by
training images. The test split's `sample_idx` refers to entirely different
images, so attaching them there would be meaningless — the code correctly wrote
NaN, but `analyse_q4_irreducibility` defaulted to the test split anyway.
**Fix** default changed to `train_holdout` (5,000 training images, augmentation
off, carries all seven). Test split retained as a robustness check.
`n_battery_scores` now recorded in the output.
**⚠ CONTAMINATION — YES, one reported number.** ΔR² = 0.254 is a *test-split,
five-score* result. A smaller battery is an **easier** test for MSC, so this
figure **overstates** irreducibility. The margin over the 0.05 gate is large
enough that the Phase 0 *decision* is unaffected, but §1.5 must be recomputed on
`train_holdout` before the number appears in the paper.
**Action** rerun NB03 Step 7 after re-uploading. Tracked in §5.

---

### D-10 · Cost estimates ~40% low
**Found** comparing predicted runtimes against Phase 0 actuals.
**Cause** `SECONDS_PER_COST_UNIT` was guessed at 5.0; measured 8.32. The
resnet32x4/wrn ratio was also wrong (1.18 hinted vs 1.54 measured).
**Fix** anchored both scale and ratio on the two measured architectures.
Predictions now match actuals to within 0.01 h. `MEASURED_ARCHS` records which
entries are real.
**Contamination** none — scheduling estimates only, no scientific quantity.

### D-09 · NB02 planned zero work and exited in 30 s
**Found** user reported the notebook finishing suspiciously fast; NB03 then
reported no per-sample tables.
**Cause** *Modelling error, not a typo.* A run passes through several stages
(train → measure → method) but the ledger carries **one** `state` per run.
`plan_work` treated `state == "completed"` as done — but that was set by
*training*. The measurement stage therefore saw all four runs as finished.
**Blast radius** would have done the same to the 45-run atlas in NB08.
**Fix** `plan_work(done_fn=, stage=)`; `Session.measured()` decides measurement
completion from **artifact existence**, not the ledger. Zero planned work is now
an `[ALARM]` when the worker owns unfinished runs. NB02/NB08 print per-run
measured status before and after.
**Contamination** none — nothing ran, so nothing was produced. All Phase 0
measurements were produced after the fix.
**Lesson** the worst failure shape is no error, plausible output, and a
consequence that surfaces two notebooks later.

### D-08 · Analysis raised `FileNotFoundError` six frames deep
**Found** NB03 run before NB02 completed.
**Cause** no prerequisite check; a missing input surfaced from inside a
statistic.
**Fix** `MissingInputs` exception type, `check_inputs()` per-run table,
`require_inputs()` hard stop naming the notebook to run next. `load_per_sample`
distinguishes *never trained* from *trained but not measured*.
**Contamination** none — crash, not corruption.

### D-07 · Twelve analysis cells assumed a non-empty DataFrame
**Found** user hit `KeyError: 'split'` in NB02; swept for the same pattern.
**Cause** `pd.DataFrame([])` has no columns; `.groupby('split')` fails.
**Fix** guarded all thirteen sites; each now explains what is missing. Added an
offline harness that executes every code cell against a stubbed empty project
and asserts no `KeyError`/`IndexError` — it caught an NB12 case the manual pass
missed, **and** a `\n` escaping bug I introduced while fixing NB02.
**Contamination** none.

### D-06 · A test that validated nothing
**Found** the resume acceptance test "failed" — but had never exercised resume.
**Cause** it simulated a kill by training a *shorter run*, which is a clean
completion followed by an extension. That path never touches the interrupt
handler, emergency flush, or resume logic. It was also correctly blocked by the
claim protocol, which refuses to restart a completed run.
**Fix** a debug hook fires a real `KeyboardInterrupt` mid-run (excluded from
`config_hash`), then resumes with an identical config. The test now compares
**per-epoch training loss after the seam**, which is where a lost RNG state
shows up — final accuracy can match by luck, the loss curve cannot.
**Contamination** none, but this is the most alarming entry in the log: for a
period we believed resume was verified when it was not.
**Lesson** a test that cannot fail for the right reason manufactures confidence.

### D-05 · Claim protocol blocked self-resume
**Found** the corrected resume test.
**Cause** the 2-hour staleness window was applied without checking *ownership*.
A session that paused at the 8.5 h limit could not be resumed by its own account
for two hours — defeating the resumability contract at exactly the moment it
exists for.
**Fix** ownership checked before freshness. Same account → always allowed, with
a note. Other accounts → unchanged.
**Contamination** none — would have blocked work, not corrupted it.

### D-04 · Shared ledger silently lost writes
**Found** auditing the live repo: two runs training, ledger listing one.
**Cause** HuggingFace has no append operation. Every worker rewrote
`registry/runs.jsonl` and pushed it; the last push destroyed the others' lines.
No error.
**Blast radius** `plan_work` reads *completion* from the ledger, so a lost
`completed` entry makes a finished 3-hour run look unfinished and it gets
retrained.
**Fix** one event shard per `(account, worker, session)`, merged on read.
Terminal states are sticky so a late heartbeat cannot resurrect finished work.
Events carry a float clock because second granularity sorts ambiguously across
shards.
**Contamination** none — occurred in the *previous* repo (`msc-kd`/`msc-kd-data`).
`msc-cifar100` was created after the fix and has per-worker shards from its first
commit. Even had it occurred, retraining is deterministic under a fixed seed.

### D-03 · Rate limiter counted per repository
**Found** design review while consolidating repos.
**Cause** HF meters writes **per user**; the limiter lived on the uploader. Two
repos × 20 commits/hour × 6 accounts = 240/hour against a ceiling near 128.
**Fix** one shared token bucket keyed by token, process-wide. Also motivated
consolidating to a single repo (one commit per cycle instead of two).
**Contamination** none — would have throttled uploads, not altered data.

### D-02 · MLP-Mixer cannot run at any other resolution
**Found** NB00 preflight: `mat1 and mat2 shapes cannot be multiplied (192x16 and 64x96)`.
**Cause** the token-mixing block is `Linear(n_tokens → hidden)` — the weight
matrix's input dimension *is* the patch count. Unlike a ViT's positional
embedding, there is no principled resampling. **This is a property of the
architecture, not a code defect.**
**Fix** `supports_native_resolution = False`; resolution axis uses the
downsample-upsample proxy and records the limitation.
**Decision consequence** the proxy became the **primary** resolution measurement
for *all 15* architectures — see DC-3.
**Contamination** none; Phase 0 Q2 already used `res_proxy`.

### D-01a · ViT could not run below 32 px
**Found** NB00 preflight: `size of tensor a (17) must match tensor b (65)`.
**Cause** the learned positional embedding is sized for the 8×8 patch grid
(64 + CLS = 65). A 16 px input gives 4×4 + CLS = 17 tokens.
**Fix** bicubic resampling of the grid portion onto whatever grid the input
needs — the standard ViT/DeiT resolution-transfer procedure. Verified all five
resolutions divide the patch size and yield square grids (16/25/36/49/64 tokens).
**Contamination** none — ViT is not in Phase 0. Would have blocked the entire
resolution axis for the architecture that makes Q3 interesting.

### D-01b · `resnet8x4` produced duplicate compute budgets
**Found** NB00 preflight — and it **passed**, which was the real problem.
**Cause** requesting 5 depth exits from a 3-block network produced cuts
`(1,2,3,3,3)` and `rho = [0.295, 0.648, 1.0, 1.0, 1.0]`. The preflight checked
monotonicity with `<=`, so duplicates slipped through.
**Blast radius** `msc_core.compute_msc` requires *strictly* ascending costs —
"the smallest sufficient budget" is undefined when two budgets cost the same.
Would have crashed ~3 hours into Phase 1b, or worse, returned an MSC depending
on which identical entry `argmax` happened to pick.
**Fix** adaptive K — take as many distinct cuts as the depth allows, record the
fractions achieved. Preflight now asserts strict ascent, distinctness, and
termination at 1.0. Cross-architecture comparison is unaffected: MSC is a cost
*fraction*, not an exit index.
**Contamination** none — resnet8x4 is not in Phase 0.

---

## 3. Design decisions changed after the plan was frozen

| | Decision | Was | Now | Why |
|---|---|---|---|---|
| DC-1 | Repository layout | Two repos (`msc-kd`, `msc-kd-data`) | **One** dataset repo `msc-cifar100`, one folder per run | HF meters per user, so two repos doubled commit consumption for no benefit (D-03). A run's artifacts belong together. Dataset repos render CSV/Parquet previews. |
| DC-2 | Work allocation | Optimistic claim protocol | **Cost-balanced deterministic split**, claims demoted to recovery | Hash sharding gave 4.91× imbalance on 45 uneven jobs; LPT bin packing gives 1.02×. A phase ends when the slowest worker ends. |
| DC-3 | Resolution axis | Native primary, proxy alongside | **Proxy primary** (all 15), native as robustness check (14/15) | MLP-Mixer cannot run natively (D-02). 14 measured one way and 1 another would make cross-architecture claims on this axis compare different quantities. `01_PHASE0_GO_NOGO.md` §3 asks us to decide and stay consistent. |
| DC-4 | Depth exits K | Fixed at 5 | **Adaptive** per architecture | A 3-block network cannot carry 5 distinct budgets (D-01b). |
| DC-5 | Notebook granularity | 5 (one per phase) | **16** | No single notebook should be too long to finish in a Kaggle session; a crash costs less. |
| DC-6 | Staging location | `/kaggle/working` (20 GB) | **`/kaggle/temp`** (~1 TB) | 240 epochs × 10 Hz power sampling + step traces is never disk-constrained there. |
| DC-7 | `NUM_WORKERS` default | 6 | **1** | Splitting across accounts is an optimisation, not a prerequisite. Defaulting to it made the simple case look complicated. |
| DC-8 | Q4 split | test | **train_holdout** | Only that split carries all seven battery scores (D-11). |
| DC-9 | Deleted loss terms | Omitted | **Columns present, value `NA`** | Uniform schema; enabling a term later needs no migration. Fabricating a number for a loss the model never computed would be worse. Protocol's 3-term objective intact. |
| DC-11 | Cost model in assignment | Refined by measured timings | **Static table only** | Measured costs change as runs finish, so ownership stopped being stable (D-12). Estimates still use measurements; allocation must not. |
| DC-10 | Telemetry breadth | 22 columns/epoch | **171** + 91 final | "We only train once." Re-running the atlas to recover a forgotten metric is unrecoverable time. |

---

## 4. Paper crib — what feeds which section

| Paper section | Source | Notes |
|---|---|---|
| §3 MSC definition | `00_RESEARCH_PROTOCOL.md` §2 | Unchanged since freeze |
| §3.1 Cost normalisation | `budgets/{arch}.json`, §1.8 here | Profiler name + version recorded per architecture |
| §3.2 Stable sufficiency | protocol §2.2 + `msc_core.compute_msc` | The suffix-closure argument |
| §3.3 Irreducible subpopulation | **§1.7** | Report |U_τ|/N per model per τ |
| §4.1 Protocol & noise ceiling | **§1.2** | ρ_seed ≈ 0.72; this is the denominator |
| §4.2 [Q2] Axis structure | **§1.3** | **H2 refuted.** PC1 = 0.503 vs predicted ≥ 0.60 |
| §4.3 [Q3] Transfer | **§1.4** | T = 0.946 within-family; awaiting CNN→token |
| §4.4 [Q4] Irreducibility | **§1.5** | ⚠ recompute on `train_holdout` first (D-11) |
| §6.1 Setup | §1.1 + `env/environment.json` | Recipe validated against published numbers |
| §6.4 Efficiency | `metrics/final.csv` | FLOPs primary; energy as methodology only |
| §7 Limitations | **D-02, D-11, §1.7**, batching caveat | See below |
| Reproducibility statement | **§2 entire**, `paper/provenance.csv` | Every artifact → run_id |

### Limitations paragraph — assembled

Draft, from the record:

> Per-sample dynamic routing yields no wall-clock speedup under batched
> inference unless the batch is split by route; our deployment claims are scoped
> to the batch-1, edge and streaming regimes and measured there. The precision
> axis is simulated by fake quantisation — no INT4/INT6 kernel exists for the T4
> — and is priced by an analytic bit-operation model, never reported as measured
> latency. The resolution axis uses a downsample-upsample proxy uniformly across
> all architectures because MLP-Mixer's token-mixing layer is dimensioned to the
> patch count and cannot be evaluated at another resolution; native-resolution
> measurements are reported alongside for the fourteen architectures that
> support it. Risk control is calibrated at ε = 0.03 rather than 0.01, because a
> Hoeffding bound at δ = 0.05 requires ≈ 14,979 calibration samples and the
> CIFAR-100 test set contains 10,000; we calibrate on a held-out training slice
> and note that the calibration distribution is train-like. All experiments are
> on a single hardware class (Kaggle T4) at CIFAR-100 scale.

### Sentences the record already supports

- *"All backbones matched or exceeded published accuracies for the standard
  recipe (Table X), confirming that measurements derive from correctly trained
  models."*
- *"A shuffled-target control yielded T = 0.007, confirming per-sample tables are
  correctly aligned across architectures."*
- *"Every statistic is reported as a curve over τ ∈ {0, 0.1, 0.2, 0.3, 0.5}; no
  conclusion in this paper depends on the choice of τ."*
- *"Contrary to our pre-registered hypothesis H2, a single component explains
  only 50.3% of the variance in per-sample compute requirements across reduction
  axes."* ← **the sentence a reviewer will remember**

---

## 5. Open items

| | Item | Blocks | Priority |
|---|---|---|---|
| O-1 | Recompute Q4 on `train_holdout` (D-11) | Q4 number in the paper | **high** |
| O-1b | **Finish `p1-resnet32x4-cifar100-base-s3`** (epoch 79/240, resumable) | resnet32x4 3rd seed | **high** |
| O-2 | Run NB05–NB07 atlas training (~64 GPU-h remaining) | Q3 across families | **high** |
| O-3 | Run NB08 measurement (~27 GPU-h) | Q1–Q4 at scale | high |
| O-4 | Confirm Q2 non-one-dimensionality across the atlas | §4.2 claim | high |
| O-5 | Read SAFE-KD (2602.03043); write the differentiation memo | Related work | **high, not started** |
| O-6 | Verify the 2026 arXiv IDs cited in the protocol | Bibliography | medium |
| O-7 | Test whether architectures agree on |U_τ| *membership*, not just size | free sub-finding | medium |
| O-8 | Width axis (slimmable) — deferred | Q2 completeness | low |

---

## 6. Timeline

| Date | Event |
|---|---|
| 2026-08-02 | D-12 found auditing NB04 — assignment drift; ownership now uses a static cost table |
| 2026-08-02 | **NB04 atlas: 14/15 ResNet runs complete**, all beating published references |
| 2026-08-02 | **Phase 0 verdict `FULL-PROGRAM`.** All gates cleared at every τ. H2 refuted. |
| 2026-08-02 | D-11 found reading the Q4 output; Q4 default moved to `train_holdout` |
| 2026-08-02 | Phase 0 measurement (NB02) completed for all four runs |
| 2026-08-02 | D-09 found — NB02 was a no-op; stage-aware completion added |
| 2026-08-02 | D-10 found — cost model recalibrated against real timings |
| 2026-08-02 | Phase 0 training completed, all four beating published references |
| 2026-08-02 | D-07, D-08 — empty-state guards and prerequisite checks |
| 2026-08-02 | D-04, D-05, D-06 — ledger sharding, self-resume, the resume test itself |
| 2026-08-02 | Repos consolidated to `msc-cifar100` (DC-1); D-03 rate limiter |
| 2026-08-02 | D-01a/b, D-02 found by the NB00 preflight before any GPU-hour was spent |
| 2026-08-02 | Pipeline built: `msc_lib` 1.0.0, 16 notebooks |

---

*Append new entries at the top of each log. Every claim here should be traceable
to a file in the HuggingFace repository or a commit in this one.*
