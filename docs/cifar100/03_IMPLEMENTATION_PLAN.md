# Implementation Plan — MSC

**Companion to** `00_RESEARCH_PROTOCOL.md`, `01_PHASE0_GO_NOGO.md`, `02_ENGINEERING_SPEC.md`
**Status:** living document — updated after implementation and first runs
**Target platform:** Kaggle dual-T4, 6 accounts, HF Hub as sole permanent store
**Owner:** Shanmuk4622

---

## Part I — What I understand

### 1. The scientific object

The project introduces a per-sample scalar, **Minimum Sufficient Compute (MSC)**, and asks a question the adaptive-inference field has assumed rather than measured: *is the computation an input requires a property of the input, or of the model?*

Three properties of the definition carry all the methodological weight, and every implementation decision below is downstream of them:

**(a) Cost normalisation.** `ρ(c) = FLOPs(f,c) / FLOPs(f,c_full) ∈ (0,1]`. Absolute FLOPs are not comparable across a ResNet and a ViT; the ratio is. This is what makes "did MSC transfer from ResNet to ViT?" a well-posed question at all. *Implementation consequence:* FLOPs must be measured once per (architecture, configuration) with a real profiler and frozen into `budgets/{arch}.json` before any oracle sweep runs. A budget table computed inconsistently between two architectures silently corrupts every transfer number in the paper.

**(b) Stable sufficiency, not naive minimum.** A configuration is sufficient only if it *and every larger budget* agree with the full-compute decision at margin ≥ τ. Predictions under compute reduction are not monotone — a model can be right at 40%, wrong at 60%, right at 100%. The naive minimum records 40%, which is an accident of the sweep. *Implementation consequence:* the oracle must sweep **every** configuration on **every** sample and store the full grid, not just the first agreeing point. There is no early-exit shortcut in the measurement.

**(c) The irreducible subpopulation `U_τ`.** When the full model's own margin is below τ, MSC = 1 trivially and the definition degenerates. These samples must be flagged, counted, excluded from correlation analysis, and analysed separately. *Implementation consequence:* every per-sample table carries an `irreducible` boolean per (axis, τ), and `clean()` masking is applied before any correlation is computed. Forgetting this inflates every transfer coefficient in the project through a shared constant.

### 2. The five questions and what each needs from the pipeline

| Q | Statistic | What the pipeline must produce |
|---|---|---|
| **Q1** Is MSC stable? | `ρ_S^seed` between two seeds of one architecture | ≥2 seeds per architecture, per-sample tables index-aligned to the canonical test-set order |
| **Q2** Is compute need one-dimensional? | PCA over the per-axis MSC matrix | ≥3 axes measured on the *same* model and the *same* samples |
| **Q3** Does MSC transfer? | `T(A,B) = ρ_S(A,B) / √(ceiling_A · ceiling_B)` | Cross-architecture per-sample tables, plus the Q1 ceilings as denominators |
| **Q4** Is MSC irreducible to difficulty? | partial ρ_S, nested ΔR² | Seven-score difficulty battery per model, including three that need training-time instrumentation |
| **Q5** Does MSC-KD beat confidence routing? | Accuracy at matched average FLOPs | Teacher MSC tables → student sufficiency head → risk-calibrated routing → FLOPs-matched comparison |

**Q4 is the one that constrains the training loop.** Three of the seven difficulty scores cannot be computed post-hoc from a final checkpoint:

- **EL2N** (Paul et al. 2021) — the during-training variant. Requires `‖softmax(f(x)) − y_onehot‖₂` recorded at a fixed early epoch. GraNd-at-init is explicitly excluded (failed reproduction, arXiv 2303.14753).
- **Forgetting events** (Toneva et al. 2019) — requires per-sample correctness on the *training* set tracked across every epoch, counting 1→0 transitions.
- **Prediction depth** (Baldock et al. 2021) — requires k-NN probes on intermediate representations; computed post-hoc but needs the exit-head feature maps.

So the backbone training loop must, at no meaningful cost, instrument the training set. This is designed in from the start rather than bolted on, because re-running the atlas to collect a forgotten score costs 110 T4-hours.

### 3. What I take from the E2AM codebase

I decoded and read `e2am.py` v6.6.0 (the 242 KB base64 payload in cell 1 of every ConvNeXtV2-Nano notebook) and all five notebooks. The following patterns are proven at ~5 T4-hours of continuous Kaggle training and are carried forward essentially unchanged, because they already solve the hard problems:

| Pattern | Why it is kept |
|---|---|
| **Base64-embedded library bootstrap** | One notebook cell writes the entire library to `/kaggle/working` and imports it. No pip install of a private package, no git clone, no path fragility. Survives Kaggle's "Run All" from a cold session. |
| **`BackgroundUploader` with batched commits** | A single worker thread and a buffer keyed by `repo_path`. Every push in a 30-minute window collapses into **one** HF commit. This is the single most important rate-limit defence — 6 files in 6 commits is 6× the quota for no benefit. |
| **Token-bucket rate limiter** | Rolling-hour window of commit timestamps, hard cap well below HF's 128/hr. When the cap is reached the worker *sleeps until the oldest commit ages out* rather than failing. |
| **Fingerprint dedup** `(repo_path, size, mtime)` | `config.yaml` and `budgets.json` never change; re-enqueueing them costs nothing because they are skipped before they reach a commit. |
| **429 `retry-after` parsing** | HF's 429 message carries a human-readable hint ("retry after 1234 seconds", "in about 2 hours"). Parsing it and sleeping exactly that long beats blind exponential backoff. |
| **`sync_from_hf_and_repair`** | On a fresh session: scoped `snapshot_download` with `allow_patterns`, wipe unrelated directories, then **rebuild the progress ledger from `history.csv`** rather than trusting `progress.json`. This is what makes resume survive a session that died mid-push. |
| **Confirm-then-delete** | Local run directory is wiped only after `list_repo_files()` confirms the artifacts are actually on HF. Prevents the 20 GB working disk from filling while never risking data loss. |
| **Broken-stub demotion** | A run marked `completed` with too few epochs is a lie left by a crash. Detect, wipe locally and on HF, retrain. |

### 4. What I am changing or adding

E2AM was an efficiency-benchmark pipeline. MSC is a *measurement* pipeline, and four things the engineering spec demands are not in E2AM:

| Gap in E2AM | Fix |
|---|---|
| Checkpoint stores no RNG state | `02_ENGINEERING_SPEC.md` §3 is explicit: without python/numpy/torch/cuda RNG state, a resumed run ≠ an uninterrupted run, augmentation order diverges, **and the seeds become meaningless** — which destroys Q1, the noise ceiling, which is the denominator of the entire project. Full RNG capture is non-negotiable here in a way it was not for E2AM. |
| No `config_hash` check on resume | Resume asserts `sha256(config.yaml)` matches. Loud failure beats silently continuing a run under an edited config. |
| Only `KeyboardInterrupt` is caught | Add `atexit` **and** a `SIGTERM` handler. Kaggle sends SIGTERM before killing a session; catching it buys the seconds needed for a final flush. |
| Datasets land on `/kaggle/working` | Move all dataset and scratch tensors to `/kaggle/temp` (~1 TB, session-local). `/kaggle/working` (20 GB) is artifact space only. |
| No session watchdog | Track elapsed wall time; past 8.5 h, force a full push and mark `state: paused` before Kaggle pulls the plug. |
| Repo layout | **One** dataset repo, `msc-cifar100`, with one folder per run. (An intermediate two-repo design was tried and reverted: HF's write limit is per *user*, so two uploaders doubled commit consumption for no benefit. See §6.2(d).) |
| No multi-account coordination | Optimistic claim protocol on `registry/runs.jsonl` + `registry/claims/{run_id}.json`, with 2-hour staleness takeover. |

One further change, on maintainability. E2AM's library exists **only** as base64 inside the notebooks, so editing it means decoding, patching, re-encoding by hand. Here the library is a readable file (`src/msc_lib.py`) and `build_notebooks.py` regenerates the `.ipynb` files from it. Edit the Python, re-run the generator. The base64 bootstrap is kept because it is robust; the opacity is not.

---

## Part II — What we are going to do

### 5. Decisions taken

| Decision | Choice | Reason |
|---|---|---|
| Scope | Full generalised pipeline — Phase 0 + Phase 1 atlas + Phase 3 method, config-driven | Phase 0 still runs first and writes its gate decision to HF; building the rest now costs nothing extra and removes a week of wall-clock from the critical path |
| Axes | **Depth + Resolution + Precision** | All three are inference-only once the backbone is trained. Three axes make Q2's PCA a real measurement rather than a two-variable preview. Width (slimmable) deferred — it is the only axis requiring a second training procedure |
| Resolution axis | **Proxy primary** (all 15 architectures), **native as robustness check** (14 of 15) | See §6.1 — revised after NB00 found that MLP-Mixer cannot run at any resolution but 32px |
| HF layout | **One repo:** `Shanmuk4622/msc-cifar100` (dataset), one folder per run | Revised — see `06_DATA_SCHEMA.md`. HF's rate limit is per *user*, so two repos doubled commit consumption; and a run's artifacts belong together |
| Model zoo | Full zoo registered on CIFAR-100 | ViT-Tiny and Mixer-Nano are what make Q3's hypothesis testable; registering them now avoids rework |
| Dataset | `shanmuk4622/dataset-cifar100-python` (Kaggle) | User-specified, fast in-datacentre download. Torchvision auto-download retained as last-resort fallback only |
| τ grid | `{0.0, 0.1, 0.2, 0.3, 0.5}`, always reported as a curve | Protocol §2.4. No single τ is ever selected |
| Seeds | 3 per architecture (Phase 0 uses 2) | Spec §8.6 — single-seed numbers do not go in the paper |

### 6. Compute configuration grid

Frozen here so `budgets/{arch}.json` is deterministic.

| Axis | K | Configurations | Cost model |
|---|---|---|---|
| Depth | 5 | Exits at fractional backbone depth {0.2, 0.4, 0.6, 0.8, 1.0}, head = GAP → BN → Linear | Measured FLOPs of backbone prefix + head |
| Resolution | 5 | r ∈ {16, 20, 24, 28, 32} px | **Native**: measured FLOPs at input r. **Proxy**: same cost model, but the network is run at 32 px on a downsample-then-upsample image — labelled *idealised* |
| Precision | 5 | {INT4, INT6, INT8, FP16, FP32} | ρ = bitwidth / 32 applied to the MAC-dominant layers, plus measured FP32 FLOPs for the rest. Bit-operation accounting documented in the budgets file |

The backbone is **frozen** while exit heads train (20 epochs, LR 0.01, cosine). This is load-bearing: if the backbone adapts, each exit is reading a different network and the "same model under reduced compute" interpretation — which the whole construct rests on — collapses.

### 6.1 Three corrections found by the NB00 preflight

The preflight caught three problems before any GPU-hour was spent on them. Recorded here because two of them changed decisions stated above.

**(a) K is adaptive, not fixed at 5.**
`resnet8x4` has 3 blocks in total, so it cannot carry five distinct depth exits. Requesting them produced cuts `(1,2,3,3,3)` and hence `rho = [0.295, 0.648, 1.0, 1.0, 1.0]`.

Those duplicate `1.0` entries are not cosmetic. `msc_core.compute_msc` requires strictly ascending costs, because "the smallest sufficient budget" is undefined when two budgets cost the same — and had the guard not been there, MSC would have depended on which of several identical entries `argmax` happened to return. **This would have surfaced as a crash roughly three hours into Phase 1b**, or worse, not surfaced at all.

Fix: take as many distinct cuts as the depth allows and record the fractions actually achieved. Cross-architecture comparison is unaffected — MSC is a cost *fraction* in (0,1], not an exit index, so different K is legitimate. The preflight now asserts strict ascent, distinctness, and termination at 1.0 for every architecture.

**(b) ViT needs positional-embedding interpolation.**
The learned positional embedding is sized for the 8×8 patch grid of a 32px input (64 patches + CLS = 65). A 16px input gives 4×4 = 16 patches + CLS = 17 tokens, and adding a 65-entry embedding to a 17-token tensor is a shape error — which is exactly the reported `size of tensor a (17) must match tensor b (65)`.

Fix: keep the CLS entry, reshape the patch entries to their square grid, bicubically resample to the grid the current input needs. This is the standard ViT/DeiT resolution-transfer procedure, not an invention. It matters because token-count reduction is *where a transformer's compute saving actually comes from* — without it the resolution axis would be undefined for the one architecture whose inductive bias makes Q3 interesting.

All five resolutions divide the patch size and yield square grids: 16→16, 20→25, 24→36, 28→49, 32→64 tokens.

**(c) MLP-Mixer cannot run at any other resolution, and this changes the primary resolution measurement.**
The token-mixing block is `Linear(n_tokens → hidden)`: the weight matrix's input dimension *is* the patch count. Unlike the ViT case there is no principled fix — a positional embedding is a lookup that can be resampled, but token-mixing weights are a learned linear map whose domain is the token grid. A trained Mixer simply cannot be evaluated at a different token count.

`01_PHASE0_GO_NOGO.md` §3 anticipated this, offering native ("cleaner; use it if the architecture tolerates it") or the downsample-upsample proxy, and instructing us to **decide in Phase 0 and stay consistent**.

Decision: **the proxy is primary, native is a robustness check.** The reasoning is consistency, not convenience. If 14 architectures were measured natively and one by proxy, every cross-architecture claim on the resolution axis would silently compare two different quantities for that model — which is a worse failure than a uniformly idealised cost model. NB03 Step 6b and NB10 Step 5 quantify the agreement between the two on the 14 architectures that support both, so the caveat is measured rather than argued.

Consequences recorded in the artifacts: `budgets/{arch}.json → axes.resolution.native_supported`, the `available_axes()` helper, and a stated limitation in the model card.

### 6.2 Further corrections found by running it

Three more problems surfaced only once real runs were going. All are documented
in full in `07_REPLICATION_PLAYBOOK.md` §13.

**(d) The rate limiter was on the wrong object.** HF meters writes per *user*;
the limiter lived on the uploader. Two repos × 20 commits/hour × 6 accounts =
240/hour against a ceiling near 128. Fixed with one shared token bucket keyed by
token, process-wide.

**(e) The shared run ledger was losing writes.** HuggingFace has no append
operation, so every worker rewriting `registry/runs.jsonl` meant the last push
silently destroyed the others' lines. Observed live: two runs training, ledger
listing one. Since work planning reads *completion* from that ledger, a lost
`completed` entry means a finished 3-hour run gets trained again. Fixed by
sharding one event file per writer and merging on read — the same collision-safe
pattern a shard writer uses.

**(f) The claim protocol blocked self-resume.** The 2-hour staleness window was
applied without checking ownership, so a session that paused at the 8.5-hour
limit could not be resumed by its own account for two hours — defeating the
entire resumability contract at exactly the moment it exists for. Ownership is
now checked before freshness.

### 7. Deliverable file layout

```
KD/
├── 00_RESEARCH_PROTOCOL.md      the science
├── 01_PHASE0_GO_NOGO.md         the gate
├── 02_ENGINEERING_SPEC.md       original contract (repo layout superseded)
├── 03_IMPLEMENTATION_PLAN.md    this file
├── 04_NOTEBOOK_RUNBOOK.md       how to run it
├── 05_PLAIN_ENGLISH_GUIDE.md    what it means, no jargon
├── 06_DATA_SCHEMA.md            repo structure + every column
├── 07_REPLICATION_PLAYBOOK.md   how to rebuild this anywhere
├── msc_core.py                  oracle + analysis statistics
├── msc_torch.py                 reference model-side components
├── src/msc_lib.py               the pipeline library (source of truth)
├── build_notebooks.py           regenerates the notebooks
└── notebooks/                   NB00 .. NB15  (16 notebooks)
```

### 8. Notebook responsibilities

Sixteen notebooks, so no single one is too long to finish in a Kaggle session,
and so a crash in one costs little. Full table in `04_NOTEBOOK_RUNBOOK.md` §2.

| # | Notebook | GPU | Time | Sharded |
|---|---|---|---|---|
| 00 | Setup & Verify | T4 | 15 min | run on every account |
| 01 | Phase 0 — train | T4 | ~12 GPU-h | 4 workers |
| 02 | Phase 0 — measure + **final evaluation** | T4 | ~2 h | 4 workers |
| 03 | Phase 0 — decision | off | 10 min | — |
| 04 | Atlas — ResNets | T4 | ~25 GPU-h | 6 workers |
| 05 | Atlas — WRN + VGG | T4 | ~25 GPU-h | 6 workers |
| 06 | Atlas — Mobile | T4 | ~15 GPU-h | 6 workers |
| 07 | Atlas — ConvNeXt / ViT / Mixer | T4 | ~45 GPU-h | 6 workers |
| 08 | Atlas — measurement | T4 | ~25 GPU-h | 6 workers |
| 09 | Q1 noise ceiling | off | 5 min | — |
| 10 | Q2 axis structure | off | 5 min | — |
| 11 | Q3 transfer | off | 15 min | — |
| 12 | Q4 irreducibility | off | 20 min | — |
| 13 | MSC-KD training | T4 | ~120 GPU-h | 6 workers |
| 14 | Method comparison | T4 | ~5 h | 3 workers |
| 15 | Paper outputs | off | 10 min | — |

NB00 is not optional and not a formality. It is the cheapest place to discover
that a ViT does not tolerate the input sizes the resolution sweep assumes — which
is exactly what it found.

NB07 is separate because ConvNeXt, ViT and Mixer need a completely different
recipe (AdamW, long warmup, label smoothing). Under SGD they do not learn at all.

### 9. What gets saved to Hugging Face

The instruction is *save every detail — we only train once*. Concretely, per run:

**One repo — `Shanmuk4622/msc-cifar100` — one folder per run.** Full schema in
`06_DATA_SCHEMA.md`; summary here:

```
runs/{run_id}/
├── config.yaml · config_hash.txt · STATUS.json · summary.json
├── metrics/      epochs.csv (171 cols) · final.csv (91 cols)
│                 confusion_matrix.csv · per_class.csv
│                 calibration.csv · inference_bench.csv
│                 exit_metrics.csv · msc_summary.csv
├── telemetry/    energy_samples.csv (10 Hz, per GPU)
│                 system_samples.csv (1 Hz, per GPU)
│                 step_traces.jsonl · console.log
├── per_sample/   test.parquet · train_holdout.parquet
│                 train_dynamics.parquet · meta.json
├── checkpoints/  ckpt_last.pt · ckpt_best.pt · exit_heads.pt
└── env/          environment.json · budgets.json

registry/events/{account}_w{id}_{session}.jsonl   one shard per writer
registry/claims/ · registry/plans/
budgets/{arch}.json
tables/all_epochs.csv · all_final.csv · atlas_summary.csv
analysis/ · paper/
```

Per-sample table schema (test set, 10,000 rows, index-aligned to canonical order):

```
sample_idx, label,
pred_d1..d5,   top1p_d1..d5,   top2p_d1..d5,   logit_entropy_d1..d5    # depth
pred_rn1..rn5, top1p_rn1..rn5, top2p_rn1..rn5                          # resolution, native
pred_rp1..rp5, top1p_rp1..rp5, top2p_rp1..rp5                          # resolution, proxy
pred_q1..q5,   top1p_q1..q5,   top2p_q1..q5                            # precision
msp, margin, entropy, ce_loss, el2n, forget_events, pred_depth         # difficulty battery
```

### 10. HF push policy

| Trigger | Pushed |
|---|---|
| 30-minute timer | `ckpt_last.pt`, `history.csv`, `STATUS.json`, `progress.json` |
| Major stage completion (backbone done / heads done / sweep done) | full run directory + per-sample Parquet |
| New best validation metric | `ckpt_best.pt` (suppressed if <3 epochs since last push — in early training every epoch is a new best, which would defeat batching) |
| `KeyboardInterrupt` / `SIGTERM` / `atexit` / uncaught exception | **immediate full push, blocking flush, then exit** |
| Elapsed > 8.5 h | immediate full push, `state: paused` |

All of it collapses into **one batched commit per cycle**. Budget: ~18 pushes per 9-hour session per account, hard-capped at 20 commits/hour per account by the token bucket. Six accounts → ~120/hr worst case against HF's 128; the cap is set at 20 rather than 30 precisely because six accounts share one quota.

### 11. Resumability contract

On a cold session with zero local state, Notebook 02 must:

1. Pull `registry/runs.jsonl`; skip any `run_id` held by a live claim (<2 h old), take over stale ones
2. Scoped `snapshot_download` of only this run's artifacts
3. Rebuild `progress.json` from `history.csv` — never trust the ledger alone
4. Load `ckpt_last.pt`, assert `config_hash` matches, restore model / optimizer / scheduler / scaler / **all four RNG streams** / cumulative energy / wall time
5. Truncate `history.csv` to the checkpointed epoch so resumed rows never duplicate
6. Continue

**Acceptance test (in Notebook 00):** train 4 epochs, hard-kill, resume, and confirm the loss curve is continuous across the seam and that epoch 5's first-batch RNG draw matches an uninterrupted reference run. A resume that "works" but perturbs the RNG stream is a silent Q1 corruption, so it is tested explicitly rather than assumed.

### 12. Known risks in the implementation

| Risk | Mitigation |
|---|---|
| Per-sample table index misalignment between models | Canonical sample order asserted by a `sha256` of the label vector, stored in every table and compared at analysis time. The shuffled-target control in Notebook 04 catches this too — nonzero shuffled transfer means a bug, not a finding |
| ViT/Mixer `forward_features` shape mismatch with `ExitHead`'s `AdaptiveAvgPool2d` | Exit heads dispatch on feature rank: `(B,C,H,W)` → spatial pool, `(B,N,C)` → token mean. Verified per-architecture in Notebook 00 |
| INT4/INT6 have no native PyTorch kernel | Simulated via fake-quantisation (quantise-dequantise). Documented as *simulated precision reduction*, and the cost model is labelled analytic — never presented as measured latency |
| FLOPs profiler disagreement (fvcore vs ptflops vs thop) | One profiler chosen, version pinned and recorded in `budgets/{arch}.json`. Cross-checked against a second in Notebook 00 and any >5% disagreement reported |
| Kaggle 20 GB working disk fills mid-atlas | Datasets and scratch on `/kaggle/temp`; confirm-then-delete local run wipe; disk-free assertion before each run |
| Six accounts double-run the same config | Optimistic claim protocol + heartbeat, and `train_one_run` skips anything the ledger marks `completed` |
| A run marked complete but truncated by a crash | Broken-stub detection: `completed` with `last_epoch < 0.9 × planned` is demoted, wiped locally and on HF, retrained |

---

## Part III — Build order

1. `src/msc_lib.py` — infrastructure, zoo, oracle, method
2. `build_notebooks.py` — generator
3. Notebooks 00 → 06
4. `04_NOTEBOOK_RUNBOOK.md`
5. Verification: library compiles, `msc_core.py` self-test passes, library self-test passes, resume simulation passes, all notebooks are valid JSON with correct execution order

Nothing in this plan requires a trial run before implementation. Every number it commits to is either fixed by the protocol or measured by the pipeline itself.
