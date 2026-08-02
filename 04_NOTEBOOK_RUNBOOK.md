# Notebook Runbook

Operating manual. For *what the project is about*, read `05_PLAIN_ENGLISH_GUIDE.md` first.

---

## 1. One-time Kaggle setup

Do this on **every** account.

### 1.1 Secret

`Add-ons → Secrets → Add a new secret`

| Label | Value |
|---|---|
| `HF_TOKEN` | A Hugging Face token with **WRITE** scope |

From <https://huggingface.co/settings/tokens>. A read-only token fails only at the first push, hours in — NB00 checks the scope explicitly for exactly this reason.

Every account needs write access to the repo:

- **`Shanmuk4622/msc-cifar100`** — a HuggingFace **dataset** repo holding everything: checkpoints, metrics, telemetry, per-sample tables, registry, analysis.

Add each teammate as a collaborator, or use an org. The repo is created automatically on first push.

*(A dataset repo rather than a model repo because HF renders CSV and Parquet previews for datasets — every metrics table becomes browsable in the browser without downloading.)*

### 1.2 Notebook settings

| Setting | Value |
|---|---|
| Accelerator | **GPU T4 × 2** — except analysis notebooks, see below |
| Internet | **On** |
| Environment | Latest Kaggle image |

**Turn the GPU OFF for NB03, NB09–NB12, NB15.** They're pure arithmetic on already-saved tables; a GPU session burns your weekly quota for nothing.

### 1.3 Attach the dataset

`+ Add Input → Datasets` → **`shanmuk4622/dataset-cifar100-python`**

Found instantly under `/kaggle/input` if attached. Otherwise it falls back to the Kaggle CLI, then torchvision. Attaching is much faster.

### 1.4 Set your identity — the two lines that matter

Near the top of every notebook:

```python
ACCOUNT     = 'acct1'      # <<< CHANGE ME
NUM_WORKERS = 1            # <<< DEFAULT: this account does everything
WORKER_ID   = 0            # <<< CHANGE ME: 0, 1, 2, ... NUM_WORKERS-1
```

**Every notebook ships with `NUM_WORKERS = 1`** — one account runs the whole
thing. That is the simplest thing that works, and nothing is lost by it except
wall-clock.

Raise it only when you have several accounts going at once. Then
**`WORKER_ID` must differ on every account** — it's what splits the work. Two
accounts sharing a `WORKER_ID` will train exactly the same models and waste
half your compute.

Each notebook prints, at the top and again at runtime, how many runs it
contains and the wall-clock at 1 / 2 / 4 / 6 workers. Decide from that.

---

## 2. Run order

Sixteen notebooks. Restart the kernel between them.

| # | Notebook | GPU | Runs | Est. GPU-h | Wall-clock @ 1 worker |
|---|---|---|---|---|---|
| 00 | Setup & Verify | T4 | — | — | ~15 min |
| 01 | Phase 0 — train | T4 | 4 | ~10 | ~10 h (2 sessions) |
| 02 | Phase 0 — measure + final eval | T4 | 4 | ~2 | ~2 h |
| 03 | Phase 0 — decision | **off** | — | — | ~10 min |
| 04 | Atlas — ResNets | T4 | 15 | ~25 | ~25 h (3 sessions) |
| 05 | Atlas — WRN + VGG | T4 | 15 | ~19 | ~19 h (3 sessions) |
| 06 | Atlas — Mobile | T4 | 6 | ~9 | ~9 h (2 sessions) |
| 07 | Atlas — ConvNeXt/ViT/Mixer | T4 | 9 | ~36 | ~36 h (5 sessions) |
| 08 | Atlas — measure + final eval | T4 | 45 | ~27 | ~27 h (4 sessions) |
| 09–12 | Analysis Q1–Q4 | **off** | — | — | 5–20 min each |
| 13 | MSC-KD training | T4 | 9 | ~30 | ~30 h |
| 14 | Method comparison | T4 | — | ~5 | ~5 h |
| 15 | Paper outputs | **off** | — | — | ~10 min |

**Total ≈ 163 GPU-hours.** Every notebook defaults to `NUM_WORKERS = 1`, meaning
one account does all of it. Raise it and run the same notebook on each account
with a different `WORKER_ID` to divide the time — the wall-clock column at
2/4/6 workers is printed inside each notebook.

Sessions assume the 8.5 h pause-and-resume limit. A notebook needing more than
one session resumes automatically: start a fresh session and re-run.

**Turn the GPU OFF for NB03, NB09–NB12, NB15.**

### ⛔ Stop after NB03. Read the verdict. Decide as a team.

NB04–NB07 are independent — run them in any order, or simultaneously on
different accounts. NB08 can start as soon as *any* model finishes.
**NB09 must run before NB11** (it produces the ceilings NB11 divides by).
NB13–NB14 only if NB11 showed transfer.

## 3. How the work splits across accounts

### The mechanism

Every account runs the **same notebook**. The only difference is `WORKER_ID`.

Each account independently computes the same job→worker assignment from the same list, then keeps only its own. Identical code + identical input = identical conclusions, **with no communication**. Hence:

- **No overlap** — a job belongs to exactly one worker
- **No gaps** — every job belongs to someone
- **Restart-proof** — assignment depends only on the job name, not on timing or crashes

### Why it isn't plain hashing

Hashing is perfect for a large open-ended universe (which is why the image-generation pipeline uses it). Here the universe is 45 jobs of very unequal cost, and hashing splits it badly:

| Mode | Runs per worker (6 accounts) | Wall-clock imbalance |
|---|---|---|
| `hash` | `[11, 7, 4, 10, 3, 10]` | **4.91×** |
| `balanced` | `[8, 8, 8, 7, 7, 7]` | 1.31× |
| **`cost`** (default) | `[7, 7, 7, 8, 8, 8]` | **1.02×** |

A phase isn't finished until the *slowest* worker finishes, so 4.91× imbalance means a 4.91×-longer phase than necessary.

`cost` mode estimates each job's GPU-hours, sorts most-expensive-first, and gives each job to whichever worker currently has least queued. It also **self-corrects**: once real per-epoch timings exist in the logs, they replace the estimates.

NB00 Step 6 prints all three so you can see it.

### If an account dies

Each notebook heartbeats every 30 minutes. After 2 hours of silence, other workers — **once they've finished their own share** — pick up the abandoned jobs automatically. Own work always comes first, so two live workers never collide.

### Using several accounts

Set `NUM_WORKERS` to how many you have and give each `WORKER_ID` 0..N-1, then run the same notebooks everywhere. The scheduler handles the rest — manual architecture assignment isn't needed and generally does worse.

The default is 1, so nothing is required of you here. It only changes wall-clock:

| | 1 account | 2 | 4 | 6 |
|---|---|---|---|---|
| Atlas training (NB04–07) | ~89 h | ~45 h | ~23 h | ~15 h |
| Whole project | ~163 h | ~82 h | ~41 h | ~28 h |

---

## 4. What each notebook needs

### NB00 — Setup & Verify

Run all cells. Two cells `assert`.

Proves: HF reachable **and writable** (pushes a probe, re-lists the repo to confirm arrival) · dataset found · all 15 architectures build/forward/backprop · FLOPs tables sane · **kill-and-resume produces an identical run** · work splits evenly.

**If Step 5 (resume) fails, stop.** A resume that reloads weights but not the RNG state changes the order images are seen in, which breaks the meaning of "same model, different seed" — and that comparison is the denominator of every number in the paper.

### NB01 — Phase 0 training

Four runs, ~10 GPU-hours total (resnet32x4 ≈ 2.9 h each, wrn-40-2 ≈ 1.9 h each). At `NUM_WORKERS = 1` that is ~10 h across two sessions; at 4 accounts, ~3 h.

Expected: **resnet32x4 ≈ 79.4%**, **wrn-40-2 ≈ 75.6%**. Step 5 audits this. Do not proceed past a warning — measurements from an under-trained model are meaningless, and an under-trained model is otherwise easy to miss.

### NB02 — Phase 0 measurement **and final evaluation**

Inference only, ~30 min per model. Two jobs in one checkpoint load:

1. **MSC measurement** — exit heads (backbone frozen), then all 20 compute configurations over every test image, plus the difficulty battery → `per_sample/*.parquet`
2. **Final evaluation** — confusion matrix, per-class precision/recall/F1, calibration (ECE/MCE/NLL/Brier), latency and throughput at batch 1/32/128 with warm-up discarded, inference energy, params/FLOPs/size, compression and energy-reduction ratios → `metrics/final.csv`

Step 5 checks row alignment across tables.

### NB03 — Phase 0 decision

**GPU off.** Step 4 asserts on the scrambled control — nonzero scrambled agreement is a bug, not a finding.

Step 8 prints the verdict and writes `analysis/phase0_decision.json`. Three of the five verdicts still lead to a paper.

**Stop here.** Write the verdict and reasoning into the repo before continuing.

### NB04–NB07 — atlas training

Long-running. Step 4 shows the plan and estimated hours before committing. Step 6 shows team-wide progress, not just yours.

NB07 (ConvNeXt/ViT/Mixer) is separate because those need AdamW, long warmup, and label smoothing. Under plain SGD they sit near 1% accuracy forever. The library picks the recipe automatically.

**Do not skip NB07.** Those three are the reason Q3 is interesting.

### NB08 — atlas measurement

Only measures fully-trained models. Step 6 audits row alignment across the whole atlas.

### NB09–NB12 — analysis

**GPU off.** NB09 first (produces `analysis/ceilings.json`). NB11 Step 3 asserts on the scrambled control across every pair.

NB12 Step 2 checks whether the difficulty battery is complete. EL2N, forgetting events and prediction depth are recorded *during* training and cannot be recovered afterward — if they're missing, re-run NB08.

### NB13 — the method

**Run with `SHUFFLE_ABLATION = True` first, then `False`.** Step 6 compares them. If scrambled targets perform as well as real ones, the compute signal isn't doing the work and the central claim is wrong — better to learn that now.

### NB14 — comparison

Step 5 tests the stated mechanism: does the advantage grow as students shrink? If not, the mechanism claim is wrong **even if the method wins**, and the paper must say so.

Step 7 prints the risk-control sample requirements. Read them:

| ε | δ | Calibration images needed |
|---|---|---|
| 0.01 | 0.05 | **14,979** |
| 0.03 | 0.05 | 1,665 |
| 0.05 | 0.05 | 600 |

CIFAR-100's test set is 10,000. **You cannot certify a 1% accuracy drop at 95% confidence on it.** We calibrate on the 5,000-image training holdout at ε=0.03 and state that in the paper. A design decision, not a bug — but report it rather than quietly using ε=0.01.

### NB15 — paper outputs

**GPU off.** Tables, figures, telemetry summary, energy totals, provenance manifest, model card.

---

## 5. What gets saved

### Per epoch, for every run

**Learning** — train/val loss, train/val accuracy, top-5, F1, precision, recall, loss min/max/std

**Optimisation health** — LR per parameter group, gradient norm mean/max/p95, gradient-clip hit rate, weight norm, update-to-weight ratio, AMP scale, NaN/Inf batch count, optimizer step count

**Speed** — epoch time, train vs eval time, **dataload vs compute split**, step-time p50/p90/p99, throughput, samples seen

**Hardware** — VRAM allocated/reserved/peak/total, GPU utilisation, GPU temperature, CPU %, RAM used, free disk

**Energy** — joules and kWh per epoch and cumulative, CO₂, mean and peak power, sample count

**Provenance** — run ID, worker ID, session ID, hostname, timestamp, architecture, seed, batch size

### Raw streams

- `energy.csv` — power samples at 10 Hz
- `system.csv` — GPU util / temperature / clock / CPU / RAM at 1 Hz
- `step_traces.jsonl` — downsampled per-step timing and loss, per epoch

### Repository layout

**`Shanmuk4622/msc-cifar100`** — one folder per run:

```
runs/{run_id}/
├── config.yaml · config_hash.txt · STATUS.json · summary.json
├── metrics/
│   ├── epochs.csv              <- 171 columns per epoch
│   ├── final.csv               <- 91 columns, one row
│   ├── confusion_matrix.csv · per_class.csv
│   ├── calibration.csv · inference_bench.csv
│   └── exit_metrics.csv · msc_summary.csv
├── telemetry/
│   ├── energy_samples.csv      10 Hz power, per GPU
│   ├── system_samples.csv      1 Hz util/temp/clock/CPU/RAM, per GPU
│   └── step_traces.jsonl · console.log
├── per_sample/
│   ├── test.parquet            <- the scientific artifact
│   ├── train_holdout.parquet
│   ├── train_dynamics.parquet  EL2N + forgetting events
│   └── meta.json
├── checkpoints/
│   ├── ckpt_last.pt            model+optim+sched+scaler+ALL RNG+energy+time
│   ├── ckpt_best.pt · exit_heads.pt
└── env/
    ├── environment.json        pip freeze, CUDA, driver, GPU
    └── budgets.json            FLOPs per configuration

registry/events/{account}_w{id}_{session}.jsonl   one shard per writer
registry/claims/ · registry/plans/
budgets/{arch}.json
tables/all_epochs.csv · all_final.csv · atlas_summary.csv
analysis/*.csv · paper/figures/*.png · paper/provenance.csv
```

Full column-by-column schema: `06_DATA_SCHEMA.md`.

---

## 6. Push policy and rate limiting

### When we push

| Trigger | What |
|---|---|
| Every 30 min | checkpoint, logs, heartbeat |
| Every 10 epochs | everything |
| New best accuracy | best checkpoint — suppressed if <3 epochs since last push |
| Stage completion | everything, including measurement tables |
| **You press Stop** | **immediate, blocking, then exit** |
| Kaggle SIGTERM | same |
| Uncaught exception | same, plus a `failed` marker |
| 8.5 h elapsed | everything, marks itself `paused` |

New-best suppression matters: early in training every epoch is a new best, which would defeat batching entirely.

### Staying under HF's ~128 commits/hour

1. **One commit per push.** Twenty files → one commit, not twenty.
2. **Token bucket**, hard-capped at **20 commits/hour per account**. 6 × 20 = 120 < 128. At the cap the uploader *sleeps until the oldest commit ages out* rather than failing.
3. **Fingerprint dedup** on `(path, size, mtime)` — unchanged files are skipped before they reach a commit.

On a 429 anyway, we parse the `retry-after` hint from the response and sleep exactly that long. A failed push is retried next cycle and never kills training.

### Disk

`/kaggle/working` (20 GB) holds **only** artifacts awaiting upload. `/kaggle/temp` (~1 TB) holds datasets and scratch. After a run finishes and HF **confirms the files by re-listing the repo** (not merely by a flush that didn't error), the local copy is deleted.

---

## 7. Resuming

Kill the session whenever. To resume: fresh session, run all cells.

1. Scoped download — never the whole repo
2. Progress rebuilt **from `metrics/epochs.csv`**, not from the status file — a session that died mid-push leaves them disagreeing, and the log is honest
3. Broken stubs (marked complete but truncated) demoted so they resume
4. Checkpoint loaded, **`config_hash` asserted**
5. `history.csv` truncated so resumed epochs don't duplicate
6. Model, optimizer, scheduler, AMP scaler, **all four RNG streams**, cumulative energy and time restored

**If the config changed, it refuses to resume and raises.** Deliberate. Silently continuing under an edited config is how you get numbers that don't reproduce and no idea why. Restore the config, or set `force_rerun=True` to discard and retrain.

---

## 8. Troubleshooting

**`no token: add 'HF_TOKEN' to Kaggle Secrets`**
Secret missing, or not enabled for *this* notebook. Add-ons → Secrets, check the toggle.

**`AUTH FAILURE -- check HF_TOKEN write scope`**
Token is read-only, or the account lacks write access to one repo. Regenerate with write scope.

**`Could not obtain CIFAR-100 from any source`**
Internet off, or dataset not attached.

**`config_hash mismatch ... The config changed since this run started`**
Working as designed. Restore the config, or `force_rerun=True`.

**`SKIP {run_id}: held by acctN (12 min ago)`**
A **different** account owns it and is actively working. Correct. If their session died, it becomes stealable after 2 hours automatically.

This never blocks you from resuming **your own** run. Ownership is checked before freshness: same account → always allowed, because "my session died and this is the new one" is by far the most common case. You'll see `resuming own run from a previous session` instead.

**`was left 'paused' by an earlier session of acctN — resuming it`**
Normal. Your previous Kaggle session ended (limit, crash, or you stopped it) and this one is picking the run back up. Only worth a second look if you genuinely have two live sessions on one account — in which case give them different `WORKER_ID`s.

**`recipe_check_skipped` in a run summary**
The run was too short (<100 epochs) to compare against a published full-recipe accuracy. Expected for smoke tests. Full atlas runs are checked normally.

**`rate-limit guard: 20 commits in the last hour, sleeping 2400s`**
Working as designed. Training continues; only the uploader waits. If constant, lower `commits_per_hour_limit` or raise `milestone_push_every_epochs`.

**Work split looks lopsided**
Check NB00 Step 6. Use `shard_mode='cost'` (the default). If still bad, your `NUM_WORKERS` may not divide the workload well — try a different count.

**Two accounts training the same model**
They have the same `WORKER_ID`. That one line must differ on every account.

**Disk full mid-atlas**
`msc.free_mb(sess.work)` to check. `sess.sync_state(run_ids=[...])` to re-scope. Confirm nothing wrote a dataset into `/kaggle/working`.

**`shuffled transfer FAILED: T=0.31 (expected ~0)`**
A bug, not a finding. The tables aren't row-aligned. Check `sample_order_hash` matches (NB08 Step 6). Usual cause: a shuffled eval loader — eval loaders must never shuffle.

**A shallower exit beats a deeper one by >2 points**
The stage partition is wrong for that architecture. Inspect `budgets/{arch}.json → axes.depth.stage_cuts` before trusting its depth measurements.

**`resnet8x4` shows `K=3` instead of 5**
Correct, not a bug. It has only 3 blocks, so it cannot carry five distinct early-exit points. MSC is a cost *fraction*, not an exit index, so different K across architectures is fine. Forcing 5 would produce duplicate budgets (`rho = [..., 1.0, 1.0, 1.0]`), which makes "the smallest sufficient budget" undefined and crashes the oracle.

**`mixer_nano` shows `res_native = False`**
Correct, not a bug. Its token-mixing layer is `Linear(n_tokens → hidden)` — the weight matrix's input dimension *is* the patch count, so a trained Mixer cannot run at a different resolution. There's no fix; it's a property of the architecture. Its resolution axis uses the shrink-then-restore proxy, which is what all 15 architectures use as the primary measurement anyway (see §4, NB10).

**`native-resolution sweep failed ... proxy only for this model`**
The library tried the native sweep, it didn't work, and it fell back cleanly. The per-image table simply won't have `res_native` columns. `msc.available_axes(df)` tells you what a table carries; the analysis notebooks already handle it.

**ViT-Tiny or Mixer-Nano stuck at ~1% accuracy**
They need AdamW, not SGD. Confirm `cfg['optimizer'] == 'adamw'`. The library sets this automatically via `TRANSFORMER_LIKE`.

**`dataload_frac` above 0.3**
The GPU spent >30% of the epoch waiting for data. Not a correctness problem, but the fix is the loader, not the model. NB15 Step 4 flags these.

**`nan_or_inf_batches` above 0**
AMP produced non-finite losses. The run continues but learns nothing from those batches. Investigate before trusting the result. NB15 Step 4 flags these.

**`LTT is underpowered: n=5000 gives a Hoeffding slack of 0.0173`**
Statistics, not a bug. See §4, NB14. Raise ε above the reported slack.

---

## 9. Changing the library

```bash
cd "D:\Documents\norse\web Applicarion\KD"
python src/msc_lib.py --selftest      # 138 offline checks, no GPU
python build_notebooks.py             # regenerate all 16 notebooks
python build_notebooks.py --check     # confirm they're current
```

Then re-upload. Never edit the base64 blob, or the `msc_lib.py` that appears in `/kaggle/working` — both are overwritten on the next bootstrap.

`msc_core.py` is embedded the same way and is the single source of truth for every statistic. `msc_lib` imports it and never reimplements one — a second copy of `compute_msc` that drifts by one index is exactly the kind of bug that produces a plausible-looking wrong answer.

---

## 10. Quick reference

```python
sess = msc.Session(account='acct1', phase='p1', dataset='cifar100',
                   worker_id=0, num_workers=6, shard_mode='cost')
sess.audit_repos()                           # what is actually on HF, per run
sess.prepare_data()                          # locate/fetch CIFAR-100
sess.sync_state()                            # scoped pull + log repair
cfg  = sess.config('resnet32x4', seed=1)     # recipe + run_id + config_hash
plan = sess.plan([c['run_id'] for c in cfgs])# this worker's share
sess.run_all(cfgs)                           # plan + train + session-limit break
sess.run_all(cfgs, fn=sess.oracle)           # same, for measurement
sess.budgets('resnet32x4')                   # FLOPs table
sess.status()                                # team-wide run log
sess.finish()                                # final blocking flush

msc.shard_report(run_ids, 6, mode='cost')    # check the split before starting
msc.free_mb(sess.work)                       # disk
msc.ltt_min_calibration_n(0.03, 0.05)        # risk-control sample requirement
msc.phase0_decision(rho, T, dR2)             # the gate table, encoded
```

**Architectures:** `resnet20 resnet56 resnet110 resnet8x4 resnet32x4 wrn_40_2 wrn_16_2 wrn_40_1 vgg13 vgg8 mobilenetv2 shufflenetv2 convnext_femto vit_tiny mixer_nano`
