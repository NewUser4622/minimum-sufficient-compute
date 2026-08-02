# Engineering Specification

> **Status note (updated after implementation).**
> This document remains the contract for the **checkpoint schema (§3)**,
> **multi-account coordination (§4)**, **dataset acquisition (§6)**,
> **benchmark reference numbers (§7)** and **reproducibility requirements (§8)**.
>
> Three sections have been superseded by implementation experience:
>
> | Section | Superseded by | Why |
> |---|---|---|
> | §2 HF repository layout (two repos) | `06_DATA_SCHEMA.md` | HF's write limit is **per user**, not per repo, so two repos doubled commit consumption for no benefit. Now one dataset repo, one folder per run. |
> | §4 claim protocol (as the primary mechanism) | `07_REPLICATION_PLAYBOOK.md` §7 | Claims are now a *recovery* mechanism. Primary allocation is a deterministic cost-balanced split needing no coordination. The shared `runs.jsonl` also lost writes (HF has no append) and is now sharded per writer. |
> | §9 notebook structure (one per phase) | `04_NOTEBOOK_RUNBOOK.md` | 16 granular notebooks instead of 5, so no single notebook is too long to finish in a session. |
>
> Everything else stands as written.

Infrastructure contract for the MSC project. Written so that six people on six Kaggle accounts can run ~1,200 GPU-hours of experiments without colliding, losing work, or producing numbers nobody can trace back to a config.

---

## 1. Design constraints

| Constraint | Consequence |
|---|---|
| Kaggle session: ~9–12 h, dual T4, 20 GB persistent `/kaggle/working` | Every run must checkpoint and resume. Nothing may depend on finishing in one session. |
| Kaggle scratch: ~1 TB under `/kaggle/temp` (session-local, wiped) | Datasets and intermediate tensors go to scratch; only artifacts go to `/kaggle/working`. |
| HF Hub write rate limit | Push on a schedule, not on every epoch. See §5. |
| 6 accounts, no shared filesystem | HF Hub is the only coordination substrate. Claim protocol required. |
| Sessions die without warning | Every push must be atomic and idempotent. Assume the process is killed at the worst moment. |

---

## 2. Hugging Face repository layout

Two repos, deliberately separated.

**`Shanmuk4622/msc-kd`** (model repo) — weights and checkpoints.
**`Shanmuk4622/msc-kd-data`** (dataset repo) — per-sample tables, logs, metrics, figures.

The split matters because per-sample MSC tables are the scientific artifact and are queried constantly during analysis; checkpoints are large and touched rarely. Keeping them apart makes the analysis repo cloneable in seconds.

```
msc-kd/                                    # model repo
├── runs/
│   └── {run_id}/
│       ├── config.yaml                    # frozen at run start, never edited
│       ├── ckpt_last.pt                   # rolling, overwritten
│       ├── ckpt_best.pt                   # best val acc
│       ├── exit_heads.pt                  # post-hoc, backbone frozen
│       └── STATUS.json                    # {state, epoch, host, updated_at}
└── README.md

msc-kd-data/                               # dataset repo
├── registry/
│   ├── runs.jsonl                         # append-only run ledger
│   └── claims/{run_id}.json               # claim files, see §4
├── budgets/{arch}.json                    # FLOPs per configuration
├── per_sample/{run_id}/
│   ├── test.parquet                       # the scientific artifact
│   └── train_holdout.parquet
├── logs/{run_id}/
│   ├── history.csv                        # per-epoch metrics
│   └── energy.csv                         # NVML samples
├── analysis/
│   ├── msc_tables/                        # computed MSC per (run, axis, tau)
│   ├── transfer/                          # T(A,B) matrices
│   └── figures/
└── paper/
    ├── tables/
    └── figures/
```

**`run_id` format:** `{phase}-{arch}-{dataset}-{method}-s{seed}`
Examples: `p0-r32x4-c100-base-s1`, `p3-r32x4to r8x4-c100-mscKD-s2`

Deterministic and collision-free by construction. Never auto-generate a UUID — you will not be able to find anything six weeks from now.

---

## 3. Checkpoint schema and the resumability contract

A checkpoint is only useful if resuming from it produces **bit-identical continuation**. Saving model weights alone does not achieve this. Save all of:

```python
{
  "run_id":        str,
  "epoch":         int,          # last COMPLETED epoch
  "global_step":   int,
  "model":         state_dict,
  "optimizer":     state_dict,
  "scheduler":     state_dict,
  "scaler":        state_dict,   # AMP GradScaler — omitting this changes loss scale on resume
  "rng": {
      "python":    random.getstate(),
      "numpy":     np.random.get_state(),
      "torch":     torch.get_rng_state(),
      "cuda":      torch.cuda.get_rng_state_all(),
  },
  "best_metric":   float,
  "config_hash":   str,          # sha256 of config.yaml
  "wall_seconds":  float,
  "energy_joules": float,        # cumulative, survives restarts
}
```

Three failure modes this prevents, all of which silently corrupt results:

- **Missing scaler state** → AMP loss scale resets, first post-resume steps behave differently.
- **Missing RNG state** → augmentation and shuffling sequence changes, so a resumed run ≠ an uninterrupted run and your seeds are meaningless.
- **Missing `config_hash`** → you resume a run under a config that has been edited since, and no one notices.

**On resume, assert `config_hash` matches.** Fail loudly on mismatch. A silent mismatch is worse than a crash.

### Atomic write

```python
torch.save(state, path + ".tmp")
os.replace(path + ".tmp", path)      # atomic on POSIX
```

Never write directly to `ckpt_last.pt`. A session killed mid-write leaves a truncated file, and the run is lost.

### Epoch boundary only

Checkpoint at epoch boundaries, not mid-epoch. Mid-epoch resumption requires dataloader state, which is not reliably serialisable with workers. Epochs are ~45 s here — the granularity is fine.

---

## 4. Multi-account coordination

No locking primitive exists on HF Hub, so use an **optimistic claim protocol**. With six people it is sufficient.

**Before starting a run:**
1. Pull `registry/runs.jsonl`.
2. If `run_id` exists with state `running` and `updated_at` within 2 hours → **skip**, someone owns it.
3. If state is `running` but stale (>2 h) → the session died; you may take it over, resuming from `ckpt_last.pt`.
4. Write `registry/claims/{run_id}.json` with `{account, started_at, session_id}`.
5. Append to `runs.jsonl`, state `running`.

**Heartbeat:** update `STATUS.json` every push cycle. Staleness detection depends on it.

**On completion:** append state `done` with final metrics.

Keep a single pinned Google Sheet or Discord channel mirroring `runs.jsonl` for humans. The ledger is the source of truth; the sheet is for coordination at a glance.

### Account role assignment (suggested)

| Account | Role |
|---|---|
| 1 | Phase 1 atlas — ResNet/WRN family |
| 2 | Phase 1 atlas — VGG/Mobile/ShuffleNet |
| 3 | Phase 1 atlas — ConvNeXt/ViT/Mixer (slowest, needs the most babysitting) |
| 4 | Phase 3 method runs — CIFAR-100 |
| 5 | Phase 3 method runs — baselines B4–B9 (the reimplementations; hardest, assign to whoever is most careful) |
| 6 | Tiny ImageNet / ImageNet-100 + rerun queue for failures |

Baseline reimplementation (account 5) is the highest-risk job in the project. An unfaithful SAFE-KD reimplementation invalidates the paper's central comparison. Assign accordingly and have a second person review it.

---

## 5. Hugging Face push policy

The rate limit is real and hitting it mid-run costs a session. Policy:

**Push triggers:**
| Trigger | What is pushed |
|---|---|
| Every 30 min (timer) | `ckpt_last.pt`, `history.csv`, `STATUS.json` |
| Major stage completion | full run directory + per-sample Parquet |
| New best validation metric | `ckpt_best.pt` |
| `KeyboardInterrupt` / `SIGTERM` | **immediate full push**, then exit |
| Session-end detection (elapsed > 8.5 h) | immediate full push, mark `state: paused` |

**Implementation notes:**

- Register both `atexit` and a `SIGTERM` handler. Kaggle sends `SIGTERM` before killing a session; catching it buys enough time for a final push.
- Wrap `KeyboardInterrupt` around the training loop so a manual stop always flushes.
- Use `huggingface_hub.CommitOperationAdd` with a **single batched commit** per push, not one commit per file. Six files in six commits is six times the rate-limit consumption for no benefit.
- Maintain a **push queue with backoff**: on `HfHubHTTPError` with 429, sleep with exponential backoff (60 s, 120 s, 240 s, cap 600 s) and retry. Never let a failed push kill training — log it and continue; the next cycle will catch up.
- Deduplicate: skip pushing a file whose sha256 is unchanged since the last successful push. Rolling checkpoints change every time, but `config.yaml`, `budgets.json`, and completed logs do not.
- Budget: at 30-minute cycles, one run over a 9-hour session makes ~18 pushes. Six concurrent accounts → ~108 commits/hour across the org. Stay under this by keeping the timer at 30 min and batching commits.

**Token:** `HF_TOKEN` in Kaggle Secrets, per account. Every account needs write access to both repos — either add each as a collaborator or use an org.

---

## 6. Dataset acquisition

Per team preference, pull from Kaggle rather than the origin servers — it is materially faster inside Kaggle.

| Dataset | Source |
|---|---|
| CIFAR-10 / CIFAR-100 | Kaggle dataset mirror, or `torchvision` with `download=True` into `/kaggle/temp` |
| Tiny ImageNet | Kaggle dataset (`tiny-imagenet`), attach to notebook |
| ImageNet-100 | Kaggle mirror; **record the exact class list in `config.yaml`** — no canonical split exists and results are not comparable without it |

Extract to `/kaggle/temp` (scratch, ~1 TB), never `/kaggle/working` (20 GB, and it is your artifact space). Cache the extracted form as a Kaggle Dataset once so subsequent sessions skip extraction.

---

## 7. Benchmark reference numbers

Compare against published values rather than re-running baselines. CIFAR-100 top-1 (%), standard CRD/DKD recipe:

| Teacher → Student | Teacher | Student | KD | CRD | ReviewKD | DKD |
|---|---|---|---|---|---|---|
| resnet56 → resnet20 | 72.34 | 69.06 | 70.66 | 71.16 | 71.89 | 71.97 |
| resnet110 → resnet32 | 74.31 | 71.14 | 73.08 | 73.48 | 73.89 | 74.11 |
| resnet32x4 → resnet8x4 | 79.42 | 72.50 | 73.33 | 75.51 | 75.63 | 76.32 |
| wrn-40-2 → wrn-16-2 | 75.61 | 73.26 | 74.92 | 75.48 | 76.12 | 76.24 |
| wrn-40-2 → wrn-40-1 | 75.61 | 71.98 | 73.54 | 74.14 | 75.09 | 74.81 |
| vgg13 → vgg8 | 74.64 | 70.36 | 72.98 | 73.94 | 74.84 | 74.68 |
| vgg13 → MobileNetV2 | 74.64 | 64.60 | 67.37 | 69.73 | 70.37 | 69.71 |
| ResNet50 → MobileNetV2 | 79.34 | 64.60 | 67.35 | 69.11 | 69.89 | 70.35 |
| resnet32x4 → ShuffleNetV1 | 79.42 | 70.50 | 74.07 | 75.11 | 77.45 | 76.45 |

Reproduce **only** what you modify. If your resnet32x4 lands below ~78.5% or your reproduced KD number is off by more than ~0.5, fix the recipe before generating any MSC tables — MSC computed from an undertrained model is meaningless.

Numbers vary ±0.3–0.5 across papers due to recipe and seed differences; cite the source you take each from (DKD paper / mdistiller repo).

---

## 8. Reproducibility requirements

Non-negotiable, because the KD field has a documented reproducibility problem and this project's credibility rests on rigour rather than on a large accuracy delta:

1. **Every number in the paper maps to a `run_id`.** Maintain `paper/tables/*.csv` with a `run_ids` column.
2. **`config.yaml` frozen at run start**, hashed, checked on resume, pushed with the checkpoint.
3. **Seeds set for python / numpy / torch / cuda**, and `torch.use_deterministic_algorithms(True)` where it does not cost more than ~10% throughput. Where determinism is disabled, record that in the config.
4. **Environment capture:** `pip freeze`, CUDA version, driver version, GPU model into `logs/{run_id}/env.json`. T4 sessions vary; record which you got.
5. **Identical hyperparameter search budget across methods**, logged as a count of trials. This is what the segmentation critique (arXiv 2309.03659) showed to be decisive: distillation gains vanished under sufficient tuning of baselines.
6. **≥3 seeds** for every headline number. Report mean ± std. Single-seed numbers do not go in the paper.

---

## 9. Notebook structure

One notebook per phase, parameterised by a config cell at the top. Structure:

```
Cell 1   Config + secrets + HF login + run_id construction
Cell 2   Registry claim (abort if already owned)
Cell 3   Environment capture + dataset acquisition to /kaggle/temp
Cell 4   Model + budget-configuration construction
Cell 5   Resume logic — pull ckpt_last.pt if it exists, verify config_hash
Cell 6   Training loop (checkpoint at epoch boundary, 30-min push timer,
         signal handlers registered)
Cell 7   Exit-head training (backbone frozen)
Cell 8   Oracle sweep -> per-sample Parquet
Cell 9   Final push + registry state -> done
```

Cell 5 must run correctly on a fresh session with no local state — that is the entire point. Test it by deliberately killing a session at epoch 30 and confirming the resumed run's loss curve continues smoothly rather than jumping.

---

## 10. Analysis pipeline

Analysis runs locally or in a CPU-only Kaggle session; it needs no GPU.

```
per_sample/*.parquet
        │
        ▼
  compute_msc()          → analysis/msc_tables/{run_id}_{axis}_{tau}.parquet
        │
        ├──► seed_ceiling()            → ρ_S^seed per architecture
        ├──► disattenuated_transfer()  → T(A,B) matrix + bootstrap CI
        ├──► top_decile_jaccard()      → hard-tail agreement
        ├──► irreducibility()          → partial ρ_S, ΔR² + CI
        └──► axis_structure()          → PCA / factor loadings
                    │
                    ▼
            paper/figures/, paper/tables/
```

`msc_core.py` implements all of these. It depends only on numpy, scipy, pandas, and scikit-learn — no torch — so analysis is fast, portable, and testable.

---

## 11. Order of build

1. `msc_core.py` — provided; run its self-test first
2. Phase 0 training script (no HF infrastructure — save locally, upload manually; four runs do not justify building the machinery)
3. **Phase 0 decision meeting**
4. *Only then:* full notebook infrastructure with registry, push policy, resume logic
5. Phase 1 atlas
6. Method

Building step 4 before step 3 is the single most likely way to waste a month.
