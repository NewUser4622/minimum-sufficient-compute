# Repository Structure & Data Schema

**Supersedes** the two-repo layout in `02_ENGINEERING_SPEC.md` §2 and `03_IMPLEMENTATION_PLAN.md` §9.
**Scope:** CIFAR-100. A second dataset gets its own repo, named the same way.

---

## 1. The repository

```
Shanmuk4622/msc-cifar100          (HuggingFace DATASET repo)
```

**One repo, not two.** Two reasons, one of which was a bug:

- HuggingFace's write limit (~128 commits/hour) is **per user**, not per repo. The old design ran two uploaders each capped at 20 commits/hour, so a single account could emit 40/hr and six accounts 240/hr — nearly double the real ceiling. One repo means one commit per push cycle, and the cap now means what it says.
- A run's artifacts belong together. Splitting weights from metrics meant reading a run's history required knowing which of two repos to look in.

**Why a *dataset* repo rather than a model repo:** HuggingFace renders CSV and Parquet previews in dataset repos. Every metrics table becomes browsable in the web UI without downloading anything. Given that the per-sample tables are the scientific artifact of this project, that is worth more than the model-repo badge.

---

## 2. Per-run folder — everything in one place

```
runs/{run_id}/
│
├── config.yaml                     frozen at run start, never edited
├── config_hash.txt                 sha256, asserted on every resume
├── STATUS.json                     heartbeat: state, epoch, worker, updated_at
├── summary.json                    the whole run in one file
│
├── metrics/
│   ├── epochs.csv                  ★ per-epoch, ~150 columns  (§15.1)
│   ├── final.csv                   ★ final evaluation, one row (§15.2)
│   ├── final.json                  same, nested
│   ├── confusion_matrix.csv        100×100, true × predicted
│   ├── per_class.csv               precision / recall / F1 / support / accuracy
│   ├── inference_bench.csv         latency & throughput at several batch sizes
│   ├── calibration.csv             reliability-diagram bins
│   ├── exit_metrics.csv            per-exit accuracy & FLOPs        (NB02)
│   └── msc_summary.csv             MSC mean/std per axis per tau    (NB02)
│
├── telemetry/
│   ├── energy_samples.csv          NVML power at 10 Hz, one row per GPU per sample
│   ├── system_samples.csv          util / temp / clock / CPU / RAM at 1 Hz, per GPU
│   ├── step_traces.jsonl           per-step timing + loss, downsampled per epoch
│   └── console.log                 stdout mirror
│
├── per_sample/
│   ├── test.parquet                MSC measurement, 10,000 rows     (NB02)
│   ├── train_holdout.parquet       5,000 rows                       (NB02)
│   ├── train_dynamics.parquet      EL2N + forgetting, 50,000 rows   (during training)
│   └── meta.json                   budgets, order hash, config hash
│
├── checkpoints/
│   ├── ckpt_last.pt                model+optim+sched+scaler+ALL RNG+energy+wall
│   └── ckpt_best.pt
│
├── exit_heads.pt                   trained exit heads          (NB02 / NB08)
│                                   ^ NOTE: at the run root, NOT in checkpoints/
│
└── env/
    ├── environment.json            pip freeze, CUDA, driver, GPU models, hostname
    └── budgets.json                FLOPs per compute configuration
```

## 3. Repository root

```
registry/
  events/{account}_w{id}_{session}.jsonl   one shard per writer — never overwritten
  claims/{run_id}.json
  plans/{account}_w{id}of{n}_{phase}.json

tables/                             cross-run concatenations, rebuilt by NB15
  all_epochs.csv                    every epoch of every run, one file
  all_final.csv                     every run's final evaluation, one row each
  atlas_summary.csv                 mean ± std across seeds

analysis/                           q1_*.csv … q5_*.csv, phase0_decision.json
paper/                              figures/*.png, tables/*.csv, provenance.csv
README.md                           auto-generated model card
```

---

## 4. Where files live while running

| Location | Size | Holds |
|---|---|---|
| `/kaggle/temp/msc/runs/{run_id}/` | ~1 TB | **everything** — the live run directory |
| `/kaggle/working/` | 20 GB | essentially nothing |

Previously the run directory sat on `/kaggle/working` and had to be policed against the 20 GB ceiling. It now lives on scratch, so a 240-epoch run with 10 Hz power sampling and full step traces is never disk-constrained. HuggingFace is the permanent store either way, so losing scratch on session end costs nothing beyond the last push interval.

---

## 5. Push policy

| Trigger | What goes |
|---|---|
| **Every 30 min** | `metrics/*`, `STATUS.json`, `config*`, `summary.json`, checkpoints, registry |
| Every 10 epochs | the above **plus** `telemetry/*` and `per_sample/*` |
| New best accuracy | checkpoints (suppressed if <3 epochs since last push) |
| Stage completion | everything |
| **Stop / SIGTERM / crash** | everything, immediately, blocking until confirmed |
| 8.5 h elapsed | everything, mark `paused` |

Between pushes, files accumulate on scratch. Each push is **one commit** containing every changed file.

The 30-minute cycle carries the metrics but not the bulky raw telemetry, because `energy_samples.csv` reaches several MB and re-uploading it every half hour would churn LFS storage for data nobody reads until the run ends. Raw streams go at 10-epoch milestones and at completion. Worst case loss on an unclean kill is one milestone of raw samples; the metrics and checkpoints are never more than 30 minutes stale.

**Rate limit:** one shared token bucket per HF token, process-wide, hard-capped at 20 commits/hour. Six accounts × 20 = 120 < 128. When the cap is reached the uploader sleeps until the oldest commit ages out; training never blocks.

---

## 6. `metrics/epochs.csv` — the per-epoch table

Column groups, with your §15.1 requirements marked ★.

### Identity & provenance
`run_id` · `epoch` · `global_step` · `timestamp_utc` · `unix_ts` · `account` · `worker_id` · `session_id` · `hostname` · `arch` · `family` · `dataset` · `seed` · `phase` · `method` · `config_hash`

### Learning ★
★`train_loss` ★`val_loss` ★`train_accuracy` ★`val_accuracy` · `train_accuracy_top5` `val_accuracy_top5` · ★`f1_macro` `f1_micro` `f1_weighted` · ★`precision_macro` `precision_micro` `precision_weighted` · ★`recall_macro` `recall_micro` `recall_weighted` · `balanced_accuracy` `cohen_kappa` `matthews_corrcoef` · `train_loss_min` `train_loss_max` `train_loss_std` `train_loss_median` · `best_val_accuracy_so_far` `epochs_since_best` `is_best`

### Calibration *(beyond spec — added)*
`val_ece` `val_mce` `val_nll` `val_brier` `val_confidence_mean` `val_entropy_mean`

> Calibration is the mechanism this project's Q5 claim rests on — "small students are miscalibrated, so their own confidence is a poor gate". Recording ECE every epoch costs nothing and turns that claim from an assertion into something measured.

### Loss components ★
★`loss_ce` ★`loss_kd` `loss_msc` `loss_total` · `loss_l1` · `alpha` `beta` `temperature`

★`loss_feature` ★`loss_attention` ★`loss_energy_boundary` ★`loss_counterfactual` ★`loss_pareto`

> **These five columns always exist and are written `NA` unless the corresponding term is enabled.** The current objective is `CE + α·KD + β·MSC` — three terms, two weights — because `00_RESEARCH_PROTOCOL.md` §1 identifies "seven loss terms and six λ's" as a specific rejection risk and deletes feature/attention/Pareto, and drops counterfactual for colliding with existing counterfactual-KD work.
>
> Keeping the columns means the schema is uniform and complete, and enabling any term later (`cfg['enable_feature_loss'] = True`) fills it without a schema migration. Writing a fabricated number into a column for a loss the model never computed would be worse than `NA`.

### Optimisation health
★`learning_rate` · `lr_min_group` `lr_max_group` `lr_groups_json` · `momentum` `weight_decay` · `grad_norm_mean` `grad_norm_max` `grad_norm_min` `grad_norm_p50` `grad_norm_p95` `grad_norm_p99` `grad_norm_std` · `grad_clip_value` `grad_clip_hit_frac` · `weight_norm` `update_norm` `update_to_weight_ratio` · `amp_scale` `amp_scale_decreases` · `n_batches` `n_optimizer_steps` `n_skipped_steps` `nan_or_inf_batches`

### Time ★
★`train_time_sec` ★`val_time_sec` · `epoch_time_sec` `cumulative_time_sec` · `dataload_time_sec` `compute_time_sec` `backward_time_sec` `optimizer_time_sec` `dataload_frac` · `step_time_mean_ms` `step_time_p50_ms` `step_time_p90_ms` `step_time_p99_ms` `step_time_max_ms` · `throughput_train_img_s` `throughput_val_img_s` · `samples_seen` `cumulative_samples_seen` · `eta_sec`

### GPU — **per device** ★
For each GPU `i` (dual T4 → `gpu0_*`, `gpu1_*`):

★`gpu{i}_util_mean_pct` `gpu{i}_util_max_pct` · ★`gpu{i}_mem_used_mb` `gpu{i}_mem_total_mb` `gpu{i}_mem_util_pct` · ★`gpu{i}_temp_mean_c` `gpu{i}_temp_max_c` · `gpu{i}_power_mean_w` `gpu{i}_power_max_w` · `gpu{i}_sm_clock_mhz` `gpu{i}_mem_clock_mhz` · `gpu{i}_energy_j` · `gpu{i}_throttle_reasons`

Plus PyTorch's own view: ★`vram_allocated_mb` `vram_reserved_mb` `peak_vram_mb` `vram_total_mb`

> Per-GPU separation is explicit in your spec and matters here: training uses one T4 while the second sits idle, and an aggregate would hide that.

### Host
`cpu_percent` `cpu_count` `ram_used_mb` `ram_total_mb` `ram_percent` `proc_rss_mb` `disk_free_scratch_mb` `disk_free_working_mb`

### Energy & carbon ★
★`epoch_energy_j` `epoch_energy_wh` `epoch_energy_kwh` · `cumulative_energy_j` `cumulative_energy_wh` `cumulative_energy_kwh` · ★`epoch_co2_g` `epoch_co2_kg` `cumulative_co2_g` `cumulative_co2_kg` · `carbon_intensity_g_per_kwh` · `power_mean_w` `power_max_w` `power_min_w` · `energy_per_sample_mj` · `energy_samples_n` `energy_sample_hz`

### Config echo
`batch_size` `effective_batch_size` `gradient_accumulation_steps` `amp_enabled` `num_epochs` `optimizer` `scheduler` `image_size` `num_classes` `label_smoothing` `deterministic` `msc_lib_version`

**~150 columns.** One CSV, self-describing, no cross-referencing needed.

---

## 7. `metrics/final.csv` — final evaluation

### Accuracy ★
★`top1_accuracy` ★`top5_accuracy` · ★`f1_macro` `f1_micro` `f1_weighted` · ★`precision_macro` `precision_micro` `precision_weighted` · ★`recall_macro` `recall_micro` `recall_weighted` · `balanced_accuracy` `cohen_kappa` `matthews_corrcoef` · ★confusion matrix → `metrics/confusion_matrix.csv` · per-class → `metrics/per_class.csv`

### Calibration *(added)*
`ece` `mce` `nll` `brier` `confidence_mean` `overconfidence_gap`

### Model ★
★`params_total` `params_trainable` `params_nonzero` `sparsity_pct` · ★`model_size_mb` `model_size_mb_fp16` `model_size_mb_int8` · ★`flops` `macs` `flops_per_param` · `n_layers` `n_conv_layers` `n_linear_layers`

### Speed ★
★`latency_bs1_mean_ms` `latency_bs1_median_ms` `latency_bs1_p90_ms` `latency_bs1_p99_ms` `latency_bs1_std_ms` · `latency_bs32_median_ms` `latency_bs128_median_ms` · ★`throughput_bs1_img_s` `throughput_bs32_img_s` `throughput_bs128_img_s` · `warmup_batches_discarded` `n_repeats`

> Batch-1 latency is the number that matters. Per-sample adaptive routing gives **no wall-clock gain under batched inference** unless the batch is split by route — stated in the paper, not buried.

### Energy ★
★`train_energy_kwh` `train_energy_j` · ★`inference_energy_j_per_image` `inference_power_mean_w` · ★`train_co2_kg` `inference_co2_g_per_1k_images` · `energy_per_accuracy_point` · `total_gpu_hours`

### Comparative ★
★`energy_reduction_pct` ★`accuracy_change_pts` ★`compression_ratio` · `speedup_vs_baseline` `flops_reduction_pct` · `baseline_run_id`

> These are **relative** and need a reference. For an atlas backbone the reference is its own full-compute configuration, so they read 0 / 0 / 1.0. For NB02's precision variants and NB13's students they compare against the fp32 full model, which is where they become meaningful. `baseline_run_id` records what each was measured against — without it these numbers are uninterpretable.

### MSC-specific *(added)*
`exit_accuracies_json` · `msc_mean_depth_tau0.1` `msc_std_depth_tau0.1` · `frac_irreducible_tau0.1` · `sample_order_hash` · per-axis, per-τ detail in `metrics/msc_summary.csv`

### Provenance
`run_id` `config_hash` `arch` `family` `seed` `dataset` `phase` `method` `num_epochs_planned` `num_epochs_run` `started_utc` `completed_utc` `account` `worker_id` `msc_lib_version` `torch_version` `cuda_version` `gpu_names` `driver_version`

---

## 8. Raw telemetry files

**`telemetry/energy_samples.csv`** — one row per GPU per sample, 10 Hz
`unix_ts` `datetime_utc` `monotonic_sec` `epoch` `phase` `gpu_index` `power_w`

**`telemetry/system_samples.csv`** — 1 Hz
`unix_ts` `datetime_utc` `monotonic_sec` `epoch` `gpu_index` `util_pct` `mem_util_pct` `mem_used_mb` `temp_c` `sm_clock_mhz` `mem_clock_mhz` `power_w` `throttle_reasons` `cpu_percent` `ram_used_mb`

**`telemetry/step_traces.jsonl`** — one JSON object per epoch, downsampled to ≤2000 points
`epoch` `step[]` `step_time_ms[]` `loss[]` `lr[]` `grad_norm[]`

---

## 9. What changes in the code

| Area | Change |
|---|---|
| `MSCHub` | single repo; `RunSync` writes every path under `runs/{run_id}/` |
| Rate limiter | process-wide shared token bucket keyed by token, not per uploader |
| Paths | run directory moves to `/kaggle/temp/msc/runs/{run_id}/` |
| `EpochTelemetry` | per-GPU sampling, timing breakdown, gradient percentiles, calibration |
| `evaluate()` | returns confusion matrix, per-class metrics, calibration |
| New `final_evaluation()` | model stats, latency sweep, inference energy, comparatives |
| `HISTORY_FIELDS` | ~150 columns, `NA` for disabled loss terms |
| NB02 | folds in the full final-evaluation pass |
| NB15 | builds `tables/all_epochs.csv` and `tables/all_final.csv` |

---

## 10. Migration

The existing `msc-kd` / `msc-kd-data` repos are left untouched. Current Phase 0 runs are ~half done under the old layout; finish them there or restart into the new repo — restarting costs ~6 GPU-hours and yields a clean, uniform dataset, which for a project whose contribution is partly *the artifact* is probably worth it.

`sess.audit_repos()` reports both.
