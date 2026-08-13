# ImageNet-100 — LIVE STATUS

**Updated:** 2026-08-13 · self-test 423/423 · all notebook cells parse as Python 3.10

## PRE-REGISTERED GATES — measured

| gate | value | threshold | verdict |
|---|---|---|---|
| ρ_seed ≥ 0.60 | resnet50 **0.822** · vit **0.649** | 0.60 | **PASS** |
| shuffled control \|z\| < 5 | **2.30** | 5.0 | **PASS** |
| Q4 partial ρ ≥ 0.30 | **0.282** | 0.30 | **MISS by 0.018** |

Q4 ΔR² = **0.0411**, CI [0.0352, 0.0468] — excludes zero. MSC *does* add
information beyond the 7-score difficulty battery; the partial Spearman just
misses its threshold. This is the gate MSC-KD (NB5) sits downstream of, and it
must be reported as a miss, not rounded up.

**Q3 transfer, resnet50 ↔ vit_small_p16 (CNN–transformer):**
ρ_raw 0.468 → **T = 0.640** [0.614, 0.664] after disattenuation.
Shuffled control gives T = 0.037 — a 17× separation, so the alignment is real.
Jaccard@top-10 = 0.259.

This is the "where are we" file. One page, updated every session.

- **What happened and why** → `22_IN100_LAB_NOTEBOOK.md` (defect log, D-37…D-70)
- **Symptom → cause → fix** → `23_IN100_RUNBOOK.md` (look here first when something breaks)
- **The plan** → `20_IN100_PORT_PLAN.md` · **What changed from CIFAR** → `21_IN100_ENGINEERING_DELTA.md`

---

## 1. The question this run exists to answer

On CIFAR-100, ViT and Mixer showed **low seed-reliability** — ρ_seed 0.547,
against 0.62–0.73 for CNNs. MSC either measures something stable about an
architecture, or it partly measures the seed.

**Does that gap survive at ImageNet scale, or was it a small-data artifact?**

Everything below is scaffolding for that one comparison. ρ_seed is the
denominator of every transfer claim in the paper; if it is unstable, nothing
above it means anything.

---

## 2. Where we are right now

```
NB1 Setup     ██████████ done    data packed, 129,395 images, fingerprint 2b6269ef…
NB2 Train     ██████████ done    4/4 runs, 100 epochs each
NB3 Measure   ██████████ done    4/4 measured, per-sample tables written
NB4 Analysis  ██████████ done    Q1 Q2 Q3 Q4 all written to analysis/
NB5 Method    ░░░░░░░░░░ ready   CONFIRM gate clean · 18 runs, seed-major
                                 first comparison after 6 runs, not 18 (D-75)
```

## FIRST RESULT — the question, answered for the pilot

**ρ_seed at τ=0.1** (the noise ceiling: Spearman between two seeds of the same
architecture; the denominator of every transfer claim)

| architecture | CIFAR-100 | **ImageNet-100** | change |
|---|---|---|---|
| CNN (`resnet50`) | 0.62–0.73 | **0.822** | up |
| ViT (`vit_small_p16`) | 0.547 | **0.649** | up |
| **gap (CNN − ViT)** | ~0.10 | **0.173** | **wider** |

Read carefully, because it cuts both ways:

- **The ordering survives.** ViT is still measurably less seed-reliable than
  the CNN at 40× the data and 49× the pixels. The CIFAR finding was not purely
  a small-data artifact.
- **But both rose**, and ViT's 0.649 at ImageNet scale now sits *inside* the
  range CNNs occupied on CIFAR (0.62–0.73). "ViT is unreliable" is not
  scale-invariant; "ViT is less reliable *than a CNN trained alongside it*" is
  what held.
- **The gap widened rather than closed** (0.10 → 0.17), which is the opposite
  of the small-data-artifact hypothesis.

Jaccard@top-10 tells the same story more starkly: 0.62 (resnet50) vs 0.28
(vit) — the two ViT seeds barely agree on *which* samples are expensive.

**Do not write this up yet.** n=2 seeds means one pair per architecture and no
error bar (§6.1), and the ViT arm has a 38-point train/val gap (§6.2). Both are
reasons this number could move. Two architectures is a pilot, not the atlas.

**Phase 0 is a 2-architecture pilot, not the study.** The 8-architecture
atlas is phase `p1` and has not been started.

### Trained runs (NB2 output — all verified on disk)

| run | state | epochs | best top-1 | img/s | GPU-h | train artifacts | measured |
|---|---|---|---|---|---|---|---|
| `resnet50-s1` | completed | 100 | **82.62%** | 80 | 41.5 | OK | **yes** |
| `resnet50-s2` | completed | 100 | **82.12%** | 80 | 41.5 | OK | **yes** |
| `vit_small_p16-s1` | completed | 100 | **60.56%** | 603 | 5.7 | OK | **yes** |
| `vit_small_p16-s2` | completed | 100 | **61.01%** | 595 | 5.7 | OK | **yes** |

Seed pairs agree to 0.50 (resnet50) and 0.45 (vit) points.

The `img/s` column is why `resnet50` cost 7× what `vit` did — see §5.

---

## 3. What each notebook does and produces

| NB | does | writes per run | reruns safely? |
|---|---|---|---|
| **NB1** | pack JPEGs → uint8 memmap; build FLOPs budgets | `msc_data/in100/*`, `budgets/*.json` | yes, skips if packed |
| **NB2** | train backbones | `epochs.csv`, `ckpt_last/best.pt`, `train_dynamics.parquet`, telemetry, `summary.json` | yes, resumes mid-run |
| **NB3** | exit heads + final eval + **oracle sweep** | `exit_heads.pt`, `final.csv`, `confusion_matrix.csv`, `per_class.csv`, `exit_metrics.csv`, **`per_sample/test.parquet`**, `train_holdout.parquet`, `meta.json` | yes, skips measured runs |
| **NB4** | Q1–Q4 analysis | `analysis/*.csv`, `tables/`, `paper/figures/` | yes, pure read |
| **NB5** | MSC-KD: trains **18 new runs** | new `p3-*` runs | yes, resumes |

**`per_sample/test.parquet` is the scientific artifact.** Everything in the
paper is computed from it. Until NB3 writes it, no analysis exists.

---

## 4. Order — and the answer to "can I go to NB5?"

```
NB1 → NB2 → NB3 → NB4 → NB5
             ^you are here
```

**NB3 is done** — all four runs measured, per-sample tables on disk. NB5 is
now *technically* unblocked (it needs a measured `resnet50` teacher, which
exists). Finish NB4 first: it is minutes, and Q3/Q4 tell you whether the
per-sample tables are sound before you spend days distilling from them.

NB5 also **trains 18 new runs** (3 students × 3 seeds × 2 arms). Two of those
architectures have never been timed at the corrected `channels_last` setting,
so the cost is unknown. Run `python tools/conv_sweep.py --arch deit_small`
first — 2 minutes — before committing days to it.

### Every time you re-run a notebook after I change something

1. **Close it in Jupyter WITHOUT saving.** Jupyter writes the open tab to disk
   when you run it, so a stale tab silently overwrites the regenerated file
   (D-68).
2. `python build_notebooks_in100.py`
3. Reopen · Kernel → Restart · Run All
4. Cell 1 must print `msc_lib build <hash> verified, and current with src/`.
   If it says `STALE NOTEBOOK`, step 1 did not happen.

---

## 5. Facts established by measurement (not estimate)

| finding | evidence |
|---|---|
| `channels_last` is **6.7× slower** on this card | `conv_sweep`: 81.6 → 550.3 img/s, RTX 4000 Ada / cuDNN 9.1. Default is now `False`. |
| The loader was **never** the bottleneck | `dataload_frac = 0.000` across all 4 runs |
| Augmentation is not the bottleneck | `augment_frac` 1.2% (resnet50), 9.4% (vit) |
| The GPU was busy, not starved | 108 W of 130 W, `gpu0_util` 98.5% |
| ViT-S/16 is 7.5× faster than ResNet-50 here | 106 vs 796 ms/batch, same loader |
| Disk is fast | RAM cache reads 23.7 GiB at 1.5–3.4 GiB/s |

**Stale numbers.** Every throughput figure for `vgg16`, `resnet18`,
`shufflenetv2_in`, `swin_tiny`, `convnext_tiny` was taken under the slow layout
and is understated by an unknown factor (`IN100_PENDING_REMEASURE`). Any atlas
budget built on them is wrong in the pessimistic direction.

---

## 6. Open scientific questions — read before writing the paper

1. **Only 2 seeds per architecture.** ρ_seed from a single pair is a point
   estimate with no error bar, on the quantity every transfer claim divides by.
   The plan called for 3.
2. **ViT overfits hard**: train 98.7% vs val 60.6% — a 38-point gap, against
   16 points for ResNet-50. That is the no-strong-augmentation arm behaving as
   designed (`DEIT_RECIPE` covers `deit_small` only), but a badly-overfit ViT
   may not be the right input to a seed-reliability comparison. Decide this
   deliberately, and write down which way and why.
3. **Equal epochs for all architectures** is a confound the user chose
   knowingly. It must be stated in the paper, not discovered by a reviewer.
4. **The pilot is 2 architectures.** No CNN/attention conclusion can rest on
   `resnet50` vs `vit_small_p16` alone.

---

## 7. Defects, most recent first

Full analysis in `22_IN100_LAB_NOTEBOOK.md`. Fixed unless marked.

| # | one line | cost |
|---|---|---|
| D-71 | `require` matched a run id against an arch-keyed dict → Q3/Q4 silently empty | `KeyError: 'passed'` |
| D-70 | `np.asarray(y)` on a CUDA tensor — CIFAR yields CPU labels, IN-100 device labels | failed 40 min into the sweep |
| D-69 | checkpoint path joined to the run root; correct spelling was in dead HF code | 4 failed measurements |
| D-68 | Jupyter saved a stale notebook over the regenerated one | ~2 rounds |
| D-67 | `run_all(fn=sess.oracle)` planned as `stage='train'` → skipped everything, reported success | NB3 did nothing, twice |
| D-66 | `analyse_*_all` defaulted to `phase="p1"` internally | misleading `KeyError` in NB4 |
| D-65 | NB3/4/5 hardcoded `PHASE='p1'` while NB2 writes `p0` | silent no-op |
| D-64 | `final.csv` required at train stage, written at measure stage | false alarm on 4 healthy runs |
| D-63 | `hash_compatible` probed a reconstruction, not the on-disk record | 3 rounds |
| D-62 | stale module: fix on disk, old code running | 2 rounds |
| D-61 | `.get(k, nan)` does not fire on a present `None` | crash after training succeeded |
| D-60 | adding to `_HASH_EXCLUDE` changed every hash → orphaned checkpoints | nearly lost 90 GPU-h |
| D-59 | `channels_last` 6.7× **slower**; D-55 had enforced it | ~35 h × 2 runs |
| D-58 | measured it: convolutions, not the pipeline | — |
| D-57 | two wrong diagnoses from aggregate numbers | ~2 days |
| D-56 | RAM-resident loader (real fix, wrong premise) | — |
| D-55 | memory format applied at 1 of 16 sites (wrong direction) | — |
| D-54 | `run_all` handed a callable it cannot call | — |

**Recurring shapes, in order of how much they have cost:**

- *A test that agrees with the author instead of the program* — D-37, D-60, D-63
- *A path spelled twice, one spelling wrong* — D-16, D-23, D-69
- *A predicate answering a different question than the work* — D-31, D-67
- *A checker that fires on healthy code* — D-61, D-69, and this file's own §5
- *Silence as a failure mode* — D-65, D-67 (the expensive ones: no traceback)

---

## 8. Next actions

1. **Finish NB4** (close without saving → rebuild → reopen → Run All). Q1 and
   Q2 already produced results; D-71 unblocks Q3 and Q4.
2. ~~`exit_metrics.csv` is 0 bytes~~ — **false alarm, my error.** It is 269
   bytes and correct (exit, depth_fraction, rho, flops, stage_cut,
   feature_dim × 5 exits). My status check rounded 269 B to "0K".
3. Decide §6.1 (2 seeds vs 3) and §6.2 (the ViT overfitting arm) **before**
   NB5 — both change what NB5 is worth doing.
4. `conv_sweep --arch deit_small` before committing to NB5's 18 runs.
5. Then either NB5 (method) or start the `p1` atlas (the actual study).
