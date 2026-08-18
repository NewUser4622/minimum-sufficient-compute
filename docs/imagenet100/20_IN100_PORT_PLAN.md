# ImageNet-100 Port — Design and Run Matrix

**Status:** design frozen 2026-08-07 · implementation in progress
**Predecessor:** the completed CIFAR-100 programme, recorded in
[`09_LAB_NOTEBOOK.md`](../cifar100/09_LAB_NOTEBOOK.md) and [`10_FINAL_RESULTS.md`](../cifar100/10_FINAL_RESULTS.md)

---

## 0. The one question

CIFAR-100 produced this (§1.2 of the lab notebook, τ = 0.1, depth axis):

| | n | ρ_seed range | mean |
|---|---|---|---|
| convolutional | 13 | 0.6217 – 0.7256 | 0.676 |
| attention / MLP-mixing | 2 | 0.5470 – 0.5475 | **0.547** |

Clean separation, margin 0.074, both non-CNNs below the pre-registered 0.60 gate.

**Was that an architectural property, or a small-data artifact?** CIFAR-100 is
50,000 images at 32 px. A ViT trained from scratch on that has no business being
well-conditioned, and the finding may say more about the regime than about
attention. This port exists to answer that and nothing else. Every other design
choice below is subordinate to it.

### What would count as an answer

| outcome | reading |
|---|---|
| non-CNN ρ_seed stays ≈ 0.55 while CNNs stay ≈ 0.68 | **architectural.** The strongest version of the CIFAR claim. |
| the gap closes at ImageNet scale | **small-data artifact.** The CIFAR headline is withdrawn and *that* becomes the finding — a measured, publishable retraction of our own result. |
| both families drop, gap preserved | reliability is scale-dependent but family ordering is not. Report the curve, not the point. |
| the gap **inverts or scrambles** | the construct is unstable across regimes and the disattenuation methodology needs re-examining before anything downstream is trusted. |

All four are publishable. That is the point of the pre-registration, and per rule
12 the *first* row is the one that gets the hardest scrutiny, because it is the
one that flatters us.

---

## 1. The zoo, and why these eight

Eight architectures × 3 seeds = **24 backbone runs**. Chosen so the CNN/non-CNN
boundary is crossed *four different ways*, which is what the CIFAR design lacked.

| arch | family | role in answering the question |
|---|---|---|
| `resnet50` | resnet | CNN anchor. Published ImageNet reference exists; the recipe-validity test. |
| `resnet18` | resnet | second CNN of the same family → gives a **within-family** ρ_seed comparison at ImageNet scale. |
| `vgg16` | vgg | CNN without residuals — the across-CNN-family intermediate that makes the H3 ordering testable. |
| `shufflenetv2` | mobile | depthwise/grouped CNN. Present in the CIFAR zoo too (ρ_seed 0.6698), so it is the **one direct CIFAR↔ImageNet bridge**. |
| `vit_small_p16` | vit | plain ViT. The direct successor to `vit_tiny` (0.5475). |
| `deit_small` | vit | **geometrically identical to `vit_small_p16`.** Differs only in training recipe. |
| `swin_tiny` | swin | attention **with** hierarchical locality and a shifted-window prior. |
| `convnext_tiny` | convnext | convolution **with** transformer design language. Successor to `convnext_femto`. |

### The four crossings, and what each isolates

This is the part the CIFAR study could not do, and it is why the eight are worth
the GPU-hours.

1. **`vit_small_p16` vs `deit_small` — isolates *recipe* at fixed architecture.**
   Same patch size, same depth, same width, same head count, same parameter
   count; built by the same function with the same arguments. Only the optimiser,
   augmentation and regularisation differ.
   *If ρ_seed differs between these two, low reliability is a property of how
   attention models are trained, not of attention.* That would substantially
   reframe the CIFAR finding rather than confirm it — and no CIFAR run could have
   detected it.

2. **`swin_tiny` vs `vit_small_p16` — isolates *spatial prior* at fixed
   attention.** Both use self-attention; only Swin has locality and hierarchy.
   If reliability tracks "attention", they agree. If it tracks "weak spatial
   prior", Swin sits with the CNNs.

3. **`convnext_tiny` vs `resnet50` — isolates *design language* at fixed
   convolution.** On CIFAR, `convnext_femto` had a CNN-like **ceiling** (0.7084)
   but a transformer-like **transfer** (0.766 vs 0.912). That n=1 observation
   is open item O-14. Here it is testable against a second modern CNN regime.

4. **`shufflenetv2` — the same architecture in both studies.** Its CIFAR ρ_seed
   is 0.6698. Whatever the ImageNet number is, the *difference* is a direct
   measurement of what changing dataset scale does to this statistic, holding
   architecture exactly fixed. It calibrates every other comparison in the table.

Crossings 2 and 3 are a 2×2: {conv, attention} × {strong prior, weak prior}.
`resnet50` and `vit_small_p16` are the pure corners; `convnext_tiny` and
`swin_tiny` are the mixed ones. If the CIFAR finding is about attention, the
split runs along one diagonal; if it is about spatial prior, along the other.
**A 2×2 with a prediction on each cell is a much stronger instrument than the
`convnext_femto`-alone argument the CIFAR paper had to lean on.**

### The confound, stated plainly

Per the decision recorded on this port: **equal epochs for all eight**, and the
accuracy confound is reported rather than engineered away.

This is weaker than matching accuracy, and it must not be dressed up. ViTs
trained from scratch on 129k images at 100 epochs will land below the ResNets.
So family and accuracy will again be partly confounded, exactly as in CIFAR
§1.2, and open item **O-9 remains open**.

Three things make the equal-epoch design defensible anyway, and all three must
appear wherever the result does:

- **Schedule is no longer confounded.** On CIFAR the three modern architectures
  got 300 epochs and the CNNs 240, so family, accuracy *and* schedule length
  moved together. Here schedule is held exactly constant, removing one of the
  three.
- **The 2×2 above does not depend on accuracy ordering.** If `swin_tiny` lands
  at CNN-level reliability while sitting at ViT-level accuracy, the accuracy
  explanation is dead regardless of the marginal means. The same argument runs
  through `convnext_tiny` in the opposite direction.
- **`vit_small_p16` vs `deit_small` will differ in accuracy while sharing an
  architecture.** That pair gives a within-architecture accuracy contrast — the
  exact thing CIFAR's `convnext_femto` argument was a weak substitute for.

The pre-registered analysis therefore reports ρ_seed against top-1 **within** the
CNN group and **within** the ViT-geometry pair, not only across the whole zoo.

---

## 2. Data

Full detail, including the exact class list and split indices, is in
[`25_IN100_DATA_CARD.md`](25_IN100_DATA_CARD.md). Summary:

| | |
|---|---|
| Source | `train/` — 100 WNID folders, **129,395 JPEGs**, original ImageNet resolution |
| Subset identity | the **first 100 WNIDs of ImageNet-1k in sort order**, `n01440764`…`n01855672` |
| Class balance | 1,071 – 1,300 per class (not uniform) |
| val split | **10,000** — 100/class, stratified, fixed RNG. The primary MSC measurement split. |
| train split | **119,395** — everything else |
| train_holdout | **15,000** slice *of* train, augmentation off, not excluded from training |
| Storage | 256×256 uint8 packed memmap, ~25.4 GB |
| Train view | GPU-side RandomResizedCrop → **224**, hflip |
| Eval view | center crop 224 from 256 (the standard ImageNet protocol) |

### Three things about this subset that must be stated in the paper

1. **It is not the CMC/`ImageNet-100` of Tian et al.**, which is a *random*
   100-class subset. Ours is the first 100 WNIDs, which is a **taxonomically
   clustered** slice: fish, birds, reptiles, snakes, spiders, insects. It is
   therefore a fine-grained problem with high inter-class similarity, not a
   representative miniature of ImageNet-1k. Protocol §5.4 already requires the
   exact subset to be documented because no canonical split exists; this is that
   documentation, and the clustering is a substantive caveat, not a footnote.
2. **There is no validation split in the source.** Ours is carved from train,
   stratified, with the indices published as an artifact. Every run uses the
   same indices, enforced by a fingerprint that participates in `config_hash`
   (§4.3).
3. **Packing squares the image before augmentation.** Stored as
   shorter-side-256 → center-crop 256, so RandomResizedCrop samples within the
   central square rather than the full frame. This mildly reduces augmentation
   diversity relative to the standard pipeline. It is applied **identically to
   every architecture**, so it cannot manufacture the family effect being
   measured — but it does mean absolute accuracies are not directly comparable
   to published ImageNet-100 numbers, and no such comparison will be claimed.

### Why pack at all, on a 24-core machine

Decode is not the bottleneck on this hardware — 24 cores will decode faster than
the GPU consumes. The packing earns its keep for three other reasons:

- **Determinism of the telemetry.** `dataload_frac` is a recorded column and one
  of the five the playbook §11 calls out as unrecoverable after the fact. With a
  memmap it measures the model; with JPEG decode it measures whatever else the
  machine was doing.
- **Windows small-file I/O.** 129k separate file opens per epoch on NTFS is slow
  and highly variable in a way that would contaminate every timing column.
- **Portability.** The same artifact mounts on Kaggle as a Dataset, so the
  notebooks run unchanged in both places (§5).

---

## 3. Compute grids

`ρ(c) = FLOPs(f,c) / FLOPs(f,c_full)` is unchanged — it is the load-bearing
normalisation that makes cross-architecture comparison well-posed (protocol
§2.1), and it is dimensionless, so nothing about it is CIFAR-specific.

| axis | CIFAR-100 | ImageNet-100 | note |
|---|---|---|---|
| depth | adaptive K from the model | **adaptive K from the model** | unchanged; K is asked of the backbone, never assumed (D-01b, D-28, D-33) |
| resolution | 16, 20, 24, 28, 32 | **96, 128, 160, 192, 224** | every value divisible by 32, so ViT patch-16 grids and Swin's four-stage /32 reduction both land on integers |
| precision | int4/6/8, fp16, fp32 | **unchanged** | analytic ρ = bits/32, simulated by fake quantisation, never reported as measured latency |
| τ | 0.0, 0.1, 0.2, 0.3, 0.5 | **unchanged** | every result a curve; no conclusion may depend on τ |

The resolution grid is the only real change and it was forced: 224 × the CIFAR
fractions gives 112/140/168/196/224, and 140 and 196 are not multiples of 32.
A ViT-S/16 cannot patchify 140 px and a Swin-T cannot reduce it four times.

**Per-resolution native support is probed, not assumed.** On CIFAR,
`supports_native_resolution` was a single boolean and MLP-Mixer's failure
(D-02) took the whole axis with it. Here the budget builder tries each
resolution independently and records a per-value result, so an architecture that
manages 128–224 but not 96 contributes what it can. Per **DC-3** the
downsample-upsample **proxy remains primary for all eight**, with native as a
robustness check — measuring 7 architectures one way and 1 another would make
cross-architecture claims on this axis compare different quantities.

---

## 4. What changes in the code, and what does not

Per the playbook §14, steps 1–6 are infrastructure and lift wholesale. They are
dataset-agnostic and stay: uploader, shared rate limiter, sharded registry,
LPT work assignment, lifecycle guards, checkpoint/resume contract.

**One library, parameterised by dataset — not a fork.** A second copy of 9,235
lines means every future fix lands twice, and the fix that lands once is the one
that costs a week. A `DATASETS` registry supplies class count, native
resolution, normalisation, resolution grid and loader backend; every site that
previously hardcoded `32` or `100` now asks it.

### 4.1 The CIFAR-coupled surfaces, enumerated

Found by audit, not by memory. Each is a place where a literal is right for one
dataset and silently wrong for another — rule 2's failure mode.

| site | coupling | fix |
|---|---|---|
| `measure_flops` default | `(1,3,32,32)` | required argument from the dataset spec |
| `build_budget_table` | three hardcoded `(1,3,32,32)` | resolution from the spec, asserted against the model |
| `_resize_proxy` | upsamples to `32` | upsamples to the native resolution |
| `sweep_all_axes` | `if r == 32` skip | compares against native resolution |
| `RESOLUTIONS` global | CIFAR grid | per-dataset |
| `CIFARTensor` / `locate_cifar100` / `build_loaders` | pickle-based CIFAR reader | new packed-memmap backend alongside |
| `base_config` | `n_classes` dict, CIFAR recipe | `imagenet100` branch |
| `REFERENCE_ACC` | CIFAR published numbers | ImageNet-100 has **no** published references — see §4.4 |
| `ARCH_COST_HINT`, `SECONDS_PER_COST_UNIT` | CIFAR timings | re-anchored; static per DC-11 |
| `N_GPU_COLUMNS = 2` | dual T4 assumed | derived from `torch.cuda.device_count()` |
| `HF_REPO` | `msc-cifar100` | `msc-imagenet100` |
| `prediction_depth` | 5k support fine at n=10k | support size scaled and recorded |

### 4.2 The twelve rules, mapped to mechanisms

Not aspirations. Each row is a thing in the code or the build.

| # | rule | mechanism |
|---|---|---|
| 1 | dry-run the **entire** path first | `backbone_dry_run`, `oracle_dry_run`, `msckd_dry_run` — each pushes one synthetic batch through training **and evaluation and artifact write**, inside the expensive function, before any setup. Sub-second. |
| 2 | never hardcode a shape/budget/exit count | `K = len(model.feature_dims)`; input shape from the dataset spec; a self-check asserts no literal `5` or `32` survives in the budget path |
| 3 | column names are data | `build_notebooks.py` **fails generation** if any notebook column literal is absent from `HISTORY_FIELDS`/`FINAL_FIELDS` |
| 4 | no repo path as a string literal | every path via `run_layout` / named accessor; the generator greps for `runs/` string literals and refuses |
| 5 | caches answer "is it VALID?" | completion predicates check the data fingerprint and router width, not just file presence |
| 6 | invalidation understood at **all** N gates | `force_rerun` set before the claim, so `plan_work` / `can_claim` / `already_finished` clear together (the D-32 shape) |
| 7 | invariants in code, not comments | no operator-remembered flags; both MSC-KD arms loop in one pass (the D-27 fix, preserved) |
| 8 | test what you **wrote** | the preflight covers `MSCStudent`, `MSCLoss` under autocast, the packing, the split fingerprint and the staging adapters — not just the eight backbones |
| 9 | trust `resolve` only | HF verification rewritten off `list_repo_files`/tree onto per-file resolve metadata |
| 10 | draining ≠ landing | session close asks the repository, per file, and reports finished / resumable / at-risk (the D-20 three-state fix) |
| 11 | "cosmetic" needs evidence | any defect closed as cosmetic requires a recorded grep of all readers — the D-16 → D-23 failure |
| 12 | scrutinise favourable results harder | the primary hypothesis is pre-registered above **with its refutation criteria**, and the confound is stated in §1 before any number exists |

### 4.3 Data fingerprint — new, and load-bearing

The CIFAR pipeline had `sample_order_hash` to prove per-sample tables were
index-aligned. Here there is a second thing that can silently drift: **the
split**. If two runs disagree about which 10,000 images are val, their
per-sample tables are aligned by index and comparing them is meaningless.

`data_fingerprint = sha256(pack_version ‖ sorted class list ‖ val indices ‖
holdout indices ‖ stored resolution)`, written into every config, **included in
`config_hash`**, and asserted on resume. A changed pack invalidates every run
that used the old one — loudly, at gate time, not at analysis time.

This is rule 5 applied to the dataset itself: presence of a per-sample table is
not evidence it is comparable to another one.

### 4.4 There are no published references, and that changes the acceptance test

On CIFAR, the acceptance test for the whole pipeline was that all four Phase 0
runs beat their published numbers — "MSC computed from an under-trained model is
meaningless, and an under-trained model is otherwise easy to miss."

**No published from-scratch number exists for this 100-class subset at 100
epochs.** `REFERENCE_ACC` is therefore `null` for all eight, exactly as it was
for the three modern CIFAR architectures, and **no Δ is claimed for anything**.
D-14 is the cautionary case: a reference without a matching parameter count and
recipe is unfalsifiable, and comparing a full-width model to a half-width
baseline produced the largest apparent win in the atlas.

Replacement acceptance test, since something must play that role:

| check | threshold | rationale |
|---|---|---|
| train/val gap sane | val top-1 within 25 pts of train top-1 | catches a run that memorised |
| val top-1 floor | ≥ 55% for CNNs, ≥ 40% for ViT-geometry | far below expectation; catches a run that never learned, not one that learned less well |
| seed spread | 3 seeds within 2.0 pts | a seed disagreeing by more indicates a broken run, not seed noise |
| `nan_or_inf_batches` | 0 | AMP silently discards these |
| `amp_scale_decreases` | recorded, flagged if > 5% of steps | |
| `update_to_weight_ratio` | in [1e-4, 1e-2] at epoch 10 | catches a dead or exploding LR before epoch 100 |

The last three are the columns playbook §11 calls out as impossible to recover
afterwards, promoted here from telemetry to gate because there is no published
number to fall back on.

---

## 5. Where this runs

**Primary: local, single RTX 4000 Ada (20 GB), 24 cores, 63 GB RAM, 773 GB free.**

Notebooks stay Kaggle-compatible and HuggingFace remains the permanent store —
a local machine can crash, and "we only train once" applies identically. The
library detects its environment rather than being told:

| | Kaggle | local |
|---|---|---|
| work root | `/kaggle/working` | repo-relative, configurable |
| scratch | `/kaggle/temp` | configurable data root |
| session watchdog | 8.5 h | off by default, still available |
| GPU columns | `device_count()` | `device_count()` — **1** here |
| dataloader workers | `cpu_count()` | `cpu_count()` |
| interrupt | SIGTERM + atexit | SIGBREAK + atexit (Windows has no real SIGTERM) |
| push cadence | 30 min | 30 min |
| push on stop | immediate | immediate |

Two Windows-specific hazards to pre-empt, both of which would present as
mysterious data loss rather than as errors:

- **`os.replace` is atomic on NTFS for same-volume renames but fails if the
  destination handle is open.** The atomic-write helpers get a bounded retry.
  Without it a checkpoint write fails at the exact moment it matters.
- **`num_workers > 0` uses spawn, not fork.** A memmap opened in the parent is
  not inherited; each worker must open its own. Otherwise the loader either
  crashes or, worse, silently serves zeros.

---

## 6. GPU budget

Anchored on `resnet50` @224, AMP, channels_last, packed memmap ≈ **420 img/s**
on this card. Everything else scaled by measured FLOPs and adjusted for kernel
efficiency — depthwise and windowed-attention models run well below their FLOPs
share, which CIFAR §1.9 measured as an 18× spread in time-per-GFLOP.

**These are estimates and are labelled as such.** Per D-10 the first guessed cost
table was 40% low; per DC-11 measured timings refine the *display* and must never
reach the assignment. `NB00` re-anchors them after the first real epoch.

Train split = 119,395 images/epoch.

| arch | est. img/s | s/epoch | at 100 ep, ×3 seeds |
|---|---|---|---|
| `resnet18` | 900 | 133 | 11.1 h |
| `shufflenetv2` | 950 | 126 | 10.5 h |
| `resnet50` | 420 | 284 | 23.7 h |
| `vit_small_p16` | 380 | 314 | 26.2 h |
| `deit_small` | 380 | 314 | 26.2 h |
| `convnext_tiny` | 300 | 398 | 33.2 h |
| `swin_tiny` | 260 | 459 | 38.3 h |
| `vgg16` | 150 | 796 | 66.3 h |
| | | **2,824 s** | **235 h** |

| stage | GPU-h at 60 ep | at 80 ep | **at 100 ep** |
|---|---|---|---|
| atlas training (24 runs) | 141 | 188 | **235** |
| measurement (24 runs, NB08) | 22 | 22 | **22** |
| MSC-KD (18 runs, Q5) | 86 | 114 | **143** |
| **total** | **249** | **324** | **400** |
| wall-clock, continuous | ~10.4 d | ~13.5 d | **~16.7 d** |

`EPOCHS` is **one named constant**. The default is 100. Lowering it is the single
lever if the budget binds, and it costs accuracy, not validity — ρ_seed is a
rank correlation between two seeds of the *same* recipe, so it remains
well-defined at any epoch count. What a lower count costs is the strength of the
"these are converged models" claim, which is exactly what §4.4's replacement
acceptance test has to carry.

`vgg16` alone is 28% of the training budget for one across-CNN-family data point.
If the budget binds, dropping it is a cheaper cut than dropping epochs — it
weakens Q3's family ordering but leaves the Q1 headline untouched. Recorded here
so the trade is explicit rather than discovered at hour 200.

---

## 7. Run matrix

| phase | content | runs | est. GPU-h |
|---|---|---|---|
| **p0** | pilot: `resnet50` + `vit_small_p16`, 2 seeds each | 4 | ~33 |
| **p1** | atlas: 8 arch × 3 seeds | 24 | 235 |
| **p1b** | exit heads + oracle sweep, all 24 | — | 22 |
| **p3** | MSC-KD: 3 students × 3 seeds × 2 arms | 18 | 143 |

**Phase 0 is not optional and is not a formality.** It is 4 runs and ~33 GPU-h
against a 400 GPU-h programme, and it answers the only question that matters
before committing the rest: *does the ρ_seed gap survive at all?* One CNN and one
ViT, two seeds each, gives one ceiling per family. If they land at
0.68 / 0.55 the CIFAR result is reproducing and the atlas is worth building. If
they land at 0.68 / 0.67 the headline is already answered — the gap was a
small-data artifact — and the remaining 20 runs are a different, cheaper paper.

Gate, pre-registered now:

| Phase 0 outcome | action |
|---|---|
| ViT ceiling < CNN ceiling − 0.05 | **CIFAR finding reproducing.** Build the full atlas. |
| \|difference\| ≤ 0.05 | **artifact.** Retract the CIFAR headline; atlas shrinks to the 2×2 (resnet50, vit_small_p16, swin_tiny, convnext_tiny) and the paper becomes about scale-dependence. |
| ViT ceiling > CNN ceiling + 0.05 | inversion. Stop and audit the measurement before spending anything further. |
| either ceiling < 0.40 | noise-dominated at this scale. Coarsen the budget grid once, then re-gate. |

---

## 8. Build order

Mirrors playbook §14 — each step verified before the next.

1. **Data card + packing tool**, with the split indices frozen and fingerprinted
2. **Library parameterisation** — `DATASETS`, every hardcoded literal removed
3. **Zoo adapters** — eight architectures into `StagedBackbone`, staging verified
4. **Dry runs** — all three, sub-second, covering evaluation and artifact write
5. **HF verification on `resolve`** — rules 9 and 10
6. **Build-time validation** — schema and path literals, generation fails on error
7. **Self-checks** — extend the 232 to cover every new surface
8. **NB00 preflight** — every model, every resolution, kill-and-resume, on the
   real machine
9. **Phase 0** — 4 runs, then re-read this document at §7 before continuing

---

## 9. Files

| file | contents |
|---|---|
| `20_IN100_PORT_PLAN.md` | this document |
| `25_IN100_DATA_CARD.md` | subset identity, split indices, packing, fingerprint |
| `21_IN100_ENGINEERING_DELTA.md` | every CIFAR→ImageNet change, with its reason |
| `22_IN100_LAB_NOTEBOOK.md` | append-only defect log, contamination analysis each |
| `23_IN100_RUNBOOK.md` | how to run, in order, with what to check after each |
| `24_IN100_STATUS.md` | results ledger, filled as runs land |

The CIFAR documents (`00`–`10`, `PAPER.md`) are the completed record of the
previous programme and are **not edited** by this port.
