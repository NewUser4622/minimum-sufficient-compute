# Engineering Delta — CIFAR-100 → ImageNet-100

Every change to `src/msc_lib.py`, with the reason. One row per surface, so that
"why does this differ from the CIFAR pipeline?" is answerable without a diff.

**The library is parameterised, not forked.** A second copy of 9,235 lines means
every future fix lands twice, and the fix that lands once is the one that costs
a week. The CIFAR path is preserved intact and remains covered by its own
self-checks.

---

## 1. What did NOT change

Per playbook §14, steps 1–6 are infrastructure and are the same in every
project. These were lifted wholesale and are untouched:

| component | why it needed nothing |
|---|---|
| `BackgroundUploader` | batching, dedup and 429 parsing are dataset-agnostic |
| `_SharedRateLimiter` | HF meters per *user*; the bucket is keyed by token (D-03) |
| `RunRegistry` | sharded events, sticky terminal states (D-04) |
| `assign_workers` / `plan_work` | LPT packing over a static cost table (D-12, DC-11) |
| `LifecycleGuard` | SIGTERM + atexit + interrupt + watchdog |
| checkpoint contract | optimizer, scheduler, scaler, all four RNG streams |
| `msc_core` statistics | ρ is dimensionless; every statistic is scale-free |
| `shuffled_control_verdict` | operates on the raw correlation against the exact permutation null (D-17) |

That list is the payoff for having built it properly the first time.

---

## 2. Dataset parameterisation

`DATASETS` registry, three accessors, and every hardcoded literal removed.

| site | was | now | failure it prevents |
|---|---|---|---|
| `measure_flops` | `input_shape=(1,3,32,32)` **default** | `shape` **required**, `ValueError` if absent | a budget table built at the wrong resolution. ρ is a ratio, so it looks entirely plausible |
| `build_budget_table` | 3 × literal `(1,3,32,32)` | `input_shape(dataset)` | same |
| `build_budget_table` | `num_classes: int = 100` | from the spec | a head of the wrong width in the FLOPs count |
| `_resize_proxy` | upsamples to literal `32` | to the native resolution, defaulting to the tensor's own | would have reshaped every ImageNet batch to thumbnail size while reporting full-resolution cost |
| `sweep_all_axes` | `if r == 32: skip` | `if r == res0` | a redundant interpolation, and a wrong skip |
| `sweep_all_axes` | `resolutions=RESOLUTIONS` | from the dataset | sweeping 16–32px while the budget table prices 96–224px. **Both halves internally consistent** |
| `preflight` | `build_model(a, 100)`, `randn(4,3,32,32)` | from the dataset | the preflight passing on a shape nothing will run |
| `msckd_dry_run` | `cfg.get("image_size", 32)` | `native_res(dataset)` | a dry run that certifies 32px for a 224px run — worse than no dry run (D-06) |
| `base_config` | `{"cifar100":100,...}[dataset]` | `num_classes_for` | KeyError on an unregistered dataset, which is the correct outcome |
| `N_GPU_COLUMNS` | literal `2` | `torch.cuda.device_count()`, floor 1 | D-36's sibling — see the lab notebook |

### The resolution grid

CIFAR: `(16, 20, 24, 28, 32)` — 0.5 to 1.0 of native.
ImageNet-100: **`(96, 128, 160, 192, 224)`**.

Not the same fractions, and the reason is a hard constraint rather than taste.
224 × the CIFAR fractions gives 112/140/168/196/224. **140 and 196 are not
multiples of 32**, and:

- ViT-S/16 must patchify the input into a square grid — needs /16
- Swin-T reduces by patch-4 then three merges = **/32**

LCM 32. The chosen grid divides cleanly by both at every point, and terminates
at native so `ρ_res` reaches exactly 1.0 (asserted at build time).

### Per-resolution native probing

CIFAR had one boolean, `supports_native_resolution`, and when MLP-Mixer failed
(D-02) it took the entire axis with it for that architecture.

At 224px the failures are **partial**: Swin-T's final stage is 7×7 at 224 and
3×3 at 96 — smaller than its own 7×7 attention window. So the budget builder now
tries each resolution independently and records `native_supported_per_res`,
falling back to the analytic quadratic model per value.

Per **DC-3** the proxy remains **primary for all eight architectures** —
measuring seven one way and one another would make cross-architecture claims on
this axis compare different quantities.

---

## 3. Data backend

New: `PackedImageDataset` + `GPUBatchLoader`, alongside the untouched
`CIFARTensor`.

### Augmentation lives in the loader

The obvious implementation puts `x = augment(x)` after every `.to(device)`.
There are **eleven** such sites: `train_backbone`, `evaluate`, the three sweeps
in `run_oracle`, `difficulty_battery`, `prediction_depth`, `train_exit_heads`,
`train_msc_kd`, and the dry runs.

**Rule 6 is exactly this shape**: when a step can be skipped at N points,
forgetting it at one is a silent wrong answer, not an error. A model trained on
normalised data and measured on un-normalised data produces a per-sample MSC
table that is well-formed and meaningless.

So `GPUBatchLoader` yields what every existing consumer already expects —
`(x_float_normalised_on_device, y, sample_idx)`. **Zero call sites changed, and
nothing downstream can forget.** Rule 7: a mechanism, not an instruction.

Crop and resize are one batched `affine_grid` + `grid_sample`, expressing
RandomResizedCrop as an affine transform — one kernel per batch rather than a
per-image Python loop, and the same code path for train (random) and eval
(fixed centre crop).

### Three details that are load-bearing

- **`sample_idx` is the global pack index**, not the position within a split.
  Per-sample tables become self-describing, val and holdout tables coexist
  unambiguously, and a split mismatch shows up as non-overlapping indices rather
  than as a plausible correlation.
- **The memmap opens lazily, per worker.** Windows spawns rather than forks, so
  a parent handle is not inherited — the worker would crash or, far worse,
  serve zeros silently.
- **The crop generator is seeded from the run seed.** Augmentation sampling has
  to be part of the reproducible RNG story or a resumed run sees a different
  stream than an uninterrupted one — the exact corruption the checkpoint
  contract's `rng` field exists to prevent.

---

## 4. The zoo

Eight architectures. **Adapters, not reimplementations**, for the convolutional
backbones: torchvision ships alongside torch and its definitions are the ones
everyone means by "ResNet-50". What is ours — and therefore what needs testing
(rule 8) — is the decomposition into `(stem, ordered blocks, classifier)`,
because that is what makes `forward_prefix(x, k)` genuinely stop at stage k
rather than run the whole network and read a mid-layer activation. An early exit
costing full compute would make every FLOPs saving in the project fictional.

| arch | source | notes |
|---|---|---|
| `resnet50`, `resnet18` | torchvision | decomposed by residual block: 16 and 8 |
| `vgg16` | torchvision `vgg16_bn` | conv stack only; conv+bn+relu grouped so a cut never lands between a convolution and its normalisation |
| `shufflenetv2_in` | torchvision | the **CIFAR↔ImageNet bridge** — same design measured in both studies |
| `convnext_tiny` | ours | reuses `_ConvNeXtBlock` / `_LayerNorm2d`, already exercised by the CIFAR checks; `stem_patch` is the only parameter that differs from femto |
| `vit_small_p16`, `deit_small` | ours | **one builder, one argument set** |
| `swin_tiny` | torchvision | `SwinBackbone` permutes NHWC→NCHW at the boundary |

### One head shape for all eight

Global average pool → Linear, everywhere.

Stock VGG-16 has a 25088→4096→4096 head worth ~124M parameters. If the final
exit carried that while exits 1..K−1 carried a GAP+Linear `ExitHead`, the
**depth-axis ρ would be measuring the head rather than the backbone** — and ρ is
what the entire project normalises by. So every architecture terminates the way
the exit heads do.

Cost: this is "VGG-16(BN) with a GAP head", not stock VGG-16. Recorded, and
harmless because no published reference is claimed for anything in this zoo.

### Swin's memory layout

torchvision's Swin speaks NHWC internally. Rather than teach `ExitHead`,
`pooled` and the FLOPs profiler about a second layout — three more places to get
it wrong — `SwinBackbone` permutes once, at the boundary where features leave
the backbone. Internals stay exactly as torchvision wrote them.

### `build_model(dataset=)` checks the zoo

A CIFAR `resnet20` fed 224px input **does not raise**. It has a stride-1 stem
and no maxpool, produces a 56×56 final feature map, runs ~40× slower than
intended, and trains to a plausible-looking number. That is the D-33 shape: a
configuration that is wrong and silent. Refused where it costs one line.

---

## 5. The recipe

| | CIFAR-100 | ImageNet-100 |
|---|---|---|
| epochs | 240 CNN / 300 modern | **100, all eight** |
| batch | 64 / 128 | 128 |
| CNN | SGD 0.05, multistep | SGD 0.1×bs/256, cosine, 5-epoch warmup |
| transformer | AdamW 1e-3, cosine, 20 warmup | AdamW 5e-4×bs/512, cosine, 5 warmup |
| label smoothing | 0 / 0.1 | 0.1 everywhere |
| references | 12 of 15 | **none** |

**Equal epochs** removes schedule length as a confounded variable. On CIFAR the
modern architectures got 300 and the CNNs 240, so family, accuracy *and*
schedule moved together — §1.2 of the CIFAR notebook had to argue "schedule
length is not the difference either" from `convnext_femto` alone.

It does **not** fix the accuracy confound. That is reported, and O-9 stays open.

### `vit_small_p16` vs `deit_small`

Same builder, same arguments, same optimiser, same LR, same weight decay, same
schedule, same warmup, same epoch count. Asserted by self-check.

Differences, and this is the exhaustive list:

| | `vit_small_p16` | `deit_small` |
|---|---|---|
| mixup / cutmix | off | 0.8 / 1.0 |
| RandomResizedCrop scale | (0.35, 1.0) | (0.08, 1.0) |
| drop-path | 0.05 | 0.1 |

If seed-reliability differs across this pair, it is a property of **training**
and not of attention — which would reframe the CIFAR finding rather than confirm
it. Building them from one function is what guarantees the comparison means
that.

Mixup is applied to backbone training only, never in `train_msc_kd`: the MSC
target is a per-sample property of a specific image, and mixing two images
produces a sample whose minimum sufficient compute is undefined.

---

## 6. Correctness machinery added

| addition | rule | replaces |
|---|---|---|
| `backbone_dry_run` | 1 | nothing — the backbone path had no dry run at all |
| `oracle_dry_run` | 1 | nothing |
| build-time assertion that dry runs are **called, and called first** | 1, 7 | a comment saying they should be |
| `budget_table_valid` | 5 | `if t and t.get("full_flops")` |
| `data_fingerprint` in `config_hash` | 5 | nothing — CIFAR's splits shipped with the dataset and could not drift |
| `resolve_meta` / `files_present` | 9 | `list_repo_files` (tree endpoint) |
| `_atomic_replace` retry | — | bare `os.replace`, which raises on Windows if any process holds the destination |
| lists + canary + floor in `_selftest` | 8 | a scalar that a tuple unpack silently destroyed (**D-37**) |
| AST-parsing source checks | — | substring greps that matched their own docstrings |

Self-checks **232 → 381**, all passing, **exit code verified against a canary**.

### Build-time name checking, in five layers

Every one was added after a real failure, and each catches what the previous
one structurally cannot. The through-line: **a name that exists is not the same
as a name used correctly.**

| layer | added after | catches |
|---|---|---|
| column literals vs `HISTORY_FIELDS`/`FINAL_FIELDS` | D-22, D-36 | a column that is not in the schema |
| repo paths must go through an accessor | D-16, D-23, D-25 | a hand-spelled `runs/...` |
| library **names** must exist | D-39 | `M.analyse_q1_all` when it was never written |
| call **signatures** must match | D-47, D-48 | `resume_acceptance_test(interrupt_after=2)` — the parameter is `kill_at` |
| result **keys** must be declared | D-51, D-52 | `res.get('passed')` when the key is `ok` |
| drive letters are forbidden | D-44 | `r'D:\...'` on a machine with no D: |

Plus, inside the library rather than at build time: an AST arity check on
internal calls, a signature check on every zoo builder, and a canary proving
the harness itself can fail.

---

## 7. Still to do

Tracked as O-23 through O-26 in
[`22_IN100_LAB_NOTEBOOK.md`](22_IN100_LAB_NOTEBOOK.md) §4.

- **`build_notebooks.py` still emits the CIFAR notebooks.** The library is
  ported; the notebooks are not.
- **Build-time schema and path validation** (rules 3, 4) — generation should
  fail on a column literal absent from `HISTORY_FIELDS`, or a repo path spelled
  as a string instead of going through `run_layout`. This is the D-22/D-36 class
  and it has recurred twice; a build-time check is the only thing that stops it.
- **`ARCH_COST_HINT` is estimated, not measured.** D-10 showed the first guess
  was 40% low. Per DC-11 these refine the display only and must never reach the
  assignment.
- **No architecture in this zoo has been built on a GPU.** Everything above is
  verified offline. The preflight is where the torchvision decompositions and
  Swin's low-resolution behaviour actually get tested.
