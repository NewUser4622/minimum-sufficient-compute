# Data Card — ImageNet-100 (this project's subset)

Protocol §5.4 requires the exact subset to be documented, because **no canonical
ImageNet-100 split exists**. This is that documentation. Everything here is
verified against the files, not against memory.

Produced and checked by [`tools/pack_imagenet100.py`](tools/pack_imagenet100.py).

---

## 1. Identity

| | |
|---|---|
| Classes | **100** |
| Images | **129,395** |
| WNID range | `n01440764` … `n01855672` |
| Selection rule | the **first 100 WNIDs of ImageNet-1k in sort order** |
| Source layout | `train/<wnid>/*.JPEG`, plus `class_mappings.json` |
| Original resolution | native ImageNet — variable, typically 500×375, some as small as 153×202 |
| Original size on disk | ~6.4 GB JPEG (~50 KB mean) |
| **Validation split in source** | **none** |

### ⚠ This is not the ImageNet-100 of Tian et al.

The widely-cited "ImageNet-100" (CMC, arXiv 1906.05849) is a **randomly sampled**
100-class subset. Ours is the first 100 WNIDs, and because WNIDs are ordered by
WordNet taxonomy that makes it a **taxonomically clustered** slice:

> tench, goldfish, sharks, rays · 40+ birds · salamanders, newts, frogs ·
> turtles, lizards, crocodilians · 18 snakes · spiders, scorpions, ticks ·
> grouse, quail, parrots, hummingbirds, toucans, ducks, geese

Every class is an animal, and large blocks are fine-grained (18 snake species,
9 lizards). Consequences that must be carried into any write-up:

- **Higher inter-class similarity** than a random 100-class subset, so absolute
  accuracies will sit lower than published ImageNet-100 numbers even for an
  identical recipe.
- **No published reference is applicable.** `REFERENCE_ACC` is `null` for all
  eight architectures and no Δ is claimed for any of them. D-14 is the
  cautionary precedent — a reference without a matching recipe and parameter
  count produced the largest apparent win in the CIFAR atlas and was withdrawn.
- **For the seed-reliability question this is neutral or helpful.** ρ_seed is a
  rank correlation between two seeds of the *same* recipe on the *same* data, so
  the subset's difficulty cancels. A harder, finer-grained problem produces a
  wider spread of per-sample compute requirements, which is if anything a
  better-conditioned measurement than a coarse one.

### Class balance

| | |
|---|---|
| Per class | **1,071 – 1,300** |
| Mean / median | 1,294.0 / 1,300 |
| Classes at the full 1,300 | **96** |
| Classes below | **4** |

| WNID | n | name |
|---|---|---|
| `n01744401` | 1,071 | rock python, rock snake, *Python sebae* |
| `n01688243` | 1,117 | frilled lizard, *Chlamydosaurus kingi* |
| `n01855032` | 1,141 | red-breasted merganser, *Mergus serrator* |
| `n01704323` | 1,266 | triceratops |

Mild imbalance, ≤ 18% below full. **No class weighting or resampling is
applied** — introducing one would be a confound that differs by nothing but
happens to interact with per-sample difficulty, which is the quantity being
measured. Instead, top-1 is reported alongside **balanced accuracy** and macro
F1, all three of which are already columns in `FINAL_FIELDS`.

The full class list with names is written to `manifest.json` as
`classes` and `class_names`, and is reproduced in `analysis/` at publication.

---

## 2. Splits

No validation split ships with the source, so one is carved and **frozen**.

| split | n | drawn from | augmentation | role |
|---|---|---|---|---|
| `val` | **10,000** | 100/class, stratified | off | the primary MSC measurement split |
| `train` | **119,395** | complement of val | on | training |
| `train_holdout` | **15,000** | sampled *from* `train` | **off** | Q4's split (DC-8) and LTT calibration |

```
val      : rng seed 12345, 100 per class, sorted indices
train    : all indices minus val minus decode failures
holdout  : rng seed 67890, 15,000 sampled from train
```

### Three decisions worth defending

**`val` is 10,000, deliberately.** CIFAR-100's test set is 10,000, and ρ_seed is
estimated at that n throughout the previous study. Matching it means the
ImageNet ceiling estimates carry the same sampling precision, so the
CIFAR↔ImageNet comparison — which is the entire point of this port — is not
partly a comparison of two different estimator variances. 100/class costs 8% of
the training data and buys direct comparability.

**`train_holdout` is *not* held out.** It is training data evaluated with
augmentation off. This is not a mistake and it mirrors the CIFAR design: EL2N
and forgetting events are *training-set* quantities indexed by training images,
and they are undefined on any split the model never trained on. Running Q4
without them handicaps the difficulty battery, which flatters MSC — that is
exactly defect **D-11**, which inflated the reported ΔR² by 2.5×. The
`train_holdout` split exists so that mistake cannot recur.

**`train_holdout` is 15,000, not 5,000.** CIFAR used 5,000 and paid for it in
the limitations paragraph: a Hoeffding bound at δ = 0.05 needs ≈ 14,979
calibration samples for ε = 0.01, so risk control had to be calibrated at
ε = 0.03 instead. With 15,000 the Learn-then-Test guarantee holds at **ε = 0.01**
and that caveat disappears. The calibration distribution is still train-like,
which remains stated.

### Reproducing the splits

They are a pure function of `(labels, seeds)`, so they regenerate identically on
any machine. Verified: two independent calls produce byte-identical index arrays
and the same fingerprint. The index sets are nonetheless **published** in
`splits.json` rather than only described, because a split that must be
recomputed to be known is a split that can silently drift.

---

## 3. Packing

| | |
|---|---|
| Format | `images_256.u8` — `(129395, 256, 256, 3)` uint8 memmap |
| Size | **23.7 GiB** |
| Resize policy | shorter side → 256 (bicubic), centre-crop 256×256, RGB |
| Train view | GPU-side RandomResizedCrop → **224**, horizontal flip |
| Eval view | centre-crop 224 from the stored 256 |

The eval view is exactly the standard ImageNet protocol (`Resize(256)` +
`CenterCrop(224)`), because the stored image *is* the resized-256 image.

### The one real cost of packing, stated

Squaring at pack time means **RandomResizedCrop samples inside the central
square**, not the full frame. On a 500×375 source, roughly 25% of the horizontal
extent is discarded before augmentation ever sees it. That is a genuine
reduction in augmentation diversity versus a standard JPEG pipeline, and it will
cost some absolute accuracy.

It is acceptable here for one reason and it should be stated that way: **it is a
constant across all eight architectures.** It cannot manufacture a difference
between CNNs and transformers, which is the only quantity this study is
measuring. What it does forbid is comparing our absolute top-1 to any published
number — which §1 already forbids for a different reason.

### Decode failures

Every image is packed, **including any that fail to decode**. `N` must not
depend on decode success: one skipped file shifts every index after it and
silently re-labels the remainder of the dataset. Failures are written as zeros,
recorded in `manifest["failures"]` with path and exception, and **excluded from
every split**, so a failed image can never enter a measurement.

`ImageFile.LOAD_TRUNCATED_IMAGES` is on, as in every published ImageNet
pipeline. The count of images that needed it is recorded rather than assumed to
be zero.

---

## 4. Fingerprint

```
fingerprint = sha256( pack_version ‖ stored_res ‖ sorted class list
                      ‖ val indices ‖ holdout indices )
```

Current value, computed against the real data:

```
2b6269ef51ff87b2c9e00fa17c44326ce634a67892c9eb550ec518a6dd2d2b6c
```

`train` is the complement, so it is implied and not hashed separately.

**This value participates in `config_hash`.** Consequences, which are the whole
reason it exists:

- A repacked or re-split dataset **invalidates every run trained against the
  old one**, at gate time, with a message naming both fingerprints.
- Two runs whose per-sample tables are correlated must agree on it. Without this
  check, two runs that disagreed about which 10,000 images are `val` would
  produce tables that align perfectly by index and are meaningless to correlate
  — and the failure would look exactly like a real result.

This is rule 5 applied to the dataset: *presence* of a per-sample table is not
evidence that it is *comparable* to another one. The CIFAR pipeline had
`sample_order_hash` for row alignment within a fixed dataset; the fingerprint is
the analogous guarantee one level up, and it is new here because CIFAR's splits
came with the dataset and could not drift.

---

## 5. Verification

`pack_imagenet100.py --verify` re-reads the pack and asserts:

| check | catches |
|---|---|
| 512 sampled images are non-zero unless recorded as failures | a chunk that silently wrote zeros — the spawn/memmap hazard |
| sampled mean pixel in a plausible range | a channel-order or normalisation error |
| `val ∩ train = ∅` | a split bug |
| `holdout ⊆ train` | a split bug |
| `len(labels) == N` | a truncated pack |
| `val` is exactly 100 per class | a stratification bug |
| recomputed fingerprint == stored | any of the above, or a hand-edited manifest |

A dry run of the identical code path — decode → resize → crop → memmap write →
**memmap read-back** → dtype and range check — runs on 32 images before the
24 GiB job starts. It takes 0.2 s. Rule 1: a partial dry run only moves where
the bug hides, so it exercises the same functions the real pass does, including
the read-back, because writing correctly and reading correctly are different
claims.

**Verified on the real data before any pack was written:**

```
[DRY]  32 images -> memmap -> read back OK (32 non-zero, mean 115.8, 0.2s, 0 failures)
index: 100 classes, 129,395 images, labels int16 [0, 99]
val 10000 / train 119395 / holdout 15000 · val per class 100..100
val ∩ train = ∅ · holdout ⊆ train · splits deterministic across calls
fingerprint 2b6269ef51ff87b2c9e00fa17c44326ce634a67892c9eb550ec518a6dd2d2b6c
```

---

## 6. How to build it

```
python tools/pack_imagenet100.py --src "C:/Users/Administrator/Desktop/New folder" \
                                 --out  "D:/msc_data/in100"
```

- ~20–40 min on 24 cores; needs **25 GB** free at the destination.
- **Resumable at chunk granularity** — an interrupted pack continues rather than
  restarting. A 30-minute job that cannot resume is its own small D-19.
- Ends by running the full verification above and exits non-zero on any problem.
- Re-check an existing pack at any time with `--verify`.

Uploading the output directory as a Kaggle Dataset makes the notebooks run
unchanged there; nothing else about them changes.
