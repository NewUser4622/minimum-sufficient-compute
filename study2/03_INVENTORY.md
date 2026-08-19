# What already exists

The reason Study 2 is cheap. Study 1's expensive phase produced artifacts that
answer a better question than the one they were collected for.

**Everything below is already trained, measured, and on HuggingFace.**

---

## Runs

| study | architectures | seeds | measured runs | repo |
|---|---|---|---|---|
| CIFAR-100 | **15** | 3 | **45** (+4 pilot) | `Shanmuk4622/msc-cifar100` |
| ImageNet-100 | 2 | 2 | **4** | `Shanmuk4622/msc-imagenet100` |
| ImageNet-100 MSC-KD students | 3 | 3 × 2 arms | **18** | same |

CIFAR architectures: ResNet ×5, WRN ×3, VGG ×2, mobile ×2, ConvNeXt, ViT,
MLP-Mixer. **13 convolutional, 2 not** — the split R1 needs.

## Per-sample scores

> ### ⚠ CORRECTED 2026-08-19 — this section was wrong, and wrong in the exact
> ### way this file was written to prevent
>
> The original text claimed **eight scores on both splits**, sourced from
> `analysis/q4_irreducibility_all.csv` — a *summary* file — rather than from a
> parquet. It shipped as an unverified claim and it cost a notebook failure and
> a false pass. What the files actually contain:

| score | `test.parquet` | `train_holdout.parquet` |
|---|---|---|
| `msp`, `margin`, `entropy`, `ce_loss`, `pred_depth` | **populated** | **populated** |
| `el2n`, `forget_events` | column exists, **entirely NaN** | **populated** |
| `msc` | **not a column at all** | **not a column at all** |

Three consequences, none of them cosmetic:

1. **`msc` is derived, never persisted.** It has to be recomputed from
   `pred_d*` / `top1p_d*` / `top2p_d*` if it is wanted. Decision **D2** (MSC as
   one of eight) therefore costs work rather than being free.
2. **`el2n` and `forget_events` are training-set quantities.** A test sample has
   no training history, so the columns are NaN there by construction. They are
   real on `train_holdout.parquet`.
3. **Only 5 scores can route unseen data.** This is a finding, not an
   inconvenience: two of the eight candidate signals are *unusable as routing
   signals on unseen samples by construction*, whatever their reliability.

So the grids are:

| grid | split | scores | used for |
|---|---|---|---|
| routing / optimism bias (NB2) | `test` | **5** | R3, R4, R5 — the centrepiece |
| reliability atlas (NB1) | `test` **and** `train_holdout` | 5 and **7** | R1, R2 |

**How this was caught, and how it is prevented.** It was *not* caught by
reading. `S2_NB1` cell 4 now probes every split and prints what it found;
`tools/s2_canaries.py` proves the check can fail. The verification snippet at
the end of this file was correct in spirit and I did not run it.

**`sample_idx` is a global pack index**, not a position within a split. This is
why 49 runs can be joined directly — and it is a Study 1 design decision (D-49)
that makes Study 2 possible at all.

## What that buys, with no new training

| question | data needed | have it? |
|---|---|---|
| ρ_seed per score per architecture | 2+ seeds × per-sample scores | **yes** — 15 archs × 3 seeds |
| score × score correlation | one run's per-sample table | **yes** |
| raw vs disattenuated cross-arch agreement | above + the ceilings | **yes** |
| oracle routing ceiling per score | per-exit logits + a saved checkpoint | **checkpoints yes**, one forward pass needed |

**R1, R2 and R4 are pure re-analysis.** R3 needs one forward pass per run
through the existing `evaluate_routing_methods` machinery, generalised from MSC
to an arbitrary score.

## Machinery that already exists and is reusable

| what | where | reuse |
|---|---|---|
| MSC definition and every statistic | `msc_core.py` | as-is |
| per-exit forward + routing curves | `evaluate_routing_methods` | **generalise `oracle_msc` → any score** |
| matched-FLOPs comparison | `sweep_operating_points`, `accuracy_at_matched_flops` | as-is |
| disattenuation | `analyse_q3_all` | as-is |
| noise ceilings | `analyse_q1_all` | **generalise from MSC to any column** |
| resumable runner, resume-safe hashing | `Session.run_all`, `hash_compatible` | as-is |
| six-layer notebook validation | `tools/` | as-is |

The generalisation in both cases is *one column name becoming a parameter*.

## What does NOT exist

1. **A third ImageNet-100 seed.** 2 seeds → ρ_seed is a point estimate with no
   error bar (`02_PROTOCOL.md` stopping rule 4).
2. **Any architecture measured on both datasets.** `shufflenetv2` and
   `convnext` exist in both zoos but only `shufflenetv2`/`convnext_femto` were
   run on CIFAR and neither on ImageNet. **No cross-scale claim is available**
   without new training.
3. **Oracle ceilings for the seven non-MSC scores.** This is R3, and it is P1.
4. **ρ_seed for the seven non-MSC scores.** This is R1, and it is P0.

## The honest catch

Items 3 and 4 are the whole study, so "no new training" does not mean "no work".
It means the work is **measurement and analysis over existing models** rather
than producing new ones — roughly a day of compute against Study 1's ~215
GPU-hours.

It also means a negative result costs a day. That is the property worth having.

## Verifying this inventory before trusting it

**This section already existed, was correct, and I did not run it.** That is the
whole defect. A verification snippet nobody executes is a comment, not a
mechanism (rule 7).

It is no longer optional: `S2_NB1` cell 4 performs this probe on every split and
refuses to continue if nothing usable survives. To check by hand:

```python
import pandas as pd
from pathlib import Path
run = sorted((Path(MSC_ROOT) / 'runs').iterdir())[0]
want = ['msp','margin','entropy','ce_loss','el2n','forget_events','pred_depth','msc']
for split in ['test', 'train_holdout']:
    d = pd.read_parquet(run / 'per_sample' / f'{split}.parquet')
    absent = [c for c in want if c not in d.columns]
    allnan = [c for c in want if c in d.columns and d[c].notna().mean() <= 0.5]
    print(f'{split:14s} usable={len([c for c in want if c not in absent + allnan])}'
          f'  absent={absent}  all-NaN={allnan}')
```

Expected, as of the correction above:

```
test           usable=5  absent=['msc']  all-NaN=['el2n', 'forget_events']
train_holdout  usable=7  absent=['msc']  all-NaN=[]
```

**Presence is not enough — check for all-NaN too.** The original snippet tested
only `need - set(df.columns)`, which `el2n` and `forget_events` pass on the test
split while carrying no data. That is precisely how they vanished from the first
atlas without a word.
