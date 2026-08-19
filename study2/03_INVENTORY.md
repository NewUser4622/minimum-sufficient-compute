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

Every measured run has `per_sample/test.parquet` and
`per_sample/train_holdout.parquet` carrying **all eight scores per sample**:

```
msp   margin   entropy   ce_loss   el2n   forget_events   pred_depth   msc
```

Confirmed from `analysis/q4_irreducibility_all.csv`: `n_battery_scores = 7`,
`battery = msp,margin,entropy,ce_loss,el2n,forget_events,pred_depth`, plus `msc`
as the eighth.

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

Run before P0 — do not take this file's word for it:

```python
import msc_lib as M, pandas as pd
sess = M.Session(account='local', phase='p1', dataset='cifar100', work_root=MSC_ROOT)
runs = [r['run_id'] for r in sess.completed_runs(phase='p1') if sess.measured(r['run_id'])]
print(len(runs), 'measured runs')

df = pd.read_parquet(M.run_layout(sess.work, runs[0])['per_sample'] / 'test.parquet')
need = {'msp','margin','entropy','ce_loss','el2n','forget_events','pred_depth','msc'}
print('missing columns:', need - set(df.columns))   # must be empty
print('sample_idx is global:', df['sample_idx'].max(), 'vs rows', len(df))
```

If `missing columns` is non-empty, P0's scope shrinks to whatever is present and
this file is wrong — which is exactly the kind of assumption Study 1 kept making
without checking.
