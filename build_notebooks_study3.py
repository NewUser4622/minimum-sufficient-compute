#!/usr/bin/env python3
"""Generate the Study 3 notebooks.

Reuses the CIFAR-100 generator's bootstrap, worker/session and data cells, so
Study 3 inherits Study 1's resume, push-batching and registry machinery rather
than reimplementing it. Duplication is what caused D-23 and D-49.

    python build_notebooks_study3.py

Writes notebooks_study3/ and runs every validation gate afterwards.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from build_notebooks import DATA_CELL, bootstrap_cell, code, md, notebook

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks_study3"

# The three architectures for Q1. Chosen for cost and family spread, with
# Study 1's MEASURED wall-clock, not guesses:
#   resnet20    1.3 h   low capacity, small existing pool
#   resnet32x4  2.9 h   the reference architecture in both studies
#   vgg8        ~2.0 h  different family; guards against a ResNet-only result
Q1_ARCHS = ("resnet20", "resnet32x4", "vgg8")


def paths_cell(phase="analysis", needs_data=False, hf=False) -> str:
    """Locate the results root the same way Study 2 did, and PROVE data is there.

    The first version of this used `build_notebooks.py`'s `worker_cell`, which
    constructs a Session WITHOUT `work_root`. The Session then chose its own
    default directory, which was not where the runs live, so `runs` came back
    empty, `pd.DataFrame([])` had no columns, and the first `meta['method']`
    raised `KeyError: 'method'` -- 40 lines from the real cause.

    Two fixes, both here:
      * resolve through `M.resolve_storage`, exactly as Study 2's notebooks do,
        and pass `work_root=MSC_ROOT` to the Session;
      * check for MEASURED RUNS, not merely that `runs/` is a directory. A
        Session creates that directory itself, so its existence proved nothing.
    """
    data_note = "" if needs_data else (
        "\n# This notebook only READS existing runs -- it trains nothing, so"
        "\n# DATA_DIR is irrelevant to it.")
    strict = ("    raise RuntimeError(\n"
              "        f'no measured runs under {MSC_ROOT}/runs.\\n'\n"
              "        'Study 3 re-analyses Study 1 output. Fetch it first with\\n'\n"
              "        'notebooks_study2/S2_NB0_Fetch.ipynb, or set MSC_ROOT above\\n'\n"
              "        'to the folder that already holds them.')")
    return f"""\
# === CELL 2 -- WHERE EVERYTHING LIVES ======================================
# Leave both as None and they are CHOSEN FOR YOU: the roomiest drive that
# actually exists on this machine. Set MSC_ROOT explicitly if your runs are
# somewhere specific -- e.g. r'C:\\msc_results'.{data_note}

DATA_DIR = None      # e.g. r'E:\\msc_data'      -- None = choose for me
MSC_ROOT = None      # e.g. r'C:\\msc_results'   -- None = choose for me

# WHERE CIFAR-100 ALREADY IS. Point this at the folder that contains
# `cifar-100-python` (or at that folder itself -- both work). It is checked
# BEFORE any download path, so nothing ever re-fetches 169 MB over a copy you
# already have.
CIFAR_DIR = r'C:\\Users\\Administrator\\Desktop\\New folder'

# ---------------------------------------------------------------------------
import os
from pathlib import Path

M = msc                       # Study 2's notebooks say `M`; same module.

if CIFAR_DIR:
    os.environ['MSC_CIFAR_DIR'] = str(CIFAR_DIR)

# Study 3 is CIFAR-100. The library's default repo is the ImageNet one, so
# without this every push would land in msc-imagenet100 -- which is exactly what
# happened on the first joint run.
os.environ.setdefault('MSC_HF_REPO', 'Shanmuk4622/msc-cifar100')

_paths = M.resolve_storage(DATA_DIR, MSC_ROOT)
if not _paths['ok']:
    raise SystemExit('storage is not usable -- see the problems listed above')

DATA_DIR = _paths['data_dir']
MSC_ROOT = _paths['results_root']
os.environ['MSC_SCRATCH'] = MSC_ROOT

# OFFLINE BY DEFAULT. This machine does not always have a network, and a
# background uploader that retries mid-epoch turns a missing connection into a
# failed run. Everything is written to disk in full; S3_NB5_Publish uploads it
# in ONE pass at the end. `enable_hf` is the only switch that decides this.
sess = M.Session(account='local', phase='{phase}', dataset='cifar100',
                 work_root=MSC_ROOT, session_limit_h=0.0,
                 enable_hf={hf!r},
                 worker_id=0, num_workers=1)
print('HuggingFace: ' + ('ON -- publishing' if {hf!r} else
                         'OFF -- fully offline, nothing is uploaded here'))

print(f'msc_lib   {{M.__version__}}')
print(f'MSC_ROOT  {{MSC_ROOT}}')
print(f'data_dir  {{sess.data_dir}}')

# Resolve CIFAR-100 now, loudly, rather than discovering mid-training that it
# is about to download. `_has_cifar100` is the same check the loader uses.
if CIFAR_DIR:
    _cd = Path(CIFAR_DIR)
    _cd = _cd if M._has_cifar100(_cd) else _cd.parent
    if M._has_cifar100(_cd):
        print(f'CIFAR-100  {{_cd}}  (found -- no download)')
    else:
        print(f'CIFAR-100  NOT at {{CIFAR_DIR}} -- expected a cifar-100-python/'
              ' folder with train/ and test/ inside. It will try to download.')

# A Session CREATES runs/, so "the directory exists" proves nothing. Count the
# runs that actually carry a measurement -- that is what every cell below reads.
_runs_dir = Path(MSC_ROOT) / 'runs'
_measured = sorted(d.name for d in _runs_dir.iterdir()
                   if d.is_dir() and (d / 'per_sample' / 'test.parquet').exists()
                   ) if _runs_dir.is_dir() else []
print(f'measured runs on disk: {{len(_measured)}}')
if not _measured:
{strict}
print(f'  e.g. {{_measured[0]}}')
"""


def runs_cell() -> str:
    """One accessor for "which runs do I analyse", used by every notebook.

    `pd.DataFrame([])` has no columns, so the first `meta['method']` raised
    `KeyError: 'method'` -- a message 40 lines from the actual cause, which was
    an empty run list. Rule 3: column names are data, and they get validated at
    build time rather than discovered mid-analysis.
    """
    return """\
# === Which runs am I analysing? ===========================================
# One accessor, so the dedupe rule and the emptiness checks live in a single
# place rather than being re-typed in each notebook (rule 4).
import numpy as np, pandas as pd

runs_dir = Path(MSC_ROOT) / 'runs'

def measured_runs(dataset='cifar100', methods=('base',), require=True):
    ids = sorted(d.name for d in runs_dir.iterdir()
                 if d.is_dir() and (d / 'per_sample' / 'test.parquet').exists())
    if not ids:
        raise RuntimeError(f'no measured runs under {runs_dir}')
    df = pd.DataFrame([{**M.parse_run_id(r), 'run_id': r} for r in ids])

    for col in ('method', 'dataset', 'arch', 'seed', 'phase'):
        if col not in df.columns:
            raise RuntimeError(
                f'parse_run_id did not yield a {col!r} column. Columns present: '
                f'{list(df.columns)}. Run ids look like: {ids[:3]}')

    sel = df[(df['dataset'] == dataset) & (df['method'].isin(methods))]
    if require and sel.empty:
        raise RuntimeError('; '.join([
            f'{len(df)} measured run(s) on disk, but NONE with '
            f'dataset={dataset!r} and method in {tuple(methods)!r}',
            f'datasets present: {sorted(df["dataset"].dropna().unique())}',
            f'methods present: {sorted(df["method"].dropna().unique())}',
            'either the wrong MSC_ROOT is set, or the runs you need have not '
            'been fetched or trained yet']))

    # p0 pilots and p1 runs share seed numbers, so pooling them counts one seed
    # twice -- the contamination Study 2 found. Keep the highest phase.
    before = len(sel)
    sel = (sel.sort_values('phase')
              .drop_duplicates(subset=['arch', 'dataset', 'method', 'seed'],
                               keep='last'))
    if len(sel) < before:
        print(f'dropped {before - len(sel)} duplicate (arch, method, seed) '
              f'run(s) -- pilot replicates')
    return sel.reset_index(drop=True)

_b = measured_runs()
print(f'{len(_b)} CIFAR-100 base run(s); '
      f'{_b["arch"].nunique()} architecture(s)')
"""


# Q3 pruning: score sources at the two ends of the saturation range measured in
# Study 2 (analysis/s2_memorisation.csv).
SATURATED = "convnext_femto"     # 99.99 % train acc, 71 % of samples > 0.99 conf
UNSATURATED = "mobilenetv2"      # 83.1 % train acc, 16 % saturated
PRUNE_TARGET = "resnet20"        # fixed retraining target, 1.3 h/run
PRUNE_RATES = (0.5, 0.3)         # fraction of the training set KEPT


# ---------------------------------------------------------------------------
# S3_NB0 -- the free gate
# ---------------------------------------------------------------------------
def nb0():
    return notebook([
        md("""
# S3_NB0 — does exit quality predict the oracle excess?

**~15 minutes · CPU only · no training · reads data already on disk**

## Why this runs first

Study 2's core claim is that the oracle early-exit bound sits **+6.86 points
above the network's own full-compute accuracy**, and that the excess is
per-exit noise. Study 1 trained its exit heads on a **frozen** backbone;
MSDNet, BranchyNet and DE3-BERT train exits **jointly**. Better exits might
shrink the excess to nothing, which would make Study 2 an artifact.

Settling that costs ~10 GPU-hours (`S3_NB1`). **This notebook tries to
pre-answer it for free.**

Across Study 1's 15 architectures, exit-head quality already varies a great
deal. If the excess **shrinks as the early exits get stronger**, joint training
would shrink it further and we can estimate by how much. If the excess is
**insensitive** to exit quality, joint training is unlikely to remove it.

> **This is a gate, not evidence.** It is observational and
> cross-architectural: architectures differ in many ways besides exit quality.
> It cannot replace `S3_NB1`. It only tells us how much `S3_NB1` is worth, and
> what to expect — which makes the GPU result falsifiable in advance instead of
> merely interesting.
"""),
        code(bootstrap_cell()),
        code(paths_cell()),
        code(runs_cell()),
        md("""
---
## The quantities

For every measured run, from `per_sample/test.parquet`:

| | |
|---|---|
| `acc_full` | accuracy of the final exit — what you get by running everything |
| `acc_k` | accuracy of exit *k* alone |
| **`excess`** | `P(correct at ANY exit) − acc_full` — Study 2's +6.86 pt |
| **`exit_quality`** | mean of `acc_k / acc_full` over the early exits |

`exit_quality` near 1.0 means the early exits are nearly as good as the full
network. Frozen post-hoc heads should sit well below that; jointly trained ones
should sit closer to it. The regression of `excess` on `exit_quality` is the
extrapolation.
"""),
        code("""
import numpy as np, pandas as pd, itertools
from pathlib import Path
from scipy.stats import spearmanr

runs_dir = Path(MSC_ROOT) / 'runs'
base = measured_runs()          # THE accessor -- checks and dedupes

rows = []
for r in sorted(base['run_id']):
    d = pd.read_parquet(runs_dir / r / 'per_sample' / 'test.parquet')
    ks = sorted(int(c.split('_d')[1]) for c in d.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    lab = d['label'].to_numpy()
    corr = np.stack([(d[f'pred_d{k}'].to_numpy() == lab) for k in ks], axis=1)
    acc_k = corr.mean(axis=0)
    acc_full = float(acc_k[-1])
    rows.append({
        'run_id': r, 'arch': M.parse_run_id(r)['arch'],
        'seed': M.parse_run_id(r)['seed'],
        'acc_full': acc_full,
        'excess': float(corr.any(axis=1).mean()) - acc_full,
        'exit_quality': float(np.mean(acc_k[:-1] / max(acc_full, 1e-9))),
        'acc_first_exit': float(acc_k[0]),
        'K': len(ks),
    })

ex = pd.DataFrame(rows)
M.save_analysis(sess.data_dir, 's3_exit_quality', ex)
per = ex.groupby('arch')[['acc_full', 'excess', 'exit_quality',
                          'acc_first_exit']].mean()
print(per.round(4).sort_values('exit_quality').to_string())
"""),
        md("""
---
## The extrapolation

`excess` regressed on `exit_quality`. The **slope** is what matters: it says how
many accuracy points the excess falls for each unit of improvement in the early
exits.

We do not know where jointly trained exits land on the `exit_quality` axis, so
the cell reports the predicted excess at several plausible values instead of
guessing one. `S3_NB1` will measure the real value and we compare.
"""),
        code("""
r_q, p_q = spearmanr(ex['exit_quality'], ex['excess'])
print(f'Spearman(exit_quality, excess) = {r_q:+.3f}   p = {p_q:.4f}   '
      f'n = {len(ex)} runs')

# THE CONFOUND, and it is not subtle: exit_quality = mean(acc_k / acc_full), so
# acc_full is its own denominator. A low-accuracy network scores high
# exit_quality with weak early exits -- and a low-accuracy network makes more
# final-layer errors, which is exactly what `excess` counts. Reporting the raw
# correlation alone would be reporting that circularity as a finding.
r_a, p_a = spearmanr(ex['acc_full'], ex['excess'])
print(f'Spearman(acc_full,     excess) = {r_a:+.3f}   p = {p_a:.4f}   '
      '<- the confound')

def _partial(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    res = lambda t: t - np.polyval(np.polyfit(rz, t, 1), rz)
    return spearmanr(res(rx), res(ry))

r_p, p_p = _partial(ex['exit_quality'], ex['excess'], ex['acc_full'])
print(f'PARTIAL, holding acc_full fixed = {r_p:+.3f}   p = {p_p:.4f}   '
      '<- the one that counts')
if abs(r_p) < 0.2:
    print('  -> the raw relationship was mostly the confound. Treat the')
    print('     extrapolation below as having no predictive power.')
print()

A = np.polyfit(ex['exit_quality'], ex['excess'], 1)
slope, intercept = float(A[0]), float(A[1])
print(f'OLS: excess = {slope:+.4f} * exit_quality {intercept:+.4f}')
print()

obs_lo, obs_hi = ex['exit_quality'].min(), ex['exit_quality'].max()
print(f'observed exit_quality range (frozen heads): '
      f'{obs_lo:.3f} to {obs_hi:.3f}')
print()
print('predicted excess if joint training reaches:')
for q in sorted({round(float(obs_hi), 4), 0.90, 0.95, 1.00}):
    pred = slope * q + intercept
    tag = '  <- best frozen run' if abs(q - obs_hi) < 1e-9 else ''
    extrap = '  (EXTRAPOLATION, outside observed range)' if q > obs_hi else ''
    print(f'    exit_quality {q:.2f}  ->  excess {pred*100:+6.2f} pt{tag}{extrap}')

print()
print('--- GATE ---')
pred_at_1 = slope * 1.0 + intercept
if p_q > 0.05:
    print('exit quality does NOT predict the excess (p > 0.05).')
    print('=> Joint training is unlikely to remove it. S3_NB1 is still needed')
    print('   to confirm, but expect the excess to SURVIVE.')
elif pred_at_1 * 100 > 2.0:
    print(f'even at exit_quality = 1.0 the predicted excess is '
          f'{pred_at_1*100:+.2f} pt, above the 2.0 pt threshold.')
    print('=> Expect Study 2 to SURVIVE joint training. Run S3_NB1 to confirm.')
else:
    print(f'at exit_quality = 1.0 the predicted excess is '
          f'{pred_at_1*100:+.2f} pt, below the 2.0 pt threshold.')
    print('=> Study 2 may be a frozen-backbone ARTIFACT. S3_NB1 is now the')
    print('   most important experiment in the project -- run it next.')
print()
print('Either way this is observational. It sets an EXPECTATION that S3_NB1')
print('can falsify, which is the point of running it first.')
"""),
        md("""
---
## Canary — this analysis must be able to give a wrong answer

Three synthetic checks. The load-bearing one is the third: if `excess` were
computed wrongly it might read ~0 everywhere, and "no excess" and "cannot
measure excess" look identical in the output.
"""),
        code("""
def _excess_of(corr):
    a = corr.mean(axis=0)
    return float(corr.any(axis=1).mean()) - float(a[-1])

rng = np.random.default_rng(0)
n, K = 5000, 5

# 1. perfect exits, all identical -> excess must be EXACTLY zero
c = np.zeros((n, K)); c[rng.random(n) < 0.7, :] = 1
e1 = _excess_of(c)
print(f'{"PASS" if abs(e1) < 1e-12 else "FAIL"}  identical exits -> excess '
      f'{e1*100:+.4f} pt (must be 0)')

# 2. early exits right where the final is wrong -> excess must be large
c = np.zeros((n, K)); hit = rng.random(n) < 0.3
c[hit, 0] = 1                      # right early, wrong at the end
c[rng.random(n) < 0.6, -1] = 1
e2 = _excess_of(c)
print(f'{"PASS" if e2 > 0.05 else "FAIL"}  early-only correctness -> excess '
      f'{e2*100:+.2f} pt (must be large)')

# 3. excess can never be negative, on any input
worst = min(_excess_of((rng.random((400, K)) < rng.random()).astype(float))
            for _ in range(300))
print(f'{"PASS" if worst >= -1e-12 else "FAIL"}  excess is never negative '
      f'over 300 random draws (worst {worst*100:+.6f} pt)')
"""),
        md("""
---
## What to do next

Read the **GATE** block above, then:

| gate says | next |
|---|---|
| excess survives at `exit_quality = 1.0` | run **`S3_NB1`** to confirm — it should agree |
| excess vanishes | run **`S3_NB1`** anyway; it is now the highest-value experiment in the project, because it decides whether Study 2 stands |
| exit quality does not predict excess | run **`S3_NB1`**; the extrapolation has no power and only the real experiment will settle it |

**In every branch the answer is `S3_NB1`.** What changes is what you should
expect — and having written the expectation down first is what makes the GPU
result meaningful rather than merely a number.

Record the outcome in `study3/03_LOG.md` before moving on.
"""),
    ])


# ---------------------------------------------------------------------------
# S3_NB1 -- joint exit training  (the blocker)
# ---------------------------------------------------------------------------
def nb1():
    return notebook([
        md(f"""
# S3_NB1 — joint exit training

**~10 GPU-hours · 3 runs · fully resumable · the blocking experiment**

## What this settles

Study 1 trained exit heads on a **frozen** backbone. The field trains them
**jointly**. If Study 2's +6.86 pt excess is an artifact of weak post-hoc
exits, it should shrink or vanish here.

**Pre-registered (`study3/01_PROTOCOL.md` H1):**

```
oracle_in − acc_full  ≥  2.0 accuracy points,  in 3 of 3 architectures
```

| measured | verdict |
|---|---|
| ≥ 2.0 pt in 3/3 | Study 2 is **not** a frozen-backbone artifact — archival paper unblocked |
| 0.5 – 2.0 pt | real but much smaller; the paper leads with the joint number |
| < 0.5 pt | Study 2 **is** an artifact — withdraw to a methodological note, Q3 becomes the main paper |

## Why this needs no seeds

The quantity is a **per-run identity**:
`oracle_in − acc_full == frac_early_saves`. It is computed from one network's
own per-exit predictions. Seeds were needed for the *bias*, not for this. So
this is **{len(Q1_ARCHS)} architectures × 1 seed**, not × 3 — roughly 10
GPU-hours instead of 30.

## The design is paired

Everything is held constant against the Study 1 runs of the same architectures.
**One variable changes:** `joint_exits`. Same data, same epochs, same optimiser,
same augmentation, same exit positions, same FLOP budgets, and — most
importantly — the **same measurement code**.

{', '.join(Q1_ARCHS)}
"""),
        code(bootstrap_cell()),
        code(paths_cell(phase="p4", needs_data=True)),
        code(runs_cell()),
        code(DATA_CELL),
        md("""
---
## Configs

`joint_exits=True` is a real configuration difference, so it is **not** in
`_HASH_EXCLUDE` and these runs get their own `config_hash`. They cannot collide
with, or resume from, a Study 1 checkpoint.

`exit_weight_scheme` is recorded explicitly because deep supervision has several
standard weightings and the result can depend on which
(`study3/02_RISKS.md` R-03). `uniform` is the MSDNet-style default.
"""),
        code(f"""
ARCHS = {list(Q1_ARCHS)!r}
SEED = 1
SCHEME = 'uniform'        # uniform | linear | final_heavy

# `sess.config`, NOT `M.base_config`. The bound method calls prepare_data() and
# fills in `data_root`; the raw function does not, so the loader fell through to
# locate_cifar100()'s download path even though the data was already on disk.
cfgs = [sess.config(a, seed=SEED, method='jointexit',
                    joint_exits=True, exit_weight_scheme=SCHEME)
        for a in ARCHS]

for c in cfgs:
    K = len(M.load_or_build_budgets(c['arch'], sess.work,
                                    'cifar100')['axes']['depth']['rho'])
    w = M.exit_loss_weights(K, SCHEME)
    print(f"{{c['run_id']:38s}} epochs={{c['num_epochs']}}  K={{K}}  "
          f"weights={{[round(x, 3) for x in w]}}")
    assert c.get('joint_exits') is True, 'joint_exits did not survive base_config'

# The frozen counterparts already exist. Confirm before training, so the paired
# comparison in S3_NB2 cannot fail after 10 GPU-hours are spent.
missing = []
for a in ARCHS:
    hits = [d.name for d in (Path(MSC_ROOT) / 'runs').iterdir()
            if d.is_dir() and f'-{{a}}-cifar100-base-s' in d.name
            and (d / 'per_sample' / 'test.parquet').exists()]
    if not hits:
        missing.append(a)
    else:
        print(f'  paired frozen run for {{a}}: {{sorted(hits)[0]}}')
if missing:
    raise RuntimeError(
        f'no frozen counterpart for {{missing}} -- the paired comparison would '
        'be impossible. Run S2_NB0_Fetch first, or change ARCHS.')
print()
print('paired frozen runs present for every architecture')
"""),
        md("""
---
## Dry run first (rule 1)

Two synthetic batches through the **entire** joint path — model construction,
the weighted multi-exit loss, backward, checkpoint save and reload — before any
real epoch. Seconds against hours.

This matters more than usual here: `joint_exits` is a **new code path**, and it
has never run on a GPU. It was written in an environment with no torch, so
every training line below has shipped unexecuted. The dry run is where that
gets caught.
"""),
        code("""
import torch, traceback

ok = True
for c in cfgs:
    try:
        r = M.backbone_dry_run(c)
        print(f"  {c['run_id']:38s} {r if not isinstance(r, tuple) else r[1]}")
    except Exception as e:
        ok = False
        print(f"  {c['run_id']:38s} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()

if not ok:
    raise RuntimeError('dry run failed -- no GPU time has been spent. Fix first.')

# Memory probe (study3/02_RISKS.md R-08). Multi-exit training holds K heads and
# K losses, so peak memory is higher than Study 1's single-head runs. A
# benchmark once crashed this machine; find the ceiling cheaply instead.
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    print()
    print(f'GPU free {free/2**30:.1f} GiB of {total/2**30:.1f} GiB')
    print(f'peak allocated during dry run: '
          f'{torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    print('If a real epoch OOMs, halve batch_size in the config above rather')
    print('than risking the machine.')
"""),
        md("""
---
## Train

**Resuming is automatic.** Re-run this cell after any interruption: finished
runs are skipped, partial runs continue from their last completed epoch with
optimiser, scheduler, AMP scaler and all four RNG streams restored.

**Safe to stop at any time.** On interrupt the session pushes everything to
HuggingFace, blocking, before it exits.
"""),
        code("""
results = sess.run_all(cfgs, title='Study 3 Q1 / joint exit training')

print()
for r in results:
    if r.get('status') == 'skipped':
        print(f"  SKIPPED   {r['run_id']}  ({r.get('reason')})")
    else:
        print(f"  {r.get('status','?'):9s} {r.get('run_id','?')}  "
              f"top1={M.fmt_metric(r.get('best_accuracy'))}  "
              f"{r.get('num_epochs_run','?')} epochs")
"""),
        md("""
---
## Measure

The joint runs wrote their heads to `exit_heads.pt` during training, so this
stage **loads them rather than training frozen ones on top**. That is the whole
reason the joint path saves through `exit_heads_path()`: every downstream
consumer — measurement, budgets, and Study 2's entire analysis — works unchanged.

The cell asserts it, because a silent fall-through to frozen head training would
produce a perfectly healthy-looking run that answers the wrong question.
"""),
        code("""
for c in cfgs:
    hp = M.exit_heads_path(sess.work, c['run_id'])
    if not hp.exists():
        raise RuntimeError(
            f"{c['run_id']}: no exit_heads.pt. Joint training should have "
            "written it at every new best. Without it the measurement stage "
            "would train FROZEN heads and silently answer the wrong question.")
    blob = torch.load(hp, map_location='cpu', weights_only=False)
    if not blob.get('joint'):
        raise RuntimeError(
            f"{c['run_id']}: exit_heads.pt is not marked joint -- these are "
            "frozen heads from an earlier run. Delete and retrain.")
    print(f"  {c['run_id']:38s} joint heads OK  "
          f"scheme={blob.get('exit_weight_scheme')}  epoch={blob.get('epoch')}")

print()
# ALL THREE arguments, which is the call D-67's own error message prescribes:
#
#   sess.run_all(cfgs, fn=sess.oracle, done_fn=sess.measured, stage='measure')
#
#   fn       what to run                     -> the oracle, not the trainer
#   done_fn  what "already done" MEANS here  -> measured, not trained
#   stage    the label the plan prints
#
# Passing `stage='oracle'` alone plans TRAINING, finds every run already
# trained, and measures nothing while printing 'MY REMAINING WORK: 0' (D-88).
# Passing `fn=sess.oracle` alone is refused by D-67, because the completion
# predicate would still ask "is it trained?".
#
# `run_all` does contain an auto-inference for this, but it compares
# `fn is getattr(self, 'oracle', None)` -- and a bound method is a NEW object on
# every attribute lookup, so that identity test is never true. The inference is
# dead code for the oracle path; D-67 is what actually catches the mistake.
# Being explicit here is correct regardless of whether that is ever repaired.
res = sess.run_all(cfgs, fn=sess.oracle, done_fn=sess.measured,
                   stage='measure', title='Study 3 Q1 / measurement')
for r in res:
    print(f"  {r.get('status','?'):9s} {r.get('run_id','?')}")

# Rule 5 / D-79: a plan that says 'nothing to do' is not evidence the artifact
# exists. Open the file the next notebook will read.
print()
missing = []
for c in cfgs:
    ps = M.run_layout(sess.work, c['run_id'])['per_sample'] / 'test.parquet'
    if ps.exists() and ps.stat().st_size > 0:
        print(f"  {c['run_id']:38s} test.parquet {ps.stat().st_size/2**20:.1f} MB")
    else:
        missing.append(c['run_id'])
if missing:
    raise RuntimeError(
        f'measurement reported success but per_sample/test.parquet is missing '
        f'for {missing}. S3_NB2 reads exactly this file, so it would fail with '
        '"no joint runs found" two steps from the real cause.')
"""),
        md("""
---
## Confirm on disk before you stop
"""),
        code("""
run_ids = [c['run_id'] for c in cfgs]
status = sess.confirm_on_disk(run_ids)
print()
for rid, st in (status.items() if isinstance(status, dict) else []):
    print(f'  {rid:38s} {st}')
print()
print('Next: S3_NB2_Compare -- the paired frozen-vs-joint verdict.')
print('Record the outcome in study3/03_LOG.md.')
"""),
    ])


# ---------------------------------------------------------------------------
# S3_NB2 -- the paired comparison  (Q1 verdict)
# ---------------------------------------------------------------------------
def nb2():
    return notebook([
        md("""
# S3_NB2 — frozen vs joint: the Q1 verdict

**~10 minutes · CPU only**

The paired comparison Study 3 exists for. One variable differs between the two
sets of runs: whether the exit heads were trained with the backbone frozen or
jointly.

**Pre-registered:** `oracle_in − acc_full ≥ 2.0 pt` in 3 of 3 architectures.

**The comparison must be at matched final accuracy** (`study3/02_RISKS.md`
R-02). Joint training changes the backbone too, so a raw excess comparison
confounds "better exits" with "different network". Both numbers are reported;
if `acc_full` moves by more than 1 pt the conditioned one is primary.
"""),
        code(bootstrap_cell()),
        code(paths_cell()),
        code(runs_cell()),
        code("""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

runs_dir = Path(MSC_ROOT) / 'runs'

def exit_stats(run_id):
    d = pd.read_parquet(runs_dir / run_id / 'per_sample' / 'test.parquet')
    ks = sorted(int(c.split('_d')[1]) for c in d.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    lab = d['label'].to_numpy()
    corr = np.stack([(d[f'pred_d{k}'].to_numpy() == lab) for k in ks], axis=1)
    acc_k = corr.mean(axis=0)
    acc_full = float(acc_k[-1])
    return {
        'acc_full': acc_full,
        'any_correct': float(corr.any(axis=1).mean()),
        'excess': float(corr.any(axis=1).mean()) - acc_full,
        'exit_quality': float(np.mean(acc_k[:-1] / max(acc_full, 1e-9))),
        'acc_first_exit': float(acc_k[0]),
        'K': len(ks),
    }

# THE accessor -- it validates the columns and applies the dedupe rule, so
# neither is re-typed here (rule 4). `require=False` because the joint arm may
# legitimately not exist yet; the explicit check below says so clearly.
sel = measured_runs(methods=('base', 'jointexit'), require=False)
if sel.empty:
    raise RuntimeError('no CIFAR-100 base or jointexit runs found')
df = pd.DataFrame([
    {**dict(row), 'arm': 'joint' if row['method'] == 'jointexit' else 'frozen',
     **exit_stats(row['run_id'])}
    for _, row in sel.iterrows()])
M.save_analysis(sess.data_dir, 's3_frozen_vs_joint', df)

joint_archs = sorted(df[df['arm'] == 'joint']['arch'].unique())
if not joint_archs:
    # Distinguish 'never trained' from 'trained but not measured'. The second
    # is what actually happened (D-88), and the two need different actions.
    trained = sorted(d.name for d in runs_dir.iterdir()
                     if d.is_dir() and 'jointexit' in d.name
                     and (d / 'summary.json').exists())
    if trained:
        raise RuntimeError(
            f'{len(trained)} joint run(s) are TRAINED but have no '
            f'per_sample/test.parquet: {trained}. Training finished; the '
            'MEASUREMENT stage did not run. Re-run the measurement cell in '
            'S3_NB1 -- it is inference only, ~30-40 min, and the checkpoints '
            'are already on disk.')
    raise RuntimeError('no joint runs at all -- run S3_NB1 first.')
print(f'{len(df)} run(s); joint architectures: {joint_archs}')
"""),
        md("""
---
## The paired table
"""),
        code("""
piv = (df[df['arch'].isin(joint_archs)]
       .groupby(['arch', 'arm'])[['acc_full', 'excess', 'exit_quality',
                                  'acc_first_exit']]
       .mean().unstack('arm'))
print((piv * 100).round(2).to_string())
print()

comp = []
for a in joint_archs:
    f = df[(df['arch'] == a) & (df['arm'] == 'frozen')]
    j = df[(df['arch'] == a) & (df['arm'] == 'joint')]
    if f.empty or j.empty:
        print(f'  {a}: missing an arm, skipped')
        continue
    comp.append({
        'arch': a,
        'excess_frozen': f['excess'].mean() * 100,
        'excess_joint': j['excess'].mean() * 100,
        'd_excess': (j['excess'].mean() - f['excess'].mean()) * 100,
        'accfull_frozen': f['acc_full'].mean() * 100,
        'accfull_joint': j['acc_full'].mean() * 100,
        'd_accfull': (j['acc_full'].mean() - f['acc_full'].mean()) * 100,
        'eq_frozen': f['exit_quality'].mean(),
        'eq_joint': j['exit_quality'].mean(),
    })
cmp_df = pd.DataFrame(comp)
M.save_analysis(sess.data_dir, 's3_q1_comparison', cmp_df)
print(cmp_df.round(2).to_string(index=False))
"""),
        md("""
---
## H1 — the verdict

Reported against the threshold fixed **before** any of this was run.
"""),
        code("""
THRESH = 2.0
n_ok = int((cmp_df['excess_joint'] >= THRESH).sum())
n_tot = len(cmp_df)

print(f'excess under JOINT training, per architecture:')
for _, r in cmp_df.iterrows():
    mark = 'PASS' if r['excess_joint'] >= THRESH else 'below'
    print(f"    {r['arch']:14s} {r['excess_joint']:+6.2f} pt   "
          f"(frozen {r['excess_frozen']:+6.2f}, change {r['d_excess']:+6.2f})  {mark}")

print()
print(f'H1 (>= {THRESH} pt in 3 of 3): {n_ok} of {n_tot}')
if n_ok == n_tot and n_tot > 0:
    print('=> H1 SUPPORTED. Study 2 is NOT a frozen-backbone artifact.')
    print('   Report both numbers; the archival paper is unblocked.')
elif cmp_df['excess_joint'].max() >= 0.5:
    print('=> PARTIAL. Real but smaller than +6.86 pt. The paper must lead with')
    print('   the JOINT number and report frozen as an upper bound.')
else:
    print('=> H1 FALSIFIED. The excess IS an artifact of post-hoc exits.')
    print('   Withdraw Study 2 to a methodological note and make Q3 the main')
    print('   paper. Report this with equal prominence -- "oracle bounds are')
    print('   sound when exits are trained jointly" is itself useful.')

print()
print('--- R-02: is the comparison confounded by backbone accuracy? ---')
mx = cmp_df['d_accfull'].abs().max()
print(f'largest change in acc_full: {mx:.2f} pt')
if mx > 1.0:
    print('EXCEEDS 1 pt -- joint training moved the backbone, so the raw excess')
    print('comparison confounds exit quality with network quality.')
    print('The conditioned number below is PRIMARY.')
    slope = np.polyfit(df['acc_full'], df['excess'], 1)[0]
    cmp_df['d_excess_adj'] = (cmp_df['d_excess']
                              - slope * 100 * cmp_df['d_accfull'] / 100)
    print()
    print(cmp_df[['arch', 'd_excess', 'd_accfull', 'd_excess_adj']]
          .round(2).to_string(index=False))
else:
    print('within 1 pt -- the raw comparison stands, no adjustment needed.')
"""),
        md("""
---
## Did joint training actually work?

If `exit_quality` barely moved, the joint runs are not meaningfully different
from the frozen ones and H1 has not been tested at all — the experiment failed
rather than the hypothesis. That is a different outcome and must not be
confused with a result.
"""),
        code("""
d_eq = (cmp_df['eq_joint'] - cmp_df['eq_frozen'])
print('exit_quality, frozen -> joint:')
for _, r in cmp_df.iterrows():
    print(f"    {r['arch']:14s} {r['eq_frozen']:.3f} -> {r['eq_joint']:.3f}  "
          f"({r['eq_joint'] - r['eq_frozen']:+.3f})")
print()
if d_eq.max() < 0.02:
    raise RuntimeError(
        'exit_quality barely moved (max +{:.3f}). Joint training did not '
        'produce meaningfully better exits, so H1 was never actually tested. '
        'This is an EXPERIMENT failure, not a hypothesis result -- check the '
        'loss weighting and that joint_exits reached the training loop.'
        .format(d_eq.max()))
print(f'exit quality improved by up to {d_eq.max():+.3f} -- joint training did')
print('change what it was supposed to change, so H1 is a real test.')

print()
print('Compare against S3_NB0s prediction (analysis/s3_exit_quality.csv):')
print('  did the measured joint excess land where the extrapolation said?')
print('  If not, the extrapolation had no power -- worth recording either way.')
"""),
        md("""
---
## Next

Record the verdict in `study3/03_LOG.md`, then:

- **H1 supported** → `S3_NB3_Router` (Q2), then update `study2/PAPER.md` with
  both numbers and submit.
- **H1 falsified** → skip Q2. Go to `S3_NB4_Pruning` (Q3), which is independent
  of all of this, and rewrite `study2/PAPER.md` as a methodological note.
"""),
    ])


# ---------------------------------------------------------------------------
# S3_NB3 -- the learned router (Q2)
# ---------------------------------------------------------------------------
def nb3():
    return notebook([
        md("""
# S3_NB3 — how much of the gap can a learned router capture?

**~5 GPU-hours · feature dump + a small gate per exit**

## The overclaim this fixes

Study 2 says the oracle excess "cannot be reached by any router". What was
actually shown is that **a second seed** cannot reach it. A learned router with
access to the input might do better, and nobody has measured it.

```
capture fraction = (router − confidence baseline) / (oracle_in − baseline)
```

**Pre-registered (H2):** a learned router captures **< 25 %** of the gap.

| outcome | reading |
|---|---|
| captures most | the field is right, the gap is real headroom, and here is a router |
| captures a little | the bound is mostly noise, now quantified |
| captures none | the strongest version of Study 2's claim |

All three are reportable and two are positive.

## The deployability constraint

A gate at exit *k* may use **only features available at exit k**. Anything else
is not a router, it is an oracle wearing a router's clothes — the exact mistake
`pred_depth` turned out to be in Study 2.

## The control that decides whether the number means anything

Train the gate on seed *i*, evaluate on seed *j*'s network. An in-seed capture
fraction alone is uninterpretable: a gate can fit one seed's noise perfectly.
**Both numbers are reported, always.**
"""),
        code(bootstrap_cell()),
        code(paths_cell(phase="p4", needs_data=True)),
        code(runs_cell()),
        code(DATA_CELL),
        md("""
---
## Dump per-exit features

One forward pass per run. ~50 MB per run at 256 dims — cheap, and it is the
only thing standing between us and a learned router.
"""),
        code("""
import numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path

ARCHS = ['resnet20', 'resnet32x4', 'vgg8']
SEEDS = [1, 2]          # two seeds: the cross-seed control needs them
feat_dir = Path(MSC_ROOT) / 'features'
feat_dir.mkdir(parents=True, exist_ok=True)

def dump_features(run_id, cfg):
    out = feat_dir / f'{run_id}.npz'
    if out.exists():
        return out
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    L = M.run_layout(sess.work, run_id)
    blob = torch.load(L['checkpoints'] / 'ckpt_best.pt', map_location=device,
                      weights_only=False)
    backbone = M.place_model(M.build_model(cfg['arch'], cfg['num_classes']),
                             device, cfg, tag='feature dump')
    backbone.load_state_dict(blob['model'], strict=True)
    me = M.place_model(M.MultiExitModel(backbone, cfg['num_classes'], freeze=True),
                       device, cfg)
    hp = M.exit_heads_path(sess.work, run_id)
    me.heads.load_state_dict(torch.load(hp, map_location=device,
                                        weights_only=False)['heads'])
    me.eval()

    _, val_loader, _, _, _ = M.build_loaders(cfg)
    feats, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device, non_blocking=True)
            fs = me.backbone.forward_features(x)
            pooled = []
            for f in fs:
                if f.dim() == 4:
                    pooled.append(nn.functional.adaptive_avg_pool2d(f, 1)
                                  .flatten(1).float().cpu().numpy())
                elif f.dim() == 3:
                    pooled.append((f[:, 0] if me.token_model
                                   else f.mean(1)).float().cpu().numpy())
                else:
                    pooled.append(f.flatten(1).float().cpu().numpy())
            feats.append(pooled)
            labels.append(batch[1].numpy())
    K = len(feats[0])
    stacked = {f'f{k}': np.concatenate([b[k] for b in feats], axis=0)
               for k in range(K)}
    np.savez_compressed(out, label=np.concatenate(labels), **stacked)
    del me, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

paths = {}
for a in ARCHS:
    for s in SEEDS:
        cfg = sess.config(a, seed=s, method='base')
        rid = cfg['run_id']
        if not (Path(MSC_ROOT) / 'runs' / rid / 'per_sample' / 'test.parquet').exists():
            print(f'  missing {rid}, skipped')
            continue
        paths[rid] = dump_features(rid, cfg)
        mb = paths[rid].stat().st_size / 2**20
        print(f'  {rid:38s} {mb:6.1f} MB')
print(f'\\n{len(paths)} feature dump(s)')
"""),
        md("""
---
## Train a gate per exit, then measure capture

The gate is deliberately small. A large one would fit the seed's noise and
inflate the in-seed number — which the cross-seed control would then expose,
but it is cheaper not to invite the problem.
"""),
        code("""
from sklearn.linear_model import LogisticRegression   # small on purpose

def gate_scores(train_rid, eval_rid):
    '''Train per-exit gates on train_rid, score eval_rid's samples.'''
    tr = np.load(paths[train_rid]); ev = np.load(paths[eval_rid])
    dtr = pd.read_parquet(Path(MSC_ROOT) / 'runs' / train_rid
                          / 'per_sample' / 'test.parquet').sort_values('sample_idx')
    ks = sorted(int(c.split('_d')[1]) for c in dtr.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    lab_tr = dtr['label'].to_numpy()
    out = []
    for i, k in enumerate(ks[:-1]):        # no gate needed at the final exit
        y = (dtr[f'pred_d{k}'].to_numpy() == lab_tr).astype(int)
        Xtr, Xev = tr[f'f{i}'], ev[f'f{i}']
        if y.min() == y.max():
            out.append(np.full(len(Xev), float(y.mean())))
            continue
        clf = LogisticRegression(max_iter=300, C=0.1)
        clf.fit(Xtr, y)
        out.append(clf.predict_proba(Xev)[:, 1])
    return np.stack(out, axis=1), ks

def correctness(rid):
    d = pd.read_parquet(Path(MSC_ROOT) / 'runs' / rid / 'per_sample'
                        / 'test.parquet').sort_values('sample_idx')
    ks = sorted(int(c.split('_d')[1]) for c in d.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    lab = d['label'].to_numpy()
    corr = np.stack([(d[f'pred_d{k}'].to_numpy() == lab) for k in ks], axis=1).astype(float)
    conf = np.stack([d[f'top1p_d{k}'].to_numpy() for k in ks], axis=1)
    return corr, conf, ks
"""),
        md("""
---
## Evaluate at matched budget

Routing helpers are **imported from Study 2's notebook logic**, re-implemented
here only because the notebook is standalone — but the canaries in
`tools/s2_routing_canaries.py` cover the same functions and must pass first.
"""),
        code("""
def _cost(k, rho):  return float(np.mean(np.asarray(rho)[k]))

def route_confidence(conf, correct, rho, target):
    n, K = correct.shape
    lo, hi = 0.0, 1.0
    for _ in range(60):
        th = (lo + hi) / 2
        fires = conf >= th; fires[:, -1] = True
        k = fires.argmax(axis=1); c = _cost(k, rho)
        if c < target: lo = th
        else: hi = th
    return float(correct[np.arange(n), k].mean()), c

def route_gate(pgate, correct, rho, target):
    '''Exit at the first exit whose gate probability clears a threshold.'''
    n, K = correct.shape
    lo, hi = 0.0, 1.0
    for _ in range(60):
        th = (lo + hi) / 2
        fires = np.concatenate([pgate >= th, np.ones((n, 1), bool)], axis=1)
        k = fires.argmax(axis=1); c = _cost(k, rho)
        if c < target: lo = th
        else: hi = th
    return float(correct[np.arange(n), k].mean()), c

def route_oracle(cc, ce, rho, target):
    rho = np.asarray(rho, float)
    lo, hi = 0.0, 100.0
    for _ in range(80):
        lam = (lo + hi) / 2
        k = (cc - lam * rho[None, :]).argmax(axis=1)
        if float(rho[k].mean()) > target: lo = lam
        else: hi = lam
    k = (cc - hi * rho[None, :]).argmax(axis=1)
    return float(ce[np.arange(len(ce)), k].mean()), float(rho[k].mean())

TARGET = 0.80
rows = []
for a in ARCHS:
    rids = [r for r in paths if M.parse_run_id(r)['arch'] == a]
    if len(rids) < 2:
        continue
    rho = M.load_or_build_budgets(a, sess.work, 'cifar100')['axes']['depth']['rho']
    i, j = sorted(rids)[0], sorted(rids)[1]
    for train_on, eval_on, kind in [(i, i, 'in-seed'), (i, j, 'cross-seed')]:
        corr, conf, ks = correctness(eval_on)
        base, _ = route_confidence(conf, corr, rho, TARGET)
        orac, _ = route_oracle(corr, corr, rho, TARGET)
        pg, _ = gate_scores(train_on, eval_on)
        gt, _ = route_gate(pg, corr, rho, TARGET)
        gap = orac - base
        rows.append({'arch': a, 'kind': kind, 'baseline': base * 100,
                     'router': gt * 100, 'oracle': orac * 100,
                     'gap': gap * 100, 'router_gain': (gt - base) * 100,
                     'capture': (gt - base) / gap if gap > 1e-9 else np.nan})

cap = pd.DataFrame(rows)
M.save_analysis(sess.data_dir, 's3_router_capture', cap)
print(cap.round(3).to_string(index=False))
print()
for kind in ['in-seed', 'cross-seed']:
    sub = cap[cap['kind'] == kind]
    if len(sub):
        print(f'  {kind:11s} median capture = {sub["capture"].median()*100:6.2f} %')
print()
cs = cap[cap['kind'] == 'cross-seed']['capture'].median()
print(f'H2 (< 25 % captured): '
      f'{"SUPPORTED" if cs < 0.25 else "FALSIFIED"}  (cross-seed {cs*100:.1f} %)')
print()
print('The CROSS-SEED number is the one that means anything. If in-seed capture')
print('is high and cross-seed is not, the gate memorised one seed s noise --')
print('which is Study 2 s finding restated, not a contradiction of it.')
"""),
        md("""
---
## Canaries — the gate must be shown to work and to fail
"""),
        code("""
rng = np.random.default_rng(0)
n, K = 4000, 5
rho = [0.2, 0.4, 0.6, 0.8, 1.0]
easy = rng.random(n) < 0.5
cc = np.zeros((n, K)); cc[easy, :] = 1.0; cc[~easy, K-1] = 1.0

# a gate handed the truth must capture ~everything
perfect = np.repeat(easy[:, None].astype(float), K-1, axis=1)
b, _ = route_confidence(rng.random((n, K)), cc, rho, 0.7)
o, _ = route_oracle(cc, cc, rho, 0.7)
g, _ = route_gate(perfect, cc, rho, 0.7)
capture = (g - b) / (o - b) if o > b else float('nan')
print(f'{"PASS" if capture > 0.8 else "FAIL"}  oracle-gate captures '
      f'{capture*100:.0f}% (must be ~100)')

# a gate handed noise must capture ~nothing
g2, _ = route_gate(rng.random((n, K-1)), cc, rho, 0.7)
cap2 = (g2 - b) / (o - b) if o > b else float('nan')
print(f'{"PASS" if cap2 < 0.25 else "FAIL"}  noise-gate captures '
      f'{cap2*100:.0f}% (must be ~0)')
"""),
        md("""
---
## Next

Record in `study3/03_LOG.md`. Then `S3_NB4_Pruning` (Q3), which is independent
of everything above.
"""),
    ])


# ---------------------------------------------------------------------------
# S3_NB4 -- pruning (Q3)
# ---------------------------------------------------------------------------
def nb4():
    return notebook([
        md(f"""
# S3_NB4 — does the memorisation collapse damage pruning?

**~18 GPU-hours · 14 runs · fully resumable · independent of Q1 and Q2**

## The claim being tested

Study 2 §4.3: softmax difficulty scores lose seed reliability on data the model
has **memorised** — `ce_loss` ρ_seed falls 0.647 → 0.108 — predicted by
saturation (ρ = +0.832) and **not** by test accuracy (−0.114).

Dataset pruning computes exactly these scores, on exactly this data, from a
single seed. **A warning is not a demonstration.** This measures the damage.

| source | train acc | > 0.99 conf | ρ_seed drop |
|---|---|---|---|
| `{SATURATED}` (saturated) | 99.99 % | 71 % | 0.558 |
| `{UNSATURATED}` (unsaturated) | 83.1 % | 16 % | 0.026 |

**Pre-registered (H3):** the saturated source is worse by **≥ 1.0 pt** at 30 %
retention, and the gap widens as retention falls.
**H3b:** the saturated source is indistinguishable from **random** pruning
(±0.5 pt) at 30 %. If true, that is the headline.

## The cheap gate runs first

If the two sources keep nearly the same samples, no downstream difference is
possible and 18 GPU-hours buy nothing. That check costs minutes.
"""),
        code(bootstrap_cell()),
        code(paths_cell(phase="p5", needs_data=True)),
        code(runs_cell()),
        code(DATA_CELL),
        md("""
---
## Gate — do the two sources even disagree?
"""),
        code(f"""
import numpy as np, pandas as pd
from pathlib import Path

SATURATED   = {SATURATED!r}
UNSATURATED = {UNSATURATED!r}
RATES       = {list(PRUNE_RATES)!r}
TARGET_ARCH = {PRUNE_TARGET!r}
SCORE       = 'ce_loss'

runs_dir = Path(MSC_ROOT) / 'runs'

def train_score(arch, seed=1):
    hits = [d.name for d in runs_dir.iterdir()
            if d.is_dir() and f'-{{arch}}-cifar100-base-s{{seed}}' in d.name
            and (d / 'per_sample' / 'train_holdout.parquet').exists()]
    if not hits:
        raise RuntimeError(f'no train_holdout parquet for {{arch}} seed {{seed}}')
    d = pd.read_parquet(runs_dir / sorted(hits)[-1] / 'per_sample'
                        / 'train_holdout.parquet')
    return d.set_index('sample_idx')[SCORE].astype(float)

s_sat = train_score(SATURATED)
s_uns = train_score(UNSATURATED)
common = s_sat.index.intersection(s_uns.index)
print(f'{{len(common):,}} samples scored by both sources')

print()
print('kept-set overlap (keeping the HARDEST samples):')
overlaps = {{}}
for rate in RATES:
    n_keep = int(len(common) * rate)
    a = set(s_sat.loc[common].nlargest(n_keep).index)
    b = set(s_uns.loc[common].nlargest(n_keep).index)
    ov = len(a & b) / n_keep
    overlaps[rate] = ov
    print(f'    keep {{rate:.0%}}:  {{ov:.1%}} of the kept samples are shared')

if min(overlaps.values()) >= 0.90:
    raise RuntimeError(
        f'overlap >= 90% at every rate ({{overlaps}}). The two sources disagree '
        'on RELIABILITY but agree on WHICH SAMPLES TO KEEP, so no downstream '
        'difference is possible. CANCEL this experiment and report the overlap '
        'itself -- that is a finding, and it cost minutes instead of 18 GPU-h.')
print()
print('the sources disagree enough for a downstream difference to be possible')
"""),
        md("""
---
## Build the subsets

Three arms per retention rate: saturated-guided, unsaturated-guided, random.
Plus a full-data control. Two seeds each for the trained arms.
"""),
        code("""
import json
subset_dir = Path(MSC_ROOT) / 'subsets'
subset_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
specs = []
for rate in RATES:
    n_keep = int(len(common) * rate)
    keeps = {
        'sat':  sorted(int(i) for i in s_sat.loc[common].nlargest(n_keep).index),
        'uns':  sorted(int(i) for i in s_uns.loc[common].nlargest(n_keep).index),
        'rand': sorted(int(i) for i in rng.choice(np.asarray(common),
                                                  n_keep, replace=False)),
    }
    for arm, idx in keeps.items():
        p = subset_dir / f'keep_{arm}_{int(rate*100)}.json'
        p.write_text(json.dumps({'arm': arm, 'rate': rate, 'score': SCORE,
                                 'n_keep': len(idx), 'keep': idx}))
        specs.append({'arm': arm, 'rate': rate, 'path': str(p), 'n': len(idx)})

print(pd.DataFrame(specs)[['arm', 'rate', 'n']].to_string(index=False))
print()
print(f'{len(specs)} subset(s) written to {subset_dir}')
"""),
        md("""
---
## Train the target on each subset

`resnet20`, ~1.3 GPU-h per run. **Resumable and safe to stop** — same machinery
as every other training notebook in this project.

`subset_path` is part of the config, so it is inside `config_hash`: two arms can
never silently share a checkpoint.
"""),
        code("""
SEEDS = [1, 2]
cfgs = []
for sp in specs:
    for sd in SEEDS:
        cfgs.append(sess.config(
            TARGET_ARCH, seed=sd,
            method=f"prune{sp['arm']}{int(sp['rate']*100)}",
            subset_path=sp['path']))
# full-data controls
for sd in SEEDS:
    cfgs.append(sess.config(TARGET_ARCH, seed=sd, method='prunefull'))

for c in cfgs:
    print(f"  {c['run_id']:44s} subset={Path(c.get('subset_path', '-')).name}")
print(f'\\n{len(cfgs)} run(s)')

# The library must actually honour subset_path. If it silently ignores an
# unknown config key, every arm trains on the full dataset and the experiment
# returns a perfect null that looks like a result.
import inspect
_src = inspect.getsource(M.build_loaders)
if 'subset_path' not in _src:
    raise RuntimeError(
        'build_loaders does not read subset_path. Every arm would train on the '
        'FULL dataset and produce identical results -- a null that looks like a '
        'finding. Implement subset support before spending 18 GPU-hours.')
print('build_loaders honours subset_path')
"""),
        code("""
ok = all(M.backbone_dry_run(c) for c in cfgs[:2])
print('dry run ok' if ok else 'DRY RUN FAILED')

results = sess.run_all(cfgs, title='Study 3 Q3 / pruning')
for r in results:
    print(f"  {r.get('status','?'):9s} {r.get('run_id','?')}  "
          f"top1={M.fmt_metric(r.get('best_accuracy'))}")
"""),
        md("""
---
## H3 — the verdict
"""),
        code("""
rows = []
for c in cfgs:
    L = M.run_layout(sess.work, c['run_id'])
    s = L['base'] / 'summary.json'
    if not s.exists():
        continue
    d = json.loads(s.read_text())
    m = M.parse_run_id(c['run_id'])
    meth = m['method']
    arm = ('full' if meth == 'prunefull'
           else ''.join(ch for ch in meth[5:] if not ch.isdigit()))
    rate = ''.join(ch for ch in meth if ch.isdigit()) or '100'
    rows.append({'arm': arm, 'rate': int(rate), 'seed': m['seed'],
                 'acc': d.get('best_accuracy') or d.get('final_acc')})

pr = pd.DataFrame(rows).dropna()
M.save_analysis(sess.data_dir, 's3_pruning', pr)
tab = pr.groupby(['rate', 'arm'])['acc'].agg(['mean', 'std', 'count'])
print((tab * 100).round(2).to_string())

print()
for rate in sorted(pr['rate'].unique()):
    if rate == 100:
        continue
    sub = pr[pr['rate'] == rate]
    g = sub.groupby('arm')['acc'].mean() * 100
    if 'sat' in g and 'uns' in g:
        gap = g['uns'] - g['sat']
        print(f'  keep {rate}%:  unsaturated {g["uns"]:.2f}  '
              f'saturated {g["sat"]:.2f}  gap {gap:+.2f} pt', end='')
        if 'rand' in g:
            print(f'   random {g["rand"]:.2f}  '
                  f'(saturated - random {g["sat"]-g["rand"]:+.2f})')
        else:
            print()

print()
print('H3  (unsaturated beats saturated by >= 1.0 pt at 30% retention)')
print('H3b (saturated indistinguishable from random, +/- 0.5 pt at 30%)')
print()
print('If H3b holds, the headline is: a single-seed difficulty score from a')
print('memorising model is worth no more than chance.')
"""),
    ])


# ---------------------------------------------------------------------------
# S3_NB5 -- publish, once, at the end
# ---------------------------------------------------------------------------
def nb5():
    return notebook([
        md("""
# S3_NB5 — publish everything to HuggingFace, in one pass

**Run this LAST, when you have a network. Nothing else in Study 3 touches
HuggingFace at all.**

## Why this notebook exists

The first joint run died like this:

```
[HF:hub] AUTH FAILURE -- check HF_TOKEN write scope
[HF:hub] BATCH FAILED after 8 attempts (17 files): 403 Forbidden
```

Two problems, and the second is the structural one.

1. **It was pushing to `msc-imagenet100`** — the library's default repo — during
   a CIFAR-100 study.
2. **It was pushing at all.** A background uploader that retries every 30
   minutes turns *"the network is down right now"* into *"the training run
   failed"*. On a machine without a permanent connection that is the wrong
   default, and it cost a run that had already passed its dry run.

So Study 3 trains **completely offline**. Every artifact is written to disk in
full — configs, epoch histories, telemetry, per-sample parquets, checkpoints,
environment records — and this notebook uploads the finished tree in one pass.

**The local tree is the source of truth.** HuggingFace is a copy of it.
"""),
        code(bootstrap_cell()),
        code(paths_cell(phase="analysis", hf=True)),
        md("""
---
## Check the token BEFORE uploading anything

`hf_token_check` was written after a 403 arrived under forty lines of
traceback (D-84). It answers three questions separately — is the token valid,
does it have **write** scope, and does it reach **this** repo — so a failure
names its own cause.
"""),
        code("""
import os
TOKEN = os.environ.get('HF_TOKEN') or ''
REPO  = os.environ.get('MSC_HF_REPO', 'Shanmuk4622/msc-cifar100')

print(f'repo  : {REPO}')
print(f'token : {"set, " + str(len(TOKEN)) + " chars" if TOKEN else "NOT SET"}')
print()

chk = M.hf_token_check(TOKEN or None, REPO)
for k in ('ok', 'valid', 'can_write', 'user', 'namespace', 'reason'):
    if k in chk:
        print(f'  {k:12s} {chk[k]}')

if not chk.get('ok'):
    raise RuntimeError(
        'token check FAILED -- fix this before uploading. The most common cause '
        'is a READ token: HuggingFace -> Settings -> Access Tokens -> New token '
        'with the WRITE role, then set HF_TOKEN. Nothing has been uploaded.')
print()
print('token OK -- safe to upload')
"""),
        md("""
---
## What is about to be uploaded

Listed and sized **before** anything moves, so a 40 GB surprise is visible
while it is still cancellable.
"""),
        code("""
from pathlib import Path
import pandas as pd

root = Path(MSC_ROOT)
INCLUDE = ['runs', 'analysis', 'budgets', 'registry', 'tables', 'paper']
SKIP_SUFFIX = {'.tmp', '.lock'}

rows = []
for top in INCLUDE:
    d = root / top
    if not d.is_dir():
        continue
    for p in d.rglob('*'):
        if p.is_file() and p.suffix not in SKIP_SUFFIX:
            rows.append({'top': top, 'path': str(p.relative_to(root)),
                         'mb': p.stat().st_size / 2**20,
                         'ckpt': p.suffix == '.pt'})

files = pd.DataFrame(rows)
if files.empty:
    raise RuntimeError(f'nothing to upload under {root}')

by_top = files.groupby('top')['mb'].agg(['count', 'sum']).rename(
    columns={'count': 'files', 'sum': 'MB'})
print(by_top.round(1).to_string())
print()
ck = files[files['ckpt']]['mb'].sum()
print(f'TOTAL      {len(files):,} files   {files["mb"].sum()/1024:.2f} GB')
print(f'  of which checkpoints (.pt): {ck/1024:.2f} GB')
print()
print('Checkpoints are ~95% of the bytes and are rarely re-opened. Set')
print('UPLOAD_CHECKPOINTS = False below to skip them if bandwidth is scarce --')
print('every analysis in Studies 2 and 3 runs from the parquets alone.')
"""),
        md("""
---
## Upload

`hf_upload_resilient` batches, retries with backoff, and respects the 128
commits/hour ceiling. It is the same path the training notebooks used to call
in the background — the difference is that here it runs **once**, deliberately,
with a network you know is up.

Interrupting is safe: it uploads in batches and re-running skips what already
landed.
"""),
        code("""
UPLOAD_CHECKPOINTS = True     # <<< False to skip the ~95% that is .pt files

# the file-level view, used for the size report and the resolve probe below
sel = files if UPLOAD_CHECKPOINTS else files[~files['ckpt']]
print(f'{len(sel):,} file(s), {sel["mb"].sum()/1024:.2f} GB')

# hf_upload_resilient takes (local_path, path_in_repo, label) triples and
# uploads FOLDER AT A TIME -- that is the unit it can retry and resume at.
items = []
for top in INCLUDE:
    d = root / top
    if not d.is_dir():
        continue
    if top == 'runs':
        for run in sorted(x for x in d.iterdir() if x.is_dir()):
            items.append((str(run), f'runs/{run.name}', run.name))
    else:
        items.append((str(d), top, top))

if not UPLOAD_CHECKPOINTS:
    print('NOTE: UPLOAD_CHECKPOINTS=False, but hf_upload_resilient uploads whole')
    print('folders. Checkpoints live inside each run folder, so skipping them')
    print('means excluding them here rather than filtering the file list.')
    import shutil, tempfile
    stage = Path(tempfile.mkdtemp(prefix='msc_nockpt_'))
    items = []
    for run in sorted(x for x in (root / 'runs').iterdir() if x.is_dir()):
        dst = stage / run.name
        shutil.copytree(run, dst,
                        ignore=shutil.ignore_patterns('*.pt', 'checkpoints'))
        items.append((str(dst), f'runs/{run.name}', run.name))
    for top in INCLUDE:
        if top != 'runs' and (root / top).is_dir():
            items.append((str(root / top), top, top))

print(f'uploading {len(items)} folder(s)')
res = M.hf_upload_resilient(token=TOKEN, repo_id=REPO, repo_type='dataset',
                            items=items)
print()
for k, v in (res or {}).items():
    print(f'  {k}: {v}')
"""),
        md("""
---
## Verify by `resolve`, not by the queue draining

**Rule 9 and rule 10.** A drained upload queue is not confirmation that the
files are there, and `list_repo_files` has lied before. The only trustworthy
check is asking HuggingFace to resolve specific paths and seeing a 200.
"""),
        code("""
# `sess.hub.resolve_meta` is THE rule-9 check: it asks HuggingFace to resolve a
# specific path and returns None only for a genuine 404, raising instead of
# reporting absence when the lookup itself failed.
probe = sorted(set(
    sel.sample(min(8, len(sel)), random_state=0)['path'].tolist()
    + [p for p in sel['path'] if p.endswith('per_sample/test.parquet')][:3]
    + [p for p in sel['path'] if p.endswith('summary.json')][:3]))

ok = miss = 0
for rel in probe:
    meta = sess.hub.resolve_meta(Path(rel).as_posix())
    print(f'  {"OK  " if meta else "MISS"} {rel}'
          + (f'  ({meta["size"]:,} B)' if meta and meta.get('size') else ''))
    ok, miss = ok + bool(meta), miss + (not meta)

print()
if miss:
    raise RuntimeError(
        f'{miss} of {ok + miss} probed files did NOT resolve. A drained upload '
        'queue is not confirmation (rule 10) -- re-run the upload cell.')
print(f'{ok} probed file(s) resolve on HuggingFace')
print()
print('The local tree remains the source of truth. This is a copy of it.')
"""),
    ])


NOTEBOOKS = {
    "S3_NB0_Extrapolate.ipynb": nb0,
    "S3_NB1_JointTrain.ipynb": nb1,
    "S3_NB2_Compare.ipynb": nb2,
    "S3_NB3_Router.ipynb": nb3,
    "S3_NB4_Pruning.ipynb": nb4,
    "S3_NB5_Publish.ipynb": nb5,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    for name, fn in NOTEBOOKS.items():
        nb = fn()
        (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        kb = (OUT / name).stat().st_size // 1024
        print(f"  {name:30s} {n_code:2d} code + {n_md:2d} md   {kb:5d} KB")

    print("\n  parsing every cell as Python 3.10")
    import ast
    bad = 0
    for name in NOTEBOOKS:
        nb = json.loads((OUT / name).read_text(encoding="utf-8"))
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code":
                continue
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                bad += 1
                print(f"  [FAIL] {name} cell {i}: {e.msg} (line {e.lineno})")
    if bad:
        print("\n  Generation refused.")
        return 1
    print("  all cells parse")
    print("\n  checking for names no earlier cell defines")
    files = [str(OUT / n) for n in NOTEBOOKS]
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_names.py"),
                        *files], capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if r.returncode:
        print("\n  Generation refused.")
        return 1

    print(f"\nOK -- {len(NOTEBOOKS)} notebook(s) in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
