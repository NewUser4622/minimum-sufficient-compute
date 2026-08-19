#!/usr/bin/env python3
"""
build_notebooks_study2.py -- generate the Study 2 notebooks.

Study 2 is a CPU re-analysis of artifacts Study 1 already produced. It needs no
GPU and trains no models: every measured run's `per_sample/test.parquet` carries
per-exit predictions (`pred_d1..dK`, `top1p_d*`, `top2p_d*`), the label, and
eight difficulty scores, all keyed by a global `sample_idx`.

    S2_NB0_Fetch        pull the CIFAR-100 runs from HuggingFace
    S2_NB1_Reliability  P-1 verify · P0a collinearity · P0b reliability atlas
    S2_NB2_Ceiling      P1 optimism bias · P1b honest ceiling

Everything that guards the Study 1 notebooks guards these: the embedded-library
bootstrap with its build stamp (D-62/D-68), the undefined-name check across
cells (D-82), the Python-3.10 parse gate (D-73), and the column/path/arity
validator.

    python build_notebooks_study2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_notebooks_in100 import (  # noqa: E402
    bootstrap, code, md, notebook, paths_cell,
)

OUT = ROOT / "notebooks_study2"

SCORES = ("msp", "margin", "entropy", "ce_loss", "el2n", "forget_events",
          "pred_depth", "msc")


# ---------------------------------------------------------------------------
def nb0():
    return notebook([
        md("""
# S2 · NB0 — fetch the Study 1 runs from HuggingFace

Study 2 re-analyses artifacts Study 1 already produced. This pulls them down
into the layout the library expects, so every later notebook reads local files.

**What is fetched, and what is not.** Only the small artifacts are needed:
`per_sample/*.parquet`, `metrics/*.csv`, `summary.json`, `config.yaml`, and
`budgets/`. **Checkpoints are skipped** — Study 2 never runs a model, and they
are ~95% of the bytes.

If a download fails part way, re-run this cell: it skips runs already complete
on disk, exactly like `NB6_Publish` does in the other direction.
"""),
        code(bootstrap()),
        code(paths_cell(phase="p1", detect=False)),
        code("""
REPO_ID   = 'Shanmuk4622/msc-cifar100'
REPO_TYPE = 'dataset'

# Study 2 reads parquet and CSV only. Checkpoints are ~95% of the repo and are
# never opened, so they are excluded rather than downloaded and ignored.
WANT = ['runs/*/per_sample/*.parquet',
        'runs/*/per_sample/meta.json',
        'runs/*/metrics/*.csv',
        'runs/*/summary.json',
        'runs/*/config.yaml',
        'runs/*/config_hash.txt',
        'budgets/*.json',
        'analysis/*.csv']

# This notebook is the only one in Study 2 that touches the network, so it turns
# the offline guard off explicitly and says so (D-83).
print('offline guard BEFORE:')
for k, v in M.offline_state().items():
    print(f'    {k:44s} {v}')
M.allow_network()
print()

import os
HF_TOKEN = os.environ.get('HF_TOKEN')          # a public dataset needs no token
print('HF_TOKEN found' if HF_TOKEN else 'no HF_TOKEN -- fine for a public repo')
"""),
        md("""
---
## Download

`snapshot_download` with `allow_patterns` fetches only what is listed above.
It resumes: files already present and complete are not re-fetched.
"""),
        code("""
from huggingface_hub import snapshot_download
from pathlib import Path

dest = Path(MSC_ROOT)
dest.mkdir(parents=True, exist_ok=True)

print(f'fetching {REPO_ID} -> {dest}')
print('(checkpoints excluded -- Study 2 never runs a model)')

path = snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                         local_dir=str(dest), allow_patterns=WANT,
                         token=HF_TOKEN, max_workers=4)
print(f'\\ndownloaded into {path}')
"""),
        md("""
---
## Verify what landed

Counting files is not verification — a truncated parquet has a size. This opens
one and checks the columns Study 2 actually depends on.
"""),
        code("""
from pathlib import Path
import pandas as pd

runs_dir = Path(MSC_ROOT) / 'runs'
runs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir()) if runs_dir.exists() else []
print(f'{len(runs)} run folder(s) under {runs_dir}')

with_ps = [r for r in runs
           if (runs_dir / r / 'per_sample' / 'test.parquet').exists()]
print(f'{len(with_ps)} have per_sample/test.parquet')

NEED = ['sample_idx', 'label', 'msp', 'margin', 'entropy', 'ce_loss',
        'el2n', 'forget_events', 'pred_depth', 'msc']

if with_ps:
    df = pd.read_parquet(runs_dir / with_ps[0] / 'per_sample' / 'test.parquet')
    missing = [c for c in NEED if c not in df.columns]
    exits = sorted(c for c in df.columns if c.startswith('pred_d'))
    print()
    print(f'sample run: {with_ps[0]}')
    print(f'  rows            {len(df):,}')
    print(f'  per-exit cols   {exits}')
    print(f'  missing scores  {missing if missing else "none"}')
    print(f'  sample_idx max  {int(df["sample_idx"].max()):,}  (global index)')
    if missing:
        print()
        print('  *** Study 2 cannot run without these columns. Stop here and')
        print('  *** tell me which are absent -- the plan changes.')
else:
    print('nothing downloaded -- check REPO_ID and the network')
"""),
        md("""
---
## The raw CIFAR-100 images

Only needed if something has to be recomputed from pixels. The re-analysis
itself does **not** touch images — every quantity Study 2 uses is already in the
parquet.

Expected: `<DATA_DIR>/cifar-100-python/` containing `train`, `test`, `meta`.
"""),
        code("""
from pathlib import Path
p = Path(DATA_DIR) / 'cifar-100-python'
if p.is_dir() and (p / 'train').exists() and (p / 'test').exists():
    print(f'CIFAR-100 present at {p}')
else:
    print(f'CIFAR-100 NOT at {p}')
    print('Set DATA_DIR in cell 2 to the folder CONTAINING cifar-100-python.')
    print(r"  e.g. DATA_DIR = r'C:\\Users\\Administrator\\Desktop\\New folder'")
    print('Not required for the re-analysis -- only if pixels are needed.')
"""),
    ])


# ---------------------------------------------------------------------------
def nb1():
    return notebook([
        md(f"""
# S2 · NB1 — reliability atlas

**P-1 verify → P0a collinearity → P0b atlas.** CPU only, no models.

Answers **R1** (does reliability vary?) and **R2** (does ignoring it distort
comparisons?), and runs the check that decides whether the study is coherent at
all: if the {len(SCORES)} scores are near-collinear they are not
{len(SCORES)} scores.

Scores: {', '.join('`' + s + '`' for s in SCORES)}
"""),
        code(bootstrap()),
        code(paths_cell(phase="p1", detect=False)),
        md("""
---
## P-1 — verify the inventory rather than trusting it

`study2/03_INVENTORY.md` claims what exists. Study 1's defects were repeatedly
"the plan said X, the artifact was Y", so this checks the artifact and refuses
to continue if it disagrees.
"""),
        code(f"""
SCORES = {list(SCORES)}

import pandas as pd
from pathlib import Path

runs_dir = Path(MSC_ROOT) / 'runs'
runs = sorted(d.name for d in runs_dir.iterdir()
              if d.is_dir() and (d / 'per_sample' / 'test.parquet').exists())
print(f'{{len(runs)}} measured run(s)')

meta = []
for r in runs:
    m = M.parse_run_id(r)
    meta.append({{'run_id': r, 'arch': m['arch'], 'seed': m['seed'],
                 'phase': m['phase'], 'method': m['method'],
                 'dataset': m['dataset']}})
mdf = pd.DataFrame(meta)
base = mdf[mdf['method'] == 'base'] if 'method' in mdf else mdf
per_arch = base.groupby('arch')['seed'].nunique().sort_values(ascending=False)
print(f'{{per_arch.size}} architecture(s); '
      f'{{(per_arch >= 2).sum()}} have >=2 seeds (needed for rho_seed)')
print(per_arch.to_string())

# 03_INVENTORY.md claimed eight scores on the test split. Checked against a
# real file, that claim is wrong twice:
#   * `msc` is not stored anywhere -- it is derived, never persisted;
#   * `el2n` and `forget_events` are TRAINING dynamics. The columns exist in
#     test.parquet but are entirely NaN -- a test sample has no training
#     history -- and are populated only in train_holdout.parquet.
# So ask each split what it actually carries. Never assume again.
SPLITS = ['test', 'train_holdout']
AVAIL = {{}}
for sp in SPLITS:
    f = runs_dir / runs[0] / 'per_sample' / f'{{sp}}.parquet'
    if not f.exists():
        print(f'{{sp:14s}} FILE MISSING'); continue
    d = pd.read_parquet(f)
    live, absent, nan_ = [], [], []
    for c in SCORES:
        if c not in d.columns:
            absent.append(c)
        elif d[c].notna().mean() <= 0.5:
            nan_.append(c)
        else:
            live.append(c)
    AVAIL[sp] = live
    print(f'{{sp:14s}} n={{len(d):>6,}}  usable={{len(live)}}/{{len(SCORES)}}  {{live}}')
    if absent:
        print(f'{{"":14s}}   not a column : {{absent}}')
    if nan_:
        print(f'{{"":14s}}   all-NaN here : {{nan_}}')

exits = sorted(c for c in pd.read_parquet(
    runs_dir / runs[0] / 'per_sample' / 'test.parquet').columns
    if c.startswith('pred_d'))
print(f'\\nper-exit columns: {{exits}}')

if not AVAIL.get('test'):
    raise RuntimeError('no usable scores on the test split -- stop')
print()
print('P-1: inventory CORRECTED against the artifacts, not trusted.')
print(f'  routing/bias (NB2) can only use test-split scores : {{AVAIL["test"]}}')
print(f'  the reliability atlas also runs on train_holdout  : '
      f'{{AVAIL.get("train_holdout", [])}}')
"""),
        md("""
---
## P0a — are eight scores really eight?

**Run before anything else.** If `msp`, `margin`, `entropy` and `ce_loss` are
near-collinear — they are all functions of the same softmax — then "eight
scores" overstates the coverage and the framing has to change
(`06_RISK_REGISTER.md` §R-02).
"""),
        code("""
import numpy as np
from scipy.stats import spearmanr

def score_matrix(run_id, split='test', cols=None):
    cols = cols or SCORES
    d = pd.read_parquet(runs_dir / run_id / 'per_sample' / f'{split}.parquet')
    return d.sort_values('sample_idx')[cols].to_numpy(dtype=float)

def collinearity(split, cols):
    # Pairwise-complete |Spearman|. The previous version used a LISTWISE mask
    # -- ~isnan(X).any(axis=1) -- which dropped EVERY row on the test split,
    # because el2n/forget_events are all-NaN there. It then printed n=0, an
    # all-NaN matrix, and the conclusion "the scores carry distinct
    # information". A statistic that returns a reassuring answer from zero
    # samples is the D-37 shape: a check that cannot fail. This one refuses.
    X = score_matrix(runs[0], split, cols)
    cm = pd.DataFrame(np.nan, index=cols, columns=cols)
    npair = pd.DataFrame(0, index=cols, columns=cols)
    for a_ in range(len(cols)):
        for b_ in range(len(cols)):
            m_ = ~(np.isnan(X[:, a_]) | np.isnan(X[:, b_]))
            npair.iloc[a_, b_] = int(m_.sum())
            if m_.sum() > 100 and X[m_, a_].std() > 0 and X[m_, b_].std() > 0:
                cm.iloc[a_, b_] = abs(spearmanr(X[m_, a_], X[m_, b_])[0])
    nmin = int(npair.values.min())
    if nmin == 0 or bool(cm.isna().all().all()):
        raise RuntimeError(
            f'P0a on {split}: no usable correlations (min pairwise n={nmin}). '
            'Refusing to conclude anything about collinearity.')
    return cm, nmin

CM = {}
for sp, cols in AVAIL.items():
    if len(cols) < 2:
        continue
    cm_, nmin_ = collinearity(sp, cols)
    CM[sp] = cm_
    print(f'|Spearman| between scores -- {sp}, run {runs[0]}  '
          f'(pairwise, min n={nmin_:,})')
    print(cm_.round(2).to_string())
    print()

# The decision is made where all the scores exist. train_holdout carries all
# seven; the test split carries five and cannot speak about el2n/forget_events.
if not CM:
    raise RuntimeError(
        'P0a: no split has 2+ usable scores, so there is nothing to correlate. '
        f'Usable per split: { {k: len(v) for k, v in AVAIL.items()} }. '
        'Refusing to continue rather than reporting an empty result as a pass.')
DECIDE = 'train_holdout' if 'train_holdout' in CM else sorted(CM)[0]
cm = CM[DECIDE]
live = list(cm.index)
print(f'collinearity decision taken on: {DECIDE}  ({len(live)} scores)')

thr = 0.90
pairs = [(a_, b_, cm.loc[a_, b_]) for i, a_ in enumerate(live)
         for b_ in live[i + 1:] if cm.loc[a_, b_] >= thr]
print()
if pairs:
    print(f'near-collinear pairs (|rho| >= {thr}):')
    for a, b, v in pairs:
        print(f'    {a:14s} {b:14s} {v:.3f}')
    print('-> report these as ONE family in the paper; do not claim independence')
else:
    print(f'no pair exceeds |rho| = {thr} -- the scores carry distinct information')
"""),
        md("""
---
## P0b — the reliability atlas

ρ_seed(score, arch): Spearman between the score from two seeds of the same
architecture, over the samples both measured. Every seed pair, then the mean —
not just (seed1, seed2), which throws away two thirds of the evidence when
three seeds exist.
"""),
        code("""
import itertools

rows, skipped = [], []
for split, cols in AVAIL.items():
  for (arch, dset), grp in base.groupby(['arch', 'dataset']):
    ids = sorted(grp['run_id'])
    if len(ids) < 2:
        continue
    frames = {r: pd.read_parquet(
                   runs_dir / r / 'per_sample' / f'{split}.parquet')
                   .set_index('sample_idx') for r in ids}
    for a, b in itertools.combinations(ids, 2):
        fa, fb = frames[a], frames[b]
        common = fa.index.intersection(fb.index)
        for s in cols:
            va = fa.loc[common, s].to_numpy(dtype=float)
            vb = fb.loc[common, s].to_numpy(dtype=float)
            m = ~(np.isnan(va) | np.isnan(vb))
            # A silently-skipped cell is how el2n and forget_events vanished
            # from this grid without a word. Record every one.
            if m.sum() < 100:
                skipped.append((split, arch, s, f'n={int(m.sum())}')); continue
            if np.std(va[m]) == 0 or np.std(vb[m]) == 0:
                skipped.append((split, arch, s, 'zero variance')); continue
            rho, _ = spearmanr(va[m], vb[m])
            rows.append({'split': split, 'arch': arch, 'dataset': dset,
                         'score': s, 'pair': f'{a[-2:]}|{b[-2:]}',
                         'rho_seed': float(rho), 'n': int(m.sum())})

if skipped:
    print(f'{len(skipped)} (arch, score) cell(s) skipped:')
    for sp_, a, sc, why in skipped[:12]:
        print(f'    {sp_:14s} {a:16s} {sc:14s} {why}')
    if len(skipped) > 12:
        print(f'    ... and {len(skipped) - 12} more')
    print()

pairs_df = pd.DataFrame(rows)
grid = (pairs_df.groupby(['split', 'arch', 'dataset', 'score'])['rho_seed']
        .agg(['mean', 'std', 'count']).reset_index()
        .rename(columns={'mean': 'rho_seed', 'std': 'sd', 'count': 'n_pairs'}))
M.save_analysis(sess.data_dir, 's2_reliability_grid', grid)
M.save_analysis(sess.data_dir, 's2_reliability_pairs', pairs_df)

# D3: CIFAR-100 is the main result. ImageNet-100 has 2 archs x 2 seeds =
# one seed pair each, which cannot carry an interval; it is reported apart
# rather than averaged into the grid.
cif = grid[grid['dataset'] == 'cifar100']
side = grid[grid['dataset'] != 'cifar100']

for sp_ in AVAIL:
    sub = cif[cif['split'] == sp_]
    if not len(sub):
        continue
    w = sub.pivot(index='arch', columns='score', values='rho_seed')
    print(f'rho_seed -- CIFAR-100, {sp_} split '
          f'[{w.shape[0]} archs x {w.shape[1]} scores]')
    print(w.round(3).to_string())
    print()

# The test split is what NB2 routes on, so it carries the main R1 verdict.
main = cif[cif['split'] == 'test']
wide = main.pivot(index='arch', columns='score', values='rho_seed')

if len(side):
    sw = (side[side['split'] == 'test']
          .pivot(index='arch', columns='score', values='rho_seed'))
    print('ImageNet-100 -- consistency note only, 1 seed pair per arch, no interval')
    print(sw.round(3).to_string())
"""),
        md("""
---
## R1 — the verdict

**H1:** ρ_seed varies across the grid by at least **0.15**.
"""),
        code("""
vals = main['rho_seed'].dropna()          # CIFAR-100 only -- decision D3
rng = float(vals.max() - vals.min())
by_score = main.groupby('score')['rho_seed'].agg(['min', 'max', 'mean'])
by_score['range'] = by_score['max'] - by_score['min']

print(by_score.round(3).sort_values('range', ascending=False).to_string())
print()
print(f'grid range  {rng:.3f}   (H1 threshold 0.15)')
print(f'H1: {"SUPPORTED" if rng >= 0.15 else "FALSIFIED"}')
if rng < 0.15:
    print()
    print('R1 is a null. Report it as one -- do not re-cut looking for a')
    print('positive (02_PROTOCOL stopping rule 1). R3 continues regardless.')
"""),
    ])


# ---------------------------------------------------------------------------
def nb2():
    return notebook([
        md("""
# S2 · NB2 — the optimism bias

**The centrepiece.** An oracle ceiling computed from the same seed it routes is
optimistically biased: it partly routes on that model's own noise, which no
deployable router could have.

```
in-seed   oracle : score from seed i  -> routes seed i's model   (optimistic)
cross-seed oracle: score from seed j  -> routes seed i's model   (honest)
optimism bias    = in-seed - cross-seed            at matched FLOPs
```

Answers **R3**, **R4** and **R5**. CPU only — per-exit predictions are already
in the parquet, so no model is loaded.
"""),
        code(bootstrap()),
        code(paths_cell(phase="p1", detect=False)),
        code("""
# 03_INVENTORY.md claimed eight scores. It was an unverified claim: `msc` is
# not a stored column, and `el2n`/`forget_events` are NaN on the test split.
# Ask the artifact instead of trusting the document.
WANTED = ['msp', 'margin', 'entropy', 'ce_loss', 'el2n', 'forget_events',
          'pred_depth', 'msc']

import numpy as np, pandas as pd, itertools
from pathlib import Path

runs_dir = Path(MSC_ROOT) / 'runs'
runs = sorted(d.name for d in runs_dir.iterdir()
              if d.is_dir() and (d / 'per_sample' / 'test.parquet').exists())
meta = pd.DataFrame([{**M.parse_run_id(r), 'run_id': r} for r in runs])

_probe = pd.read_parquet(runs_dir / runs[0] / 'per_sample' / 'test.parquet')
absent = [c for c in WANTED if c not in _probe.columns]
allnan = [c for c in WANTED if c in _probe.columns
          and _probe[c].notna().mean() <= 0.5]
SCORES = [c for c in WANTED if c not in absent and c not in allnan]
if absent:
    print(f'not a column at all      : {absent}')
if allnan:
    print(f'NaN on the test split    : {allnan}  (training-set quantities)')
print(f'usable scores            : {SCORES}  ({len(SCORES)} of {len(WANTED)})')
if not SCORES:
    raise RuntimeError('no usable score columns -- refusing to continue')
base = meta[meta['method'] == 'base']
print(f'{len(base)} base run(s), {base["arch"].nunique()} architecture(s)')
"""),
        md("""
---
## Routing, from the parquet alone

`pred_dk == label` gives per-exit correctness; `budgets/{arch}.json` gives the
cost ρ of each exit. Routing by any score is a sort on that column. No model is
needed, which is why this is minutes rather than GPU-hours.

**Lower score = route earlier**, so scores where *high* means *easy*
(`msp`, `margin`) are negated. The direction is asserted, not assumed — a
sign error here would invert the whole result.
"""),
        code("""
HIGH_MEANS_EASY = {'msp', 'margin'}      # everything else: high = hard

def exit_tables(run_id):
    d = pd.read_parquet(runs_dir / run_id / 'per_sample' / 'test.parquet')
    d = d.sort_values('sample_idx').reset_index(drop=True)
    ks = sorted(int(c.split('_d')[1]) for c in d.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    correct = np.stack([(d[f'pred_d{k}'].to_numpy() == d['label'].to_numpy())
                        for k in ks], axis=1).astype(float)
    conf = np.stack([d[f'top1p_d{k}'].to_numpy() for k in ks], axis=1)
    return d, correct, conf, ks

def route_by(rank, correct, rho, target_rho):
    '''Route each sample to an exit so the MEAN cost equals target_rho.
    Samples with the lowest `rank` exit earliest.'''
    n, K = correct.shape
    order = np.argsort(rank, kind='stable')
    lo, hi = 0.0, 1.0
    for _ in range(40):                       # bisect the fraction sent deep
        frac = (lo + hi) / 2
        k_assign = np.full(n, 0, dtype=int)
        n_deep = int(round(frac * n))
        k_assign[order[n - n_deep:]] = K - 1
        cost = np.mean([rho[k] for k in k_assign])
        if cost < target_rho: lo = frac
        else: hi = frac
    return float(correct[np.arange(n), k_assign].mean()), cost

print('routing helpers defined -- correctness and cost come from the parquet')
"""),
        md("""
---
## The bias, both directions

R-03 in the risk register: seeds differ in accuracy, so routing seed *i* with
seed *j*'s score could look worse simply because *j* is a worse model. A real
optimism bias is **symmetric**; an accuracy confound is not. Both directions are
computed and reported.
"""),
        code("""
# The corpus is MIXED: 15 CIFAR-100 architectures and 2 ImageNet-100 ones.
# `sess.budgets(arch)` uses the SESSION's dataset, which paths_cell set to
# imagenet100 -- so it asked the imagenet zoo for `convnext_femto` and raised.
# A budget belongs to the RUN, not to the session.
_bud = {}
def rho_for(arch, dataset):
    if (arch, dataset) not in _bud:
        b = M.load_or_build_budgets(arch, sess.work, dataset)
        _bud[(arch, dataset)] = list(b['axes']['depth']['rho'])
    return _bud[(arch, dataset)]

TARGET_RHO = 0.80          # the operating point; the full curve comes next

rows = []
for (arch, dset), grp in base.groupby(['arch', 'dataset']):
    ids = sorted(grp['run_id'])
    if len(ids) < 2:
        continue
    try:
        rho = rho_for(arch, dset)
    except Exception as e:
        print(f'  SKIP {arch} ({dset}): {type(e).__name__}: {str(e)[:70]}')
        continue
    tab = {r: exit_tables(r) for r in ids}
    for i, j in itertools.permutations(ids, 2):
        di, ci, confi, _ = tab[i]
        dj, _, _, _ = tab[j]
        common = di['sample_idx'].isin(dj['sample_idx']).to_numpy()
        base_conf, _ = route_by(-confi[:, -1][common], ci[common], rho, TARGET_RHO)
        for s in SCORES:
            sign = -1.0 if s in HIGH_MEANS_EASY else 1.0
            in_seed = sign * di[s].to_numpy(dtype=float)[common]
            cross   = sign * dj.set_index('sample_idx').loc[
                di['sample_idx'][common], s].to_numpy(dtype=float)
            if np.isnan(in_seed).all() or np.isnan(cross).all():
                continue
            a_in, _ = route_by(np.nan_to_num(in_seed, nan=np.inf), ci[common], rho, TARGET_RHO)
            a_cx, _ = route_by(np.nan_to_num(cross,  nan=np.inf), ci[common], rho, TARGET_RHO)
            rows.append({'arch': arch, 'dataset': dset, 'score': s,
                         'model_seed': i[-2:],
                         'score_seed': j[-2:], 'in_seed': a_in,
                         'cross_seed': a_cx, 'bias': a_in - a_cx,
                         'msp_baseline': base_conf,
                         'headroom_honest': a_cx - base_conf})

bias = pd.DataFrame(rows)
M.save_analysis(sess.data_dir, 's2_optimism_bias', bias)
print(f'{len(bias)} (arch, score, seed-pair) rows')
print(bias.groupby('score')[['bias', 'headroom_honest']].mean().round(4).to_string())
"""),
        md("""
---
## R3 — is the in-seed oracle optimistic?

**H3:** median bias ≥ **0.5 accuracy points** and > 0 for at least 6 of 8 scores.

Either answer is reportable: a large bias means the field's oracle bounds are
inflated; a bias of zero validates a practice nobody had checked.
"""),
        code("""
med = bias['bias'].median() * 100
per_score = bias.groupby('score')['bias'].median() * 100
n_pos = int((per_score > 0).sum())

print(per_score.round(3).sort_values(ascending=False).to_string())
print()
print(f'median bias over the whole grid : {med:+.3f} accuracy points')
print(f'scores with positive bias       : {n_pos} of {len(per_score)}')
print(f'H3 (>= 0.5 pt AND >= 6 of 8)    : '
      f'{"SUPPORTED" if (med >= 0.5 and n_pos >= 6) else "NOT SUPPORTED"}')
print()
print('symmetry check (R-03): a real bias is direction-symmetric;')
print('an accuracy confound is not.')
sym = bias.groupby(['arch', 'score']).apply(
    lambda g: g['bias'].std(), include_groups=False)
print(f'  mean within-pair sd of bias: {sym.mean()*100:.3f} pt')
"""),
        md("""
---
## R5 — after correction, is there headroom?

**H5:** no score's **cross-seed** oracle beats `msp` by more than 1.0 point.

This is the gate. If none clears it, no method is built and the bound is the
result (`02_PROTOCOL.md` stopping rule 2).
"""),
        code("""
hon = bias.groupby('score')['headroom_honest'].median() * 100
print(hon.round(3).sort_values(ascending=False).to_string())
n = int(bias['arch'].nunique())
se2 = 2 * (0.5 / np.sqrt(10000)) * 100      # 2 SE on an accuracy diff, 10k samples
print()
print(f'noise floor (2 SE, 10k samples): +/-{se2:.3f} pt')
print(f'best honest headroom           : {hon.max():+.3f} pt  ({hon.idxmax()})')
print(f'H5 (nothing clears +1.0 pt)    : '
      f'{"SUPPORTED -- no method is built" if hon.max() < 1.0 else "FALSIFIED"}')
if hon.max() >= 1.0:
    print()
    print(f'  -> {hon.idxmax()} clears the gate. Amend 02_PROTOCOL.md in writing')
    print('     with the date and reason, then scope a method to that score.')
"""),
        md("""
---
## R4 — does the bias follow reliability?

**H4:** bias correlates with (1 − ρ_seed) at Spearman ≥ 0.5.

If it holds, two observations become one mechanism — and ρ_seed becomes a cheap
predictor of how inflated a published oracle bound is.

n is small. The scatter is reported, not just the coefficient.
"""),
        code("""
from scipy.stats import spearmanr
grid = pd.read_csv(Path(sess.data_dir) / 'analysis' / 's2_reliability_grid.csv')
grid = grid[(grid['dataset'] == 'cifar100') & (grid['split'] == 'test')]  # D3
j = (bias[bias['dataset'] == 'cifar100']
     .groupby(['arch', 'score'])['bias'].median().reset_index()
     .merge(grid[['arch', 'score', 'rho_seed']], on=['arch', 'score']))
j['unreliability'] = 1 - j['rho_seed']
M.save_analysis(sess.data_dir, 's2_bias_vs_reliability', j)

m = j[['unreliability', 'bias']].dropna()
r, p = spearmanr(m['unreliability'], m['bias'])
print(f'n = {len(m)} (arch, score) cells')
print(f'Spearman(1 - rho_seed, bias) = {r:+.3f}   p = {p:.4f}')
print(f'H4 (>= 0.5): {"SUPPORTED" if r >= 0.5 else "NOT SUPPORTED"}')
print()
print('per-score medians (the scatter behind the number):')
print(j.groupby('score')[['rho_seed', 'bias']].median().round(4).to_string())
"""),
    ])


NOTEBOOKS = {
    "S2_NB0_Fetch.ipynb": nb0,
    "S2_NB1_Reliability.ipynb": nb1,
    "S2_NB2_Ceiling.ipynb": nb2,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in NOTEBOOKS.items():
        nb = fn()
        (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        nc = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        nm = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        kb = (OUT / name).stat().st_size / 1024
        print(f"  {name:28s} {nc:2d} code + {nm:2d} md   {kb:6.0f} KB")

    print("\n  checking for names no earlier cell defines")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_names.py")]
                       + [str(f) for f in sorted(OUT.glob("S2_*.ipynb"))],
                       capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if r.returncode != 0:
        print("  Generation refused.")
        return 1

    print("\n  parsing every cell as Python 3.10")
    import ast
    bad = 0
    for f in sorted(OUT.glob("S2_*.ipynb")):
        for ci, c in enumerate(json.loads(f.read_text(encoding="utf-8"))["cells"]):
            if c.get("cell_type") != "code":
                continue
            try:
                ast.parse("".join(c.get("source", [])), feature_version=(3, 10))
            except SyntaxError as e:
                bad += 1
                print(f"  [FAIL] {f.name} cell {ci}: {e.msg} (line {e.lineno})")
                print(f"         {(e.text or '').strip()[:88]}")
    if bad:
        print(f"\n  {bad} cell(s) do not parse. Generation refused.")
        return 1
    print("  all cells parse")
    print(f"\nOK -- {len(NOTEBOOKS)} notebook(s) in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
