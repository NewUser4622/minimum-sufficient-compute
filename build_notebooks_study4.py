#!/usr/bin/env python3
"""Generate the Study 4 notebooks.

Reuses Study 3's `paths_cell` and `runs_cell` (extended for dataset choice) so
Study 4 inherits every fix those cost: explicit storage resolution, offline
sessions, the measured-runs accessor, the dedupe rule, and the empty-frame
guards.

    python build_notebooks_study4.py

Writes notebooks_study4/ and runs every validation gate afterwards.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from build_notebooks import bootstrap_cell, code, md, notebook

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks_study4"

IN100_ARCHS = ("resnet50", "vit_small_p16")


# ---------------------------------------------------------------------------
# shared cells
# ---------------------------------------------------------------------------
def paths_cell(phase="analysis", dataset="cifar100", hf=False) -> str:
    """Locate storage, resolve the dataset, and PROVE the runs are there.

    Carries every fix Study 3 paid for:
      * `M.resolve_storage` + `work_root=MSC_ROOT`, so the Session points at the
        directory that actually holds the runs (D-88's neighbour);
      * counts MEASURED RUNS rather than trusting that `runs/` exists, because a
        Session creates that directory itself;
      * offline by default -- a background uploader retrying mid-epoch turns a
        missing network into a failed run (R-09);
      * the CIFAR repo, not the library's ImageNet default.
    """
    env = ("os.environ['MSC_IN100_DIR'] = DATA_DIR"
           if dataset == "imagenet100" else
           "if CIFAR_DIR:\n    os.environ['MSC_CIFAR_DIR'] = str(CIFAR_DIR)")
    return f"""\
# === CELL 2 -- WHERE EVERYTHING LIVES ======================================
# Leave DATA_DIR / MSC_ROOT as None and the roomiest usable drive is chosen.
# Set them explicitly if your data is somewhere specific.

DATA_DIR  = None      # e.g. r'C:\\\\msc_data\\\\in100'
MSC_ROOT  = None      # e.g. r'C:\\\\msc_results'
CIFAR_DIR = r'C:\\\\Users\\\\Administrator\\\\Desktop\\\\New folder'   # holds cifar-100-python

# ---------------------------------------------------------------------------
import os
from pathlib import Path

M = msc

os.environ.setdefault('MSC_HF_REPO', 'Shanmuk4622/msc-cifar100')

_paths = M.resolve_storage(DATA_DIR, MSC_ROOT)
if not _paths['ok']:
    raise SystemExit('storage is not usable -- see the problems above')
DATA_DIR = _paths['data_dir']
MSC_ROOT = _paths['results_root']
os.environ['MSC_SCRATCH'] = MSC_ROOT
{env}

# OFFLINE. Nothing is uploaded here; S4_NB3_Publish does that once, at the end.
sess = M.Session(account='local', phase='{phase}', dataset='{dataset}',
                 work_root=MSC_ROOT, session_limit_h=0.0,
                 enable_hf={hf!r}, worker_id=0, num_workers=1)

print(f'msc_lib   {{M.__version__}}')
print(f'MSC_ROOT  {{MSC_ROOT}}')
print(f'dataset   {dataset}')
print('HuggingFace: ' + ('ON -- publishing' if {hf!r} else 'OFF -- fully offline'))

# A Session CREATES runs/, so its existence proves nothing. Count runs that
# actually carry a measurement -- that is what every cell below reads.
_runs_dir = Path(MSC_ROOT) / 'runs'
_measured = sorted(d.name for d in _runs_dir.iterdir()
                   if d.is_dir() and (d / 'per_sample' / 'test.parquet').exists()
                   ) if _runs_dir.is_dir() else []
print(f'measured runs on disk: {{len(_measured)}}')
if not _measured:
    raise RuntimeError(
        f'no measured runs under {{MSC_ROOT}}/runs. Fetch Study 1-3 output '
        'first (notebooks_study2/S2_NB0_Fetch.ipynb), or set MSC_ROOT above.')
"""


def runs_cell() -> str:
    """THE accessor for "which runs do I analyse". One place for the dedupe rule
    and the emptiness checks, so neither is re-typed (rule 4)."""
    return """\
# === Which runs am I analysing? ===========================================
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
                f'parse_run_id gave no {col!r} column. Present: '
                f'{list(df.columns)}. Ids look like {ids[:3]}')
    sel = df[(df['dataset'] == dataset) & (df['method'].isin(methods))]
    if require and sel.empty:
        raise RuntimeError('; '.join([
            f'{len(df)} measured run(s) on disk, but NONE with '
            f'dataset={dataset!r} and method in {tuple(methods)!r}',
            f'datasets present: {sorted(df["dataset"].dropna().unique())}',
            f'methods present: {sorted(df["method"].dropna().unique())}']))
    before = len(sel)
    sel = (sel.sort_values('phase')
              .drop_duplicates(subset=['arch', 'dataset', 'method', 'seed'],
                               keep='last'))
    if len(sel) < before:
        print(f'dropped {before - len(sel)} duplicate (arch, method, seed) run(s)')
    return sel.reset_index(drop=True)

def exit_tables(run_id):
    d = (pd.read_parquet(M.run_layout(sess.work, run_id)['per_sample']
                         / 'test.parquet')
         .sort_values('sample_idx').reset_index(drop=True))
    ks = sorted(int(c.split('_d')[1]) for c in d.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    lab = d['label'].to_numpy()
    correct = np.stack([(d[f'pred_d{k}'].to_numpy() == lab) for k in ks],
                       axis=1).astype(float)
    conf = np.stack([d[f'top1p_d{k}'].to_numpy() for k in ks], axis=1)
    second = np.stack([d[f'top2p_d{k}'].to_numpy() for k in ks], axis=1)
    pred = np.stack([d[f'pred_d{k}'].to_numpy() for k in ks], axis=1)
    return correct, conf, second, pred, ks

print('accessors defined')
"""


ROUTING_CELL = """\
# === Routing, identical to Study 2/3 ======================================
# tools/s2_routing_canaries.py (18/18) covers these; they are repeated here
# only because the notebook is standalone.

def _cost(k, rho):  return float(np.mean(np.asarray(rho)[k]))

def route_threshold(score, correct, rho, target, higher_exits=True):
    '''Exit at the first exit whose per-exit SCORE clears a threshold.
    `higher_exits=True` means a LARGER score means "exit here".

    D-89. The first version had the bisection INVERTED. Cost is non-decreasing
    in the threshold -- a higher bar means fewer early exits and more compute --
    so when the achieved cost is below target the bar must go UP. The code
    lowered it, and the search never tracked the budget at all: it returned the
    same cost (0.594) for every target from 0.30 to 0.95, which produced the
    impossible result that ACCURACY FELL as the budget rose.

    It survived because the canary only asserted `cost <= target`, and a
    constant 0.594 satisfies that at every loose budget. A canary that a broken
    function passes is not a canary.
    '''
    n, K = correct.shape
    s = np.asarray(score, float)
    if not higher_exits:
        s = -s                       # now LARGER always means "exit here"

    def cost_at(th):
        fires = s >= th
        fires = np.concatenate([fires[:, :-1], np.ones((n, 1), bool)], axis=1)
        k = fires.argmax(axis=1)
        return k, _cost(k, rho)

    lo, hi = float(s.min()) - 1e-6, float(s.max()) + 1e-6   # cheapest .. dearest
    for _ in range(60):
        th = (lo + hi) / 2
        if cost_at(th)[1] < target:
            lo = th                  # under budget -> raise the bar
        else:
            hi = th
    k, c = cost_at(lo)               # `lo` side is the one that never overspends
    return float(correct[np.arange(n), k].mean()), c

def route_patience(pred, correct, rho, target):
    '''PABEE: exit at the first exit where the last `n` predictions agree.
    `n` is the only knob, so the budget is met by choosing it -- a coarser
    control than a threshold, which is a property of the method, not a bug.'''
    N, K = correct.shape
    best = None
    for patience in range(1, K + 1):
        k = np.full(N, K - 1)
        for i in range(patience - 1, K):
            agree = np.ones(N, bool)
            for j in range(i - patience + 1, i):
                agree &= (pred[:, j] == pred[:, i])
            fresh = agree & (k == K - 1)
            k = np.where(fresh, i, k)
        c = _cost(k, rho)
        acc = float(correct[np.arange(N), k].mean())
        if best is None or abs(c - target) < abs(best[2] - target):
            best = (acc, patience, c)
    return best[0], best[2], best[1]

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

print('routing helpers defined')
"""


# ---------------------------------------------------------------------------
# S4_NB0 -- figures and intervals
# ---------------------------------------------------------------------------
def nb0():
    return notebook([
        md("""
# S4_NB0 — Figure 1 and bootstrap intervals

**CPU only · minutes · no training · no network**

Two things a reviewer will ask for that the existing data already answers.

## Figure 1 — honest headroom versus compute budget

`s2_headroom_sweep.csv` already spans ρ = 0.40–0.95. It is currently a table
buried in a log. Plotted, with the in-seed bound above it and zero marked, it
carries the paper's whole argument in one panel.

## Bootstrap intervals

Study 3's Q1 used **one seed per architecture**. That is correct — the excess is
a per-run identity, not a seed average — but it leaves the numbers without error
bars, and reviewers expect them.

Resampling the **10,000 test images** gives a legitimate interval for a per-run
quantity.

> **It is a sample interval, not a seed interval.** It captures sampling noise
> over the test set, not training variation. Every label in this notebook says
> so, and `01_PROTOCOL.md` requires it. Conflating the two would overstate the
> result — which is the failure mode this project keeps finding.

**Refusal condition (pre-registered):** if any joint run's interval includes
zero, that architecture's excess is not established at n = 10,000 and must be
reported as such regardless of the point estimate.
"""),
        code(bootstrap_cell()),
        code(paths_cell()),
        code(runs_cell()),
        md("""
---
## Preflight

Every Study 3 failure was code written against an assumed state. This lists
what the notebook needs and whether it is there, **before** doing anything.
"""),
        code("""
AN = Path(sess.data_dir) / 'analysis'
need = {'s2_headroom_sweep.csv': 'Figure 1',
        's3_q1_comparison.csv': 'the joint/frozen pairs'}
print(f"{'file':30s} {'exists':>7s} {'rows':>7s}  used for")
missing = []
for f, why in need.items():
    p = AN / f
    n = len(pd.read_csv(p)) if p.exists() else 0
    print(f'{f:30s} {str(p.exists()):>7s} {n:7d}  {why}')
    if not p.exists() or n == 0:
        missing.append(f)
if missing:
    raise RuntimeError(f'missing or empty: {missing}. Run Study 2/3 first.')

joint = measured_runs(methods=('jointexit',))
print()
print(f'{len(joint)} joint run(s): {sorted(joint["run_id"])}')
"""),
        md("""
---
## Figure 1
"""),
        code("""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sw = pd.read_csv(AN / 's2_headroom_sweep.csv')
rhos = sorted(sw['target_rho'].unique())
lo, hi = min(rhos), max(rhos)
if not (lo <= 0.45 and hi >= 0.90):
    raise RuntimeError(
        f'sweep spans only rho {lo}-{hi}; Figure 1 claims 0.40-0.95. '
        'Re-run S2_NB2 rather than plotting a narrower range under a wider '
        'caption.')

piv = sw.groupby(['target_rho', 'score'])['headroom'].median().unstack()
fig, ax = plt.subplots(figsize=(7, 4.5))
for c in piv.columns:
    ax.plot(piv.index, piv[c], marker='o', label=c)
ax.axhline(0, color='k', lw=1)
ax.set_xlabel('compute budget  rho  (fraction of full-network FLOPs)')
ax.set_ylabel('honest headroom over confidence baseline  (accuracy points)')
ax.set_title('Cross-seed headroom is negative at every operating point')
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=.3)
fig.tight_layout()

figdir = Path(sess.data_dir) / 'paper' / 'figures'
figdir.mkdir(parents=True, exist_ok=True)
out = figdir / 'fig1_headroom.png'
fig.savefig(out, dpi=200)
print(f'wrote {out}  ({out.stat().st_size/1024:.0f} KB)')
print()
print(piv.round(2).to_string())
"""),
        md("""
---
## Bootstrap intervals on the excess

For each joint run: resample the test set with replacement, recompute
`P(correct at any exit) − P(correct at the final exit)`, and take the 2.5/97.5
percentiles.
"""),
        code("""
RNG = np.random.default_rng(0)
DRAWS = 1000

def excess_ci(run_id, draws=DRAWS):
    correct, _, _, _, _ = exit_tables(run_id)
    n = correct.shape[0]
    any_ok = correct.any(axis=1).astype(float)
    fin_ok = correct[:, -1]
    point = float(any_ok.mean() - fin_ok.mean())
    idx = RNG.integers(0, n, size=(draws, n))
    boot = any_ok[idx].mean(axis=1) - fin_ok[idx].mean(axis=1)
    return point * 100, float(np.percentile(boot, 2.5)) * 100, \\
           float(np.percentile(boot, 97.5)) * 100, n

rows = []
for rid in sorted(joint['run_id']):
    p, lo_, hi_, n = excess_ci(rid)
    rows.append({'run_id': rid, 'arch': M.parse_run_id(rid)['arch'],
                 'excess': p, 'ci_lo': lo_, 'ci_hi': hi_, 'n_samples': n})
ci = pd.DataFrame(rows)
M.save_analysis(sess.data_dir, 's4_bootstrap', ci)

print(f'{"arch":16s} {"excess":>8s}  95% CI over TEST SAMPLES (not seeds)')
for _, r in ci.iterrows():
    print(f'{r["arch"]:16s} {r["excess"]:+8.2f}  [{r["ci_lo"]:+.2f}, {r["ci_hi"]:+.2f}]'
          f'   n={int(r["n_samples"]):,}')

print()
bad = ci[(ci['ci_lo'] <= 0) & (ci['ci_hi'] >= 0)]
if len(bad):
    print('REFUSAL CONDITION MET -- these intervals include zero:')
    print(bad[['arch', 'excess', 'ci_lo', 'ci_hi']].to_string(index=False))
    print('Their excess is NOT established at this sample size and must be')
    print('reported as such, whatever the point estimate says.')
else:
    print(f'all {len(ci)} intervals exclude zero')
print()
_n = int(ci['n_samples'].iloc[0])
print(f'LABEL FOR THE MANUSCRIPT: "95% CI over test samples (n={_n:,});')
print('NOT a seed interval." Seed variation is Study 2 s 90 pairs and is')
print('reported separately. The n above is READ FROM THE DATA, not assumed --')
print('quoting 10,000 when the split is smaller would be a fabricated detail.')
"""),
        md("""
---
## Canaries — the interval must widen and narrow for the right reasons
"""),
        code("""
rng = np.random.default_rng(1)

def _ci_of(any_ok, fin_ok, draws=400):
    n = len(any_ok)
    idx = RNG.integers(0, n, size=(draws, n))
    b = any_ok[idx].mean(axis=1) - fin_ok[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)) * 100, float(np.percentile(b, 97.5)) * 100

res = []
def chk(tag, ok, detail=''):
    res.append(ok); print(f'{"PASS" if ok else "FAIL"}  {tag}'
                          + (f'  -- {detail}' if not ok and detail else ''))

# 1. no excess -> interval must contain zero
n = 10000
a = np.ones(n); f = np.ones(n)
lo_, hi_ = _ci_of(a, f)
chk(f'zero excess -> CI contains 0  [{lo_:+.3f}, {hi_:+.3f}]', lo_ <= 0 <= hi_)

# 2. large excess -> interval must EXCLUDE zero
f2 = np.zeros(n); f2[:int(.7 * n)] = 1
lo2, hi2 = _ci_of(a, f2)
chk(f'30 pt excess -> CI excludes 0  [{lo2:+.2f}, {hi2:+.2f}]', lo2 > 0)

# 3. more samples -> narrower interval.
#    NOTE: sample RANDOMLY. Slicing f2[:500] takes 500 leading ones from a
#    sorted array, giving zero excess and a zero-width interval -- which looks
#    like the canary failing when it is the canary being wrong.
sub = rng.choice(n, 500, replace=False)
lo_s, hi_s = _ci_of(a[sub], f2[sub])
lo_b, hi_b = _ci_of(a, f2)
w_small, w_big = hi_s - lo_s, hi_b - lo_b
chk(f'CI narrows with n  ({w_small:.2f} at n=500 -> {w_big:.2f} at n={n})',
    w_big < w_small)

print(f'\\n{sum(res)}/{len(res)} canaries pass')
if not all(res):
    raise RuntimeError('a bootstrap canary failed -- do not report the intervals')
"""),
        md("""
---
## Next

`S4_NB1_Baselines` — margin and patience. It is also free, and it can
**invalidate the headline**: if either beats confidence thresholding, every
headroom number in Studies 2–3 was computed against a weak comparator.

Record the outcome in `study4/03_LOG.md`.
"""),
    ])


# ---------------------------------------------------------------------------
# S4_NB1 -- baselines
# ---------------------------------------------------------------------------
def nb1():
    return notebook([
        md("""
# S4_NB1 — is the conclusion baseline-independent?

**CPU only · minutes · no training · no network**

Studies 2 and 3 compare against **one** baseline: confidence thresholding on
`top1p_dk`. Reviewers will name entropy, patience-based exiting (PABEE), and
learned policies.

## What is actually computable, checked against the parquet

| baseline | rule | available |
|---|---|---|
| confidence (existing) | exit when `top1p_dk ≥ τ` | yes |
| **margin** | exit when `top1p_dk − top2p_dk ≥ τ` | **yes** |
| **patience / PABEE** | exit when *n* consecutive exits agree | **yes** |
| learned gate | Study 3 Q2 | done — 1.7 % capture |
| entropy | needs the full per-exit softmax | **no** |

**Entropy is omitted, and the notebook says why.** Only the top-2 probabilities
were stored, never logits. Approximating entropy from `top1p`/`top2p` would be a
fabricated baseline. Margin is its closest available relative.

## H6 (pre-registered)

> With margin and patience substituted for confidence, the honest (cross-seed)
> headroom **remains negative at every operating point** ρ = 0.40–0.95.

**If a new baseline is stronger, the headline is recomputed against it.** A
limits paper that picks a weak comparator is worthless — so this notebook
firing against us improves the paper, and it costs CPU minutes to find out.
"""),
        code(bootstrap_cell()),
        code(paths_cell()),
        code(runs_cell()),
        code(ROUTING_CELL),
        md("""
---
## Preflight — do the columns each baseline needs exist?
"""),
        code("""
base = measured_runs()
probe = sorted(base['run_id'])[0]
correct, conf, second, pred, ks = exit_tables(probe)
print(f'probe run: {probe}   K = {len(ks)} exits, n = {correct.shape[0]:,}')
print()
checks = {
    'confidence  (top1p_dk)': conf.shape == correct.shape,
    'margin      (top1p_dk - top2p_dk)': second.shape == correct.shape,
    'patience    (pred_dk)': pred.shape == correct.shape,
}
for k, v in checks.items():
    print(f'  {"OK  " if v else "MISS"} {k}')
if not all(checks.values()):
    raise RuntimeError('a baseline is missing its columns; see above')
print()
print('entropy: NOT computable -- only top-2 probabilities were stored.')
print('It is omitted and reported as omitted, not approximated.')
"""),
        md("""
---
## All baselines, all budgets

Cross-seed evaluation: the routing rule is fitted on nothing (these are
threshold rules), but the **oracle** is scored from a different seed, exactly as
in Study 2, so "honest headroom" means the same thing it did there.
"""),
        code("""
BUDGETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
rows = []
for (arch, dset), grp in base.groupby(['arch', 'dataset']):
    if dset != 'cifar100':
        continue
    ids = sorted(grp['run_id'])
    if len(ids) < 2:
        continue
    try:
        rho = M.load_or_build_budgets(arch, sess.work, dset)['axes']['depth']['rho']
    except Exception as e:
        print(f'  skip {arch}: {type(e).__name__}'); continue
    i_, j_ = ids[0], ids[1]
    ci_, cf_, s2_, pr_, _ = exit_tables(i_)
    cj_, _, _, _, _ = exit_tables(j_)
    n = min(len(ci_), len(cj_))
    ci_, cf_, s2_, pr_, cj_ = ci_[:n], cf_[:n], s2_[:n], pr_[:n], cj_[:n]
    for tr in BUDGETS:
        # MEASURE the cost, never echo the target. The first version recorded
        # `tr` for the threshold rules, so the overspend check could not fire
        # for them by construction -- which is what hid D-89 for a whole run.
        b_conf, c_conf = route_threshold(cf_, ci_, rho, tr)
        b_marg, c_marg = route_threshold(cf_ - s2_, ci_, rho, tr)
        b_pat, c_pat, pat_n = route_patience(pr_, ci_, rho, tr)
        orc_cross, _ = route_oracle(cj_, ci_, rho, tr)
        for name, acc, cost in [('confidence', b_conf, c_conf),
                                ('margin', b_marg, c_marg),
                                ('patience', b_pat, c_pat)]:
            rows.append({'arch': arch, 'target_rho': tr, 'baseline': name,
                         'baseline_acc': acc * 100,
                         'achieved_cost': cost,
                         'oracle_cross': orc_cross * 100,
                         'honest_headroom': (orc_cross - acc) * 100,
                         'patience_n': pat_n if name == 'patience' else np.nan})

bl = pd.DataFrame(rows)
if bl.empty:
    raise RuntimeError('no baseline rows produced -- every table below would '
                       'be a KeyError on an empty frame')
M.save_analysis(sess.data_dir, 's4_baselines', bl)
print(f'{len(bl)} rows, {bl["arch"].nunique()} architectures, '
      f'{bl["target_rho"].nunique()} budgets, {bl["baseline"].nunique()} baselines')
"""),
        md("""
---
## Which baseline is strongest?

If confidence is not the best, every headroom number in Studies 2–3 was computed
against a weak comparator and must be recomputed against this one.
"""),
        code("""
strength = (bl.groupby(['target_rho', 'baseline'])['baseline_acc']
              .median().unstack())
print('baseline accuracy (median over architectures), by budget')
print(strength.round(2).to_string())

# PABEE has only K discrete operating points, so it often cannot hit the
# requested budget. Comparing its accuracy against a threshold rule that DID
# hit the budget would be unfair in whichever direction the miss went, so the
# achieved cost is reported beside it rather than assumed equal.
cost = (bl.groupby(['target_rho', 'baseline'])['achieved_cost']
          .median().unstack())
print()
print('cost actually ACHIEVED (threshold rules hit the target; patience cannot)')
print(cost.round(3).to_string())
off = (cost['patience'] - cost.index.to_series()).abs()
if (off > 0.05).any():
    print()
    print(f'patience misses the target by up to {off.max():.2f} at some budgets.')
    print('Its accuracy is therefore NOT directly comparable to the threshold')
    print('rules there, and the paper must say so rather than tabulating them')
    print('side by side as if the budgets matched.')
print()
# CHOOSE ON MATCHED COST, NOT RAW ACCURACY. A baseline that overshoots the
# budget gets more compute and therefore more accuracy; ranking on the raw
# number rewards missing the target. Only rows that actually hit the requested
# budget are eligible.
# The rule is ASYMMETRIC on purpose. A baseline may spend LESS than the budget
# -- that is conservative and only hurts it. Spending MORE buys accuracy it was
# not entitled to, so overshooting disqualifies the row. A symmetric tolerance
# let patience win at a 0.70 target while actually spending 0.747.
TOL = 0.01
fair = bl[bl['achieved_cost'] <= bl['target_rho'] + TOL]
excluded = sorted(set(bl['baseline']) - set(fair['baseline']))
partial = [b_ for b_ in fair['baseline'].unique()
           if len(fair[fair['baseline'] == b_]) < len(bl['target_rho'].unique())
           * bl['arch'].nunique()]
if excluded:
    print()
    print(f'EXCLUDED from the ranking (never hit the budget within {TOL}): {excluded}')
if partial:
    print(f'PARTIAL (hit the budget at only some points): {partial}')

if fair.empty:
    raise RuntimeError('no baseline hit its budget -- nothing is comparable')

# COMMON BUDGETS ONLY. Ranking medians computed over different budget subsets
# is not like-for-like: a baseline that qualifies only at the cheap end would be
# compared against one measured across the whole range. Restrict to the budgets
# where EVERY baseline has a qualifying row.
n_base = bl['baseline'].nunique()
per_rho = fair.groupby('target_rho')['baseline'].nunique()
common = sorted(per_rho[per_rho == n_base].index)
print()
if not common:
    print('NO budget has a qualifying row for every baseline, so no overall')
    print('ranking is honest. Compare per budget instead:')
    print(fair.groupby(['target_rho', 'baseline'])['baseline_acc']
            .median().unstack().round(2).to_string())
    winner = 'confidence'      # unchanged; nothing displaced it
    print()
    print(f'keeping {winner} as the comparator -- nothing beat it on equal terms')
else:
    print(f'budgets comparable across all {n_base} baselines: {common}')
    cm = fair[fair['target_rho'].isin(common)]
    rank = cm.groupby('baseline')['baseline_acc'].median()
    print(rank.round(2).to_string())
    winner = rank.idxmax()
    print()
    print(f'strongest, AT MATCHED COST ON COMMON BUDGETS: {winner}')
if winner != 'confidence':
    print()
    print('*** CONFIDENCE IS NOT THE STRONGEST BASELINE. ***')
    print(f'Studies 2-3 quoted headroom against confidence; {winner} is better,')
    print('so every headroom figure must be recomputed against it before the')
    print('paper quotes one. This is R-04 firing, and it improves the paper.')
else:
    print('confidence remains the strongest -- the existing headroom numbers')
    print('were computed against the right comparator.')
"""),
        md("""
---
## H6 — the verdict
"""),
        code("""
hh = (bl.groupby(['target_rho', 'baseline'])['honest_headroom']
        .median().unstack())
print('honest (cross-seed) headroom, accuracy points')
print(hh.round(2).to_string())
print()
pos = hh[hh > 0].stack()
if len(pos):
    print('POSITIVE headroom found at:')
    print(pos.round(2).to_string())
    print()
    print('H6: FALSIFIED -- the conclusion is NOT baseline-independent.')
    print('The paper must lead with the baseline that shows headroom and')
    print('explain the difference, not average over them.')
else:
    print(f'negative at all {hh.shape[0]} budgets for all {hh.shape[1]} baselines')
    print('H6: SUPPORTED -- the conclusion does not depend on the baseline.')
"""),
        md("""
---
## Canaries — the baselines must be distinguishable
"""),
        code("""
rng = np.random.default_rng(0)
n, K = 4000, 5
rho = [.2, .4, .6, .8, 1.]
easy = rng.random(n) < .5
cc = np.zeros((n, K)); cc[easy, :] = 1.0; cc[~easy, K-1] = 1.0

res = []
def chk(t, ok, d=''):
    res.append(ok); print(f'{"PASS" if ok else "FAIL"}  {t}' + (f'  -- {d}' if not ok and d else ''))

# informative confidence beats uninformative confidence
cf_good = np.where(easy[:, None], .95, .15) * np.ones((1, K))
cf_bad = rng.random((n, K))
a_good, _ = route_threshold(cf_good, cc, rho, .7)
a_bad, _ = route_threshold(cf_bad, cc, rho, .7)
chk(f'informative confidence beats noise ({a_good*100:.1f} vs {a_bad*100:.1f})',
    a_good > a_bad + .02)

# Patience has only K discrete operating points, so it frequently CANNOT hit
# the target budget -- both arms then land near full compute and an
# accuracy-based test cannot tell them apart. That is a real property of PABEE,
# not a bug, so the canary tests the thing that IS discriminating: with
# agreeing predictions it should reach a LOWER cost, because agreement lets it
# exit early.
pr_good = np.where(easy[:, None], 7, rng.integers(0, 100, (n, K)))
a_pat, c_pat, npat = route_patience(pr_good, cc, rho, .7)
a_noi, c_noi, _ = route_patience(rng.integers(0, 100, (n, K)), cc, rho, .7)
chk(f'patience exploits agreement to exit earlier '
    f'(cost {c_pat:.3f} vs {c_noi:.3f} on noise)', c_pat < c_noi - .01)

# margin: a wide margin on easy samples must beat a random margin
mg_good = np.where(easy[:, None], .8, .02) * np.ones((1, K))
a_mg, _ = route_threshold(mg_good, cc, rho, .7)
a_mg_noise, _ = route_threshold(rng.random((n, K)), cc, rho, .7)
chk(f'margin uses the gap ({a_mg*100:.1f} vs {a_mg_noise*100:.1f} on noise)',
    a_mg > a_mg_noise + .02)

# THE CANARY THAT MATTERS (D-89). 'cost <= target' is satisfied by a CONSTANT,
# so it passed while route_threshold ignored the budget entirely. The cost must
# TRACK the target, and accuracy must never fall as the budget rises.
costs, accs = [], []
for t in (.3, .4, .5, .6, .7, .8, .9, .95):
    a_, c_ = route_threshold(cf_good, cc, rho, t)
    costs.append(c_); accs.append(a_)
    if c_ > t + 1e-6:
        chk(f'never overspends at target {t}', False, f'cost {c_:.3f}'); break
else:
    chk('never overspends at any budget', True)
chk(f'cost TRACKS the budget (spread {max(costs)-min(costs):.3f}, not constant)',
    max(costs) - min(costs) > 0.15)
chk('accuracy never falls as the budget rises',
    all(b >= a - 1e-9 for a, b in zip(accs, accs[1:])),
    f'{[round(x*100,1) for x in accs]}')

print(f'\\n{sum(res)}/{len(res)} canaries pass')
if not all(res):
    raise RuntimeError('a baseline canary failed -- the comparison is not valid')
"""),
        md("""
---
## Next

If H6 is supported and confidence is still strongest, **Paper A is submittable**
— `S4_NB0` produced Figure 1 and the intervals, this notebook closed the
baseline objection, and no GPU time was spent.

Then `S4_NB2_ImageNet` for scale and a transformer.

Record both outcomes in `study4/03_LOG.md`.
"""),
    ])


# ---------------------------------------------------------------------------
# S4_NB2 -- ImageNet-100 + a transformer
# ---------------------------------------------------------------------------
def nb2():
    return notebook([
        md(f"""
# S4_NB2 — does the excess hold at scale, and on a transformer?

**~20 GPU-hours · 2 runs · fully resumable · offline**

## What this settles

Every number so far is CIFAR-100 at 32px and convolutional. Early-exit research
is now largely transformer work. This is the standard reviewer objection.

**Pre-registered:**

- **H4** — `oracle_in − acc_full ≥ 2.0 pt` in **2 of 2** architectures
- **H4b** — it holds on **`vit_small_p16` specifically**

H4b is separated deliberately. A convolution-only result leaves the claim
exactly where a reviewer attacks it; if conv holds and the transformer does not,
**the claim becomes architecture-conditional and the title must say so**.

## The cost warning that matters

| run | measured | use for planning? |
|---|---|---|
| `p0-resnet50-imagenet100-base-s1` | **41.5 GPU-h** | **NO** |
| `p0-vit_small_p16-imagenet100-base-s1` | 5.7 GPU-h | yes |

The 41.5 h figure is the **D-59 `channels_last` run** — four times the
documented ~6 h, because the model was NHWC while the batches were NCHW. Nothing
looked broken: the loss fell, the accuracy climbed, every epoch simply took four
times too long, and the flat throughput was the only tell.

**So this notebook gates on throughput during epoch 1** and aborts if the
regression has returned. One lost epoch beats thirty-five lost hours.
"""),
        code(bootstrap_cell()),
        code(paths_cell(phase="p6", dataset="imagenet100")),
        md("""
---
## Preflight
"""),
        code(f"""
ARCHS = {list(IN100_ARCHS)!r}
SEED = 1
SCHEME = 'uniform'

print('measured throughput benchmarks (img/s, contiguous):')
for a in ARCHS:
    v = M.IN100_MEASURED_IMG_S.get(a)
    print(f'  {{a:18s}} {{v if v else "unknown"}}')
print()

cfgs = []
for a in ARCHS:
    c = sess.config(a, seed=SEED, method='jointexit',
                    joint_exits=True, exit_weight_scheme=SCHEME)
    assert c.get('joint_exits') is True, 'joint_exits did not survive config()'
    assert c.get('channels_last') is False, (
        'channels_last is True -- D-59 measured that 6.7x SLOWER on this card, '
        'and D-87 was the same flag with two defaults. Refusing to start.')
    cfgs.append(c)
    K = len(M.load_or_build_budgets(a, sess.work, 'imagenet100')['axes']['depth']['rho'])
    print(f"{{c['run_id']:44s}} epochs={{c['num_epochs']}}  K={{K}}  "
          f"bs={{c.get('batch_size')}}  weights="
          f"{{[round(x,3) for x in M.exit_loss_weights(K, SCHEME)]}}")

# the FROZEN counterparts must exist, or the paired comparison is impossible
runs_dir = Path(MSC_ROOT) / 'runs'
print()
for a in ARCHS:
    hits = [d.name for d in runs_dir.iterdir()
            if d.is_dir() and f'-{{a}}-imagenet100-base-s' in d.name
            and (d / 'per_sample' / 'test.parquet').exists()]
    if not hits:
        raise RuntimeError(
            f'no measured frozen run for {{a}} -- the paired comparison would be '
            'impossible AFTER the GPU time is spent. Fetch it first.')
    print(f'  paired frozen run for {{a}}: {{sorted(hits)[0]}}')
"""),
        md("""
---
## Dry run, then a memory probe

Two synthetic batches through the entire joint path before any real epoch.
`joint_exits` at 224px has never run on this machine — it was written where
there is no torch, so every training line below ships unexecuted until now.
"""),
        code("""
import torch, traceback
ok = True
for c in cfgs:
    try:
        r = M.backbone_dry_run(c)
        print(f"  {c['run_id']:44s} {r if not isinstance(r, tuple) else r[1]}")
    except Exception as e:
        ok = False
        print(f"  {c['run_id']:44s} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
if not ok:
    raise RuntimeError('dry run failed -- no GPU time spent. Fix first.')

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    free, total = torch.cuda.mem_get_info()
    print()
    print(f'GPU free {free/2**30:.1f} of {total/2**30:.1f} GiB;  '
          f'dry-run peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    print('K heads + K losses at 224px is materially heavier than 32px.')
    print('If a real epoch OOMs, halve batch_size rather than risk the machine.')
"""),
        md("""
---
## Train, with the throughput gate armed

**Safe to stop at any time** — resume restores optimiser, scheduler, AMP scaler
and all four RNG streams.

After the first run finishes an epoch, the next cell checks its speed. Run the
gate cell **before** leaving this unattended.
"""),
        code("""
results = sess.run_all(cfgs, title='Study 4 P2 / ImageNet-100 joint exits')
print()
for r in results:
    print(f"  {r.get('status','?'):9s} {r.get('run_id','?')}  "
          f"top1={M.fmt_metric(r.get('best_accuracy'))}  "
          f"{r.get('num_epochs_run','?')} epochs")
"""),
        md("""
---
## R-02 — the throughput gate

`resnet50` at 41.5 GPU-h looked healthy the whole way. A layout tax is a **fixed**
cost per convolution, so the signature is a flat, slow throughput rather than a
crash. This compares measured img/s against the contiguous benchmark.
"""),
        code("""
import pandas as pd
ALARM = 2.0          # abort if this many times slower than benchmark

for c in cfgs:
    L = M.run_layout(sess.work, c['run_id'])
    hist = L['metrics'] / 'epochs.csv'
    if not hist.exists():
        print(f"  {c['run_id']:44s} no epochs.csv yet"); continue
    h = pd.read_csv(hist)
    tcol = next((x for x in ('epoch_time_sec', 'time_sec', 'wall_sec')
                 if x in h.columns), None)
    if tcol is None or h.empty:
        print(f"  {c['run_id']:44s} no timing column in epochs.csv"); continue
    sec = float(h[tcol].iloc[0])
    n_train = 130000                      # ImageNet-100 train size, approx
    img_s = n_train / max(sec, 1e-9)
    bench = M.IN100_MEASURED_IMG_S.get(c['arch'])
    tag = ''
    if bench:
        ratio = bench / max(img_s, 1e-9)
        tag = f'   benchmark {bench:.0f}  ratio {ratio:.2f}x'
        if ratio > ALARM:
            raise RuntimeError(
                f"{c['run_id']}: {img_s:.0f} img/s against a benchmark of "
                f"{bench:.0f} -- {ratio:.1f}x slower. This is the D-59 "
                "signature. ABORT, check channels_last and assert_layout_match, "
                "and do not spend 35 hours confirming it.")
    print(f"  {c['run_id']:44s} epoch 1: {sec:.0f}s  ~{img_s:.0f} img/s{tag}")
print()
print('throughput gate passed')
"""),
        md("""
---
## Measure

`fn=sess.oracle`, `done_fn=sess.measured`, `stage='measure'` — **all three**.
`stage` only labels the plan; `fn` selects the work and the completion
predicate. Passing fewer is refused by D-67/D-88, which cost two rounds in
Study 3.
"""),
        code("""
for c in cfgs:
    hp = M.exit_heads_path(sess.work, c['run_id'])
    if not hp.exists():
        raise RuntimeError(f"{c['run_id']}: no exit_heads.pt -- joint training "
                           "should have written it at every new best.")
    blob = torch.load(hp, map_location='cpu', weights_only=False)
    if not blob.get('joint'):
        raise RuntimeError(f"{c['run_id']}: exit_heads.pt is not marked joint.")
    print(f"  {c['run_id']:44s} joint heads OK  epoch={blob.get('epoch')}")

print()
res = sess.run_all(cfgs, fn=sess.oracle, done_fn=sess.measured,
                   stage='measure', title='Study 4 P2 / measurement')
for r in res:
    print(f"  {r.get('status','?'):9s} {r.get('run_id','?')}")

# rule 5 / D-79: a plan that says 'nothing to do' is not evidence
print()
missing = []
for c in cfgs:
    ps = M.run_layout(sess.work, c['run_id'])['per_sample'] / 'test.parquet'
    if ps.exists() and ps.stat().st_size > 0:
        print(f"  {c['run_id']:44s} test.parquet {ps.stat().st_size/2**20:.1f} MB")
    else:
        missing.append(c['run_id'])
if missing:
    raise RuntimeError(f'measurement reported success but test.parquet is '
                       f'missing for {missing}')
"""),
        md("""
---
## H4 and H4b — the verdict
"""),
        code("""
import numpy as np
def excess_of(run_id):
    d = pd.read_parquet(M.run_layout(sess.work, run_id)['per_sample'] / 'test.parquet')
    ks = sorted(int(c.split('_d')[1]) for c in d.columns
                if c.startswith('pred_d') and c.split('_d')[1].isdigit())
    lab = d['label'].to_numpy()
    corr = np.stack([(d[f'pred_d{k}'].to_numpy() == lab) for k in ks], axis=1)
    acc_full = float(corr[:, -1].mean())
    return (float(corr.any(axis=1).mean()) - acc_full) * 100, acc_full * 100

rows = []
for c in cfgs:
    ex, af = excess_of(c['run_id'])
    rows.append({'arch': c['arch'], 'excess': ex, 'acc_full': af})
    print(f"  {c['arch']:18s} excess {ex:+6.2f} pt   full compute {af:.2f} %")
q = pd.DataFrame(rows)
M.save_analysis(sess.data_dir, 's4_imagenet_excess', q)

print()
n_ok = int((q['excess'] >= 2.0).sum())
print(f'H4  (>= 2.0 pt, 2 of 2): {n_ok} of {len(q)} -> '
      f'{"SUPPORTED" if n_ok == len(q) else "NOT SUPPORTED"}')

vit = q[q['arch'] == 'vit_small_p16']
if len(vit):
    v = float(vit['excess'].iloc[0])
    print(f'H4b (transformer alone):  {v:+.2f} pt -> '
          f'{"SUPPORTED" if v >= 2.0 else "NOT SUPPORTED"}')
    if v < 2.0:
        print()
        print('*** The transformer does NOT show the excess. ***')
        print('The claim is architecture-conditional. Say so in the TITLE --')
        print('do not average conv and transformer into a mean that describes')
        print('neither.')
"""),
    ])


# ---------------------------------------------------------------------------
# S4_NB3 -- publish
# ---------------------------------------------------------------------------
def nb3():
    return notebook([
        md("""
# S4_NB3 — publish, once, at the end

**Run last, with a network.** Nothing else in Study 4 touches HuggingFace.

Same design as `S3_NB5_Publish`, with the fix that notebook needed:
`resolve_meta` lives on `BackgroundUploader`, not `MSCHub`, so the verification
call is made directly.
"""),
        code(bootstrap_cell()),
        code(paths_cell(phase="analysis", hf=True)),
        md("""
---
## Check the token BEFORE uploading anything
"""),
        code("""
import os
TOKEN = os.environ.get('HF_TOKEN') or ''
REPO = os.environ.get('MSC_HF_REPO', 'Shanmuk4622/msc-cifar100')
print(f'repo  : {REPO}')
print(f'token : {"set, " + str(len(TOKEN)) + " chars" if TOKEN else "NOT SET"}')
print()
chk = M.hf_token_check(TOKEN or None, REPO)
for k in ('ok', 'valid', 'can_write', 'user', 'namespace', 'reason'):
    if k in chk:
        print(f'  {k:12s} {chk[k]}')
if not chk.get('ok'):
    raise RuntimeError(
        'token check FAILED. Most common cause is a READ token: HuggingFace -> '
        'Settings -> Access Tokens -> New token with the WRITE role, then set '
        'HF_TOKEN. Nothing has been uploaded.')
print()
print('token OK -- safe to upload')
"""),
        md("""
---
## Size it before moving a byte
"""),
        code("""
import pandas as pd
root = Path(MSC_ROOT)
INCLUDE = ['runs', 'analysis', 'budgets', 'registry', 'tables', 'paper']
rows = []
for top in INCLUDE:
    d = root / top
    if not d.is_dir():
        continue
    for p in d.rglob('*'):
        if p.is_file() and p.suffix not in {'.tmp', '.lock'}:
            rows.append({'top': top, 'path': str(p.relative_to(root)),
                         'mb': p.stat().st_size / 2**20, 'ckpt': p.suffix == '.pt'})
files = pd.DataFrame(rows)
if files.empty:
    raise RuntimeError(f'nothing to upload under {root}')
print(files.groupby('top')['mb'].agg(['count', 'sum']).round(1).to_string())
print()
print(f'TOTAL {len(files):,} files   {files["mb"].sum()/1024:.2f} GB')
print(f'  checkpoints (.pt): {files[files["ckpt"]]["mb"].sum()/1024:.2f} GB')
"""),
        code("""
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

**Rules 9 and 10.** A drained upload queue is not confirmation, and a *truncated
listing* is not evidence of absence — that mistake was made twice while checking
Study 3's upload. Resolve specific paths.
"""),
        code("""
from huggingface_hub import get_hf_file_metadata, hf_hub_url

def resolve_meta(rel):
    url = hf_hub_url(repo_id=REPO, filename=rel, repo_type='dataset',
                     revision='main')
    try:
        m = get_hf_file_metadata(url, token=TOKEN or None)
    except Exception as e:
        s = str(e).lower()
        if '404' in s or 'not found' in s or 'entrynotfound' in s:
            return None
        raise RuntimeError(f'could not determine whether {rel} exists: {e}. '
                           'Refusing to report absence on a failed lookup.') from e
    return {'path': rel, 'size': getattr(m, 'size', None)}

probe = sorted(set(
    files.sample(min(8, len(files)), random_state=0)['path'].tolist()
    + [p for p in files['path'] if p.endswith('per_sample/test.parquet')][:3]
    + [p for p in files['path'] if p.startswith('analysis') and 's4_' in p][:4]))

ok = miss = 0
for rel in probe:
    meta = resolve_meta(Path(rel).as_posix())
    print(f'  {"OK  " if meta else "MISS"} {rel}'
          + (f'  ({meta["size"]:,} B)' if meta and meta.get('size') else ''))
    ok, miss = ok + bool(meta), miss + (not meta)
print()
if miss:
    raise RuntimeError(f'{miss} of {ok+miss} probed files did NOT resolve -- '
                       're-run the upload cell.')
print(f'{ok} probed file(s) resolve on HuggingFace')
print('The local tree remains the source of truth. This is a copy of it.')
"""),
    ])


NOTEBOOKS = {
    "S4_NB0_Figures.ipynb": nb0,
    "S4_NB1_Baselines.ipynb": nb1,
    "S4_NB2_ImageNet.ipynb": nb2,
    "S4_NB3_Publish.ipynb": nb3,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in NOTEBOOKS.items():
        nb = fn()
        (OUT / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        nc = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        nm = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        print(f"  {name:28s} {nc:2d} code + {nm:2d} md   "
              f"{(OUT / name).stat().st_size // 1024:5d} KB")

    # PARSE GATE FIRST. A cell that does not parse defines no names, so the
    # name check would otherwise report a phantom NameError three cells later.
    print("\n  parsing every cell as Python 3.10")
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
        print("\n  Generation refused."); return 1
    print("  all cells parse")

    print("\n  checking for names no earlier cell defines")
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_names.py"),
                        *[str(OUT / n) for n in NOTEBOOKS]],
                       capture_output=True, text=True)
    print(r.stdout.rstrip() or r.stderr.rstrip())
    if r.returncode:
        print("\n  Generation refused."); return 1

    # Every M.* and sess.* must EXIST. check_names cannot see attributes.
    print("\n  checking every M.* and sess.* against the library AST")
    src = (ROOT / "src" / "msc_lib.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top = set()
    def walk(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name): top.add(t.id)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                top.add(n.target.id)
            elif isinstance(n, ast.If):
                walk(n.body); walk(n.orelse)
            elif isinstance(n, ast.Try):
                walk(n.body); walk(n.orelse); walk(n.finalbody)
                for h in n.handlers: walk(h.body)
    walk(tree.body)
    sess_cls = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.ClassDef) and n.name == "Session")
    smeth = {m.name for m in sess_cls.body
             if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    sattr = {n.attr for n in ast.walk(sess_cls) if isinstance(n, ast.Attribute)
             and isinstance(n.value, ast.Name) and n.value.id == "self"}
    missing = []
    for name in NOTEBOOKS:
        nb = json.loads((OUT / name).read_text(encoding="utf-8"))
        codes = "\n".join("".join(c["source"]) for c in nb["cells"]
                          if c["cell_type"] == "code")
        for m in sorted(set(re.findall(r"\bM\.([A-Za-z_]\w*)", codes))):
            if m not in top and m != "__version__":
                missing.append(f"{name}: M.{m}")
        for m in sorted(set(re.findall(r"\bsess\.([A-Za-z_]\w*)", codes))):
            if m not in smeth and m not in sattr:
                missing.append(f"{name}: sess.{m}")
    if missing:
        for x in missing: print(f"  [FAIL] {x} does not exist")
        print("\n  Generation refused."); return 1
    print("  every referenced library symbol exists")

    print(f"\nOK -- {len(NOTEBOOKS)} notebook(s) in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
