#!/usr/bin/env python3
"""Execute S2_NB1's real notebook cells against synthetic frames that
reproduce what is actually on disk (el2n/forget_events all-NaN on test,
populated on train_holdout, `msc` absent everywhere).

Why this exists: my environment has no pyarrow and no scipy, so every parquet
line in Study 1 shipped unexecuted. This runs the decidable part -- the
masking, coverage and refusal logic -- against the exact failure shape.

Usage:  python tools/s2_cell_harness.py
"""

import json, sys, types, numpy as np, pandas as pd
from pathlib import Path

def spearmanr(a, b=None):
    if b is None:
        A = np.asarray(a, float)
        R = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, A)
        return np.corrcoef(R, rowvar=False), None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]), 0.0
sp = types.ModuleType("scipy.stats"); sp.spearmanr = spearmanr
scipy = types.ModuleType("scipy"); scipy.stats = sp
sys.modules["scipy"] = scipy; sys.modules["scipy.stats"] = sp

ARCH = ['resnet32x4','resnet8x4','mobilenetv2','vit_tiny','mixer_nano','vgg8']
RUNS = [f'p0-{a}-cifar100-base-s{s}' for a in ARCH for s in (1,2,3)] + \
       [f'p0-{a}-imagenet100-base-s{s}' for a in ('resnet50','vit_small_p16') for s in (1,2)]
SCORES7 = ['msp','margin','entropy','ce_loss','el2n','forget_events','pred_depth']
rng = np.random.default_rng(0)

def make(run_id, split):
    n = 2000
    d = pd.DataFrame({'sample_idx': np.arange(n), 'label': rng.integers(0,100,n)})
    for k in range(1,6):
        d[f'pred_d{k}'] = rng.integers(0,100,n)
        d[f'top1p_d{k}'] = rng.random(n); d[f'top2p_d{k}'] = rng.random(n)*.5
    lat = rng.random(n)
    for s in SCORES7:
        if s in ('el2n','forget_events') and split == 'test':
            d[s] = np.nan                      # <- the observed on-disk shape
        else:
            d[s] = lat + rng.normal(0,.4,n)
    d['pred_depth'] = rng.integers(1,6,n).astype(float)
    return d                                    # note: NO 'msc' column

class FakePath(type(Path('.'))): pass
_real_rp = pd.read_parquet
def fake_rp(path, *a, **k):
    path = Path(path); return make(path.parent.parent.name, path.stem)
pd.read_parquet = fake_rp

class FakeDir:
    def __init__(s, runs): s.runs = runs
    def exists(s): return True
    def iterdir(s):
        for r in s.runs:
            d = Path('/tmp/fakeruns')/r; (d/'per_sample').mkdir(parents=True, exist_ok=True)
            (d/'per_sample'/'test.parquet').touch()
            (d/'per_sample'/'train_holdout.parquet').touch()
            yield d
for r in RUNS:
    d = Path('/tmp/fakeruns/runs')/r/'per_sample'; d.mkdir(parents=True, exist_ok=True)
    (d/'test.parquet').touch(); (d/'train_holdout.parquet').touch()

M = types.SimpleNamespace()
def parse_run_id(r):
    ph, arch, ds, meth, sd = r.split('-')
    return {'run_id': r,'phase':ph,'arch':arch,'dataset':ds,'method':meth,'seed':int(sd[1:])}
M.parse_run_id = parse_run_id
M.save_analysis = lambda *a, **k: None
sess = types.SimpleNamespace(data_dir='/tmp', work='/tmp')

nb = json.loads(Path('notebooks_study2/S2_NB1_Reliability.ipynb').read_text(encoding='utf-8'))
cells = [ ''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' ]
ns = {'M':M,'sess':sess,'MSC_ROOT':'/tmp/fakeruns','pd':pd,'np':np,'Path':Path,
      '__name__':'__main__'}
for idx, src in enumerate(cells):
    if 'CELL 1 -- unpack the library' in src or 'CELL 2 -- WHERE EVERYTHING LIVES' in src:
        continue
    print(f'\n{"="*72}\nCELL {idx}\n{"="*72}')
    try:
        exec(compile(src, f'<cell{idx}>', 'exec'), ns)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f'>>> CELL {idx} RAISED {type(e).__name__}')
        break
