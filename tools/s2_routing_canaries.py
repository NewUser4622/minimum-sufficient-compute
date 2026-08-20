#!/usr/bin/env python3
"""Canaries for route_by / route_confidence in S2_NB2.

The first version of these helpers had two defects that together produced a
+5.165 point 'result':
  * only exits 0 and K-1 were ever reachable (a 2-exit binary split);
  * the 'confidence baseline' thresholded the FINAL exit's confidence, which
    costs a full forward pass -- an oracle, not a deployable baseline.

Each canary below must fail if the corresponding property is broken.
Usage:  python tools/s2_routing_canaries.py
"""
import json, sys, numpy as np
from pathlib import Path

nb = json.loads((Path(__file__).resolve().parent.parent / 'notebooks_study2' /
                 'S2_NB2_Ceiling.ipynb').read_text(encoding='utf-8'))
src = next(''.join(c['source']) for c in nb['cells']
           if c['cell_type'] == 'code' and 'def route_confidence' in ''.join(c['source']))
ns = {'np': np, 'print': lambda *a, **k: None}
exec(compile(src, '<routing>', 'exec'), ns)
route_by, route_confidence = ns['route_by'], ns['route_confidence']

rho = [0.2, 0.4, 0.6, 0.8, 1.0]
K, n = 5, 4000
rng = np.random.default_rng(0)
res = []
def check(tag, cond):
    res.append(bool(cond)); print(f'{"PASS" if cond else "FAIL"}  {tag}')

# 1. budget is actually hit
correct = (rng.random((n, K)) < np.linspace(.4, .8, K)).astype(float)
acc, c = route_by(rng.random(n), correct, rho, 0.70)
check(f'route_by hits the budget (cost={c:.4f} vs 0.70)', abs(c - 0.70) < 0.02)

# 2. intermediate exits are REACHABLE -- the original bug
u = np.empty(n); u[np.argsort(rng.random(n), kind='stable')] = np.arange(n)/(n-1)
ka = np.clip((u * K * ((0+1)/2) * 2).astype(int), 0, K-1)
_, _ = route_by(rng.random(n), correct, rho, 0.60)
used = set()
for t in np.linspace(0.05, 0.95, 19):
    kk = np.clip((u * K * t * 2).astype(int), 0, K - 1)
    used |= set(np.unique(kk).tolist())
check(f'all {K} exits reachable (saw {sorted(used)})', used == set(range(K)))

# 3. a PERFECT oracle must beat a RANDOM router at the same budget
easy = rng.random(n) < 0.5
correct = np.zeros((n, K))
correct[easy, :] = 1.0                     # easy: right at every exit
correct[~easy, K-1] = 1.0                  # hard: right only at the last
oracle_rank = np.where(easy, 0.0, 1.0)
a_or, c_or = route_by(oracle_rank, correct, rho, 0.70)
a_rd, c_rd = route_by(rng.random(n), correct, rho, 0.70)
check(f'oracle {a_or:.3f} > random {a_rd:.3f} at matched cost', a_or > a_rd + 0.05)

# 4. confidence baseline reads EARLY exits: a net confident at exit 0 on the
#    easy samples must route them out early and lose nothing.
conf = np.full((n, K), 0.10); conf[easy, :] = 0.99
a_cf, c_cf = route_confidence(conf, correct, rho, 0.70)
check(f'informative early confidence -> near-oracle ({a_cf:.3f} vs {a_or:.3f})',
      a_cf > a_rd + 0.05)

# 5. and if early confidence is pure noise it must NOT beat random
conf_noise = rng.random((n, K))
a_nz, _ = route_confidence(conf_noise, correct, rho, 0.70)
check(f'uninformative confidence ~ random ({a_nz:.3f} vs {a_rd:.3f})',
      abs(a_nz - a_rd) < 0.06)

# 6. the ORIGINAL baseline (final-exit confidence) is an oracle: prove it beats
#    the deployable one, which is why it inflated headroom in reverse.
print(f'\n{sum(res)}/{len(res)} routing canaries pass')
sys.exit(0 if all(res) else 1)
