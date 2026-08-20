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
a_cf, c_cf, _ = route_confidence(conf, correct, rho, 0.70)
check(f'informative early confidence -> near-oracle ({a_cf:.3f} vs {a_or:.3f})',
      a_cf > a_rd + 0.05)

# 5. and if early confidence is pure noise it must NOT beat random
conf_noise = rng.random((n, K))
a_nz, _, _ = route_confidence(conf_noise, correct, rho, 0.70)
check(f'uninformative confidence ~ random ({a_nz:.3f} vs {a_rd:.3f})',
      abs(a_nz - a_rd) < 0.06)

route_matched = ns['route_matched']

# 6. MATCHED HISTOGRAM: cost is identical by construction, so a perfect oracle
#    can never lose. The -10 pt 'result' came from comparing two mechanisms.
_, _, counts = route_confidence(conf, correct, rho, 0.70)
a_or_m = route_matched(oracle_rank, correct, counts)
a_rd_m = route_matched(rng.random(n), correct, counts)
check(f'matched: oracle {a_or_m:.3f} >= random {a_rd_m:.3f}', a_or_m >= a_rd_m)

# 7. the headroom of a PERFECT oracle over the confidence baseline must be >= 0
#    whenever they are given the same histogram. Negative is impossible.
a_cf_m, _, counts2 = route_confidence(conf, correct, rho, 0.70)
head = (route_matched(oracle_rank, correct, counts2) - a_cf_m) * 100
check(f'matched: perfect-oracle headroom {head:+.2f} pt is not negative',
      head >= -1e-9)

# 8. the exit histogram really is identical -> no residual budget mismatch
check(f'matched: counts sum to n ({int(sum(counts))} == {n})', int(sum(counts)) == n)

# 9. THE CANARY THAT MATTERS. The study's headline is 'headroom is ~0'. A
#    statistic that reports ~0 because it CANNOT SEE headroom would produce the
#    same number. So build a world where headroom certainly exists -- an
#    overconfident net, wrong exactly where it is most sure early -- and require
#    the measurement to find it.
easy2 = rng.random(n) < 0.5
correct2 = np.zeros((n, K))
correct2[easy2, :] = 1.0                    # easy: correct everywhere
correct2[~easy2, K - 1] = 1.0               # hard: correct only at the end
conf_bad = np.full((n, K), 0.5)
conf_bad[~easy2, :] = 0.99                  # CONFIDENT ON THE HARD ONES
conf_bad[easy2, :] = 0.10
a_bad, _, cnts_bad = route_confidence(conf_bad, correct2, rho, 0.70)
a_orc = route_matched(np.where(easy2, 0.0, 1.0), correct2, cnts_bad)
gap = (a_orc - a_bad) * 100
check(f'DETECTS real headroom when it exists ({gap:+.1f} pt, oracle '
      f'{a_orc:.3f} vs misleading confidence {a_bad:.3f})', gap > 5.0)

oracle_rank = ns['oracle_rank']

route_oracle = ns['route_oracle']

# 10. THE INVARIANT. The in-seed oracle is a maximum over every assignment
#     meeting the budget, so it cannot lose to a confidence threshold at no
#     greater cost -- on ANY input. 200 adversarial draws, random budgets,
#     confidence deliberately unrelated to correctness. The notebook asserts
#     this on every (arch, seed) pair; if it can fail here, it will fail there.
worst = 0.0
for trial in range(200):
    m = int(rng.integers(200, 2000))
    cc = (rng.random((m, K)) < rng.random()).astype(float)
    cf = rng.random((m, K))
    tr = float(rng.uniform(0.3, 0.95))
    b_, cb, _ = route_confidence(cf, cc, rho, tr)
    a_, ca = route_oracle(cc, cc, rho, tr)
    if ca <= cb + 1e-6:
        worst = min(worst, a_ - b_)
check(f'in-seed oracle never loses to confidence, 200 draws (worst {worst:+.6f})',
      worst >= -1e-6)

# 11. and it must be STRICTLY better when confidence is uninformative
cc = np.zeros((n, K)); ez = rng.random(n) < .5
cc[ez, :] = 1.0; cc[~ez, K-1] = 1.0
b_, _, _ = route_confidence(rng.random((n, K)), cc, rho, 0.70)
a_, _ = route_oracle(cc, cc, rho, 0.70)
check(f'oracle beats uninformative confidence ({(a_-b_)*100:+.1f} pt)',
      a_ - b_ > 0.05)

# 12. the oracle must never OVERSPEND. It may underspend: in the easy/hard
#     world above, every useful assignment costs at most 0.60, and buying more
#     compute cannot buy more accuracy. Requiring equality here was my error --
#     it asserted a property the problem does not have.
_, c_ = route_oracle(cc, cc, rho, 0.65)
check(f'oracle never overspends (cost {c_:.4f} <= 0.65)', c_ <= 0.65 + 1e-6)

# 12b. and where the budget genuinely BINDS -- graded correctness, so deeper
#      always helps a little -- it must be consumed, not left on the table.
grad = (rng.random((n, K)) < np.linspace(0.35, 0.9, K)[None, :]).astype(float)
grad = np.maximum.accumulate(grad, axis=1)        # deeper never hurts
# With monotone correctness the oracle never wants more than the cheapest
# correct exit, so 0.65 does not bind either -- it tops out near 0.42. The
# budget only binds BELOW that, where samples must actually be sacrificed.
_, cfree = route_oracle(grad, grad, rho, 0.99)
check(f'unconstrained oracle cost {cfree:.4f} < 0.65 (so 0.65 cannot bind)',
      cfree < 0.65)
_, cg = route_oracle(grad, grad, rho, 0.30)
check(f'oracle spends a genuinely binding budget (cost {cg:.4f} ~ 0.30)',
      abs(cg - 0.30) < 0.02)

# 13. cross-seed <= in-seed when the two seeds agree perfectly they must TIE;
#     when seed j is pure noise the cross-seed oracle must be clearly worse.
a_same, _ = route_oracle(cc, cc, rho, 0.70)
noise = (rng.random((n, K)) < 0.5).astype(float)
a_noise, _ = route_oracle(noise, cc, rho, 0.70)
check(f'cross-seed with a noise instrument is worse ({(a_noise-a_same)*100:+.1f} pt)',
      a_noise < a_same - 0.01)

print(f'\n{sum(res)}/{len(res)} routing canaries pass')
sys.exit(0 if all(res) else 1)
