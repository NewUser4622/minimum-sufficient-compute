#!/usr/bin/env python3
"""Canaries for the S2_NB1 statistics: each must be shown CAPABLE of
reporting a wrong answer (risk register R-06, defect D-37).

A  every score NaN            -> must refuse, not reassure
B  train-only scores NaN on test -> pairwise n>0 (the original false pass)
C  two identical scores       -> collinearity must be detected

Usage:  python tools/s2_canaries.py   (exit 1 if any canary fails)
"""

import re, sys, json, numpy as np, pandas as pd
from pathlib import Path
base = (Path(__file__).resolve().parent / 's2_cell_harness.py').read_text()

def run(tag, patch, expect):
    src = patch(base)
    Path('/tmp/_s2canary.py').write_text(src)
    import subprocess
    r = subprocess.run([sys.executable,'/tmp/_s2canary.py'], capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = expect(out)
    print(f'{"PASS" if ok else "FAIL"}  {tag}')
    if not ok:
        print('\n'.join(out.splitlines()[-14:])); print('-'*60)
    return ok

res = []

# A. every score all-NaN on BOTH splits -> must refuse, not reassure
res.append(run('A  all scores NaN everywhere -> refuses',
    lambda s: s.replace("            d[s] = lat + rng.normal(0,.4,n)",
                        "            d[s] = np.nan"),
    lambda o: 'RuntimeError' in o and 'Refusing to continue' in o
              and 'reporting an empty result as a pass' in o))

# B. the ORIGINAL bug shape: el2n/forget_events all-NaN on test.
#    Listwise would give n=0; pairwise must give n=2000 and real numbers.
res.append(run('B  train-only scores NaN on test -> pairwise n>0, no false pass',
    lambda s: s,
    lambda o: 'min n=2,000' in o and 'n=0' not in o
              and 'all-NaN here : [\'el2n\', \'forget_events\']' in o))

# C. two scores identical -> must be REPORTED as near-collinear, not missed
res.append(run('C  margin := msp exactly -> collinearity detected',
    lambda s: s.replace("    d['pred_depth'] = rng.integers(1,6,n).astype(float)",
                        "    d['pred_depth'] = rng.integers(1,6,n).astype(float)\n"
                        "    d['margin'] = d['msp']"),
    lambda o: 'near-collinear pairs' in o and 'margin' in o))

print(f'\n{sum(res)}/{len(res)} canaries pass')
sys.exit(0 if all(res) else 1)
