#!/usr/bin/env python3
"""Execute S3_NB0's real analysis cells against synthetic frames.

No pyarrow or scipy here, so pd.read_parquet and scipy.stats are stubbed and
the notebook's own code is run verbatim on frames whose answer is known.

Usage:  python tools/s3_nb0_harness.py
"""
import json, sys, types
from pathlib import Path
import numpy as np, pandas as pd

def spearmanr(a, b=None):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    n = len(ra)
    r = float(np.corrcoef(ra, rb)[0, 1]) if n > 2 else 0.0
    return r, 0.001 if abs(r) > 0.5 else 0.5
sp = types.ModuleType("scipy.stats"); sp.spearmanr = spearmanr
sc = types.ModuleType("scipy"); sc.stats = sp
sys.modules["scipy"], sys.modules["scipy.stats"] = sc, sp

ARCHS = ["resnet20", "resnet32x4", "vgg8", "mixer_nano", "convnext_femto"]
RUNS = [f"p1-{a}-cifar100-base-s{s}" for a in ARCHS for s in (1, 2, 3)]
rng = np.random.default_rng(0)
N, K = 2000, 5

# Ground truth we build in: exit quality VARIES by architecture, and excess
# falls as exit quality rises. The notebook must recover a negative slope.
QUAL = dict(zip(ARCHS, [0.55, 0.65, 0.75, 0.85, 0.95]))

def make(run_id):
    arch = run_id.split("-")[1]
    q = QUAL[arch]
    lab = rng.integers(0, 100, N)
    d = {"sample_idx": np.arange(N), "label": lab}
    final_ok = rng.random(N) < 0.72
    for k in range(1, K + 1):
        frac = q + (1 - q) * (k - 1) / (K - 1)
        ok = final_ok & (rng.random(N) < frac)
        # the noise pool: right early, wrong at the end -- shrinks as q rises
        if k < K:
            ok = ok | ((~final_ok) & (rng.random(N) < 0.18 * (1 - q)))
        else:
            ok = final_ok
        pred = np.where(ok, lab, (lab + 1) % 100)
        d[f"pred_d{k}"] = pred
        d[f"top1p_d{k}"] = rng.random(N)
    return pd.DataFrame(d)

pd.read_parquet = lambda path, *a, **k: make(Path(path).parent.parent.name)

for r in RUNS:
    (Path("/tmp/s3fake/runs") / r / "per_sample").mkdir(parents=True, exist_ok=True)
    (Path("/tmp/s3fake/runs") / r / "per_sample" / "test.parquet").touch()

M = types.SimpleNamespace()
def parse_run_id(r):
    ph, arch, ds, meth, sd = r.split("-")
    return {"run_id": r, "phase": ph, "arch": arch, "dataset": ds,
            "method": meth, "seed": int(sd[1:])}
M.parse_run_id = parse_run_id
# D-90: measured_runs consults ZOO to keep probe architectures (atlas=False)
# out of the study population. The real ZOO carries atlas=False on msdnet.
M.ZOO = {a: {'atlas': True} for a in ARCHS}
M.ZOO['msdnet'] = {'atlas': False}
M.save_analysis = lambda *a, **k: None
sess = types.SimpleNamespace(data_dir="/tmp", work="/tmp/s3fake")

nb = json.loads(Path("notebooks_study3/S3_NB0_Extrapolate.ipynb").read_text(encoding="utf-8"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
ns = {"M": M, "sess": sess, "MSC_ROOT": "/tmp/s3fake", "pd": pd, "np": np,
      "Path": Path, "msc": M, "__name__": "__main__"}
for idx, c in enumerate(cells):
    src = "".join(c["source"])
    if "unpack the library" in src or "WHERE EVERYTHING LIVES" in src:
        continue
    print(f"\n{'='*66}\nCODE CELL {idx}\n{'='*66}")
    try:
        exec(compile(src, f"<c{idx}>", "exec"), ns)
    except Exception:
        import traceback; traceback.print_exc()
        print(f">>> CELL {idx} RAISED"); sys.exit(1)

slope = ns.get("slope")
print(f"\n--- harness verdict ---")
print(f"built-in truth: excess FALLS as exit_quality rises")
print(f"recovered slope: {slope:+.4f}")
print("PASS" if slope is not None and slope < 0 else "FAIL",
      " the notebook recovers the sign of the relationship")
sys.exit(0 if (slope is not None and slope < 0) else 1)
