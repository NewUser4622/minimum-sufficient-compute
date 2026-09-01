#!/usr/bin/env python3
"""Execute S3_NB3's real cells against synthetic frames.

No pyarrow / sklearn / scipy here, so those are stubbed and the notebook's own
code runs verbatim on data whose answer is known:

  * a gate given an INFORMATIVE confidence signal must capture a good share
  * a gate given PURE NOISE must capture ~nothing

Usage:  python tools/s3_nb3_harness.py
"""
import json, sys, types
from pathlib import Path
import numpy as np, pandas as pd

# --- stubs ---------------------------------------------------------------
class LogisticRegression:
    """Ridge-ish logistic fit by IRLS; enough to be genuinely predictive."""
    def __init__(self, max_iter=400, C=1.0): self.max_iter, self.C = max_iter, C
    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), np.asarray(X, float)]
        y = np.asarray(y, float)
        w = np.zeros(X.shape[1])
        for _ in range(25):
            p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
            W = np.clip(p * (1 - p), 1e-6, None)
            H = X.T @ (X * W[:, None]) + np.eye(X.shape[1]) / max(self.C, 1e-6)
            try: w += np.linalg.solve(H, X.T @ (y - p))
            except np.linalg.LinAlgError: break
        self.w = w; return self
    def predict_proba(self, X):
        X = np.c_[np.ones(len(X)), np.asarray(X, float)]
        p = 1 / (1 + np.exp(-np.clip(X @ self.w, -30, 30)))
        return np.c_[1 - p, p]
lm = types.ModuleType("sklearn.linear_model"); lm.LogisticRegression = LogisticRegression
sk = types.ModuleType("sklearn"); sk.linear_model = lm
sys.modules["sklearn"], sys.modules["sklearn.linear_model"] = sk, lm

ARCH_LIST = ["resnet20", "resnet32x4", "vgg8"]
RUNS = [f"p1-{a}-cifar100-base-s{s}" for a in ARCH_LIST for s in (1, 2, 3)]
N, K = 4000, 5
INFORMATIVE = "--noise" not in sys.argv

def make(run_id):
    g = np.random.default_rng(abs(hash(run_id)) % 2**31)
    lab = g.integers(0, 100, N)
    easy = g.random(N) < 0.5                    # shared difficulty structure
    d = {"sample_idx": np.arange(N), "label": lab}
    for k in range(1, K + 1):
        ok = easy | (g.random(N) < 0.15 * k)
        d[f"pred_d{k}"] = np.where(ok, lab, (lab + 1) % 100)
        if INFORMATIVE:
            # The discriminating world: raw top1p is nearly USELESS, but the
            # MARGIN (top1p - top2p) tracks correctness. The baseline can only
            # threshold top1p; the gate also sees the margin. If the gate
            # cannot beat the baseline here, its extra features do nothing and
            # the whole Q2 measurement is inert.
            p1 = g.uniform(.55, .75, N)
            p2 = np.where(ok, p1 - g.uniform(.30, .45, N),
                              p1 - g.uniform(.01, .05, N))
            p2 = np.clip(p2, 1e-3, None)
        else:
            p1 = g.uniform(.1, .99, N)          # nothing tells you anything
            p2 = p1 * g.uniform(.1, .6, N)
        d[f"top1p_d{k}"] = p1
        d[f"top2p_d{k}"] = p2
    return pd.DataFrame(d)

pd.read_parquet = lambda path, *a, **k: make(Path(path).parent.parent.name)

M = types.SimpleNamespace()
M.run_layout = lambda work, rid: {
    "per_sample": Path("/tmp/s3nb3") / rid / "per_sample",
    "checkpoints": Path("/tmp/s3nb3") / rid / "checkpoints"}
M.load_or_build_budgets = lambda a, w, d: {"axes": {"depth": {"rho": [.2, .4, .6, .8, 1.]}}}
M.save_analysis = lambda *a, **k: None
def parse(r):
    ph, arch, ds, meth, sd = r.split("-")
    return {"run_id": r, "phase": ph, "arch": arch, "dataset": ds,
            "method": meth, "seed": int(sd[1:])}
M.parse_run_id = parse
# D-90: measured_runs consults ZOO to keep probe architectures (atlas=False)
# out of the study population. The real ZOO carries atlas=False on msdnet.
M.ZOO = {'msdnet': {'atlas': False}}   # anything absent defaults to atlas=True
sess = types.SimpleNamespace(work="/tmp/s3nb3", data_dir="/tmp")
for r in RUNS:
    (Path("/tmp/s3nb3") / r / "per_sample").mkdir(parents=True, exist_ok=True)
    (Path("/tmp/s3nb3") / r / "per_sample" / "test.parquet").touch()

def measured_runs(dataset="cifar100", methods=("base",), require=True):
    return pd.DataFrame([{**parse(r), "run_id": r} for r in RUNS])

nb = json.loads(Path("notebooks_study3/S3_NB3_Router.ipynb").read_text(encoding="utf-8"))
ns = {"M": M, "sess": sess, "MSC_ROOT": "/tmp/s3nb3", "pd": pd, "np": np,
      "Path": Path, "msc": M, "measured_runs": measured_runs, "__name__": "__main__"}
for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c["source"])
    if "unpack the library" in src or "WHERE EVERYTHING LIVES" in src: continue
    if "Which runs am I analysing" in src: continue
    if "Get CIFAR-100" in src: continue
    print(f"\n{'='*64}\nCELL {idx}\n{'='*64}")
    try:
        exec(compile(src, f"<c{idx}>", "exec"), ns)
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)

cap = ns.get("cap")
cs = cap[cap["kind"] == "cross-seed"]["capture"].median()
print(f"\n--- harness verdict ({'INFORMATIVE' if INFORMATIVE else 'NOISE'}) ---")
print(f"cross-seed capture = {cs*100:.1f} %")
if INFORMATIVE:
    ok = cs > 0.05
    print(("PASS" if ok else "FAIL"),
          " a gate seeing MARGIN beats a threshold that cannot (>5% capture)")
else:
    ok = abs(cs) < 0.15
    print(("PASS" if ok else "FAIL"), " a gate on pure noise captures ~nothing")
sys.exit(0 if ok else 1)
