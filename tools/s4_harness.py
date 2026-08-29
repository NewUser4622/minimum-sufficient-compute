#!/usr/bin/env python3
"""Execute S4_NB0 and S4_NB1's real cells against synthetic data.

This environment has no pyarrow, so pd.read_parquet is stubbed with frames whose
answer is known. Everything else -- the plotting, the bootstrap, the routing, the
verdict logic -- is the notebook's own code, run verbatim.

Ground truth built in:
  * a KNOWN excess, so the bootstrap interval must exclude zero and bracket it
  * confidence is INFORMATIVE and margin is NOISE, so H6's "which baseline is
    strongest" must pick confidence -- if it picks margin, the comparison is
    inverted somewhere

Usage:  python tools/s4_harness.py [--nb 0|1]
"""
import json, sys, types
from pathlib import Path
import numpy as np, pandas as pd

WHICH = sys.argv[sys.argv.index("--nb") + 1] if "--nb" in sys.argv else "01"
ROOT = Path(__file__).resolve().parent.parent
TMP = Path("/tmp/s4h"); (TMP / "analysis").mkdir(parents=True, exist_ok=True)

ARCHS = ["resnet20", "resnet32x4", "vgg8"]
BASE = [f"p1-{a}-cifar100-base-s{s}" for a in ARCHS for s in (1, 2, 3)]
JOINT = [f"p4-{a}-cifar100-jointexit-s1" for a in ARCHS]
N, K = 3000, 5
TRUE_EXCESS = 0.09           # 9 accuracy points, built in

def make(run_id):
    g = np.random.default_rng(abs(hash(run_id)) % 2**31)
    lab = g.integers(0, 100, N)
    final_ok = g.random(N) < 0.70
    # exactly TRUE_EXCESS of samples are right early and wrong at the end
    rescue = (~final_ok) & (g.random(N) < TRUE_EXCESS / max(1 - 0.70, 1e-9))
    d = {"sample_idx": np.arange(N), "label": lab}
    for k in range(1, K + 1):
        ok = (final_ok & (g.random(N) < 0.5 + 0.1 * k)) | (rescue & (k < K))
        if k == K:
            ok = final_ok
        d[f"pred_d{k}"] = np.where(ok, lab, (lab + 1) % 100)
        # confidence is INFORMATIVE; margin (top1p-top2p) is deliberately NOISE
        p1 = np.where(ok, g.uniform(.75, .99, N), g.uniform(.05, .45, N))
        d[f"top1p_d{k}"] = p1
        d[f"top2p_d{k}"] = p1 - g.uniform(.01, .5, N)   # gap carries nothing
    return pd.DataFrame(d)

pd.read_parquet = lambda path, *a, **k: make(Path(path).parent.parent.name)

for r in BASE + JOINT:
    (TMP / "runs" / r / "per_sample").mkdir(parents=True, exist_ok=True)
    (TMP / "runs" / r / "per_sample" / "test.parquet").touch()

# the sweep CSV NB0 plots
rows = [{"target_rho": r, "arch": a, "score": s, "headroom": -5 - 3 * r}
        for r in (.4, .5, .6, .7, .8, .9, .95) for a in ARCHS
        for s in ("ce_loss", "entropy", "msp")]
pd.DataFrame(rows).to_csv(TMP / "analysis" / "s2_headroom_sweep.csv", index=False)
pd.DataFrame([{"arch": a, "excess_frozen": 6.5, "excess_joint": 9.0,
               "d_excess": 2.5, "accfull_frozen": 70, "accfull_joint": 71,
               "d_accfull": 1.0, "eq_frozen": .5, "eq_joint": .8}
              for a in ARCHS]).to_csv(TMP / "analysis" / "s3_q1_comparison.csv",
                                      index=False)

M = types.SimpleNamespace()
def parse(r):
    ph, arch, ds, meth, sd = r.split("-")
    return {"run_id": r, "phase": ph, "arch": arch, "dataset": ds,
            "method": meth, "seed": int(sd[1:])}
M.parse_run_id = parse
M.run_layout = lambda w, rid: {"per_sample": TMP / "runs" / rid / "per_sample",
                               "metrics": TMP / "runs" / rid / "metrics",
                               "checkpoints": TMP / "runs" / rid / "checkpoints"}
M.load_or_build_budgets = lambda a, w, d: {"axes": {"depth": {"rho": [.2, .4, .6, .8, 1.]}}}
M.save_analysis = lambda *a, **k: None
sess = types.SimpleNamespace(work=str(TMP), data_dir=str(TMP))

def run(nbname, skip):
    nb = json.loads((ROOT / "notebooks_study4" / nbname).read_text(encoding="utf-8"))
    ns = {"M": M, "sess": sess, "MSC_ROOT": str(TMP), "pd": pd, "np": np,
          "Path": Path, "msc": M, "__name__": "__main__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if any(s in src for s in skip):
            continue
        print(f"\n{'='*66}\n{nbname}  CELL {i}\n{'='*66}")
        try:
            exec(compile(src, f"<{nbname}:{i}>", "exec"), ns)
        except Exception:
            import traceback; traceback.print_exc()
            print(f">>> CELL {i} RAISED"); sys.exit(1)
    return ns

SKIP = ["unpack the library", "WHERE EVERYTHING LIVES"]
fails = []

if "0" in WHICH:
    ns = run("S4_NB0_Figures.ipynb", SKIP)
    ci = ns["ci"]
    lo, hi = ci["ci_lo"].mean(), ci["ci_hi"].mean()
    inside = lo <= TRUE_EXCESS * 100 <= hi
    print(f"\n--- NB0 verdict ---")
    print(f"built-in excess {TRUE_EXCESS*100:.1f} pt; mean CI [{lo:.2f}, {hi:.2f}]")
    print(("PASS" if inside else "FAIL"), " the CI brackets the true excess")
    print(("PASS" if lo > 0 else "FAIL"), " and excludes zero")
    fails += [not inside, not (lo > 0)]
    fig = Path(sess.data_dir) / "paper" / "figures" / "fig1_headroom.png"
    print(("PASS" if fig.exists() else "FAIL"), f" Figure 1 written ({fig})")
    fails.append(not fig.exists())

if "1" in WHICH:
    ns = run("S4_NB1_Baselines.ipynb", SKIP)
    # Read the NOTEBOOK's answer, not a re-derivation. The first version of
    # this harness re-ran idxmax on the raw accuracy table and so kept testing
    # the very logic the notebook had just fixed.
    win = ns["winner"]
    bl = ns["bl"]
    print(f"\n--- NB1 verdict ---")
    print("built in: confidence INFORMATIVE, margin NOISE.")
    print("Patience ALSO has perfect signal here (predictions agree on the easy")
    print("samples), so it legitimately wins -- my first expectation that")
    print("confidence must win ignored that. The discriminating test is that")
    print("the NOISE baseline must never win.")
    print(f"strongest baseline chosen (at matched cost): {win}")
    ok = win != "margin"
    print(("PASS" if ok else "FAIL"), " the noise baseline (margin) does NOT win")
    fails.append(not ok)

    # and a baseline with signal must beat the noise one on equal terms
    fair = bl[bl["achieved_cost"] <= bl["target_rho"] + 0.01]
    med = fair.groupby("baseline")["baseline_acc"].median()
    gap = med.get("confidence", 0) - med.get("margin", 0)
    print(("PASS" if gap >= -1e-9 else "FAIL"),
          f" confidence (signal) >= margin (noise): {gap:+.2f} pt")
    fails.append(gap < -1e-9)

    # and the exclusion must actually have happened
    tol = (bl["achieved_cost"] - bl["target_rho"]).abs()
    over = bl[tol > 0.05]["baseline"].unique().tolist()
    print(("PASS" if "patience" in over else "FAIL"),
          f" patience is detected as missing the budget ({over})")
    fails.append("patience" not in over)

print(f"\n{'ALL HARNESS CHECKS PASS' if not any(fails) else 'HARNESS FAILURES PRESENT'}")
sys.exit(1 if any(fails) else 0)
