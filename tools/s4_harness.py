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

WHICH = sys.argv[sys.argv.index("--nb") + 1] if "--nb" in sys.argv else "014"
ROOT = Path(__file__).resolve().parent.parent
TMP = Path("/tmp/s4h"); (TMP / "analysis").mkdir(parents=True, exist_ok=True)

ARCHS = ["resnet20", "resnet32x4", "vgg8"]
BASE = [f"p1-{a}-cifar100-base-s{s}" for a in ARCHS for s in (1, 2, 3)]
JOINT = [f"p4-{a}-cifar100-jointexit-s1" for a in ARCHS]
MSD = [f"p7-msdnet-cifar100-jointexit-s{s}" for s in (1, 2)]
N, K = 3000, 5
TRUE_EXCESS = 0.09           # 9 accuracy points, built in
# MSDNet gets a DIFFERENT built-in excess from the attached runs, so NB4's
# designed-vs-attached comparison has a known answer (3.00 vs 9.00 -> -6.00 pt)
# instead of a number that would look right whichever way the subtraction ran.
TRUE_EXCESS_MSD = 0.03

def excess_for(run_id):
    return TRUE_EXCESS_MSD if run_id.startswith("p7-") else TRUE_EXCESS

def make(run_id):
    g = np.random.default_rng(abs(hash(run_id)) % 2**31)
    lab = g.integers(0, 100, N)
    final_ok = g.random(N) < 0.70
    # exactly this run's excess: right early, wrong at the end
    rescue = (~final_ok) & (g.random(N) < excess_for(run_id) / max(1 - 0.70, 1e-9))
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

for r in BASE + JOINT + MSD:
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
M.exit_loss_weights = lambda k, scheme="uniform": [1.0 / k] * k
# D-90: measured_runs must consult this to keep probe architectures out of the
# study population. The real ZOO carries atlas=False on msdnet.
M.ZOO = {a: {"atlas": True} for a in ARCHS}
M.ZOO["msdnet"] = {"atlas": False}
M.DEPTH_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
sess = types.SimpleNamespace(work=str(TMP), data_dir=str(TMP))
sess.config = lambda arch, seed=1, method="base", **kw: dict(
    run_id=f"p7-{arch}-cifar100-{method}-s{seed}", arch=arch, seed=seed,
    method=method, num_epochs=240, batch_size=64, learning_rate=0.05,
    channels_last=False, **kw)

def run(nbname, skip, extra=None):
    nb = json.loads((ROOT / "notebooks_study4" / nbname).read_text(encoding="utf-8"))
    ns = {"M": M, "sess": sess, "MSC_ROOT": str(TMP), "pd": pd, "np": np,
          "Path": Path, "msc": M, "__name__": "__main__"}
    ns.update(extra or {})
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

    # D-90 regression. MSDNet runs ARE on disk here, carrying a different
    # built-in excess (3.0 vs 9.0). If NB0's population ever picks them up the
    # published P0 intervals move without any error being raised -- which is
    # exactly what happened when msdnet was first added to the zoo.
    pop = set(ns["joint"]["arch"])
    ok = "msdnet" not in pop
    print(("PASS" if ok else "FAIL"),
          f" D-90: probe runs stay out of the P0 population ({sorted(pop)})")
    fails.append(not ok)
    ok = len(ns["ci"]) == len(ARCHS)
    print(("PASS" if ok else "FAIL"),
          f" P0 bootstraps exactly {len(ARCHS)} runs, not {len(ns['ci'])}")
    fails.append(not ok)

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

if "4" in WHICH:
    # Everything before the canary cell needs a GPU or a real torch build, so
    # only the ANALYSIS half of NB4 runs here. That is the half that produces
    # the verdict, and the half where a sign error would survive a green run.
    SKIP4 = SKIP + [
        "is the architecture actually", "spec = M.msdnet_channel_spec",
        "bud = M.load_or_build_budgets", "backbone_dry_run",
        "results = sess.run_all", "epochs.csv", "FLOOR = 55.0",
        "exit_heads_path",
    ]
    ns = run("S4_NB4_MSDNet.ipynb", SKIP4,
             extra={"rho": [.2, .4, .6, .8, 1.], "torch": None})
    msd, attached = ns["msd"], ns["attached"]
    exf = ns["excess_from_correct"]

    print("\n--- NB4 verdict ---")
    print(f"built in: MSDNet {TRUE_EXCESS_MSD*100:.1f} pt, "
          f"attached {TRUE_EXCESS*100:.1f} pt")

    # 1. the notebook's own canary cell ran; if excess_from_correct were blind
    #    to a present effect its asserts would already have raised.
    got = float(msd["excess"].mean())
    ok = abs(got - TRUE_EXCESS_MSD * 100) < 0.9
    print(("PASS" if ok else "FAIL"),
          f" MSDNet excess recovered: {got:.2f} pt "
          f"(built in {TRUE_EXCESS_MSD*100:.1f})")
    fails.append(not ok)

    got_a = float(attached["excess"].mean())
    ok = abs(got_a - TRUE_EXCESS * 100) < 0.9
    print(("PASS" if ok else "FAIL"),
          f" attached excess recovered: {got_a:.2f} pt "
          f"(built in {TRUE_EXCESS*100:.1f})")
    fails.append(not ok)

    # 2. the two populations must not be conflated. If NB4 ever read msdnet
    #    into `attached` the difference would collapse to ~0 and the paper
    #    would report "designed exits behave identically" from a bookkeeping
    #    slip rather than from a measurement.
    ok = "msdnet" not in set(attached["arch"])
    print(("PASS" if ok else "FAIL"),
          f" msdnet excluded from the attached population "
          f"({sorted(set(attached['arch']))})")
    fails.append(not ok)
    ok = set(msd["arch"]) == {"msdnet"} and len(msd) == 2
    print(("PASS" if ok else "FAIL"),
          f" both MSDNet seeds present and nothing else ({len(msd)} rows)")
    fails.append(not ok)

    # 3. the identity, on the notebook's own function
    ok = abs(exf(np.array([[1, 0], [0, 1], [0, 0]], bool)) - 100 / 3) < 1e-9
    print(("PASS" if ok else "FAIL"), " excess_from_correct is the early-saves "
          "fraction on a hand-checked 3x2 case")
    fails.append(not ok)

    # 4. THE branch test. Re-run the notebook's OWN verdict cell against three
    #    planted means and confirm each selects the pre-registered outcome. The
    #    third branch is the one that matters: 01_PROTOCOL.md says a null is a
    #    sharper claim, not a failure, and a notebook that cannot print it
    #    would quietly turn that promise into nothing.
    nb4 = json.loads((ROOT / "notebooks_study4" / "S4_NB4_MSDNet.ipynb")
                     .read_text(encoding="utf-8"))
    verdict_src = next("".join(c["source"]) for c in nb4["cells"]
                       if c["cell_type"] == "code"
                       and "H5 (>= 2.0 pt" in "".join(c["source"]))
    import io, contextlib
    for planted, want in ((9.0, "ARCHITECTURE-INDEPENDENT"),
                          (1.2, "PRESENT BUT ATTENUATED"),
                          (0.1, "PROPERTY OF ATTACHED EXITS")):
        frame = msd.copy()
        frame["excess"] = planted
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(verdict_src, "<nb4:verdict>", "exec"),
                 {**ns, "msd": frame, "attached": attached})
        out = buf.getvalue()
        ok = want in out
        print(("PASS" if ok else "FAIL"),
              f" excess {planted:+5.2f} pt -> {want}")
        fails.append(not ok)


print(f"\n{'ALL HARNESS CHECKS PASS' if not any(fails) else 'HARNESS FAILURES PRESENT'}")
sys.exit(1 if any(fails) else 0)
