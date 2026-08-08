"""
msc_core.py -- Minimum Sufficient Compute: oracle and analysis statistics.

Reference implementation for the MSC project. Deliberately depends only on
numpy / scipy / pandas / scikit-learn (no torch), so that analysis is fast,
portable, and runnable on a CPU-only session.

Everything here operates on per-sample tables produced by the oracle sweep.
The torch-side pieces (exit heads, ordinal sufficiency head, MSC loss) live
in msc_torch.py.

Run `python msc_core.py` to execute the self-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# 1. The MSC oracle
# ---------------------------------------------------------------------------

@dataclass
class MSCResult:
    """Per-sample MSC along one axis, at one margin threshold."""

    msc: np.ndarray                 # (N,) normalised cost in (0, 1]
    exit_index: np.ndarray          # (N,) index of the sufficient config, K-1 if none
    irreducible: np.ndarray         # (N,) bool -- full model itself below margin tau
    tau: float
    rho: np.ndarray                 # (K,) normalised costs, ascending, rho[-1] == 1
    axis: str = ""

    @property
    def n_irreducible(self) -> int:
        return int(self.irreducible.sum())

    @property
    def frac_irreducible(self) -> float:
        return float(self.irreducible.mean())

    def clean(self) -> np.ndarray:
        """MSC with irreducible samples masked to NaN.

        Correlation analyses must run on this, not on `msc`: irreducible
        samples all carry MSC == 1 by convention, and including them inflates
        agreement between any two models purely through a shared constant.
        """
        out = self.msc.astype(float).copy()
        out[self.irreducible] = np.nan
        return out


def compute_msc(
    preds: np.ndarray,
    top1p: np.ndarray,
    top2p: np.ndarray,
    rho: Sequence[float],
    tau: float = 0.1,
    axis: str = "",
) -> MSCResult:
    """Minimum Sufficient Compute under the stable-sufficiency definition.

    A configuration k is *stably sufficient* for sample i iff, for every
    j >= k, the decision agrees with the full-compute decision AND the
    top1-top2 margin is at least tau. MSC is the normalised cost of the
    smallest such k.

    The universal quantifier over larger budgets is the point. Predictions
    under compute reduction are not monotone -- a model can agree at 40%
    compute, disagree at 60%, and agree again at 100%. A naive
    `min over agreeing k` records the 40% point, which is an accident of
    the sweep rather than a property of the sample. The suffix closure
    records the point past which the decision has settled, and it makes
    the sufficiency indicator sequence monotone by construction.

    Parameters
    ----------
    preds  : (N, K) int   argmax class per configuration, ascending cost
    top1p  : (N, K) float top-1 softmax probability
    top2p  : (N, K) float top-2 softmax probability
    rho    : (K,)   float normalised cost, ascending, rho[-1] == 1.0
    tau    : float        margin threshold
    """
    preds = np.asarray(preds)
    top1p = np.asarray(top1p, dtype=float)
    top2p = np.asarray(top2p, dtype=float)
    rho = np.asarray(rho, dtype=float)

    n, k = preds.shape
    if rho.shape != (k,):
        raise ValueError(f"rho must have shape ({k},), got {rho.shape}")
    if not np.all(np.diff(rho) > 0):
        raise ValueError("rho must be strictly ascending")
    if not np.isclose(rho[-1], 1.0):
        raise ValueError("rho[-1] must be 1.0 (full compute reference)")

    reference = preds[:, -1]
    agree = preds == reference[:, None]
    margin_ok = (top1p - top2p) >= tau
    ok = agree & margin_ok                                   # (N, K)

    # Suffix-AND: suffix[:, j] is True iff ok[:, j:] is all True.
    suffix = np.ones_like(ok)
    suffix[:, -1] = ok[:, -1]
    for j in range(k - 2, -1, -1):
        suffix[:, j] = ok[:, j] & suffix[:, j + 1]

    any_ok = suffix.any(axis=1)
    exit_index = np.where(any_ok, suffix.argmax(axis=1), k - 1)
    msc = np.where(any_ok, rho[exit_index], 1.0)

    # The full model's own margin fails tau -> the definition degenerates.
    # These samples are a distinct population, not MSC == 1 observations.
    irreducible = ~ok[:, -1]

    return MSCResult(
        msc=msc,
        exit_index=exit_index,
        irreducible=irreducible,
        tau=tau,
        rho=rho,
        axis=axis,
    )


def compute_msc_from_frame(
    df: pd.DataFrame,
    axis: str,
    rho: Sequence[float],
    tau: float = 0.1,
    n_configs: int | None = None,
) -> MSCResult:
    """Convenience wrapper over the per-sample Parquet schema.

    Expects columns named `pred_{axis}{i}`, `top1p_{axis}{i}`,
    `top2p_{axis}{i}` for i in 1..K.
    """
    k = n_configs if n_configs is not None else len(rho)
    preds = np.stack([df[f"pred_{axis}{i}"].to_numpy() for i in range(1, k + 1)], axis=1)
    top1p = np.stack([df[f"top1p_{axis}{i}"].to_numpy() for i in range(1, k + 1)], axis=1)
    top2p = np.stack([df[f"top2p_{axis}{i}"].to_numpy() for i in range(1, k + 1)], axis=1)
    return compute_msc(preds, top1p, top2p, rho, tau=tau, axis=axis)


# ---------------------------------------------------------------------------
# 2. Correlation with a measurement-noise ceiling
# ---------------------------------------------------------------------------

def _paired_valid(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation over jointly-finite entries."""
    a, b = _paired_valid(np.asarray(a, float), np.asarray(b, float))
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def seed_ceiling(msc_seed1: np.ndarray, msc_seed2: np.ndarray) -> float:
    """Noise ceiling: MSC agreement between two seeds of the SAME architecture.

    This is the denominator of every transfer claim in the project. A
    cross-architecture correlation of 0.6 means something entirely different
    when seed-to-seed agreement is 0.95 than when it is 0.62. The example-
    difficulty literature routinely omits this, which makes its raw
    cross-architecture numbers hard to interpret.
    """
    return spearman(msc_seed1, msc_seed2)


def disattenuated_transfer(
    msc_a: np.ndarray,
    msc_b: np.ndarray,
    ceiling_a: float,
    ceiling_b: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Reliability-corrected transfer coefficient T(A, B).

        T = rho_S(A, B) / sqrt(ceiling_A * ceiling_B)

    This is Spearman's classical correction for attenuation. T ~ 1 means
    transfer is as complete as the measurement noise permits; T well below 1
    means genuine architecture-specific structure, not just noise.

    Returns raw correlation, T, and a bootstrap CI on T.
    """
    a, b = _paired_valid(np.asarray(msc_a, float), np.asarray(msc_b, float))
    raw = spearman(a, b)

    denom = np.sqrt(max(ceiling_a, 1e-9) * max(ceiling_b, 1e-9))
    t_point = raw / denom if denom > 0 else float("nan")

    n = a.size
    if n_boot <= 0:
        # Callers that only need the point estimate -- the shuffled control, for
        # one -- pass n_boot=0 rather than paying for a CI they discard.
        lo = hi = float("nan")
    else:
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            boots[i] = spearman(a[idx], b[idx]) / denom
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])

    return {
        "spearman_raw": raw,
        "ceiling_a": ceiling_a,
        "ceiling_b": ceiling_b,
        "T": t_point,
        "T_ci95": (float(lo), float(hi)),
        "n": int(n),
    }


def top_decile_jaccard(msc_a: np.ndarray, msc_b: np.ndarray, q: float = 0.9) -> float:
    """Jaccard overlap of the highest-MSC samples.

    For a routing application this matters more than global rank correlation:
    the router's job is identifying the expensive tail, not ordering the
    easy bulk correctly.
    """
    a = np.asarray(msc_a, float)
    b = np.asarray(msc_b, float)
    m = np.isfinite(a) & np.isfinite(b)
    idx = np.flatnonzero(m)
    a, b = a[m], b[m]
    if a.size == 0:
        return float("nan")

    ta, tb = np.quantile(a, q), np.quantile(b, q)
    sa = set(idx[a >= ta].tolist())
    sb = set(idx[b >= tb].tolist())
    union = sa | sb
    return len(sa & sb) / len(union) if union else float("nan")


# ---------------------------------------------------------------------------
# 3. Irreducibility to classical difficulty scores  (Q4 -- the main threat)
# ---------------------------------------------------------------------------

def partial_spearman(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> float:
    """Spearman correlation of x and y after linearly removing `controls`.

    Rank-transform everything, then correlate the residuals of x and y
    regressed on the ranked controls. If MSC is a monotone reparameterisation
    of classical difficulty, this collapses toward zero.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(controls, float)
    if c.ndim == 1:
        c = c[:, None]

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c).all(axis=1)
    x, y, c = x[m], y[m], c[m]
    if x.size < 10:
        return float("nan")

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rc = np.column_stack([stats.rankdata(c[:, j]) for j in range(c.shape[1])])
    rc = np.column_stack([np.ones(len(rc)), rc])

    beta_x, *_ = np.linalg.lstsq(rc, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(rc, ry, rcond=None)
    ex = rx - rc @ beta_x
    ey = ry - rc @ beta_y

    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(stats.pearsonr(ex, ey).statistic)


def irreducibility(
    msc_source: np.ndarray,
    msc_target: np.ndarray,
    difficulty: pd.DataFrame,
    n_splits: int = 5,
    n_boot: int = 500,
    seed: int = 0,
) -> dict:
    """Does MSC carry information beyond classical difficulty scores?

    Two tests, both needed:

      (a) partial Spearman of MSC_source and MSC_target controlling for the
          difficulty battery measured on the source model;
      (b) nested predictive comparison -- cross-validated R^2 for predicting
          MSC_target from the battery alone versus battery + MSC_source.

    If both collapse, MSC is difficulty renamed. That is a publishable
    finding, not a failure -- but it changes the paper, so the test runs
    early and its result is reported either way.
    """
    src = np.asarray(msc_source, float)
    tgt = np.asarray(msc_target, float)
    d = difficulty.to_numpy(dtype=float)

    m = np.isfinite(src) & np.isfinite(tgt) & np.isfinite(d).all(axis=1)
    src, tgt, d = src[m], tgt[m], d[m]

    partial = partial_spearman(src, tgt, d)

    def cv_r2(x: np.ndarray) -> np.ndarray:
        """Out-of-fold predictions from a gradient-boosted regressor."""
        oof = np.empty_like(tgt)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in kf.split(x):
            mdl = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.1, random_state=seed
            )
            mdl.fit(x[tr], tgt[tr])
            oof[te] = mdl.predict(x[te])
        return oof

    oof_base = cv_r2(d)
    oof_full = cv_r2(np.column_stack([d, src]))

    def r2(pred: np.ndarray, y: np.ndarray) -> float:
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    r2_base = r2(oof_base, tgt)
    r2_full = r2(oof_full, tgt)

    # Bootstrap the *difference* on the shared out-of-fold predictions, so the
    # CI reflects sampling noise rather than refit noise.
    rng = np.random.default_rng(seed)
    n = tgt.size
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = r2(oof_full[idx], tgt[idx]) - r2(oof_base[idx], tgt[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    return {
        "partial_spearman": partial,
        "r2_difficulty_only": r2_base,
        "r2_difficulty_plus_msc": r2_full,
        "delta_r2": r2_full - r2_base,
        "delta_r2_ci95": (float(lo), float(hi)),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# 4. Axis structure  (Q2 -- is compute need one-dimensional?)
# ---------------------------------------------------------------------------

def axis_structure(msc_by_axis: dict[str, np.ndarray]) -> dict:
    """Is per-sample compute need a single scalar factor across axes?

    Takes {axis_name: msc_vector} for depth / width / resolution / precision
    and asks how much of the joint variation one component explains.

    Never asked in this literature. Every adaptive-inference paper picks one
    axis and treats it as THE compute axis. If PC1 dominates, that implicit
    assumption is validated. If it does not, results on depth-based early
    exit do not license claims about width- or precision-adaptive inference,
    and routing has to be multi-dimensional.
    """
    names = list(msc_by_axis)
    mat = np.column_stack([np.asarray(msc_by_axis[k], float) for k in names])
    m = np.isfinite(mat).all(axis=1)
    mat = mat[m]

    if mat.shape[0] < 10:
        raise ValueError("too few jointly-valid samples for factor analysis")

    z = (mat - mat.mean(0)) / (mat.std(0) + 1e-12)
    pca = PCA(n_components=mat.shape[1]).fit(z)

    corr = np.corrcoef(
        np.column_stack([stats.rankdata(mat[:, j]) for j in range(mat.shape[1])]),
        rowvar=False,
    )

    return {
        "axes": names,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc1_loadings": dict(zip(names, pca.components_[0].tolist())),
        "spearman_matrix": pd.DataFrame(corr, index=names, columns=names),
        "n": int(mat.shape[0]),
    }


# ---------------------------------------------------------------------------
# 5. Sweep helper
# ---------------------------------------------------------------------------

def tau_sweep(
    preds: np.ndarray,
    top1p: np.ndarray,
    top2p: np.ndarray,
    rho: Sequence[float],
    taus: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5),
    axis: str = "",
) -> dict[float, MSCResult]:
    """MSC at every margin threshold.

    Every headline statistic in this project is reported as a curve over tau.
    A conclusion that survives only one tau is not a conclusion.
    """
    return {
        t: compute_msc(preds, top1p, top2p, rho, tau=t, axis=axis) for t in taus
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _synth(n=4000, k=5, latent=None, noise=0.0, seed=0):
    """Synthetic sweep where a latent 'compute need' drives the exit point."""
    rng = np.random.default_rng(seed)
    if latent is None:
        latent = rng.uniform(0, 1, n)
    obs = np.clip(latent + rng.normal(0, noise, n), 0, 1) if noise else latent
    true_exit = np.clip((obs * k).astype(int), 0, k - 1)

    preds = np.zeros((n, k), dtype=int)
    top1p = np.zeros((n, k))
    top2p = np.zeros((n, k))
    true_class = rng.integers(0, 100, n)

    for i in range(n):
        for j in range(k):
            if j >= true_exit[i]:
                preds[i, j] = true_class[i]
                top1p[i, j], top2p[i, j] = 0.9, 0.05
            else:
                preds[i, j] = rng.integers(0, 100)
                top1p[i, j], top2p[i, j] = 0.4, 0.35
    return preds, top1p, top2p, latent


def _selftest():
    rho = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("compute_msc")
    preds, t1, t2, latent = _synth(seed=1)
    r = compute_msc(preds, t1, t2, rho, tau=0.1)
    check("recovers latent compute need", spearman(r.msc, latent) > 0.95,
          f"rho_S={spearman(r.msc, latent):.3f}")
    check("MSC within (0, 1]", r.msc.min() > 0 and r.msc.max() <= 1.0)
    check("no spurious irreducibles", r.frac_irreducible == 0.0)

    print("stable-sufficiency closure")
    p = np.array([[1, 9, 1, 1]])                       # agrees, flips, agrees, agrees
    a = np.array([[0.9, 0.9, 0.9, 0.9]])
    b = np.array([[0.05, 0.05, 0.05, 0.05]])
    r2_ = compute_msc(p, a, b, [0.25, 0.5, 0.75, 1.0], tau=0.1)
    check("ignores the accidental early agreement", np.isclose(r2_.msc[0], 0.75),
          f"MSC={r2_.msc[0]}")

    print("irreducible subpopulation")
    p = np.array([[3, 3, 3]])
    a = np.array([[0.9, 0.9, 0.40]])
    b = np.array([[0.05, 0.05, 0.38]])                 # full-compute margin 0.02 < tau
    r3 = compute_msc(p, a, b, [0.3, 0.6, 1.0], tau=0.1)
    check("flags low-margin full-compute samples", r3.irreducible[0])
    check("masks them in clean()", np.isnan(r3.clean()[0]))

    print("transfer with noise ceiling")
    rng = np.random.default_rng(7)
    lat = rng.uniform(0, 1, 4000)
    a1 = compute_msc(*_synth(latent=lat, noise=0.10, seed=11)[:3], rho, tau=0.1).msc
    a2 = compute_msc(*_synth(latent=lat, noise=0.10, seed=12)[:3], rho, tau=0.1).msc
    b1 = compute_msc(*_synth(latent=lat, noise=0.25, seed=13)[:3], rho, tau=0.1).msc
    b2 = compute_msc(*_synth(latent=lat, noise=0.25, seed=14)[:3], rho, tau=0.1).msc
    ca, cb = seed_ceiling(a1, a2), seed_ceiling(b1, b2)
    tr = disattenuated_transfer(a1, b1, ca, cb, n_boot=200)
    check("T exceeds raw correlation", tr["T"] > tr["spearman_raw"],
          f"raw={tr['spearman_raw']:.3f} T={tr['T']:.3f} ceilings={ca:.3f}/{cb:.3f}")
    check("T is bounded sensibly", 0 < tr["T"] < 1.35)

    print("shuffled-target control")
    perm = np.random.default_rng(3).permutation(len(b1))
    sh = disattenuated_transfer(a1, b1[perm], ca, cb, n_boot=200)
    check("shuffled transfer ~ 0", abs(sh["T"]) < 0.05, f"T={sh['T']:.4f}")

    print("top-decile Jaccard")
    j = top_decile_jaccard(a1, b1)
    check("hard tails overlap above chance", j > 0.10, f"J10={j:.3f}")

    print("irreducibility")
    n = len(a1)
    rng = np.random.default_rng(5)
    diff = pd.DataFrame({
        "msp": 1 - lat + rng.normal(0, 0.05, n),
        "margin": 1 - lat + rng.normal(0, 0.08, n),
        "entropy": lat + rng.normal(0, 0.05, n),
    })
    irr = irreducibility(a1, b1, diff, n_boot=100)
    check("delta R^2 is finite", np.isfinite(irr["delta_r2"]),
          f"R2 {irr['r2_difficulty_only']:.3f} -> {irr['r2_difficulty_plus_msc']:.3f} "
          f"(d={irr['delta_r2']:+.3f})")
    check("partial Spearman is finite", np.isfinite(irr["partial_spearman"]),
          f"partial={irr['partial_spearman']:.3f}")

    print("axis structure")
    ax = axis_structure({"depth": a1, "resolution": b1, "precision": a2})
    check("PC1 dominates for a shared latent", ax["pc1_variance"] > 0.5,
          f"PC1={ax['pc1_variance']:.3f}")

    print("tau sweep")
    sw = tau_sweep(preds, t1, t2, rho)
    check("MSC is monotone in tau", all(
        sw[t].msc.mean() <= sw[u].msc.mean() + 1e-9
        for t, u in zip([0.0, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.5])
    ), " ".join(f"tau={t}:{r.msc.mean():.3f}" for t, r in sw.items()))

    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES PRESENT"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
