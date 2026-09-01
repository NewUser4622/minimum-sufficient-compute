#!/usr/bin/env python3
"""Canaries for MSDNet's channel arithmetic -- and for the canaries themselves.

    python tools/s4_msdnet_canaries.py

MSDNet is the one architecture this project writes rather than borrows, so its
channel bookkeeping has no external reference to check against. Multi-scale
dense growth is the kind of arithmetic that is wrong by a factor of two, or
off by one scale, and still builds and still trains and still reports a number.

D-89 is the reason this file exists in this shape. That defect survived because
the canary guarding it asserted `cost <= target`, which a constant satisfies:
**a canary a broken function passes is not a canary.** So every predicate below
is run twice -- once against the real spec, where it must PASS, and once
against a spec deliberately corrupted in the specific way the predicate exists
to catch, where it must FAIL. A predicate that passes both is reported as
broken, and this script exits non-zero.

No torch required. That is deliberate: the arithmetic must be checkable on any
machine, including the one that plans the run rather than the one that trains
it.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import msc_lib as M  # noqa: E402

FAILURES: List[str] = []
N_RUN = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global N_RUN
    N_RUN += 1
    if ok:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# The predicates. Each is a pure function of a spec dict, so it can be pointed
# at a corrupted one.
# ---------------------------------------------------------------------------
def p_cuts_on_fractions(sp: Dict[str, Any]) -> bool:
    """Exits land exactly on DEPTH_FRACTIONS -- no exit lost to rounding."""
    return (tuple(c / sp["n_layers"] for c in sp["cuts"])
            == tuple(M.DEPTH_FRACTIONS))


def p_running_channels(sp: Dict[str, Any]) -> bool:
    """Every layer's declared input equals what the previous layer emitted."""
    acc = list(sp["stem_out"])
    for layer in sp["layers"]:
        for d in layer:
            if d["cin"] != acc[d["scale"]]:
                return False
        acc = [d["cout"] for d in layer]
    return True


def p_closed_form(sp: Dict[str, Any]) -> bool:
    """The walk and the formula agree: 2**s * (base + n_layers*growth)."""
    acc = list(sp["stem_out"])
    for layer in sp["layers"]:
        acc = [d["cout"] for d in layer]
    want = [(2 ** s) * (sp["base"] + sp["n_layers"] * sp["growth"])
            for s in range(sp["n_scales"])]
    return acc == want


def p_coarsest(sp: Dict[str, Any]) -> bool:
    """Exits read the COARSEST scale.

    This is the architectural claim under test in H5. If a wiring slip made
    the exits read scale 0, MSDNet would degenerate into exactly the attached
    head it is supposed to be contrasted with, and P3 would answer nothing.
    """
    coarse = sp["n_scales"] - 1
    acc = list(sp["stem_out"])
    dims, seen = [], 0
    for i, layer in enumerate(sp["layers"], start=1):
        acc = [d["cout"] for d in layer]
        if seen < len(sp["cuts"]) and i == sp["cuts"][seen]:
            dims.append(acc[coarse])
            seen += 1
    return tuple(dims) == tuple(sp["feature_dims"])


def p_ascending(sp: Dict[str, Any]) -> bool:
    """Strictly ascending feature dims.

    Not cosmetic. `msc_core.compute_msc` refuses non-ascending rho, because
    "the smallest sufficient budget" is ill-defined when two budgets tie.
    """
    fd = sp["feature_dims"]
    return all(b > a for a, b in zip(fd, fd[1:]))


def p_parts_sum(sp: Dict[str, Any]) -> bool:
    """Each layer's branches add up to its growth, and growth is dense."""
    return all(sum(p["cout"] for p in d["parts"]) == d["growth"]
               and d["cout"] == d["cin"] + d["growth"]
               for layer in sp["layers"] for d in layer)


def p_finer_source(sp: Dict[str, Any]) -> bool:
    """The strided branch reads scale s-1, at that scale's real width.

    The off-by-one that this catches produces a network that builds, trains,
    and is not MSDNet.
    """
    for layer in sp["layers"]:
        for d in layer[1:]:
            for p in d["parts"]:
                if p["src"] == "finer" and p["cin"] != layer[d["scale"] - 1]["cin"]:
                    return False
    return True


def p_finer_strided(sp: Dict[str, Any]) -> bool:
    """The cross-scale branch downsamples; the same-scale branch does not.

    If the finer branch were stride 1 its output would be twice the spatial
    size of the tensor it is concatenated with, and torch.cat would raise --
    but many layers later, naming neither the scale nor the resolution.
    """
    for layer in sp["layers"]:
        if len(layer[0]["parts"]) != 1 or layer[0]["parts"][0]["src"] != "same":
            return False
        for d in layer[1:]:
            got = {p["src"]: p["stride"] for p in d["parts"]}
            if got != {"same": 1, "finer": 2}:
                return False
    return True


def p_scale_widths(sp: Dict[str, Any]) -> bool:
    """Coarser scales are wider -- growth doubles per scale, as MSDNet's do."""
    for layer in sp["layers"]:
        g = [d["growth"] for d in layer]
        if g != [sp["growth"] * (2 ** s) for s in range(sp["n_scales"])]:
            return False
    return True


def p_stem_chain(sp: Dict[str, Any]) -> bool:
    """Scale 0 reads the image; every coarser scale reads the one above it."""
    cins = [d["cin"] for d in sp["stem"]]
    return cins == [3] + list(sp["stem_out"][:-1])


def p_resolutions(sp: Dict[str, Any]) -> bool:
    """Resolution halves per scale and stays integral."""
    return (sp["resolutions"]
            == [sp["in_res"] // (2 ** s) for s in range(sp["n_scales"])]
            and all(r >= 1 for r in sp["resolutions"]))


# ---------------------------------------------------------------------------
# The mutations. Each corrupts the spec in the exact way one predicate exists
# to catch, so "does this canary fire?" is itself a test.
# ---------------------------------------------------------------------------
def m_finer_reads_own_scale(sp: Dict[str, Any]) -> Dict[str, Any]:
    """The classic off-by-one: strided branch reads scale s, not s-1."""
    sp = copy.deepcopy(sp)
    for layer in sp["layers"]:
        for d in layer[1:]:
            for p in d["parts"]:
                if p["src"] == "finer":
                    p["cin"] = d["cin"]
    return sp


def m_exits_read_finest(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Exits read scale 0 -- MSDNet collapsed into an attached head."""
    sp = copy.deepcopy(sp)
    acc, dims, seen = list(sp["stem_out"]), [], 0
    for i, layer in enumerate(sp["layers"], start=1):
        acc = [d["cout"] for d in layer]
        if seen < len(sp["cuts"]) and i == sp["cuts"][seen]:
            dims.append(acc[0])
            seen += 1
    sp["feature_dims"] = tuple(dims)
    return sp


def m_growth_flat(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Growth does not widen with scale -- coarse features starved.

    Labels only: `growth` is relabelled but the widths are untouched. Paired
    with `p_scale_widths`, which reads exactly that label.
    """
    sp = copy.deepcopy(sp)
    for layer in sp["layers"]:
        for d in layer:
            d["growth"] = sp["growth"]
    return sp


def m_widths_drift(sp: Dict[str, Any]) -> Dict[str, Any]:
    """The realistic slip: `growth` written where `growth * 2**s` was meant.

    Rebuilds the whole walk consistently, so per-layer bookkeeping still
    agrees with itself -- only the closed form disagrees. This mutation exists
    because the first version of this file paired `p_closed_form` with
    `m_growth_flat`, which relabels `growth` without touching any width, and
    the predicate was blind to it. The suite caught that on its first run,
    which is the entire argument for section 2.
    """
    sp = copy.deepcopy(sp)
    ch = list(sp["stem_out"])
    for layer in sp["layers"]:
        for d in layer:
            d["cin"] = ch[d["scale"]]
            d["growth"] = sp["growth"]
            d["cout"] = d["cin"] + sp["growth"]
        ch = [d["cout"] for d in layer]
    return sp


def m_stem_reads_image(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Every stem conv reads the raw image instead of the scale above."""
    sp = copy.deepcopy(sp)
    for d in sp["stem"]:
        d["cin"] = 3
    return sp


def m_finer_not_strided(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Cross-scale branch forgets to downsample."""
    sp = copy.deepcopy(sp)
    for layer in sp["layers"]:
        for d in layer[1:]:
            for p in d["parts"]:
                if p["src"] == "finer":
                    p["stride"] = 1
    return sp


def m_drop_a_channel(sp: Dict[str, Any]) -> Dict[str, Any]:
    """One branch emits one channel fewer -- growth silently short."""
    sp = copy.deepcopy(sp)
    sp["layers"][7][1]["parts"][0]["cout"] -= 1
    return sp


def m_break_running(sp: Dict[str, Any]) -> Dict[str, Any]:
    """A layer's input width drifts from the previous layer's output."""
    sp = copy.deepcopy(sp)
    sp["layers"][11][2]["cin"] += 8
    return sp


def m_tie_two_exits(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Two exits end up the same width -- ties break compute_msc."""
    sp = copy.deepcopy(sp)
    fd = list(sp["feature_dims"])
    fd[2] = fd[1]
    sp["feature_dims"] = tuple(fd)
    return sp


def m_ragged_cuts(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Cuts drift off DEPTH_FRACTIONS, so exit k means different depths."""
    sp = copy.deepcopy(sp)
    sp["cuts"] = tuple(c + 1 if i == 1 else c for i, c in enumerate(sp["cuts"]))
    return sp


def m_odd_resolution(sp: Dict[str, Any]) -> Dict[str, Any]:
    """Resolutions stop halving cleanly."""
    sp = copy.deepcopy(sp)
    sp["resolutions"] = [32, 16, 7]
    return sp


# predicate, human name, the mutation it must catch
SUITE: List[Tuple[Callable[[Dict[str, Any]], bool], str,
                  Callable[[Dict[str, Any]], Dict[str, Any]]]] = [
    (p_cuts_on_fractions, "exits land exactly on DEPTH_FRACTIONS", m_ragged_cuts),
    (p_running_channels, "layer inputs match the previous layer's outputs",
     m_break_running),
    (p_closed_form, "the walk agrees with the closed form", m_widths_drift),
    (p_coarsest, "exits read the coarsest scale", m_exits_read_finest),
    (p_ascending, "feature dims strictly ascend", m_tie_two_exits),
    (p_parts_sum, "branches sum to growth, growth is dense", m_drop_a_channel),
    (p_finer_source, "the strided branch reads scale s-1",
     m_finer_reads_own_scale),
    (p_finer_strided, "cross-scale downsamples, same-scale does not",
     m_finer_not_strided),
    (p_scale_widths, "growth doubles per scale", m_growth_flat),
    (p_stem_chain, "the stem is a chain, not four reads of the image",
     m_stem_reads_image),
    (p_resolutions, "resolution halves per scale and stays integral",
     m_odd_resolution),
]


def _safe_parse(src: str):
    """Parse a cell, or return an empty module so one odd cell cannot abort."""
    import ast as _ast
    try:
        return _ast.parse(src)
    except SyntaxError:
        return _ast.Module(body=[], type_ignores=[])


def main() -> int:
    print("MSDNet channel-spec canaries")
    print(f"msc_lib {M.__version__}\n")

    spec = M.msdnet_channel_spec()
    print(f"spec: {spec['n_scales']} scales x {spec['n_layers']} layers, "
          f"base={spec['base']} growth={spec['growth']}")
    print(f"cuts {spec['cuts']}  feature_dims {spec['feature_dims']}\n")

    print("1. every predicate holds on the real spec")
    for pred, name, _ in SUITE:
        check(name, pred(spec))

    print("\n2. every predicate FIRES on the defect it exists to catch")
    print("   (a predicate that passes a corrupted spec is not a canary)")
    for pred, name, mut in SUITE:
        broken = mut(spec)
        fired = not pred(broken)
        check(f"{name} <- {mut.__name__}", fired,
              "" if fired else "PREDICATE IS BLIND to its own mutation")

    print("\n3. the spec refuses configurations it cannot honour")
    for kw, why in ((dict(n_scales=4, in_res=32), None),
                    (dict(n_scales=4, in_res=30), "in_res not divisible by 8"),
                    (dict(n_scales=0), "n_scales < 1"),
                    (dict(n_steps=0), "n_steps < 1"),
                    (dict(step=0), "step < 1"),
                    (dict(base=0), "base < 1"),
                    (dict(growth=0), "growth < 1")):
        try:
            M.msdnet_channel_spec(**kw)
            check(f"accepts {kw}" if why is None else f"refuses {why}",
                  why is None,
                  "" if why is None else "it accepted a config it cannot build")
        except ValueError as e:
            check(f"accepts {kw}" if why is None else f"refuses {why}",
                  why is not None, str(e)[:70])

    print("\n4. the spec scales to other configurations")
    for kw in (dict(n_scales=2, in_res=32), dict(n_steps=3, step=2),
               dict(base=8, growth=4), dict(n_scales=4, in_res=64)):
        sp = M.msdnet_channel_spec(**kw)
        bad = [name for pred, name, _ in SUITE
               if name != "exits land exactly on DEPTH_FRACTIONS"
               and not pred(sp)]
        check(f"all invariants hold for {kw}", not bad, "; ".join(bad[:2]))

    print("\n5. the atlas boundary holds")
    check("msdnet is NOT in the study population",
          "msdnet" not in M.zoo_for_dataset("cifar100"),
          "Studies 1-3 claim 15 architectures; a 16th with no runs is a "
          "silent change to the sample")
    check("msdnet IS reachable by name",
          "msdnet" in M.zoo_for_dataset("cifar100", include_probes=True))
    check("the CIFAR atlas is still exactly 15",
          len(M.zoo_for_dataset("cifar100")) == 15,
          f"{len(M.zoo_for_dataset('cifar100'))}")

    print("\n6. D-90 -- every notebook's measured_runs filters probes")
    # `atlas=False` governs what gets PLANNED. `measured_runs` WALKS runs/, so
    # it needs its own filter or a trained msdnet joins whatever population the
    # notebook is analysing. Study 3's notebooks are not covered by
    # tools/s4_harness.py, so they are checked at the source level here.
    import ast as _ast
    import json as _json
    root = Path(__file__).resolve().parent.parent
    nbs = sorted(root.glob("notebooks_study[34]/*.ipynb"))
    check("generated notebooks were found", bool(nbs), f"{len(nbs)} found")
    defines = blind = 0
    for nb in nbs:
        cells = _json.loads(nb.read_text(encoding="utf-8"))["cells"]
        src = "\n".join("".join(c["source"]) for c in cells
                        if c["cell_type"] == "code")
        if "def measured_runs" not in src:
            continue
        defines += 1
        # Parse rather than grep: a prose mention of `atlas` in a docstring
        # explaining the rule is not the rule. That exact confusion is why
        # msc_lib's HF check was rewritten to walk the AST.
        fn = next((n for c in cells if c["cell_type"] == "code"
                   for n in _ast.walk(_safe_parse("".join(c["source"])))
                   if isinstance(n, _ast.FunctionDef)
                   and n.name == "measured_runs"), None)
        args = [a.arg for a in fn.args.args] if fn else []
        body = _ast.dump(fn) if fn else ""
        ok = ("include_probes" in args and "'atlas'" in body)
        if not ok:
            blind += 1
        check(f"{nb.name}: measured_runs excludes probes", ok,
              "" if ok else f"args={args}")
    check("at least one notebook defines measured_runs", defines > 0,
          f"{defines} do")
    check("no notebook defines a probe-blind measured_runs", blind == 0,
          f"{blind} blind")

    print(f"\n{N_RUN} checks run, {len(FAILURES)} failed")
    if FAILURES:
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
