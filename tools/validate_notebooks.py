#!/usr/bin/env python3
"""
validate_notebooks.py -- refuse to ship a notebook that names a column or a
repo path that does not exist.

Rules 3 and 4, as mechanisms rather than intentions:

    3. Column names are data. Validate every string literal against the schema
       at BUILD time, not at run time. Five wrong names once killed 9 runs at
       the end of epoch 0.
    4. Never spell a repo path as a string literal in a notebook. Go through
       one accessor, so a wrong path is an import error not a silent miss.

Both have already cost this project real time, twice each:

  D-22  `train_msc_kd` wrote five column names that are not columns
        (`f1_score`, `precision`, `recall`, `grad_norm`, `throughput_img_s`).
        `csv.DictWriter` raised -- at the END of the first epoch, after the
        training was done and the time unrecoverable. Nine runs.
  D-36  NB15 asked for three telemetry columns that do not exist. GPU fields
        are per device (`gpu0_*`), so there is no un-suffixed name. The build
        loop guarded every access and skipped them silently; the display two
        lines later did not, and that is where it died.
  D-16  `exit_heads.pt` written outside the documented layout. Closed as
        "cosmetic -- nothing reads the path by convention."
  D-23  Three things read it by convention. Every MSC-KD run retrained the
        teacher's exit heads for a file that had been on HuggingFace since NB08.

The pattern in all four is the same: a literal that nothing compared against
the thing it refers to. Comparing them is mechanical, takes milliseconds, and
is therefore the only sane place to do it.

The schema is IMPORTED from msc_lib rather than re-parsed out of it. Re-parsing
would create a second copy of the truth, which is the D-16 defect in a new
costume.

Usage
-----
    python tools/validate_notebooks.py                    # check notebooks/
    python tools/validate_notebooks.py --dir other/       # check elsewhere
    python tools/validate_notebooks.py --strict-paths     # rule 4 is fatal too
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import msc_lib as M                                            # noqa: E402

# ---------------------------------------------------------------------------
# What a legitimate name may be.
# ---------------------------------------------------------------------------
SCHEMA = set(M.HISTORY_FIELDS) | set(M.FINAL_FIELDS)

# Keys that live in summary.json / meta blocks rather than the CSV schema.
# Enumerated rather than pattern-matched: an allowlist you have to edit is a
# place where someone has to think, which is the point.
SUMMARY_KEYS = {
    "run_id", "arch", "family", "dataset", "dataset_name", "seed", "phase",
    "method", "status", "config_hash", "data_fingerprint", "input_res",
    "num_epochs", "num_epochs_run", "num_epochs_planned", "best_accuracy",
    "reference_accuracy", "recipe_ok", "num_parameters", "full_flops",
    "total_time_sec", "total_energy_kwh", "total_energy_j", "total_co2_kg",
    "sample_order_hash", "msc_lib_version", "created_utc", "acc_pct",
    "exit_count", "resolutions", "precisions", "tau_grid", "budgets",
    "params_M", "gflops", "state", "ts", "account", "worker", "session",
    "repaired", "teacher_rho", "rho", "axes", "depth", "resolution",
    "precision", "fractions", "stage_cuts", "feature_dims", "flops", "K",
}

# Per-sample table columns are generated (pred_d1, top1p_rn3, ...), so they are
# validated by shape rather than enumeration.
PER_SAMPLE_RE = re.compile(
    r"^(pred|top1p|top2p)_(d|rn|rp|q)\d+$|"
    r"^(sample_idx|label|split|msc|msc_[a-z_]+|tau|irreducible|"
    r"msp|margin|entropy|ce_loss|el2n|forget_events|pred_depth)$")

# Analysis outputs, which are computed columns rather than schema columns.
ANALYSIS_RE = re.compile(
    r"^(rho_seed|rho_raw|T|T_lo|T_hi|j10|pc1|pc2|pc3|delta_r2|partial_rho|"
    r"n|n_boot|pair|pair_type|arch_a|arch_b|run_a|run_b|kind|z|sd|"
    r"ceiling|ceiling_a|ceiling_b|axis|battery|n_battery_scores)$")

REPO_PREFIXES = ("runs/", "registry/", "analysis/", "tables/", "paper/",
                 "budgets/", "per_sample/", "checkpoints/", "metrics/",
                 "telemetry/", "env/", "console/", "logs/")

# Accessors that are the sanctioned way to name a repo path. A literal inside
# one of these calls is how the accessor itself is defined and is fine.
PATH_ACCESSORS = {"run_layout", "exit_heads_path", "find_exit_heads",
                  "repo_path", "enqueue", "push_root", "push_data_path"}


def _cells(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        # The base64 library bootstrap is a data blob, not code to analyse.
        if "_LIB = (" in src or src.count("'") > 400:
            continue
        yield i, src


def _sub_keys(node):
    """The string keys of one subscript: `df["x"]` or `df[["x","y"]]`."""
    sl = node.slice
    items = sl.elts if isinstance(sl, (ast.List, ast.Tuple)) else [sl]
    return [it.value for it in items
            if isinstance(it, ast.Constant) and isinstance(it.value, str)]


def _defined_and_read(tree: ast.AST):
    """Split subscript string keys into names the notebook DEFINES and names it
    merely READS.

    The first version of this checker flagged every subscript key absent from
    the schema and produced 73 hits across the CIFAR notebooks, essentially all
    of them false: `n_runs`, `wall_clock_hours`, `fam_a`, `delta_r2_lo` are
    columns the notebooks compute for themselves. That is worse than no checker.
    A check that fires on healthy data teaches you to ignore it, and the next
    alarm is the real one -- which is precisely what D-17 and D-20 cost.

    The real defect has a sharper shape. In D-36, `tel` was built by a guarded
    loop that silently skipped three names, and a display two lines later read
    one of them BY NAME. In D-22 the row was written with five keys the schema
    does not have. Both are: **a name that is read, is not in the schema, and is
    never assigned anywhere in this notebook.** That is what gets flagged, and
    a locally-computed column is defined by construction so it cannot trip it.
    """
    defined, read = set(), []

    def _mark_defined(target):
        if isinstance(target, ast.Subscript):
            defined.update(_sub_keys(target))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                _mark_defined(e)

    for nd in ast.walk(tree):
        # df["new"] = ..., and augmented/annotated forms
        if isinstance(nd, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = nd.targets if isinstance(nd, ast.Assign) else [nd.target]
            for t in tgts:
                _mark_defined(t)
        # {"a": 1, "b": 2} -- a frame or row the notebook constructs
        elif isinstance(nd, ast.Dict):
            defined.update(k.value for k in nd.keys
                           if isinstance(k, ast.Constant)
                           and isinstance(k.value, str))
        # .rename(columns={...}) values, .assign(x=...), columns=[...]
        elif isinstance(nd, ast.Call):
            for kw in nd.keywords:
                if kw.arg == "columns" and isinstance(kw.value, ast.List):
                    defined.update(e.value for e in kw.value.elts
                                   if isinstance(e, ast.Constant)
                                   and isinstance(e.value, str))
                elif kw.arg and getattr(nd.func, "attr", None) == "assign":
                    defined.add(kw.arg)

    for nd in ast.walk(tree):
        if isinstance(nd, ast.Subscript) and isinstance(nd.ctx, ast.Load):
            for k in _sub_keys(nd):
                read.append((k, getattr(nd, "lineno", 0)))
        elif isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute) \
                and nd.func.attr in ("get", "setdefault"):
            if nd.args and isinstance(nd.args[0], ast.Constant) \
                    and isinstance(nd.args[0].value, str):
                read.append((nd.args[0].value, getattr(nd, "lineno", 0)))
    return defined, read


def _path_literals(tree: ast.AST):
    """Repo-path-shaped literals not inside a sanctioned accessor call."""
    inside = set()
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Call):
            fname = getattr(nd.func, "attr", None) or getattr(nd.func, "id", None)
            if fname in PATH_ACCESSORS:
                for sub in ast.walk(nd):
                    inside.add(id(sub))
    out = []
    for nd in ast.walk(tree):
        vals = []
        if isinstance(nd, ast.Constant) and isinstance(nd.value, str):
            vals = [nd.value]
        elif isinstance(nd, ast.JoinedStr):                    # f-string
            vals = ["".join(p.value for p in nd.values
                            if isinstance(p, ast.Constant)
                            and isinstance(p.value, str))]
        for v in vals:
            if id(nd) in inside or not v:
                continue
            # A path, not prose that happens to start with one. `f"tables/
            # all_final.csv : {n} rows"` is a print, and flagging it is the
            # noise that gets a checker ignored.
            if (any(v.startswith(p) for p in REPO_PREFIXES)
                    and " " not in v.strip()
                    and (re.search(r"\.(json|csv|parquet|pt|yaml|txt|png|jsonl)$", v)
                         or v.endswith("/**") or v.endswith("/*"))):
                out.append((v.strip(), getattr(nd, "lineno", 0)))
    return out


def _library_defined() -> set:
    """Every string key `msc_lib` itself creates.

    `analyse_q3_transfer` returns rows with `spearman_raw` and `jaccard_top10`;
    `shard_report` produces `wall_clock_hours` and `per_worker_hours`. Those are
    real names -- they are simply defined in the library rather than in the CSV
    schema or in the notebook. Without this the checker flagged 61 of them and
    would have been switched off within a day.

    The legitimate universe is therefore: the CSV schema, plus everything the
    library creates, plus everything the notebook creates. A name outside all
    three exists NOWHERE, which is exactly what the five D-22 names and the
    three D-36 names were.
    """
    # EVERY source file, not just msc_lib. `partial_spearman`,
    # `r2_difficulty_only` and `r2_difficulty_plus_msc` are produced by
    # `msc_core.py` -- the reference implementation of the statistics -- and
    # harvesting only the library reported all three as unknown. A checker whose
    # notion of "everything this project defines" omits a source file will keep
    # producing false positives until someone switches it off.
    srcs = []
    for f in ("src/msc_lib.py", "msc_core.py", "msc_torch.py",
              "tools/pack_imagenet100.py"):
        fp = ROOT / f
        if fp.exists():
            srcs.append(fp.read_text(encoding="utf-8"))
    try:
        tree = ast.parse("\n".join(srcs))
    except SyntaxError:
        return set()
    # `_selftest` is EXCLUDED. It deliberately contains the wrong names -- the
    # D-22 regression check asserts that `f1_score` and `grad_norm` are absent
    # from the schema. Harvesting from it would teach the checker that the
    # exact strings it exists to catch are legitimate.
    body = [n for n in tree.body
            if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == "_selftest")]
    d, _ = _defined_and_read(ast.Module(body=body, type_ignores=[]))
    # `dict(spearman_raw=..., n=...)` is a definition too -- but ONLY for the
    # dict/DataFrame constructors. Sweeping every keyword argument in the file
    # was the first attempt and it made `f1_score` and `grad_norm` known,
    # because they appear as parameter names elsewhere. The checker's own
    # self-test caught that, which is the whole reason it has one.
    for nd in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(nd, ast.Call):
            fn = getattr(nd.func, "id", None) or getattr(nd.func, "attr", None)
            if fn in ("dict", "DataFrame", "Series", "assign"):
                d.update(kw.arg for kw in nd.keywords if kw.arg)
    return d


LIB_DEFINED = _library_defined()


def _known(name: str) -> bool:
    return (name in SCHEMA or name in SUMMARY_KEYS or name in LIB_DEFINED
            or bool(PER_SAMPLE_RE.match(name))
            or bool(ANALYSIS_RE.match(name)))


# ---------------------------------------------------------------------------
# Does every library call a notebook makes actually exist?
# ---------------------------------------------------------------------------
# Rule 3 is "column names are data -- validate them against the schema at BUILD
# time". Function names are data in exactly the same way, and the failure is
# worse: a wrong column name yields a KeyError with a suggestion, while a wrong
# function name yields an AttributeError several cells into a run that may have
# already spent GPU-hours.
#
# This project has now hit it twice. `MultiExit` instead of `MultiExitModel`
# cost an offline-verification run. Writing these five notebooks produced SIX
# invented names in one sitting -- `analyse_q1_all`, `analyse_q2_all`,
# `analyse_q3_all`, `analyse_q3_shuffled_control_all`, `analyse_q4_all`,
# `compare_routing_methods` -- every one of which would have surfaced only when
# the user ran the cell.
#
# The schema was sitting right there both times.
def _source_level_names(path: Path, cls: str | None = None) -> set:
    """Names defined at module scope (or in one class) IN THE SOURCE.

    `dir(M)` is the wrong universe. Half of msc_lib -- ExitHead,
    MultiExitModel, MSCLoss, MSCStudent -- lives under `if _TORCH_OK:`, so on a
    machine without torch those names are absent from the module object while
    being perfectly real on the machine that runs the experiment. Using dir()
    alone reported `MultiExitModel` as nonexistent, which is the checker
    producing exactly the false negative it exists to prevent.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out: set = set()

    def walk(body, want_cls=None, depth=0):
        for nd in body:
            if isinstance(nd, ast.ClassDef):
                if want_cls and nd.name == want_cls:
                    for sub in nd.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            out.add(sub.name)
                    # `self.data_dir = ...` is as much a member as a method,
                    # and it is what a notebook actually reaches for. dir() on
                    # the CLASS never sees instance attributes.
                    for sub in ast.walk(nd):
                        if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                            tg = (sub.targets if isinstance(sub, ast.Assign)
                                  else [sub.target])
                            for t in tg:
                                if isinstance(t, ast.Attribute) \
                                        and getattr(t.value, "id", None) == "self":
                                    out.add(t.attr)
                elif not want_cls:
                    out.add(nd.name)
                    walk(nd.body, want_cls, depth + 1)
            elif isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not want_cls:
                    out.add(nd.name)
            elif isinstance(nd, ast.Assign) and not want_cls:
                out.update(t.id for t in nd.targets if isinstance(t, ast.Name))
            elif isinstance(nd, ast.AnnAssign) and not want_cls \
                    and isinstance(nd.target, ast.Name):
                out.add(nd.target.id)
            elif isinstance(nd, (ast.If, ast.Try)):
                walk(nd.body, want_cls, depth)
                walk(getattr(nd, "orelse", []) or [], want_cls, depth)
                for h in getattr(nd, "handlers", []) or []:
                    walk(h.body, want_cls, depth)
    walk(tree.body, cls)
    return out


_LIBSRC = ROOT / "src" / "msc_lib.py"
LIB_PUBLIC = ({n for n in dir(M) if not n.startswith("__")}
              | _source_level_names(_LIBSRC))
SESSION_PUBLIC = ({n for n in dir(M.Session) if not n.startswith("_")}
                  | _source_level_names(_LIBSRC, cls="Session"))


def _signatures(path: Path, cls: str | None = None) -> dict:
    """Parameter names and positional counts for every function in a source
    file, or for the methods of one class.

    D-47, one layer out. The notebook validator already checked that `M.x`
    EXISTS. It did not check that the call matches `x`'s signature, so
    `M.resume_acceptance_test(..., interrupt_after=2)` shipped -- the parameter
    is `kill_at`. Every name in that line is real; the call is still wrong.

    Names being real is not the same as calls being right, and this is the
    third distinct place that distinction has cost something.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}
    out: dict = {}

    def sig(nd, drop_self=False):
        aa = nd.args
        pos = list(aa.posonlyargs) + list(aa.args)
        if drop_self and pos and pos[0].arg in ("self", "cls"):
            pos = pos[1:]
        return {"min": len(pos) - len(aa.defaults), "max": len(pos),
                "star": aa.vararg is not None, "kwargs": aa.kwarg is not None,
                "names": {x.arg for x in pos + list(aa.kwonlyargs)},
                "order": [x.arg for x in pos]}

    def walk(body, want=None):
        for nd in body:
            if isinstance(nd, ast.ClassDef):
                if want and nd.name == want:
                    for sub in nd.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            out[sub.name] = sig(sub, drop_self=True)
                elif not want:
                    walk(nd.body, want)
            elif isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not want:
                    out.setdefault(nd.name, sig(nd))
            elif isinstance(nd, (ast.If, ast.Try)):
                walk(nd.body, want)
                walk(getattr(nd, "orelse", []) or [], want)
                for h in getattr(nd, "handlers", []) or []:
                    walk(h.body, want)
    walk(tree.body, cls)
    return out


LIB_SIGS = _signatures(ROOT / "src" / "msc_lib.py")
SESSION_SIGS = _signatures(ROOT / "src" / "msc_lib.py", cls="Session")


def _call_problems(tree: ast.AST):
    """Calls to `M.x(...)` / `sess.x(...)` whose arguments do not fit."""
    bad = []
    for nd in ast.walk(tree):
        if not isinstance(nd, ast.Call) or not isinstance(nd.func, ast.Attribute):
            continue
        base = getattr(nd.func.value, "id", None)
        if base in ("M", "msc_lib"):
            sigs, label = LIB_SIGS, "msc_lib"
        elif base in ("sess", "session"):
            sigs, label = SESSION_SIGS, "Session"
        else:
            continue
        sg = sigs.get(nd.func.attr)
        if not sg:
            continue
        if any(isinstance(x, ast.Starred) for x in nd.args):
            continue
        npos = len(nd.args)
        kws = {k.arg for k in nd.keywords if k.arg}
        if npos > sg["max"] and not sg["star"]:
            bad.append((f"{label}.{nd.func.attr}(): {npos} positional, "
                        f"max {sg['max']}", getattr(nd, "lineno", 0)))
        elif npos + len(kws & sg["names"]) < sg["min"]:
            bad.append((f"{label}.{nd.func.attr}(): too few arguments, "
                        f"needs {sg['min']}", getattr(nd, "lineno", 0)))
        for k in sorted(kws - sg["names"]):
            if not sg["kwargs"]:
                near = difflib.get_close_matches(k, sorted(sg["names"]), n=2,
                                                cutoff=0.5)
                bad.append((f"{label}.{nd.func.attr}() has no parameter "
                            f"'{k}'" + (f" -- did you mean {near}?" if near
                                        else ""), getattr(nd, "lineno", 0)))
    return bad


# Parameters that take a CALLABLE. Passing a function reference to anything
# else -- or a non-function to one of these -- is D-53: the argument count is
# right, the keyword names are right, and the order is wrong.
CALLABLE_PARAMS = {"fn", "func", "callable", "done_fn", "kind_fn", "on_flush",
                   "feature_dim_fn", "criterion"}


def _arg_order_problems(tree: ast.AST):
    """A function reference passed where a value belongs, or vice versa.

    D-53. `sess.run_all(M.train_backbone, cfgs)` -- the signature is
    `run_all(cfgs, fn)`. Two positional arguments, both present, so the arity
    check (D-47/D-48) passes: it counts arguments and names keywords, and both
    were correct. Only the ORDER was wrong.

    `M.something` used as a value is unambiguous evidence of a function
    reference, and a parameter named `fn`/`done_fn`/`criterion` unambiguously
    wants one. Comparing those two facts costs nothing and catches the whole
    class.
    """
    bad = []
    for nd in ast.walk(tree):
        if not isinstance(nd, ast.Call) or not isinstance(nd.func, ast.Attribute):
            continue
        base = getattr(nd.func.value, "id", None)
        if base in ("M", "msc_lib"):
            sigs = LIB_SIGS
        elif base in ("sess", "session"):
            sigs = SESSION_SIGS
        else:
            continue
        sg = sigs.get(nd.func.attr)
        if not sg or not sg.get("order"):
            continue
        for i, arg in enumerate(nd.args):
            if i >= len(sg["order"]):
                break
            param = sg["order"][i]
            # `M.train_backbone` as a value == a function reference.
            is_ref = (isinstance(arg, ast.Attribute)
                      and getattr(arg.value, "id", None) in ("M", "msc_lib")
                      and arg.attr in LIB_SIGS)
            wants = param in CALLABLE_PARAMS
            if is_ref and not wants:
                bad.append((f"{base}.{nd.func.attr}(): argument {i + 1} is the "
                            f"function `M.{arg.attr}` but parameter {i + 1} is "
                            f"`{param}`. Arguments look swapped -- the "
                            f"signature is ({', '.join(sg['order'][:3])}, ...)",
                            getattr(nd, "lineno", 0)))
            elif wants and isinstance(arg, (ast.List, ast.Dict, ast.Constant)):
                bad.append((f"{base}.{nd.func.attr}(): parameter {i + 1} is "
                            f"`{param}` and wants a callable, got a literal",
                            getattr(nd, "lineno", 0)))
    return bad


def _result_key_problems(tree: ast.AST):
    """Keys read off a library result that the library does not declare.

    D-51/D-52. `res.get('passed')` where the key is `ok` reported a PASSING
    resume test as a failure. `ctrl['passes']` where the column is `passed`
    would have raised KeyError during analysis, after every GPU-hour was spent.

    Existence checks (D-39), signature checks (D-47/D-48) and schema checks
    (D-22/D-36) all pass on both. None of them can see a key read off a
    returned dict or frame -- so this does.

    Single-assignment dataflow, per cell: `x = M.f(...)` then `x['k']` or
    `x.get('k')`. That covers how every notebook here is written and stays
    simple enough to trust.
    """
    src_of: dict = {}
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Assign) and isinstance(nd.value, ast.Call) \
                and isinstance(nd.value.func, ast.Attribute):
            base = getattr(nd.value.func.value, "id", None)
            if base in ("M", "msc_lib", "sess", "session"):
                for tg in nd.targets:
                    if isinstance(tg, ast.Name):
                        src_of[tg.id] = nd.value.func.attr

    bad = []
    for nd in ast.walk(tree):
        var = key = None
        if isinstance(nd, ast.Subscript) and isinstance(nd.value, ast.Name):
            sl = nd.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                var, key = nd.value.id, sl.value
        elif isinstance(nd, ast.Call) and isinstance(nd.func, ast.Attribute) \
                and nd.func.attr == "get" and isinstance(nd.func.value, ast.Name):
            if nd.args and isinstance(nd.args[0], ast.Constant) \
                    and isinstance(nd.args[0].value, str):
                var, key = nd.func.value.id, nd.args[0].value
        if not var or var not in src_of:
            continue
        fn = src_of[var]
        if M.result_key_ok(fn, key):
            continue
        near = difflib.get_close_matches(key, list(M.RESULT_KEYS.get(fn, ())),
                                         n=2, cutoff=0.5)
        bad.append((f"{var} = {fn}(...) then {var}[{key!r}], but {fn} declares "
                    f"no such key" + (f" -- did you mean {near}?" if near else ""),
                    getattr(nd, "lineno", 0)))
    return bad


def _library_calls(tree: ast.AST):
    """(`attribute`, `line`, `kind`) for every `M.x` and `sess.x` reference."""
    out = []
    for nd in ast.walk(tree):
        if not isinstance(nd, ast.Attribute):
            continue
        base = getattr(nd.value, "id", None)
        if base in ("M", "msc_lib"):
            out.append((nd.attr, getattr(nd, "lineno", 0), "msc_lib"))
        elif base in ("sess", "session"):
            out.append((nd.attr, getattr(nd, "lineno", 0), "Session"))
    return out


def self_test() -> bool:
    """Prove the checker can FAIL before believing it when it passes.

    Rule 12 and rule 8. A validator reporting "0 problems" is worth exactly the
    evidence that it would have said otherwise, and D-37 was this project
    discovering its entire test harness could not fail.

    ---------------------------------------------------------------------------
    THE TWO DEFECTS HAVE DIFFERENT OWNERS, and finding that out is what this
    self-test was for.

    Writing this checker, the obvious move was to make it catch both D-22 and
    D-36. It cannot, and the reason is worth recording: `grad_norm` and
    `f1_score` are LEGITIMATE internal dict keys inside `msc_lib` -- see
    `EpochTelemetry`, which returns `{"grad_norm": ...}`. They are simply not
    CSV column names; the columns are `grad_norm_mean` and `f1_macro`. D-22 was
    using the internal key where the column name was needed. No static check
    over string literals can tell those apart, because they are the same string.

    So:

      D-22 (WRITING a row with keys the schema lacks)
          owned by `append_history_row(strict=True)`, which raises and names the
          column you probably meant. Asserted below against the real five.

      D-36 (READING a column that exists nowhere)
          owned by this checker. Asserted below against the real three.

    Splitting them is the honest answer. Claiming this checker covered both
    would be the kind of "it's handled" that D-16 was closed with.
    ---------------------------------------------------------------------------
    """
    ok = True

    # -- D-36: reading a name that exists nowhere. This checker's job. --------
    d36 = ["gpu_util_mean_pct", "gpu_temp_max_c"]
    for n in d36:
        if _known(n):
            print(f"  [SELFTEST FAIL] '{n}' is a real D-36 name and the "
                  f"checker considers it known")
            ok = False
    for n in ("throughput_train_img_s", "gpu0_util_mean_pct", "f1_macro",
              "grad_norm_mean", "val_accuracy", "spearman_raw"):
        if not _known(n):
            print(f"  [SELFTEST FAIL] '{n}' is legitimate and the checker "
                  f"rejects it -- a checker that cries wolf gets switched off")
            ok = False

    # -- D-22: writing a row with wrong keys. append_history_row's job. -------
    import tempfile
    d22 = ["f1_score", "precision", "recall", "grad_norm", "throughput_img_s"]
    for n in d22:
        with tempfile.TemporaryDirectory() as td:
            try:
                M.append_history_row(Path(td) / "e.csv",
                                     {"run_id": "x", "epoch": 0, n: 1.0},
                                     strict=True)
                print(f"  [SELFTEST FAIL] append_history_row accepted '{n}', "
                      f"one of the five names that killed nine runs (D-22)")
                ok = False
            except Exception:
                pass

    # -- the library-name check must be able to fail ------------------------
    for bad in ("analyse_q1_all_typo", "MultiExit", "confirm_on_hf_typo"):
        if bad in LIB_PUBLIC:
            print(f"  [SELFTEST FAIL] '{bad}' should not resolve")
            ok = False
    for good in ("analyse_q1_all", "MultiExitModel", "backbone_dry_run",
                 "oracle_dry_run", "compare_routing_methods"):
        if good not in LIB_PUBLIC:
            print(f"  [SELFTEST FAIL] msc_lib.{good} is referenced by the "
                  f"notebooks and does not exist")
            ok = False

    # -- the argument-order check must be able to fail (D-53) ---------------
    if not _arg_order_problems(ast.parse("sess.run_all(M.train_backbone, cfgs)")):
        print("  [SELFTEST FAIL] the arg-order check does not catch the exact "
              "D-53 line, sess.run_all(M.train_backbone, cfgs)")
        ok = False
    for good in ("sess.run_all(cfgs, M.train_backbone)",
                 "sess.run_all(cfgs, M.run_oracle)",
                 "M.build_model('resnet50', 100, dataset='imagenet100')"):
        if _arg_order_problems(ast.parse(good)):
            print(f"  [SELFTEST FAIL] the arg-order check rejects a VALID "
                  f"call: {good} -> {_arg_order_problems(ast.parse(good))}")
            ok = False

    # -- the result-key check must be able to fail (D-51, D-52) -------------
    for bad_src, why in (
            ("res = M.resume_acceptance_test(sess)\nprint(res.get('passed'))",
             "the exact D-51 line"),
            ("ctrl = M.analyse_q3_shuffled_control_all(sess)\n"
             "bad = ctrl[~ctrl['passes']]", "the exact D-52 line")):
        if not _result_key_problems(ast.parse(bad_src)):
            print(f"  [SELFTEST FAIL] the result-key check does not catch "
                  f"{why}")
            ok = False
    for good_src in ("res = M.resume_acceptance_test(sess)\nprint(res['ok'])",
                     "ctrl = M.analyse_q3_shuffled_control_all(sess)\n"
                     "bad = ctrl[~ctrl['passed']]",
                     "q = M.analyse_q1_all(sess)\nprint(q['rho_seed_tau0.1'])"):
        if _result_key_problems(ast.parse(good_src)):
            print(f"  [SELFTEST FAIL] the result-key check rejects a VALID "
                  f"read: {_result_key_problems(ast.parse(good_src))}")
            ok = False

    # -- the arity check must be able to fail (D-47) -------------------------
    _probe = ast.parse("M.resume_acceptance_test(sess, arch='x', "
                       "epochs=4, interrupt_after=2)")
    if not _call_problems(_probe):
        print("  [SELFTEST FAIL] the arity check does not catch "
              "`interrupt_after`, which is not a parameter of "
              "resume_acceptance_test")
        ok = False
    _good = ast.parse("M.resume_acceptance_test(sess, arch='x', epochs=4, "
                      "kill_at=2)")
    if _call_problems(_good):
        print(f"  [SELFTEST FAIL] the arity check rejects a VALID call: "
              f"{_call_problems(_good)}")
        ok = False

    print(f"  [SELFTEST {'PASS' if ok else 'FAIL'}] "
          f"this checker catches the {len(d36)} D-36 read-names; "
          f"append_history_row(strict) catches all {len(d22)} D-22 write-names; "
          f"6 legitimate names accepted")
    return ok


def _looks_like_a_column(name: str) -> bool:
    """Only judge things shaped like schema names.

    A subscript may legitimately hold a filename, a URL fragment, a dict key of
    our own making. The check applies to lower_snake_case identifiers, which is
    what every column in this project is -- and crucially what all five D-22
    names and all three D-36 names were.
    """
    return bool(re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)+", name))


def validate(nb_dir: Path, strict_paths: bool = False) -> int:
    if not self_test():
        print("the checker itself is broken; its verdict below means nothing")
        return 1
    problems, warnings_ = [], []
    nbs = sorted(nb_dir.glob("*.ipynb"))
    if not nbs:
        print(f"no notebooks under {nb_dir}")
        return 1

    for nb in nbs:
        # Whole-notebook scope on purpose: a column assigned in cell 5 and read
        # in cell 9 is defined, and per-cell scope would flag it.
        trees, defined = [], set()
        for ci, src in _cells(nb):
            try:
                t = ast.parse(src)
            except SyntaxError:
                continue                       # magics etc.; not our business
            d, _ = _defined_and_read(t)
            defined |= d
            trees.append((ci, t))

        for ci, tree in trees:
            _, reads = _defined_and_read(tree)
            for name, ln in reads:
                if (not _looks_like_a_column(name) or _known(name)
                        or name in defined):
                    continue
                near = difflib.get_close_matches(name, sorted(SCHEMA), n=3,
                                                 cutoff=0.7)
                problems.append(
                    f"{nb.name} cell {ci} line {ln}: reads column '{name}', "
                    f"which is not in the schema and is never assigned in this "
                    f"notebook"
                    + (f" -- did you mean {near}?" if near else ""))

            for attr, ln, kind in _library_calls(tree):
                pool = LIB_PUBLIC if kind == "msc_lib" else SESSION_PUBLIC
                if attr in pool:
                    continue
                near = difflib.get_close_matches(attr, sorted(pool), n=3, cutoff=0.6)
                problems.append(
                    f"{nb.name} cell {ci} line {ln}: {kind}.{attr} does not "
                    f"exist" + (f" -- did you mean {near}?" if near else ""))

            for msg, ln in _arg_order_problems(tree):
                problems.append(f"{nb.name} cell {ci} line {ln}: {msg}")

            for msg, ln in _result_key_problems(tree):
                problems.append(f"{nb.name} cell {ci} line {ln}: {msg}")

            for msg, ln in _call_problems(tree):
                problems.append(f"{nb.name} cell {ci} line {ln}: {msg}")

            # D-44: a literal drive letter is wrong on any machine without
            # that drive, and the failure is a WinError deep inside pathlib
            # that names neither the setting nor the file to change.
            for nd in ast.walk(tree):
                if isinstance(nd, ast.Constant) and isinstance(nd.value, str):
                    if re.match(r"^[A-Za-z]:[\\/]", nd.value):
                        problems.append(
                            f"{nb.name} cell {ci} line "
                            f"{getattr(nd, 'lineno', 0)}: hardcoded drive "
                            f"'{nd.value[:40]}'. Use resolve_storage(None, None) "
                            f"or an explicit setting the operator edits (D-44)")

            for p, ln in _path_literals(tree):
                msg = (f"{nb.name} cell {ci} line {ln}: repo path '{p}' is "
                       f"spelled as a literal; use run_layout() or a named "
                       f"accessor (rule 4, defects D-16/D-23/D-25)")
                (problems if strict_paths else warnings_).append(msg)

    print(f"checked {len(nbs)} notebook(s) against {len(SCHEMA)} schema "
          f"columns ({len(M.HISTORY_FIELDS)} history + {len(M.FINAL_FIELDS)} "
          f"final), {len(LIB_PUBLIC)} library names and {len(SESSION_PUBLIC)} "
          f"Session methods")
    for w in warnings_:
        print(f"  [WARN] {w}")
    for p in problems:
        print(f"  [FAIL] {p}")
    if problems:
        print(f"\n{len(problems)} problem(s). Generation refused.\n"
              f"A wrong column name is invisible until it is fatal: D-22 died "
              f"at the end of epoch 0 on nine runs, D-36 at a display two lines "
              f"after a guarded write skipped it silently.")
        return 1
    print(f"\nOK -- no unknown column names"
          + (f", {len(warnings_)} path warning(s)" if warnings_ else
             ", no literal repo paths"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "notebooks"))
    ap.add_argument("--strict-paths", action="store_true")
    a = ap.parse_args()
    return validate(Path(a.dir), a.strict_paths)


if __name__ == "__main__":
    sys.exit(main())
