#!/usr/bin/env python3
"""
check_names.py -- a name used in a function that is defined nowhere.

D-77. A guard added to `train_msc_kd` referenced `step`, a variable that exists
in `train_backbone`'s loop and not in this one. `ast.parse` accepts it, the
notebook-syntax gate (D-73) accepts it, the self-test imports the module fine --
and it would have raised `NameError` on the first batch of an 18-run job.

Python resolves names at CALL time, so a typo in a branch nobody exercises in
CI survives every check this project has until it reaches the GPU. That is the
same shape as the defects that make up most of this log: valid syntax, wrong
meaning, discovered late and expensively.

Deliberately narrow. It reports a name only when it is:
  * loaded inside a function, and
  * never assigned anywhere in that function or its enclosing functions, and
  * not a module-level name (INCLUDING inside `if`/`try`/`with` at module
    level -- this file's first draft missed those and produced nine false
    positives), and
  * not a builtin.

A checker that fires on healthy code teaches you to ignore it, which this
project has paid for three times (D-17, D-20, D-61).

    python tools/check_names.py src/msc_lib.py
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path


# Always present in a module namespace. Omitting these made the first run
# report `__file__` -- one false positive, which is one too many for a checker
# whose value depends on being believed.
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "__loader__", "__builtins__", "__debug__", "__path__",
                  "__class__"}


def _module_names(tree: ast.Module) -> set:
    """Every name bound at module level, however deeply nested in if/try/with."""
    out = set()

    def visit(stmts):
        for n in stmts:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for al in n.names:
                    out.add((al.asname or al.name).split(".")[0])
            elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                tgts = (n.targets if isinstance(n, ast.Assign) else [n.target])
                for t in tgts:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            out.add(x.id)
            elif isinstance(n, ast.If):
                visit(n.body); visit(n.orelse)
            elif isinstance(n, ast.Try):
                visit(n.body); visit(n.orelse); visit(n.finalbody)
                for h in n.handlers:
                    if h.name:
                        out.add(h.name)
                    visit(h.body)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars:
                        for x in ast.walk(item.optional_vars):
                            if isinstance(x, ast.Name):
                                out.add(x.id)
                visit(n.body)
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.While)):
                if isinstance(n, (ast.For, ast.AsyncFor)):
                    for x in ast.walk(n.target):
                        if isinstance(x, ast.Name):
                            out.add(x.id)
                visit(n.body); visit(n.orelse)

    visit(tree.body)
    return out


def _bound_in(fn) -> set:
    """Names bound anywhere inside `fn`, including nested defs and comprehensions."""
    out = set()
    for a in list(fn.args.args) + list(fn.args.kwonlyargs) + list(fn.args.posonlyargs):
        out.add(a.arg)
    if fn.args.vararg:
        out.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        out.add(fn.args.kwarg.arg)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Global):
            out.update(n.names)
        elif isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out


def check(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mod = _module_names(tree) | MODULE_DUNDERS
    problems = []

    def walk_fns(node, enclosing):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound = _bound_in(n) | enclosing
                for x in ast.walk(n):
                    if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                        if (x.id not in bound and x.id not in mod
                                and not hasattr(builtins, x.id)):
                            problems.append((n.name, x.id, x.lineno))
                walk_fns(n, bound)
            else:
                walk_fns(n, enclosing)

    walk_fns(tree, set())
    seen = set()
    for fname, name, line in problems:
        key = (fname, name)
        if key in seen:
            continue
        seen.add(key)
        print(f"  [FAIL] {path.name}:{line} in {fname}(): "
              f"name '{name}' is never defined here")
    return len(seen)


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or [
        Path(__file__).resolve().parent.parent / "src" / "msc_lib.py"]
    bad = 0
    for f in files:
        bad += check(f)
    if bad:
        print(f"\n  {bad} undefined name(s). These are NameErrors waiting for "
              f"the branch to be taken.")
        return 1
    print(f"  no undefined names in {', '.join(f.name for f in files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
