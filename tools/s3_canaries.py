#!/usr/bin/env python3
"""Canaries for Study 3's new library code.

My environment has no torch, so every training line ships unexecuted -- the
structural gap behind D-70/D-76/D-77. The response is the same as in Study 2:
push the decidable part into pure functions and test THOSE, then assert the
rest inside the notebook where a GPU exists.

What is checked here:
  * exit_loss_weights -- the only new pure function, fully exercised
  * the joint/frozen branch is guarded so Study 1 runs are bit-identical
  * ckpt_best stays in the format run_oracle demands (source-level check)
  * evaluate() unwraps a list of per-exit logits

Usage:  python tools/s3_canaries.py     (exit 1 if any canary fails)
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "msc_lib.py"
src = SRC.read_text(encoding="utf-8")
res = []


def check(tag, cond, detail=""):
    res.append(bool(cond))
    print(f'{"PASS" if cond else "FAIL"}  {tag}{("  -- " + detail) if detail and not cond else ""}')


# --- exit_loss_weights, executed for real --------------------------------
ns = {"List": list}
tree = ast.parse(src)
fn = next(n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "exit_loss_weights")
exec(compile(ast.Module(body=[fn], type_ignores=[]), "<w>", "exec"), ns)
w = ns["exit_loss_weights"]

for scheme in ("uniform", "linear", "final_heavy"):
    for K in (1, 2, 3, 5, 8):
        v = w(K, scheme)
        ok = len(v) == K and abs(sum(v) - 1.0) < 1e-9 and all(x > 0 for x in v)
        if not ok:
            check(f"{scheme} K={K} sums to 1, all positive, length K", False, str(v))
            break
    else:
        check(f"{scheme}: sums to 1.0, len==K, all positive for K in 1..8", True)

check("uniform is actually uniform", all(abs(x - 0.2) < 1e-9 for x in w(5, "uniform")))
check("linear increases with depth", all(a < b for a, b in zip(w(5, "linear"), w(5, "linear")[1:])))
check("final_heavy puts 0.5 on the last exit", abs(w(5, "final_heavy")[-1] - 0.5) < 1e-9)
check("final_heavy K=1 degenerates to [1.0]", w(1, "final_heavy") == [1.0])

for bad in ("uniforn", "", "msdnet", None):
    try:
        w(5, bad); check(f"rejects unknown scheme {bad!r}", False)
    except (ValueError, TypeError):
        check(f"rejects unknown scheme {bad!r}", True)
try:
    w(0, "uniform"); check("rejects K=0", False)
except ValueError:
    check("rejects K=0", True)

# --- the guarded branch: Study 1 must be untouched -----------------------
check("joint_exits defaults to False", 'cfg.get("joint_exits", False)' in src)
check("frozen path keeps `model = _backbone_only`", "model = _backbone_only" in src
      and "_ew = None" in src)
check("joint loss only fires when _ew is not None", "if _ew is not None:" in src)

# --- ckpt_best format: run_oracle does strict=True on a PLAIN backbone ---
check("ckpt_best saves the unwrapped backbone on joint runs",
      "_best_model = (_backbone_only.state_dict() if _joint" in src)
check("joint run writes exit_heads via THE accessor (D-23)",
      "atomic_save_torch(exit_heads_path(work, run_id)" in src)
check("run_oracle still loads ckpt_best strictly",
      'backbone.load_state_dict(blob["model"], strict=True)' in src)

# --- evaluate() unwraps per-exit logits ----------------------------------
ev = src[src.index("def evaluate(model, loader"):]
ev = ev[:ev.index("\ndef ")]
check("evaluate() unwraps a list/tuple of per-exit logits",
      "isinstance(logits, (list, tuple))" in ev and "logits = logits[-1]" in ev)

# --- resume: ckpt_last must keep the FULL wrapped state ------------------
check("ckpt_last still saves model.state_dict() (heads included, for resume)",
      "save_checkpoint(ckpt_last, cfg, model, optimizer, scheduler, scaler," in src)

# --- joint_exits must NOT be hash-excluded: it changes the model ---------
ex = src[src.index("_HASH_EXCLUDE"):src.index("_HASH_EXCLUDE") + 3000]
check("joint_exits is NOT in _HASH_EXCLUDE (it changes what is trained)",
      "joint_exits" not in ex)
check("exit_weight_scheme is NOT in _HASH_EXCLUDE",
      "exit_weight_scheme" not in ex)

# --- _subset_train with an explicit keep-list, EXECUTED --------------------
# Pure enough to run with stubs, so it does not have to ship unexercised.
import json, types, tempfile, numpy as np

fn2 = next(n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "_subset_train")
stub = types.SimpleNamespace()
class _Sub:
    def __init__(self, ds, idx): self.ds, self.indices = ds, list(idx)
    def __len__(self): return len(self.indices)
stub.utils = types.SimpleNamespace(data=types.SimpleNamespace(Subset=_Sub))
ns2 = {"np": np, "json": json, "Path": Path, "torch": stub,
       "log": lambda *a, **k: None, "Dict": dict, "Any": object}
exec(compile(ast.Module(body=[fn2], type_ignores=[]), "<s>", "exec"), ns2)
_subset = ns2["_subset_train"]

class _DS:
    def __init__(self, n): self._n = n; self.index_space = n; self.classes = ["a"]
    def __len__(self): return self._n

ds = _DS(50000)
tmp = Path(tempfile.mkdtemp())

def write(keep, name="k.json"):
    q = tmp / name
    q.write_text(json.dumps({"arm": "sat", "score": "ce_loss", "keep": list(keep)}))
    return str(q)

sub = _subset(ds, {"subset_path": write(range(0, 15000))})
check("keep-list of 15000 yields a 15000-item subset", len(sub) == 15000)
check("index_space is PRESERVED, not renumbered (D-49)",
      getattr(sub, "index_space", None) == 50000)
check("keep order is sorted and deduplicated",
      sub.indices == sorted(set(sub.indices)))

sub2 = _subset(ds, {"subset_path": write([9, 3, 3, 1], "dup.json")})
check("duplicates collapse, order normalised", sub2.indices == [1, 3, 9])

check("no subset_path and no frac -> dataset returned untouched",
      _subset(ds, {}) is ds)
check("train_subset_frac still works alongside",
      len(_subset(ds, {"train_subset_frac": 0.1, "seed": 1})) == 5000)

for bad_cfg, why, exc in [
        ({"subset_path": str(tmp / "nope.json")}, "missing file", FileNotFoundError),
        ({"subset_path": write([], "empty.json")}, "empty keep-list", ValueError),
        ({"subset_path": write([0, 999999], "oob.json")}, "out-of-range index", IndexError)]:
    try:
        _subset(ds, bad_cfg)
        check(f"refuses {why}", False, "did not raise -- would train on FULL data")
    except exc:
        check(f"refuses {why} ({exc.__name__})", True)

check("subset_path is NOT hash-excluded (it changes the training set)",
      "subset_path" not in ex)

print(f"\n{sum(res)}/{len(res)} Study 3 canaries pass")
sys.exit(0 if all(res) else 1)
