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

# --- locate_cifar100 must be TELLABLE, and check before downloading -------
# The notebooks re-downloaded 169 MB at 17 kB/s over a copy already on disk,
# because CIFAR-100 had no equivalent of ImageNet's MSC_IN100_DIR.
import os as _os, tempfile as _tf
fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
ns3 = {"Path": Path, "os": _os, "log": lambda *a, **k: None,
       "subprocess": None, "sys": sys, "shell": None, "shutil": None,
       "ensure_dir": lambda q: q, "SCRATCH_ROOT": Path("/nonexistent"),
       "WORK_ROOT": Path("/nonexistent"), "KAGGLE_CIFAR100_SLUG": "x"}
for _f in ("_has_cifar100", "locate_cifar100"):
    exec(compile(ast.Module(body=[fns[_f]], type_ignores=[]), "<f>", "exec"), ns3)

_root = Path(_tf.mkdtemp())
_cd = _root / "cifar-100-python"
_cd.mkdir()
(_cd / "train").touch(); (_cd / "test").touch()

_os.environ["MSC_CIFAR_DIR"] = str(_root)
check("MSC_CIFAR_DIR pointing at the PARENT resolves",
      str(ns3["locate_cifar100"](verbose=False)) == str(_root))
_os.environ["MSC_CIFAR_DIR"] = str(_cd)
check("MSC_CIFAR_DIR pointing at cifar-100-python ITSELF resolves",
      str(ns3["locate_cifar100"](verbose=False)) == str(_root))
_os.environ["MSC_CIFAR_DIR"] = str(_root / "nope")
try:
    ns3["locate_cifar100"](verbose=False)
    check("a wrong MSC_CIFAR_DIR falls through rather than lying", False)
except Exception:
    check("a wrong MSC_CIFAR_DIR falls through rather than lying", True)
_os.environ.pop("MSC_CIFAR_DIR", None)
check("the explicit check runs BEFORE the Kaggle/torchvision download",
      src.index("AN EXPLICIT LOCATION") < src.index("KAGGLE_CIFAR100_SLUG} via Kaggle CLI"))

# --- notebooks must use sess.config, which fills in data_root -------------
import json as _json
for _nb in sorted(Path("notebooks_study3").glob("*.ipynb")):
    _code = "\n".join("".join(c["source"]) for c in
                       _json.loads(_nb.read_text(encoding="utf-8"))["cells"]
                       if c["cell_type"] == "code")
    check(f"{_nb.name}: no bare M.base_config (skips data_root)",
          "M.base_config(" not in _code)

# --- D-87: ONE flag must not have TWO defaults ---------------------------
# place_model defaulted channels_last True while build_loaders defaulted it
# False. Every CIFAR config omitted the key, so the model went NHWC and the
# batches stayed NCHW -- caught by assert_layout_match on batch one of the
# first joint run, having silently applied to every CIFAR run before that.
_pm = src[src.index("def place_model"):]
_pm = _pm[:_pm.index("\ndef ")]
_bl = src[src.index("def build_loaders"):]
_bl = _bl[:_bl.index("\ndef ")]
check("place_model defaults channels_last to False",
      'cfg.get("channels_last", False)' in _pm)
check("place_model and the loader now agree on the default",
      ('cfg.get("channels_last", True)' not in _pm)
      and ('cfg.get("channels_last", True)' not in _bl))

# and the CIFAR recipe now STATES it, so nothing depends on a default
_cif = src[src.index('"train_holdout_n": 5000'):]
_cif = src[:src.index('"train_holdout_n": 5000')]
check("the CIFAR recipe states channels_last explicitly",
      _cif.rindex('"channels_last": False') > _cif.rindex('"deterministic": False'))

# --- offline-first: training notebooks must not touch HuggingFace ---------
import json as _j
_nbs = {q.name: "\n".join("".join(c["source"]) for c in
                           _j.loads(q.read_text(encoding="utf-8"))["cells"]
                           if c["cell_type"] == "code")
        for q in sorted(Path("notebooks_study3").glob("*.ipynb"))}
for _name, _code in _nbs.items():
    if _name.endswith("NB5_Publish.ipynb"):
        check(f"{_name}: HF is ON (it is the publisher)", "enable_hf=True" in _code)
    else:
        check(f"{_name}: HF is OFF (offline training)", "enable_hf=False" in _code)

check("every notebook points at the CIFAR repo, not the ImageNet one",
      all("msc-cifar100" in c for c in _nbs.values()))
check("the publisher checks the token BEFORE uploading",
      _nbs["S3_NB5_Publish.ipynb"].index("hf_token_check")
      < _nbs["S3_NB5_Publish.ipynb"].index("hf_upload_resilient"))
check("the publisher verifies by resolve, not by the queue draining (rules 9/10)",
      "resolve_meta" in _nbs["S3_NB5_Publish.ipynb"])
check("hf_upload_resilient is called with `items=`, its real parameter",
      "items=items" in _nbs["S3_NB5_Publish.ipynb"])

# --- D-88: every run_all that measures must pass fn=sess.oracle ----------
import re as _re
for _name, _code in _nbs.items():
    for _call in _re.findall(r"sess\.run_all\([^)]*\)", _code, _re.S):
        if "oracle" in _call or "measure" in _call:
            # D-67 requires all three; D-88 requires fn to match the stage.
            check(f"{_name}: measuring run_all passes fn/done_fn/stage",
                  ("fn=sess.oracle" in _call and "done_fn=sess.measured" in _call
                   and "stage='measure'" in _call), _call[:110])

# and NB1 must OPEN the artifact rather than trust the plan (rule 5 / D-79)
check("S3_NB1 verifies test.parquet exists after measuring",
      "test.parquet" in _nbs["S3_NB1_JointTrain.ipynb"]
      and "reported success but" in _nbs["S3_NB1_JointTrain.ipynb"])
check("S3_NB2 distinguishes 'not trained' from 'trained but not measured'",
      "TRAINED but have no" in _nbs["S3_NB2_Compare.ipynb"])

# --- run ids are FOUND, never CONSTRUCTED from the session phase ----------
# NB3 built p4-{arch}-cifar100-base-s{seed} from the session phase while the
# base runs are p1. Every id missed, '0 feature dump(s)', and the empty frame
# surfaced as KeyError: 'kind' four cells later.
for _name, _code in _nbs.items():
    check(f"{_name}: no run_id built from an f-string phase prefix",
          not _re.search(r"f['\"]p\d+-\{", _code))

# --- every notebook that builds a DataFrame from a scan guards it empty ---
for _name, _code in _nbs.items():
    if "pd.DataFrame(rows)" in _code:
        _tail = _code.split("pd.DataFrame(rows)", 1)[1][:400]
        check(f"{_name}: guards the empty frame before indexing a column",
              ".empty" in _tail or "len(" in _tail, _tail[:80])

print(f"\n{sum(res)}/{len(res)} Study 3 canaries pass")
sys.exit(0 if all(res) else 1)
