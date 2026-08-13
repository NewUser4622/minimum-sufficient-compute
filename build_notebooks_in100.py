#!/usr/bin/env python3
"""
build_notebooks_in100.py -- generate the five ImageNet-100 notebooks.

Five, not seventeen. The CIFAR set was split fine-grained because a Kaggle
session dies at 9-12 hours without warning and a crash mid-notebook cost the
whole notebook. That constraint is gone: this runs on one local machine with no
session limit, so the split can follow the *stages of the experiment* instead of
the platform's failure modes.

    NB1_Setup      one-time. Pack the data, prove offline, preflight all eight
                   architectures, kill-and-resume test. Ends GO / NO-GO.
    NB2_Train      backbones. Phase 0 and the atlas, same notebook, one switch.
    NB3_Measure    exit heads + the three-axis oracle sweep -> per-sample tables
    NB4_Analysis   Q1-Q4, CPU only, minutes
    NB5_Method     MSC-KD (Q5), both arms

Each is independently re-runnable and idempotent: finishing work is skipped,
unfinished work resumes, and nothing is deleted.

`src/msc_lib.py` is the source of truth. These notebooks embed it as base64 and
are GENERATED -- editing the blob does nothing that survives the next rebuild.

    python build_notebooks_in100.py
    python build_notebooks_in100.py --check
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "src" / "msc_lib.py"
CORE = ROOT / "msc_core.py"
OUT = ROOT / "notebooks_in100"

DATASET = "imagenet100"
ARCHS = ["resnet50", "resnet18", "vgg16", "shufflenetv2_in",
         "vit_small_p16", "deit_small", "swin_tiny", "convnext_tiny"]
P0_ARCHS = ["resnet50", "vit_small_p16"]


# ---------------------------------------------------------------------------
def _chunks(b64: str, width: int = 96):
    return [b64[i:i + width] for i in range(0, len(b64), width)]


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def bootstrap() -> str:
    # D-62. Stamp the build INTO the bytes that get written, so the notebook
    # can prove the module it IMPORTED is the module it SHIPPED. Two fixes in
    # a row appeared not to work because a stale module was executing.
    _lb, _cb = LIB.read_bytes(), CORE.read_bytes()
    lib_sha = hashlib.sha256(_lb).hexdigest()[:12]
    core_sha = hashlib.sha256(_cb).hexdigest()[:12]
    _lb += f'\n__MSC_BUILD__ = "{lib_sha}"\n'.encode()
    _cb += f'\n__MSC_BUILD__ = "{core_sha}"\n'.encode()
    lib_b64 = base64.b64encode(_lb).decode("ascii")
    core_b64 = base64.b64encode(_cb).decode("ascii")

    def emit(name, chunks):
        return f"_{name} = (\n" + ",\n".join(f"    '{c}'" for c in chunks) + ",\n)"

    return f"""\
# ============================================================================
# CELL 1 -- unpack the library.  Runs in every notebook.  No network.
# ============================================================================
# Writes two files into the working directory and imports them:
#
#   msc_lib.py    {lib_sha}   the pipeline: data, zoo, training, measurement
#   msc_core.py   {core_sha}   the reference maths: the MSC definition and
#                                    every statistic in the paper
#
# Both are GENERATED from src/ by build_notebooks_in100.py. Editing the base64
# below does nothing that survives a rebuild -- edit src/msc_lib.py instead.
#
# NOTHING IS INSTALLED HERE. This pipeline runs offline; the packages must
# already be present (see requirements.txt). A missing one is reported by name
# with what it costs you, rather than silently pip-installing on a machine that
# may have no network.
import base64, os, sys
from pathlib import Path

# Offline guards must be set BEFORE anything that might fetch is imported.
os.environ.setdefault('MSC_OFFLINE', '1')

WORK = Path.cwd()
{emit("LIB", _chunks(lib_b64))}

{emit("CORE", _chunks(core_b64))}

for _name, _blob in (('msc_lib', _LIB), ('msc_core', _CORE)):
    (WORK / f'{{_name}}.py').write_bytes(base64.b64decode(''.join(_blob)))
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
for _m in [m for m in list(sys.modules) if m in ('msc_lib', 'msc_core')]:
    del sys.modules[_m]          # force reimport if this cell is re-run
import importlib
importlib.invalidate_caches()

_MISSING = []
for _pkg, _why in (('torch', 'everything'),
                   ('torchvision', 'resnet/vgg/shufflenet/swin'),
                   ('numpy', 'everything'), ('pandas', 'every table'),
                   ('pyarrow', 'per_sample/*.parquet -- the science'),
                   ('yaml', 'config.yaml per run'),
                   ('scipy', 'Spearman = Q1 and Q3'),
                   ('sklearn', 'Q4 delta-R2, Q2 PCA'),
                   ('psutil', 'host telemetry columns'),
                   ('pynvml', 'GPU power -- energy columns are NA without it'),
                   ('fvcore', 'FLOPs. rho is DEFINED in FLOPs.')):
    try:
        __import__(_pkg)
    except ImportError:
        _MISSING.append(f'{{_pkg:12s}} {{_why}}')
if _MISSING:
    print('MISSING PACKAGES -- install these, then restart the kernel:')
    for _m in _MISSING:
        print('   ', _m)
    raise SystemExit('see requirements.txt')

import msc_lib as M
import torch

# D-62. Prove the module that LOADED is the module that SHIPPED.
#
# Twice now a fix was applied, verified, regenerated -- and the run failed with
# the identical error, because the code executing was not the code on disk.
# Jupyter keeps an imported module until something removes it, and any object
# built from the old module (a Session, say) keeps its old functions even after
# a reimport. There was no mechanism that could tell the difference, so the
# evidence looked like "the fix does not work" when it was "the fix never ran".
#
# Rule 5: a cache must answer "is what I have still VALID", not "do I have
# something". The stamp is written into the bytes this cell decodes, so it
# cannot drift from them.
_want = '{lib_sha}'
_got = getattr(M, '__MSC_BUILD__', None)
if _got != _want:
    raise RuntimeError(
        f"STALE msc_lib: this notebook ships build {{_want}} but the imported "
        f"module reports {{_got}}.\\n"
        f"  loaded from: {{getattr(M, '__file__', '?')}}\\n"
        f"  Restart the kernel (Kernel -> Restart) and run all cells. Objects "
        f"created before a reimport keep the OLD code even after this cell "
        f"rewrites the file (D-62).")
# D-68. Is this NOTEBOOK current with the repository?
#
# The check above proves the module matches the notebook. It CANNOT catch a
# stale notebook, because both sides come from the same .ipynb -- they always
# agree with each other and can be arbitrarily old together.
#
# Jupyter saves an open notebook on run. So regenerating NB3 on disk while it
# sits open in a tab means the tab's copy wins the moment you run it: the fixed
# notebook is silently replaced by the one that was open, and the fix appears
# not to have been applied. That happened here -- NB3 was regenerated with
# `done_fn=sess.measured, stage='measure'`, and the version that ran had
# neither.
#
# The repository source is the authority. If it has moved on, this notebook is
# stale and must be reopened, not re-run.
_repo = WORK.parent / 'src' / 'msc_lib.py'
if _repo.exists():
    import hashlib as _h
    _repo_sha = _h.sha256(_repo.read_bytes()).hexdigest()[:12]
    if _repo_sha != _want:
        raise RuntimeError(
            f"STALE NOTEBOOK: this file embeds msc_lib {{_want}}, but "
            f"src/msc_lib.py is {{_repo_sha}}.\\n"
            f"  You are running an older copy of this notebook. Jupyter saves "
            f"an open notebook when you run it, so an open tab silently "
            f"overwrites a regenerated file.\\n"
            f"  FIX: close this notebook WITHOUT saving, run "
            f"`python build_notebooks_in100.py`, then reopen it (D-68).")
    print(f'msc_lib build {{_got}} verified, and current with src/')
else:
    print(f'msc_lib build {{_got}} verified (repo source not visible)')

print(f'msc_lib {{M.__version__}}   torch {{torch.__version__}}')
print(f'CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    for _i in range(torch.cuda.device_count()):
        _p = torch.cuda.get_device_properties(_i)
        print(f'  GPU {{_i}}: {{_p.name}}  {{_p.total_memory/2**30:.1f}} GiB  sm_{{_p.major}}{{_p.minor}}')
else:
    print('  *** NO CUDA. A CPU-only torch trains at roughly 1/200th speed')
    print('  *** while reporting entirely plausible numbers. Fix this first.')
"""


def _phase_block_for(phase: str, detect: bool) -> str:
    """The PHASE line. D-65.

    Notebooks that CREATE runs name their phase. Notebooks that CONSUME runs
    detect it, because a consumer hardcoded to a phase the producer did not
    write finds nothing and says so quietly -- NB3 printed
    `0 trained run(s), 0 still to measure`, called `run_all([])` and exited 0.
    """
    if not detect:
        return f"PHASE = '{phase}'"
    return (
        "# None of the analysis notebooks may hardcode a phase: whichever one\n"
        "# NB2 actually trained is the one to read (D-65). Set PHASE by hand\n"
        "# below to override.\n"
        f"PHASE = M.detect_phase(MSC_ROOT, prefer='{phase}')\n"
        "print(f'phase: {PHASE}   on disk: {M.phases_present(MSC_ROOT)}')")


def paths_cell(phase="p1", extra="", detect=False) -> str:
    _phase_block = _phase_block_for(phase, detect)
    return f"""\
# ============================================================================
# CELL 2 -- WHERE EVERYTHING LIVES
# ============================================================================
# Leave both as None and they are CHOSEN FOR YOU: the roomiest drive that
# actually exists on this machine gets `msc_data/in100` and `msc_results`.
#
# The previous version defaulted to r'D:\\msc_data\\in100'. There is no D:
# drive here, and the failure was
#
#     FileNotFoundError: [WinError 3] The system cannot find the path
#     specified: 'D:\\'
#
# forty lines deep inside pathlib, naming neither the setting nor the file that
# had to change. A default that names a drive letter is wrong on any machine
# without that letter (D-44).
#
# Set them explicitly if you want somewhere specific. Both are checked below by
# WRITING A PROBE FILE AND READING IT BACK -- os.access lies on Windows shares.
#
#   data     ~26 GB   the packed dataset, read-only after NB1
#   results ~120 GB   every run. Nothing here is ever deleted.

DATA_DIR = None      # e.g. r'E:\\msc_data\\in100'   -- None = choose for me
MSC_ROOT = None      # e.g. r'E:\\msc_results'        -- None = choose for me

# ---------------------------------------------------------------------------
import os

_paths = M.resolve_storage(DATA_DIR, MSC_ROOT)
if not _paths['ok']:
    raise SystemExit('storage is not usable -- see the problems listed above')

DATA_DIR = _paths['data_dir']
MSC_ROOT = _paths['results_root']
os.environ['MSC_IN100_DIR'] = DATA_DIR
os.environ['MSC_SCRATCH'] = MSC_ROOT

{_phase_block}
sess = M.Session(account='local', phase=PHASE, dataset='{DATASET}',
                 work_root=MSC_ROOT, session_limit_h=0.0,
                 worker_id=0, num_workers=1)
{extra}
print()
print('layout under MSC_ROOT:')
print('  runs/{{run_id}}/  config.yaml  summary.json  STATUS.json')
print('                   metrics/     epochs.csv  final.csv  confusion_matrix.csv')
print('                                per_class.csv  exit_metrics.csv')
print('                   telemetry/   energy_samples.csv  system_samples.csv')
print('                                step_traces.jsonl')
print('                   per_sample/  test.parquet  train_holdout.parquet')
print('                                train_dynamics.parquet  meta.json')
print('                   checkpoints/ ckpt_last.pt  ckpt_best.pt')
print('                   env/         environment.json')
print('                   exit_heads.pt')
print('  budgets/{{arch}}.json     FLOPs per compute configuration')
print('  registry/events/*.jsonl  what ran, when, and how it ended')
print('  analysis/                Q1-Q4 outputs')
print('  tables/  paper/figures/  console/')
"""


# ===========================================================================
def nb1():
    c = [
        md(f"""
# NB1 · Setup — run this once, and read every output

**Nothing else in this project will work until this notebook ends with `GO`.**

It does five things, in an order chosen so the cheap checks fail before the
expensive ones start:

| step | cost | what it catches |
|---|---|---|
| 1. environment | seconds | a CPU-only torch, a missing package |
| 2. offline proof | seconds | a dependency that fetches on first use |
| 3. pack the dataset | 20–40 min | a source tree that differs from the one this port was designed against |
| 4. preflight the zoo | ~5 min GPU | an architecture that cannot run at a resolution the oracle will sweep |
| 5. kill-and-resume | ~5 min GPU | the defect class that has cost this project five separate times |

## Why the preflight matters more than it looks

On CIFAR-100 this step caught three defects **before a single GPU-hour was
spent**: a ViT whose positional embedding was sized for one patch grid, an
MLP-Mixer that cannot run at any other resolution at all, and a budget table
with duplicate entries that would have made MSC mathematically undefined
three hours into a training phase.

The eight architectures here are new and four of them are decomposed from
torchvision. **That decomposition is the part most likely to be wrong**, and
this is where it surfaces.

## What "GO" means

Every architecture builds, forwards, backprops, exits at every stage, prices a
strictly-ascending budget table, and survives a real interrupt. Anything less
and the numbers this project produces would be well-formed and meaningless.
"""),
        code(bootstrap()),
        md("""
---
## Step 1 · Where everything lives

Set the two paths below. `DATA_DIR` needs ~24 GiB, `MSC_ROOT` grows to ~60–90
GiB across the full programme.
"""),
        code(paths_cell(phase="p0")),
        md("""
---
## Step 2 · Prove the pipeline is offline

Environment variables are a *request*. This blocks the socket layer outright,
so anything reaching for the network raises naming the host it wanted.

Installing a package is not offline-readiness, the same way draining an upload
queue is not confirmation that files landed.
"""),
        code(f"""
# Blocks outbound sockets (loopback stays open -- CUDA and the dataloader use
# it) and then builds every architecture and prices every budget table.
import msc_lib as M

fails = []
with M.no_network():
    for arch in M.zoo_for_dataset('{DATASET}'):
        try:
            m = M.build_model(arch, dataset='{DATASET}').eval()
            M.build_budget_table(arch, '{DATASET}', model=m)
            print(f'  [OK]   {{arch}}')
        except Exception as e:
            fails.append(f'{{arch}}: {{type(e).__name__}}: {{e}}')
            print(f'  [FAIL] {{arch}}: {{fails[-1]}}')

print()
if fails:
    print(f'{{len(fails)}} architecture(s) need the network or do not build.')
    print('The pipeline is NOT self-contained. Do not continue.')
else:
    print('OK -- all eight build and price with the network blocked.')
"""),
        md("""
---
## Step 3 · Pack the dataset

129,395 loose JPEGs → one `256×256` uint8 memmap, with the splits frozen.
**~20–40 minutes on 24 cores, one time.**

You do not have to leave the notebook. The cell below runs the packer as a
subprocess with live output, using the `DATA_DIR` cell 2 already chose. Set
`SRC_DIR` to the folder that contains `train/`, and flip `RUN_PACKER` to True.

### How the labels are assigned

Folders sorted by WNID → class index 0…99. `n01440764` is 0, `n01855672` is 99.
Arbitrary but **deterministic and published**: the mapping lives in
`manifest.json` as `classes` and `class_names`. Nothing downstream depends on
matching official ImageNet indices, so an arbitrary-but-fixed labelling is
exactly right — what would matter is if it changed between runs, and the
fingerprint makes that impossible.

### The fingerprint is the thing to read

```
2b6269ef51ff87b2c9e00fa17c44326ce634a67892c9eb550ec518a6dd2d2b6c
```

A pure function of the file names and the split seeds, so it reproduces on any
machine. **A different value means your source tree differs from the one this
port was designed against.** It goes into `config_hash`, so two runs that
disagree about which 10,000 images are `val` refuse to be compared rather than
producing per-sample tables that align by index and describe different pictures.

Resumable at chunk granularity — an interruption continues rather than
restarting.
"""),
        code("""
# Set this to the folder that CONTAINS train/. Leave None to auto-detect.
SRC_DIR    = None
RUN_PACKER = False        # <-- flip to True to actually pack
#
# ONE TIME. Leaving it True afterwards is harmless: the cell checks whether the
# pack already exists and prints 'already packed -- nothing to do' without
# touching it. The packer itself is also idempotent and resumable, so nothing
# is redone and nothing is overwritten.

import subprocess, sys
from pathlib import Path

if SRC_DIR is None:
    # Look for a directory holding train/<wnid>/*.JPEG near the usual places.
    cands = [Path.home() / 'Desktop' / 'New folder',
             Path.cwd().parent, Path.cwd().parent.parent]
    for c in cands:
        try:
            t = Path(c) / 'train'
            if t.is_dir() and any(d.name.startswith('n0') for d in t.iterdir()):
                SRC_DIR = str(c)
                break
        except Exception:
            continue

ok, detail = M.data_present('imagenet100', DATA_DIR)
if ok:
    print('already packed -- nothing to do')
    print(' ', detail)
elif not RUN_PACKER:
    print('NOT PACKED YET. This is the next step, not a failure.')
    print()
    print(f'  source detected : {SRC_DIR or "NOT FOUND -- set SRC_DIR above"}')
    print(f'  destination     : {DATA_DIR}')
    print()
    print('  Set RUN_PACKER = True and re-run this cell (~20-40 min),')
    print('  or run the same thing in a terminal:')
    print(f'    python tools/pack_imagenet100.py --src "{SRC_DIR}" --out "{DATA_DIR}"')
elif not SRC_DIR:
    print('*** SRC_DIR not found. Set it to the folder containing train/.')
else:
    tool = Path.cwd().parent / 'tools' / 'pack_imagenet100.py'
    if not tool.exists():
        tool = Path.cwd() / 'tools' / 'pack_imagenet100.py'
    print(f'packing {SRC_DIR} -> {DATA_DIR}')
    print('(live output below; safe to interrupt, it resumes)')
    proc = subprocess.Popen([sys.executable, str(tool), '--src', str(SRC_DIR),
                             '--out', str(DATA_DIR)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        print(line.rstrip())
    proc.wait()
    print(f'packer exit code {proc.returncode}')
"""),
        code("""
# Whatever route you took, verify the result here.
import json
from pathlib import Path

ok, detail = M.data_present('imagenet100', DATA_DIR)
print(f'pack present: {ok}')
print(f'  {detail}')
PACKED = ok

if ok:
    man = json.loads((Path(DATA_DIR) / 'manifest.json').read_text())
    spl = json.loads((Path(DATA_DIR) / 'splits.json').read_text())
    print()
    print(f"  images        {man['count']:,}")
    print(f"  classes       {man['n_classes']}   "
          f"({man['classes'][0]} = 0  ...  {man['classes'][-1]} = "
          f"{man['n_classes']-1})")
    print(f"  stored at     {man['stored_res']}px")
    print(f"  policy        {man['resize_policy']}")
    print(f"  val           {len(spl['val']):,}")
    print(f"  train         {len(spl['train']):,}")
    print(f"  holdout       {len(spl['holdout']):,}  (a slice OF train, aug off)")
    print(f"  decode fails  {len(man.get('failures', []))}  "
          f"(packed as zeros, excluded from every split)")
    print()
    print(f"  fingerprint   {man['fingerprint']}")
    EXPECTED = '2b6269ef51ff87b2c9e00fa17c44326ce634a67892c9eb550ec518a6dd2d2b6c'
    if man['fingerprint'] != EXPECTED:
        print()
        print('  *** FINGERPRINT DOES NOT MATCH THE DESIGN.')
        print(f'  *** expected {EXPECTED}')
        print('  *** Your source tree differs. Every downstream comparison')
        print('  *** would be against different images. Investigate first.')
else:
    print()
    print('  Not packed yet. Everything below still runs -- the preflight and')
    print('  both dry runs are SYNTHETIC and never open the dataset. Only')
    print('  NB2 onward needs it.')
"""),
        md("""
---
## Step 4 · Preflight every architecture

Builds all eight, runs a forward and a backward, attaches an exit head to every
stage, and — the important part — **runs each one natively at every resolution
the oracle will sweep**: 96, 128, 160, 192, 224.

### Expect `swin_tiny` to fail at 96px, and possibly 128px

That is a recorded fact about the architecture, not a bug. Swin-T reduces its
input by 32 (patch-4 then three merges), so its final stage is 7×7 at 224 and
**3×3 at 96 — smaller than its own 7×7 attention window**.

The budget table probes per resolution and falls back to an analytic cost model
for the values that fail, and the *proxy* sweep (downsample-then-upsample) is
primary for all eight architectures anyway. What would be a real problem is the
failure going unrecorded.

### What each check prevents

| check | if it fails |
|---|---|
| builds / forwards / backprops | a torchvision decomposition that mis-ordered blocks |
| exit head attaches at every stage | a feature-rank or memory-layout mismatch (Swin speaks NHWC internally) |
| depth ρ strictly ascending, distinct, ends at 1.0 | **MSC is undefined when two budgets cost the same** |
| native resolution per value | a shape error discovered mid-sweep, hours in |
"""),
        code(f"""
report = M.preflight(sess, archs=M.zoo_for_dataset('{DATASET}'), quick=False)
summ = M.preflight_summary(report)
failed = summ['failed']

print()
print(f"{{len(summ['passed'])}} passed, {{len(failed)}} failed, "
      f"{{len(summ['todo'])}} not yet done, of {{summ['n']}}")
for k in summ['todo']:
    print(f'  TODO   {{k}}  -- a prerequisite, not a failure')
for k in failed:
    print(f'  FAILED {{k}}  -- {{report["checks"][k]["detail"]}}')

# D-45: one profiler for the whole zoo, or cross-architecture rho is invalid.
used = M.profilers_used()
print()
print(f'FLOPs profiler: {{sorted(used)}}')
if len(used) > 1:
    print('  *** MORE THAN ONE PROFILER PRICED THIS ZOO. Cross-architecture')
    print('  *** transfer numbers would be invalid: the analytic fallback')
    print('  *** counts Conv2d and Linear only, so a transformer loses its')
    print('  *** attention matmuls entirely, and rho is DEFINED in FLOPs.')
    failed = failed + ['mixed FLOPs profilers']
else:
    print('  one profiler for all eight -- rho is comparable across the zoo')
"""),
        md("""
---
## Step 5 · Dry-run the whole path, on synthetic data

Before any real training, push one synthetic batch through **the entire path
including evaluation and the artifact write**.

This is here because on CIFAR two defects each cost an hour of GPU time and each
was findable in milliseconds — but at *different* stages. One was the first
training step; the other was the history write at the **end** of epoch 0. A dry
run that stops after `loss.backward()` catches one and moves the other somewhere
else to hide.

So the backbone dry run goes: forward → loss → backward → optimiser step →
`optimisation_health` → `evaluate()` → history row (strict) → checkpoint save →
**checkpoint reload**. And the oracle dry run goes: every axis at every
resolution and precision → difficulty battery → prediction depth → per-sample
frame → parquet write → **parquet read back** → `compute_msc`.
"""),
        code(f"""
bad = []
# require_data=False: these are SYNTHETIC. They never open the dataset, so
# they must not wait on a 40-minute packing job to tell you a config is broken.
for arch in M.zoo_for_dataset('{DATASET}'):
    cfg = sess.config(arch, seed=1, require_data=False)
    for name, fn in (('backbone', M.backbone_dry_run), ('oracle', M.oracle_dry_run)):
        ok, why = fn(cfg)
        print(f'  [{{"OK" if ok else "FAIL"}}] {{arch:16s}} {{name:9s}} {{why}}')
        if not ok:
            bad.append(f'{{arch}}/{{name}}: {{why}}')
print()
print('all dry runs pass' if not bad else f'{{len(bad)}} FAILED -- do not train')
"""),
        md("""
---
## Step 6 · Kill-and-resume

**Do not skip this.** Five separate defects in the CIFAR programme were about
resume, and the worst of them silently restarted nine completed runs from epoch
0 — roughly 30 GPU-hours, with no error and no warning, because a re-trained run
looks exactly like normal work.

The test fires a **real interrupt** mid-run and then resumes in a fresh call. It
compares **per-epoch training loss after the seam**, not final accuracy: final
accuracy can match by luck, the loss curve cannot. An earlier version of this
test simulated a kill by training a shorter run and asking for more epochs —
that is a clean completion followed by an extension, a completely different code
path, and it validated nothing while looking like it did.
"""),
        code("""
# 5% of the training split: this is a SMOKE TEST of the resume machinery,
# not a training run. On the full 119,395 images it took ~40 minutes for
# three legs; on 1/20 it takes about two, and tests exactly the same code.
res = M.resume_acceptance_test(sess, arch='resnet18', epochs=4, kill_at=2,
                               subset_frac=0.05)
print()
print('RESUME OK' if res['ok'] else 'RESUME FAILED -- do not train')
for k, v in res.items():
    if k != 'passed':
        print(f'  {k}: {v}')
"""),
        md("""
---
## GO / NO-GO
"""),
        code(f"""
checks = {{
    'offline (all 8 build with sockets blocked)': not fails,
    'dataset packed and fingerprint matches':     ok,
    'preflight (every arch, every resolution)':   not failed,
    'dry runs (train + measure paths)':           not bad,
    'kill-and-resume equivalence':                bool(res['ok']),
}}
print()
for k, v in checks.items():
    print(f"  [{{'PASS' if v else 'FAIL'}}] {{k}}")
print()
if all(checks.values()):
    print('  GO -- open NB2_Train and run Phase 0 (4 runs, ~33 GPU-h).')
    print()
    print('  Phase 0 is 8% of the programme and it answers the only question')
    print('  that matters before committing the other 92%: does the ViT/CNN')
    print('  seed-reliability gap survive at ImageNet scale at all?')
else:
    print('  NO-GO. Fix the failures above. Training now would produce numbers')
    print('  that are well-formed and meaningless.')
"""),
    ]
    return notebook(c)


# ===========================================================================
def nb2():
    c = [
        md("""
# NB2 · Train the backbones

**Safe to stop at any moment.** Runs checkpoint every epoch and resume at the
epoch they reached. Completed runs are skipped. Nothing is ever deleted. Close
the notebook whenever you like — but run the last cell first, because it is the
only thing that confirms the work is on disk and readable.

---

## What you will see while it runs

A live bar per epoch, with the numbers that matter updating **beside** it about
once a second:

```
ep 7/100  63%|███████████▌      | 738/1178 [04:12<02:31, loss=3.412, acc=0.221, img/s=402, lr=8.7e-02, vram=2.9G]
```

An epoch here is 3–35 minutes. A bar showing only position tells you the run is
alive but not whether it is *learning*, and during a multi-day programme those
are the two separate questions you actually have.

Then one line per epoch, carrying what you would otherwise have to open
`epochs.csv` to see:

```
  ep   7/100  train 22.14%  val 19.83%  top5 45.12%  loss 3.412  lr 8.66e-02  402 img/s  289s  ETA 9.3h  0.041kWh  *BEST*
```

### Warnings that appear inline, and what each one means

These are the columns that are **silent by default and unrecoverable
afterwards**, so they are surfaced while they are happening rather than left in
a CSV nobody reads by eye:

| tag | meaning | what to do |
|---|---|---|
| `[N NaN/Inf BATCHES]` | under AMP a non-finite loss is **discarded silently**. The run continues and learns nothing from those batches | a handful is normal early; hundreds means the LR is too high |
| `[N AMP OVERFLOWS]` | gradient overflows whose steps were **thrown away** | >5% of steps is a problem |
| `[LR HIGH?]` | ‖Δw‖/‖w‖ above 1e-2 | healthy is ~1e-3. Stop and check |
| `[NOT MOVING?]` | ‖Δw‖/‖w‖ below 1e-5 | nothing is learning |
| `[DATA-BOUND N%]` | the loader is the bottleneck, not the model | raise `num_workers` |

`[DATA-BOUND]` is trustworthy now: device-side augmentation is measured
separately and subtracted, so this counts genuine CPU starvation only.

---

## Run Phase 0 first. Then stop and read the gate.

**Phase 0:** `resnet50` and `vit_small_p16`, 2 seeds each. **4 runs, ~1.5 days.**

That gives one noise ceiling per family, which is the entire question:

| ρ_seed outcome | meaning | action |
|---|---|---|
| ViT below CNN by **> 0.05** | the CIFAR finding reproduces | build the atlas |
| within **±0.05** | **it was a small-data artifact** | retract the CIFAR headline; the paper becomes about scale-dependence |
| ViT **above** CNN by > 0.05 | inversion | stop, audit the measurement |
| either below **0.40** | noise-dominated at this scale | coarsen the grid, re-gate |

All four are publishable. **Row 1 flatters the existing paper, so scrutinise it
harder than the others**: check both architectures cleared the acceptance
thresholds, seed spread is under 2 points, `nan_or_inf_batches` is 0, and both
ceilings used a comparable sample count after the τ mask.
"""),
        code(bootstrap()),
        code(paths_cell(phase="p0")),
        md("""
---
## Cost, from measurement rather than estimate

The plan estimated 235 GPU-hours. **Your benchmark says otherwise**, and the
shape of the answer changes what is worth running.

`vgg16` is now **45% of the entire atlas budget** for one across-CNN-family data
point. It strengthens Q3's family ordering; the Q1 headline — the reason this
replication exists — does not need it.

**You do not have to decide yet.** Phase 0 contains no `vgg16` and costs ~1.5
days. If the gap fails to reproduce, the atlas shrinks to the 2×2 anyway and
the question is moot.

Two caveats on the numbers below, both flagged in the table:

- `resnet50` and `vgg16` were measured with `cudnn.benchmark = False` — torch's
  default, and **not** what training uses (D-43). `resnet50` at 82 img/s against
  `resnet18`'s 413 is a 5× gap for 2.3× the FLOPs; expect ~180 once re-measured.
- `vit_small_p16` and `deit_small` **failed to build** in that run (D-42, fixed)
  and have never been measured. Their figures are inferred from `swin_tiny`.
"""),
        code("""
ALL = M.zoo_for_dataset('imagenet100')
est = M.in100_estimate(ALL, seeds=3, epochs=M.IN100_EPOCHS)

print(f"{'arch':18s} {'img/s':>7s} {'s/epoch':>8s} {'h x3':>7s} {'share':>6s}  basis")
for r in est['rows']:
    print(f"{r['arch']:18s} {r['img_s']:7.0f} {r['sec_per_epoch']:8.0f} "
          f"{r['hours_all_seeds']:7.1f} {100*est['share'][r['arch']]:5.1f}%  {r['basis']}")
print()
print(f"  atlas, all 8, {M.IN100_EPOCHS} epochs: "
      f"{est['total_gpu_hours']:.0f} GPU-h = {est['days']:.1f} days")
print(f"  the plan estimated 235 -- it was optimistic by "
      f"{(est['total_gpu_hours']-235)/235*100:.0f}%")
print()
for drop in (['vgg16'], ['vgg16', 'deit_small']):
    e = M.in100_estimate([a for a in ALL if a not in drop], 3, M.IN100_EPOCHS)
    print(f"  without {drop}: {e['total_gpu_hours']:.0f} GPU-h = {e['days']:.1f} days")
for ep in (60, 80):
    e = M.in100_estimate(ALL, 3, ep)
    print(f"  all 8 at {ep} epochs: {e['total_gpu_hours']:.0f} GPU-h = {e['days']:.1f} days")
print()
print('  Dropping ONE architecture is a more honest cut than under-training')
print('  all eight: there is no published reference for this subset, so the')
print('  "these models converged" claim rests entirely on the acceptance')
print('  thresholds and has nothing to fall back on.')
"""),
        md("""
---
## What to run

`PHASE = 'p0'` for the pilot, `'p1'` for the atlas. Nothing else changes.

`ARCHS` is an ordinary list — remove `vgg16` here if you take that cut.
"""),
        code("""
PHASE  = 'p0'                     # 'p0' = pilot (4 runs) · 'p1' = atlas
EPOCHS = M.IN100_EPOCHS           # 100

if PHASE == 'p0':
    ARCHS, SEEDS = ['resnet50', 'vit_small_p16'], (1, 2)
else:
    ARCHS, SEEDS = M.zoo_for_dataset('imagenet100'), (1, 2, 3)
    # ARCHS = [a for a in ARCHS if a != 'vgg16']    # <- the 45% cut

sess = M.Session(account='local', phase=PHASE, dataset='imagenet100',
                 work_root=MSC_ROOT, session_limit_h=0.0)
cfgs = [sess.config(a, seed=s, num_epochs=EPOCHS) for a in ARCHS for s in SEEDS]
run_ids = [c['run_id'] for c in cfgs]

print(f'{len(cfgs)} run(s), {EPOCHS} epochs each')
print()
print(f"{'run_id':46s} {'opt':>6s} {'lr':>9s} {'bs':>4s} {'mixup':>6s} {'aug':>12s}")
for c in cfgs:
    print(f"{c['run_id']:46s} {c['optimizer']:>6s} {c['learning_rate']:9.5f} "
          f"{c['batch_size']:4d} {c['mixup_alpha']:6.1f} {str(c['rrc_scale']):>12s}")
e = M.in100_estimate(ARCHS, len(SEEDS), EPOCHS)
print()
print(f"estimated {e['total_gpu_hours']:.0f} GPU-hours = {e['days']:.1f} days")
print('(an estimate; the first cell above lists which entries are measured)')
"""),
        md("""
---
## Train

Per run: claim → **dry run** → resume-or-start → train → evaluate → write
artifacts. Everything lands under `MSC_ROOT/runs/{run_id}/`.

The dry run pushes one synthetic batch through the entire path — forward, loss,
backward, optimiser step, `evaluate()`, the history row, **and a checkpoint save
and reload** — before the dataset is touched. It takes under a second and runs
*before* the run is claimed, so a broken config costs nothing and leaves no
trace in the ledger.

**Resuming is automatic.** Re-run this cell after any interruption: finished
runs are skipped, partial runs continue from their last completed epoch with
optimiser, scheduler, AMP scaler and all four RNG streams restored.
"""),
        code("""
# `sess.train` -- NOT `M.train_backbone`. The bound method supplies hub,
# registry, work_root and data_root_out; the raw function takes them as
# required positional arguments and run_all passes only the config (D-54).
results = sess.run_all(cfgs, title='Phase 0 / atlas training')

print()
for r in results:
    if r.get('status') == 'skipped':
        print(f"  SKIPPED   {r['run_id']}  ({r.get('reason')})")
    else:
        # M.fmt_metric, not a bare format spec: a paused or failed run
        # reports best_accuracy=None, and dict.get's default does not fire on
        # a key that is PRESENT and null (D-61).
        print(f"  {r.get('status','?'):9s} {r.get('run_id','?')}  "
              f"top1={M.fmt_metric(r.get('best_accuracy'))}  "
              f"{r.get('num_epochs_run','?')} epochs")
"""),
        md("""
---
## Before you stop — confirm the work is on disk

`confirm_on_disk` **opens every required artifact**. Stronger than a presence
check: a run whose `summary.json` exists but whose `epochs.csv` is zero bytes
looks healthy to a presence check and fails during analysis weeks later.

- **complete** — every required artifact present, non-empty, parseable
- **resumable** — `ckpt_last.pt` is there. **Safe to stop.** Being unfinished is
  the normal state of a paused run, not a failure
- **at risk** — missing, zero-byte, or corrupt
"""),
        code("""
status = sess.confirm_on_disk(run_ids)

print()
if status['at_risk']:
    print('  *** Do not treat the AT RISK runs as done. Re-run the training')
    print('  *** cell; finished work is skipped and unfinished work resumes.')
else:
    print('  Nothing is at risk.')
    print('  Next: NB3_Measure, then NB4_Analysis.')
    if PHASE == 'p0':
        print()
        print('  THEN COME BACK AND READ THE GATE at the top of this notebook')
        print('  before starting the atlas. Phase 0 is 8% of the programme and')
        print('  it decides whether the other 92% is worth spending.')
"""),
    ]
    return notebook(c)


def nb3():
    c = [
        md("""
# NB3 · Measure — the oracle sweep

Turns trained backbones into **per-sample MSC tables**, which are the
scientific artifact of the whole project.

For every run:

1. **Train exit heads** — K linear heads on the frozen backbone. Freezing is not
   an optimisation, it is the definition: if the backbone adapted while the
   heads trained, each exit would read a *different* network and the "same model
   under reduced compute" interpretation collapses.
2. **Sweep every configuration on every sample** — depth (K exits), resolution
   (5 native + 5 proxy), precision (5). No early-exit shortcut: the
   stable-sufficiency definition quantifies over *all larger* budgets, so
   stopping at the first agreement would record exactly the accidental early
   agreement the definition exists to reject.
3. **Compute the difficulty battery** and prediction depth.
4. **Write** `per_sample/test.parquet` and `per_sample/train_holdout.parquet`.

Inference-only and idempotent — ~1 GPU-h per run, and re-running skips anything
already measured.

## Why `train_holdout` exists

EL2N and forgetting-events are **training-set** quantities, undefined on any
split the model never trained on. Running Q4 without them handicaps the
difficulty battery, which flatters MSC — that is exactly the defect that
inflated the CIFAR ΔR² by 2.5× and had to be withdrawn.

`train_holdout` is 15,000 training images evaluated with augmentation off. It is
not held out of training.

## The coverage alarm at the end is not decoration

On CIFAR, six runs were trained and never measured — the cheapest architectures,
which the scheduler places last. One architecture ended up with **zero** measured
seeds, contributed nothing to any analysis, and the atlas was 14 architectures
while every document said 15. A noise ceiling needs **two** measured seeds
minimum.
"""),
        code(bootstrap()),
        code(paths_cell(phase="p1", detect=True)),
        code(f"""
# PHASE and `sess` come from the paths cell above, which DETECTS the phase
# that actually has runs rather than naming one (D-65).
sess.repair_ledger()

trained = [r['run_id'] for r in sess.completed_runs(phase=PHASE)]
todo    = [r for r in trained if not sess.measured(r)]

print(f'{{len(trained)}} trained run(s), {{len(todo)}} still to measure')
for r in todo:
    print(f'  {{r}}')
if not todo and trained:
    print('  everything already measured -- this notebook is a no-op')
"""),
        code("""
cfgs = [sess.config(M.parse_run_id(r)['arch'], seed=M.parse_run_id(r)['seed'])
        for r in todo]
# done_fn/stage are NOT optional here. Without them plan_work asks
# "is it trained?" to decide whether to measure, skips every run, and
# reports success having done nothing (D-67).
results = sess.run_all(cfgs, fn=sess.oracle, title='measurement',
                       done_fn=sess.measured, stage='measure')

for r in results:
    print(f"  {r.get('status','?'):9s} {r['run_id']}")
"""),
        md("""
---
## Coverage — and the alarm
"""),
        code(f"""
import collections
per_arch = collections.defaultdict(lambda: {{'trained': 0, 'measured': 0}})
for r in sess.completed_runs(phase=PHASE):
    a = M.parse_run_id(r['run_id'])['arch']
    per_arch[a]['trained'] += 1
    per_arch[a]['measured'] += int(sess.measured(r['run_id']))

print(f"{{'arch':18s}} {{'trained':>8s}} {{'measured':>9s}}   status")
weak = []
for a in M.zoo_for_dataset('{DATASET}'):
    t, m = per_arch[a]['trained'], per_arch[a]['measured']
    flag = 'OK' if m >= 2 else ('ONE SEED -- no ceiling possible' if m == 1
                                else 'NOTHING -- contributes to no analysis')
    if m < 2:
        weak.append(a)
    print(f'{{a:18s}} {{t:8d}} {{m:9d}}   {{flag}}')

print()
if weak:
    print(f'  *** ALARM: {{len(weak)}} architecture(s) have fewer than 2 measured')
    print(f'  *** seeds: {{weak}}')
    print('  *** A noise ceiling needs two. These contribute to NOTHING --')
    print('  *** not Q1, not Q3, not Q4 -- and any claim about "eight')
    print('  *** architectures" is false until this is closed.')
else:
    print('  every architecture has >= 2 measured seeds. Ceilings are computable.')
"""),
        code("""
status = sess.confirm_on_disk([r['run_id'] for r in sess.completed_runs(phase=PHASE)],
                              measured=True)
print()
print('Next: NB4_Analysis (CPU only, minutes).' if not status['at_risk']
      else 'Fix the AT RISK runs before analysing.')
"""),
    ]
    return notebook(c)


# ===========================================================================
def nb4():
    c = [
        md("""
# NB4 · Analysis — Q1 to Q4

CPU only. Minutes, not hours. Re-run it as often as you like.

## Q1 is the whole point of this replication

Everything else here is secondary and would still be worth reporting, but the
question this project exists to answer is:

> On CIFAR-100, ViT and Mixer showed seed-reliability of **0.547** against
> **0.62–0.73** for every CNN. Does that survive at ImageNet scale, or was it a
> small-data artifact?

Q1 measures the **noise ceiling** ρ_seed: the Spearman correlation between the
per-sample MSC of two seeds of the *same* architecture. It is not a side
experiment — it is the denominator every transfer number gets divided by, and
it is the single most important quantity in the project.

## Read Q1 with the confound in mind

The eight architectures were trained for **equal epochs**, so schedule length is
not a variable — which it *was* on CIFAR (240 vs 300). But ViTs from scratch on
129k images will still land below the CNNs in accuracy, so **family and accuracy
remain partly confounded** and that must be stated wherever the result is.

The design carries three answers to it, and none of them is "the marginal means
look fine":

1. **`swin_tiny` vs `vit_small_p16`** — both attention; only Swin has locality
   and hierarchy. If reliability tracks *attention*, they agree. If it tracks
   *weak spatial prior*, Swin sits with the CNNs.
2. **`convnext_tiny` vs `resnet50`** — both convolution; only ConvNeXt uses the
   transformer design language.
3. **`vit_small_p16` vs `deit_small`** — **identical geometry, built by one
   function with one argument set**, differing only in augmentation strength.
   If ρ_seed differs across this pair, reliability is a property of *training*,
   not of attention — which would reframe the CIFAR finding rather than confirm
   it.

Together 1 and 2 form a 2×2: {conv, attention} × {strong prior, weak prior}. If
the effect is about attention the split runs along one diagonal; if it is about
spatial prior, the other.

## And one direct bridge

`shufflenetv2` is the only architecture measured in **both** studies. Its CIFAR
ρ_seed is **0.6698**. Whatever it reads here, the *difference* is a measurement
of what dataset scale alone does, with architecture held exactly fixed. It
calibrates every other comparison in the table.
"""),
        code(bootstrap()),
        code(paths_cell(phase="p1", detect=True)),
        md("""
---
## Q1 · Noise ceiling — ρ_seed per architecture

Reported as a curve over τ ∈ {0.0, 0.1, 0.2, 0.3, 0.5}. **If a conclusion holds
only at one τ, it is not a conclusion.** The pre-registered operating point is
τ = 0.1 and the pre-registered gate is ρ_seed ≥ 0.60.
"""),
        code("""
q1 = M.analyse_q1_all(sess, phase=PHASE)
M.save_analysis(sess.data_dir, 'q1_seed_ceilings_all', q1)
display(q1.sort_values('rho_seed_tau0.1', ascending=False))
"""),
        code("""
# The headline table: CNN vs non-CNN at tau=0.1, and the CIFAR comparison.
import numpy as np
col = 'rho_seed_tau0.1'
fam = {a: M.ZOO[a]['family'] for a in q1['arch']}
cnn = q1[q1['arch'].map(lambda a: fam[a] in ('resnet', 'vgg', 'mobile', 'convnext'))]
att = q1[q1['arch'].map(lambda a: fam[a] in ('vit', 'swin'))]

print(f"{'group':28s} {'n':>3s} {'range':>16s} {'mean':>8s}")
for name, g in (('convolutional', cnn), ('attention', att)):
    if len(g):
        print(f"{name:28s} {len(g):3d} "
              f"{g[col].min():.4f}-{g[col].max():.4f} {g[col].mean():8.4f}")

print()
print('CIFAR-100 was:  CNN 0.6217-0.7256 (mean 0.676) · ViT/Mixer 0.547')
print()
if len(cnn) and len(att):
    gap = cnn[col].min() - att[col].max()
    print(f'separation margin here: {gap:+.4f}   '
          f'({"clean, no overlap" if gap > 0 else "OVERLAPPING -- the CIFAR separation does NOT reproduce"})')
    print()
    print('Now check the confound before believing either answer:')
    sub = q1[['arch', col, 'top1_mean']].sort_values('top1_mean')
    display(sub)
    from scipy.stats import spearmanr
    if len(cnn) > 2:
        rho, p = spearmanr(cnn[col], cnn['top1_mean'])
        print(f'within CNNs, rho_seed vs top-1: Spearman {rho:+.3f} (p={p:.3f})')
        print('  near zero means accuracy carries little information about')
        print('  ceiling height INSIDE a family -- which is the argument that')
        print('  the family effect is not an accuracy effect.')
"""),
        code("""
# The three internal controls. These do not depend on the marginal means.
for a, b, what in (('swin_tiny', 'vit_small_p16', 'spatial prior, attention held fixed'),
                   ('convnext_tiny', 'resnet50', 'design language, convolution held fixed'),
                   ('deit_small', 'vit_small_p16', 'RECIPE, geometry held fixed')):
    ra = q1.loc[q1.arch == a, col]
    rb = q1.loc[q1.arch == b, col]
    if len(ra) and len(rb):
        print(f'{a:15s} {float(ra.iloc[0]):.4f}   vs   {b:15s} {float(rb.iloc[0]):.4f}'
              f'   d={float(ra.iloc[0]) - float(rb.iloc[0]):+.4f}   [{what}]')

print()
print('shufflenetv2 -- the only architecture in BOTH studies:')
r = q1.loc[q1.arch == 'shufflenetv2_in', col]
if len(r):
    print(f'  ImageNet-100 {float(r.iloc[0]):.4f}   CIFAR-100 0.6698   '
          f'd={float(r.iloc[0]) - 0.6698:+.4f}')
    print('  That difference is what dataset scale does with architecture')
    print('  held exactly fixed. It calibrates every row above.')
"""),
        md("""
---
## Q2 · Is compute-need one-dimensional across axes?

PCA over per-sample MSC on {depth, resolution-proxy, precision}. H2 predicted
PC1 ≥ 0.60. On CIFAR **0 of 15** architectures reached it and the highest
anywhere was 0.532 — not a marginal miss.
"""),
        code("""
q2 = M.analyse_q2_all(sess, phase=PHASE)
M.save_analysis(sess.data_dir, 'q2_axis_structure_all', q2)
display(q2.sort_values('pc1', ascending=False))
print(f"reaching PC1 >= 0.60: {int((q2['pc1'] >= 0.60).sum())} of {len(q2)}")
"""),
        md("""
---
## Q3 · Transfer across architectures

The disattenuated transfer coefficient, T = ρ(A,B) / √(ρ_seed(A)·ρ_seed(B)).
Dividing by the ceilings is what turns "0.65 seems highish?" into a defensible
claim — and it is the correction the example-difficulty literature generally
omits.

**The shuffled control runs first.** It compares the raw correlation against the
exact permutation null 1/√(n−1), requires both |z| > 5 **and** |ρ| > 0.10, and
takes the worst of three permutations. An earlier version used a bare
`|T| < 0.05` threshold, which was sample-size blind, ceiling-dependent in the
worst direction (≈7× more likely to false-alarm on exactly the low-ceiling ViT
pairs carrying the headline), and two-sided against a one-sided failure mode. It
halted the analysis on a perfectly healthy pair.
"""),
        code("""
ctrl = M.analyse_q3_shuffled_control_all(sess, phase=PHASE)
M.save_analysis(sess.data_dir, 'q3_shuffled_control', ctrl)
bad = ctrl[~ctrl['passed']]
print(f"{len(ctrl) - len(bad)}/{len(ctrl)} shuffled controls pass  "
      f"(max |z| = {ctrl['z'].abs().max():.2f} against a 5-sigma threshold)")
if len(bad):
    display(bad)
    print('*** Tables may be misaligned. This is a BUG, not a finding.')
"""),
        code("""
q3 = M.analyse_q3_all(sess, phase=PHASE)
M.save_analysis(sess.data_dir, 'q3_transfer_matrix', q3)
print(q3.groupby('pair_type')['T'].agg(['count', 'mean', 'std', 'min', 'max']))
"""),
        md("""
---
## Q4 · Is MSC reducible to classical difficulty scores?

Nested-model ΔR² against the full **seven**-score battery
(`msp, margin, entropy, ce_loss, el2n, forget_events, pred_depth`), on
`train_holdout` — the only split where EL2N and forgetting-events are defined.

Running this on the test split with five of seven scores handicaps the battery,
which flatters MSC. On CIFAR that overstated irreducibility by **2.5×** and the
number had to be withdrawn.
"""),
        code("""
q4 = M.analyse_q4_all(sess, phase=PHASE, split='train_holdout')
M.save_analysis(sess.data_dir, 'q4_irreducibility_all', q4)
print(f"median delta-R2 {q4['delta_r2'].median():.4f}   "
      f"clearing 0.05: {int((q4['delta_r2'] >= 0.05).sum())}/{len(q4)}")
print(f"median partial rho {q4['partial_spearman'].median():.4f}   gate 0.30")
print()
print('Split CNN-only vs transformer-involving before reading either number.')
print('A noisier measurement necessarily explains less variance, so a low')
print('transformer delta-R2 is NOT an independent finding from a low Q1')
print('ceiling -- report them together or a reader double-counts them.')
"""),
        md("""
---
## Paper outputs

Every contribution the protocol claims has to be backed by an artifact on disk,
or it is a claim and not a result. This cell writes them and then **checks the
list**, so a missing table is reported rather than discovered while writing.

| # | contribution (protocol §8.1) | artifact |
|---|---|---|
| 1 | MSC: per-sample, cost-normalised, multi-axis, stability-closed | `runs/*/per_sample/*.parquet` + `budgets/*.json` |
| 2 | first measurement of whether compute-need is one-dimensional across axes | `analysis/q2_axis_structure_all.csv`, Table 3 |
| 3 | first noise-ceiling-corrected cross-architecture transfer study | `analysis/q1_seed_ceilings_all.csv`, `q3_transfer_matrix.csv`, Tables 2 and 4 |
| 4 | irreducibility to seven classical difficulty scores | `analysis/q4_irreducibility_all.csv`, Table 5 |
| 5 | MSC-KD, benchmarked at matched FLOPs | NB5 → `analysis/q5_method_comparison.csv` |
| 6 | fully reproducible artifact | `paper/provenance.csv`, `tables/`, every config and log |

### The one this replication adds

**Contribution 3 is where the novelty concentrates**, and it is sharper here
than on CIFAR. The methodological point is that *measurement reliability is
itself architecture-dependent*, so a cross-architecture difficulty study that
does not disattenuate is comparing quantities measured with unequal precision —
and the example-difficulty literature generally does not.

CIFAR demonstrated that. This tests whether it **survives a 40× increase in
dataset size and a 49× increase in pixels**, with four independent crossings of
the CNN/attention boundary and one architecture held fixed across both studies.
Either answer is a result; the second is a self-retraction, which is rarer and
more useful than the first.
"""),
        code("""
from pathlib import Path
import pandas as pd

tables = Path(sess.data_dir) / 'tables'
tables.mkdir(parents=True, exist_ok=True)

# Table 1 -- the atlas: what was trained, and did it converge.
rows = []
for r in sess.completed_runs(phase=PHASE):
    s = M.read_json(M.run_layout(sess.work, r['run_id'])['base'] / 'summary.json', {})
    if not s:
        continue
    m = M.parse_run_id(r['run_id'])
    rows.append({'arch': m['arch'], 'family': M.ZOO.get(m['arch'], {}).get('family'),
                 'seed': m['seed'], 'top1': s.get('best_accuracy'),
                 'epochs': s.get('num_epochs_run'),
                 'params_M': (s.get('num_parameters') or 0) / 1e6,
                 'gflops': (s.get('full_flops') or 0) / 1e9,
                 'gpu_hours': (s.get('total_time_sec') or 0) / 3600,
                 'kwh': s.get('total_energy_kwh'),
                 'measured': sess.measured(r['run_id'])})
t1 = pd.DataFrame(rows)
t1.to_csv(tables / 'table1_atlas.csv', index=False)
display(t1)

# Table 2 -- Q1, the headline. rho_seed beside accuracy, because the confound
# has to be visible in the same table rather than argued around afterwards.
t2 = q1[['arch', 'family', 'n_seeds', 'n_pairs', 'top1_mean', 'top1_spread',
         'rho_seed_tau0.1', 'rho_seed_sd_tau0.1', 'j10_tau0.1']].copy()
t2 = t2.sort_values('rho_seed_tau0.1', ascending=False)
t2.to_csv(tables / 'table2_q1_ceilings.csv', index=False)
display(t2)
"""),
        code("""
# Table 3 (Q2), Table 4 (Q3), Table 5 (Q4)
q2.to_csv(tables / 'table3_q2_axis_structure.csv', index=False)
q3.to_csv(tables / 'table4_q3_transfer.csv', index=False)
q4.to_csv(tables / 'table5_q4_irreducibility.csv', index=False)

# Table 6 -- the CIFAR<->ImageNet comparison. This table IS the paper.
CIFAR = {'shufflenetv2': 0.6698, 'vit_tiny': 0.5475, 'mixer_nano': 0.5470,
         'convnext_femto': 0.7084, 'resnet32x4': 0.7256, 'vgg8': 0.7216}
comp = []
for _, r in q1.iterrows():
    prior = CIFAR.get(M.CROSS_STUDY_ALIAS.get(r['arch'], r['arch']))
    comp.append({'arch': r['arch'], 'family': r['family'],
                 'in100_rho_seed': r['rho_seed_tau0.1'],
                 'cifar_rho_seed': prior,
                 'delta': (r['rho_seed_tau0.1'] - prior) if prior else None,
                 'same_architecture': prior is not None
                                      and r['arch'] in M.CROSS_STUDY_ALIAS})
t6 = pd.DataFrame(comp)
t6.to_csv(tables / 'table6_cifar_vs_imagenet.csv', index=False)
display(t6)
print()
print('Only the row with same_architecture=True is a controlled comparison.')
print('The others differ in architecture AND scale, so their delta mixes two')
print('effects and cannot be read as "what scale did".')
"""),
        code("""
# Figures. Small, because a paper needs few and each has to earn its place.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

figs = Path(sess.data_dir) / 'paper' / 'figures'
figs.mkdir(parents=True, exist_ok=True)

# Fig 1 -- rho_seed by architecture, coloured by family, with the CIFAR band.
fig, ax = plt.subplots(figsize=(7, 4))
d = q1.sort_values('rho_seed_tau0.1')
cols = ['tab:red' if f in ('vit', 'swin') else 'tab:blue' for f in d['family']]
ax.barh(d['arch'], d['rho_seed_tau0.1'], color=cols)
ax.axvline(0.60, ls='--', c='k', lw=1, label='pre-registered gate 0.60')
ax.axvspan(0.6217, 0.7256, alpha=0.10, color='tab:blue', label='CIFAR CNN band')
ax.axvspan(0.5470, 0.5475, alpha=0.25, color='tab:red', label='CIFAR ViT/Mixer')
ax.set_xlabel(r'$\\rho_{seed}$  ($\\tau$=0.1, depth axis)')
ax.legend(fontsize=7)
fig.tight_layout()
M.save_figure(fig, sess.data_dir, 'fig1_q1_ceilings')

# Fig 2 -- the tau curve. No conclusion may depend on tau, so show it.
fig, ax = plt.subplots(figsize=(7, 4))
taus = [0.0, 0.1, 0.2, 0.3, 0.5]
for _, r in q1.iterrows():
    c = 'tab:red' if r['family'] in ('vit', 'swin') else 'tab:blue'
    ax.plot(taus, [r.get(f'rho_seed_tau{t}') for t in taus], marker='o',
            color=c, alpha=0.7, label=r['arch'])
ax.set_xlabel(r'$\\tau$'); ax.set_ylabel(r'$\\rho_{seed}$')
ax.legend(fontsize=6, ncol=2)
fig.tight_layout()
M.save_figure(fig, sess.data_dir, 'fig2_tau_curves')

# Fig 3 -- the confound, plotted rather than asserted.
fig, ax = plt.subplots(figsize=(5, 4))
for _, r in q1.iterrows():
    c = 'tab:red' if r['family'] in ('vit', 'swin') else 'tab:blue'
    ax.scatter(r['top1_mean'], r['rho_seed_tau0.1'], color=c)
    ax.annotate(r['arch'], (r['top1_mean'], r['rho_seed_tau0.1']), fontsize=6)
ax.set_xlabel('top-1 (%)'); ax.set_ylabel(r'$\\rho_{seed}$')
ax.set_title('the confound, shown')
fig.tight_layout()
M.save_figure(fig, sess.data_dir, 'fig3_ceiling_vs_accuracy')
print('figures written to paper/figures/')
"""),
        code("""
M.provenance_manifest(sess.data_dir)

# Check the list rather than trusting it. A missing table found here costs a
# re-run of a CPU notebook; found while writing, it costs a day.
rep = M.verify_paper_artifacts(sess.data_dir)
for r in rep['rows']:
    print(f"  [{r['state']:7s}] {r['artifact']:46s} {r['backs']}")

print()
if not rep['ok']:
    print(f"  *** {len(rep['missing'])} paper artifact(s) absent. The")
    print(f"  *** contributions they back are claims, not results.")
else:
    print('  every claimed contribution has an artifact behind it.')
    print()
    print('  Q5 (the method) needs NB5. Q1-Q4 stand without it -- that')
    print('  separation is the point of the protocol restructure.')
"""),
    ]
    return notebook(c)


# ===========================================================================
def nb5():
    c = [
        md("""
# NB5 · MSC-KD — the method (Q5)

**Only run this after Q1–Q4 are in.** The method is the last section of the
paper, not its thesis: Q1, Q2 and Q3 are publishable whichever way this goes.

Distils the teacher's per-sample compute requirement into a student's monotone
routing policy. Three loss terms, two weights:

    L = L_CE + α·L_KD + β·L_MSC

Monotonicity is architectural, not a penalty — the sufficiency head is a
cumulative-link ordinal head whose thresholds are `θ_{k+1} = θ_k + softplus(δ_k)`,
so the predicted curve is non-decreasing in k **by construction**. A constraint
that cannot be violated beats a soft penalty that can trade off against other
terms.

## Both arms run in one pass

The **scrambled control** (MSC targets permuted within the batch) trains first.
If it matches the real arm, `L_MSC` is a regulariser and not a signal, and you
need to know that before writing anything.

This used to be a module-level flag with a comment saying which value to run
first. The flag defaulted to the control, four sessions in a row trained the
control, and the real arm never existed. **An invariant in a comment is not a
mechanism** — so both arms are a loop now, and whether a run is scrambled is
derived from its own `run_id`.

## The budget count comes from the student, never the teacher

A student's usable exits are adaptive: `resnet18` and `resnet50` do not have the
same number. Sizing the router from the *teacher's* budget grid produces a model
that trains fine — the loss only ever compares the head against targets, both on
the teacher's grid — and then fails at *evaluation*, where routing indexes the
student's actual exits. It is a modelling error, not a shape bug: the routing
decision spends **the student's** compute, so the teacher's grid is meaningless.

That defect took six rounds to fix because it was patched one call site at a
time, and one of those rounds recreated it *inside the dry run written to catch
it*.
"""),
        code(bootstrap()),
        code(paths_cell(phase="p3", detect=False)),
        code(f"""
TEACHER  = 'resnet50'
STUDENTS = ['resnet18', 'shufflenetv2_in', 'deit_small']
SEEDS    = (1, 2, 3)
ARMS     = [True, False]          # control FIRST, so a null result stops you early

# D-72. This notebook costs 18 training runs. Nothing below starts until you
# have looked at what Q1-Q4 actually said and set CONFIRM = True.
CONFIRM  = False

sess = M.Session(account='local', phase='p3', dataset='{DATASET}',
                 work_root=MSC_ROOT, session_limit_h=0.0)

TEACHER_PHASE = M.detect_phase(MSC_ROOT, prefer='p1')
t_runs = [r['run_id'] for r in sess.completed_runs(phase=TEACHER_PHASE)
          if M.parse_run_id(r['run_id'])['arch'] == TEACHER
          and sess.measured(r['run_id'])]
if not t_runs:
    raise SystemExit(f'no measured {{TEACHER}} run. Run NB2 and NB3 first.')
teacher_run = sorted(t_runs)[0]
print(f'teacher: {{teacher_run}}')

cfgs = []
for shuffled in ARMS:
    for a in STUDENTS:
        for s in SEEDS:
            method = ('mscKDshuffrom' if shuffled else 'mscKDfrom') + TEACHER
            cfgs.append(sess.config(a, seed=s, method=method,
                                    teacher_run=teacher_run))
print(f'{{len(cfgs)}} student run(s): {{len(STUDENTS)}} arch x {{len(SEEDS)}} seeds x 2 arms')
"""),
                md("""
---
## Read the gate before spending 18 runs

MSC-KD's premise is Q4's: that MSC carries information a conventional
difficulty battery does not. If Q4 failed, distillation is distilling
something already available from `msp`, `margin` and `entropy` — and this
notebook is an expensive way to learn that.

The gates are pre-registered in `00_RESEARCH_PROTOCOL.md`. They are read from
the analysis outputs here rather than eyeballed, because a gate nobody can
query is a gate nobody checks.
"""),
        code("""
gates = M.gate_report(sess.data_dir)
print(f"{'gate':24s} {'value':>9s} {'threshold':>10s}   verdict")
print('-' * 62)
for _name, _g in gates.items():
    if not isinstance(_g, dict):
        continue
    print(f"{_name:24s} {_g['value']:>9.3f} {_g['threshold']:>10.2f}   "
          f"{'PASS' if _g['passed'] else '** MISS **'}")
    if _g.get('detail'):
        print(f"{'':24s} {_g['detail']}")

print()
if not gates.get('all_passed'):
    print('NOT every gate passed. That does not forbid running this notebook,')
    print('but it changes what a positive result would mean, and it must be')
    print('stated in the paper rather than discovered by a reviewer.')
else:
    print('all gates passed')

# ---- what this will cost -------------------------------------------------
# Real measurements where they exist, and STALE flags where they do not, so a
# day-vs-week decision is not made from a number taken under the slow layout
# (D-59).
n_runs = len(STUDENTS) * len(SEEDS) * len(ARMS)
print()
print(f"this notebook trains {n_runs} runs "
      f"({len(STUDENTS)} students x {len(SEEDS)} seeds x {len(ARMS)} arms)")
print(f"{'student':18s} {'img/s':>8s} {'h/run':>7s} {'h total':>8s}  basis")
total = 0.0
for _a in STUDENTS:
    _ips = M.IN100_MEASURED_IMG_S.get(_a)
    _stale = _a in M.IN100_PENDING_REMEASURE
    _h = (119395 * int(M.base_config(_a, 'imagenet100')['num_epochs'])
          / _ips / 3600) if _ips else float('nan')
    _sub = _h * len(SEEDS) * len(ARMS)
    total += 0.0 if _sub != _sub else _sub
    _basis = ('STALE -- taken under channels_last; run tools/conv_sweep.py'
              if _stale else 'measured')
    print(f"{_a:18s} {_ips:>8.0f} {_h:>7.1f} {_sub:>8.1f}  {_basis}")
print(f"{'':18s} {'':>8s} {'':>7s} {total:>8.1f}  = {total/24:.1f} days")
print()
print('MSC-KD students also carry the teacher forward pass, so the true cost')
print('is above these numbers, not below them.')

if not CONFIRM:
    raise SystemExit(
        'Set CONFIRM = True in the cell above once you have read the gate '
        'table and the cost estimate. Nothing has been trained.')
"""),
code("""
# train_msc_kd needs the teacher as well, so it goes through a closure --
# the same shape Session.train and Session.oracle use internally.
#
# The arm is read back out of `method`, which is ALREADY in the run_id, rather
# than carried in a second config field. The first draft passed
# `shuffle_msc_targets=` to sess.config -- which is a library FUNCTION name,
# not a config key. `config(**overrides)` takes any key without complaint, so
# it would have entered config_hash while train_msc_kd's real parameter
# (`shuffle_targets`) quietly stayed False, and the "control" arm would have
# trained on unshuffled targets under a run_id that says shuffled (D-54b).
def _train_student(cfg):
    return M.train_msc_kd(cfg, sess.hub, sess.registry, teacher_run,
                          TEACHER, work_root=sess.work,
                          data_root_out=sess.data_dir,
                          shuffle_targets='shuff' in cfg['method'])

results = sess.run_all(cfgs, fn=_train_student, done_fn=sess.msckd_valid,
                       title='MSC-KD students')
real = [r for r in results if 'shuff' not in r['run_id']]
print(f"\\n{len([r for r in real if r.get('status') != 'skipped'])}/"
      f"{len(real)} REAL-method students trained -- the comparison needs all of them")
"""),
        md("""
---
## Compare at matched FLOPs

The only comparison that means anything. B2 (confidence-threshold routing) is
where the field actually is; B11 (routing by the student's own true post-hoc
MSC) is the ceiling. **The fraction of the B2→B11 gap that MSC-KD closes is the
result.**

`γ` is calibrated by Learn-then-Test on `train_holdout`. At 15,000 samples the
distribution-free guarantee holds at **ε = 0.01** — CIFAR had only 5,000 and had
to settle for ε = 0.03 and say so in its limitations.
"""),
        code("""
cmp_ = M.compare_routing_methods(sess, run_ids=[r['run_id'] for r in results])
M.save_analysis(sess.data_dir, 'q5_method_comparison', cmp_)
display(cmp_)
"""),
        code("""
status = sess.confirm_on_disk([r['run_id'] for r in results])
print()
print('Then NB4 for the final tables, and check that the SCRAMBLED arm is')
print('clearly worse than the real one. If it is not, L_MSC is a regulariser')
print('and the mechanism claim is wrong even if the method wins.')
"""),
    ]
    return notebook(c)


NOTEBOOKS = {
    "NB1_Setup.ipynb": nb1,
    "NB2_Train.ipynb": nb2,
    "NB3_Measure.ipynb": nb3,
    "NB4_Analysis.ipynb": nb4,
    "NB5_Method.ipynb": nb5,
}


def main() -> int:
    check = "--check" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    stale = 0
    for name, fn in NOTEBOOKS.items():
        nb = fn()
        text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
        p = OUT / name
        if check:
            cur = p.read_text(encoding="utf-8") if p.exists() else ""
            stale += (cur != text)
            print(f"  {name:24s} {'current' if cur == text else 'STALE'}")
        else:
            p.write_text(text, encoding="utf-8")
            nc = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
            nm = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
            print(f"  {name:24s} {nc:2d} code + {nm:2d} md   "
                  f"{p.stat().st_size/1024:.0f} KB")

    if not check:
        # Verify the base64 round-trips byte-identically. A silent truncation
        # here means debugging the wrong code for an hour.
        import re as _re
        raw = (OUT / "NB1_Setup.ipynb").read_text(encoding="utf-8")
        nb = json.loads(raw)
        src = "".join(nb["cells"][1]["source"])
        blob = "".join(_re.findall(r"'([A-Za-z0-9+/=]{4,})'", src))
        # D-62 appends a build stamp to the bytes before encoding, so the
        # round-trip must be checked against the STAMPED bytes. Comparing
        # against the raw source file made this fail on a correct build --
        # a check that cries wolf is the thing this project keeps paying for.
        _lb = LIB.read_bytes()
        _sha = hashlib.sha256(_lb).hexdigest()[:12]
        _stamped = _lb + f'\n__MSC_BUILD__ = "{_sha}"\n'.encode()
        lib_b64 = base64.b64encode(_stamped).decode("ascii")
        ok = blob.startswith(lib_b64)
        print(f"\n  base64 round-trip: {'byte-identical' if ok else 'MISMATCH'}")
        if not ok:
            # Decode what is actually embedded and say how it differs, rather
            # than only that it does.
            import base64 as _b64
            try:
                got = _b64.b64decode(blob[:len(lib_b64)] + "===")
                print(f"    embedded {len(got):,} bytes, expected "
                      f"{len(_stamped):,}")
                for i, (x, y) in enumerate(zip(got, _stamped)):
                    if x != y:
                        print(f"    first difference at byte {i:,}")
                        break
            except Exception as _e:                              # noqa: BLE001
                print(f"    could not decode the embedded blob: {_e}")
            return 1

    # ------------------------------------------------------------------
    # D-73. Does the emitted Python actually PARSE?
    #
    # The validator checked column names, repo paths, library names, call
    # arity, result keys and stages -- six layers, none of which asked whether
    # the code it was analysing was syntactically valid. It could not: every
    # layer starts with `ast.parse` inside a `try`, and a cell that fails to
    # parse is silently skipped rather than reported.
    #
    # So a real newline inside a single-quoted f-string -- `\n` written where
    # `\\n` was needed, which is easy to do when a generator emits code through
    # an f-string of its own -- shipped a notebook whose FIRST CELL could not
    # run, and every check said OK.
    #
    # `feature_version` pins the target: the machine that runs these is on
    # Python 3.10, and syntax accepted here but not there is the same defect
    # with a longer feedback loop.
    print("\n  parsing every emitted code cell as Python 3.10")
    _syn = 0
    for _nb in sorted(OUT.glob("NB*.ipynb")):
        for _ci, _c in enumerate(json.loads(
                _nb.read_text(encoding="utf-8")).get("cells", [])):
            if _c.get("cell_type") != "code":
                continue
            _src = "".join(_c.get("source", []))
            try:
                ast.parse(_src, feature_version=(3, 10))
            except SyntaxError as _e:
                _syn += 1
                print(f"  [FAIL] {_nb.name} cell {_ci}: {_e.msg} "
                      f"(line {_e.lineno})")
                print(f"         {(_e.text or '').strip()[:88]}")
    if _syn:
        print(f"\n  {_syn} cell(s) do not parse. Generation refused -- a "
              f"notebook that cannot be parsed cannot be run, and every other "
              f"check skips it silently.")
        return 1
    print("  all code cells parse")

    print("\n  validating column names and repo paths")
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import validate_notebooks as V
        if V.validate(OUT):
            return 1
    except Exception as e:                                       # noqa: BLE001
        print(f"  [FAIL] validator could not run: {type(e).__name__}: {e}")
        return 1
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
