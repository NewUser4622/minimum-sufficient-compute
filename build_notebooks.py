#!/usr/bin/env python3
"""
build_notebooks.py -- regenerate the Kaggle notebooks from src/msc_lib.py.

Why a generator instead of hand-written notebooks
-------------------------------------------------
The E2AM notebooks embed their library as a base64 blob in cell 1. That is
robust -- no pip install of a private package, no git clone, no path fragility,
and it survives "Run All" from a cold Kaggle session. But it is also write-only:
changing one line means decoding, patching and re-encoding by hand.

So we keep the base64 bootstrap and remove the opacity. `src/msc_lib.py` is the
readable source of truth; this script embeds it (plus `msc_core.py`, the
reference implementation of every statistic) and writes the .ipynb files.

Sixteen notebooks, so that any single one is short enough to finish, and any
notebook with more than one independent model to train is worker-sharded.

Usage
-----
    python build_notebooks.py            # regenerate
    python build_notebooks.py --check    # verify the .ipynb files are current
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "src" / "msc_lib.py"
CORE = ROOT / "msc_core.py"
OUT = ROOT / "notebooks"

HF_REPO = "Shanmuk4622/msc-cifar100"
SESSION_LIMIT_H = 8.5


def _cost_model():
    """Read the cost model out of msc_lib without importing it -- the generator
    must run with nothing but the stdlib.

    Handles annotated assignments (`X: Dict[str, float] = {...}`), which is how
    ARCH_COST_HINT is declared. Missing that silently yields an empty table and
    every architecture prices at the fallback, which is how five ResNets of very
    different sizes all reported the same estimate.
    """
    tree = ast.parse(LIB.read_text(encoding="utf-8"))
    hints, sec, measured = {}, None, set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        else:
            continue
        try:
            if name == "ARCH_COST_HINT":
                hints = ast.literal_eval(value)
            elif name == "SECONDS_PER_COST_UNIT":
                sec = float(ast.literal_eval(value))
            elif name == "MEASURED_ARCHS":
                # frozenset({...})
                measured = set(ast.literal_eval(value.args[0])
                               if isinstance(value, ast.Call)
                               else ast.literal_eval(value))
        except Exception:
            continue
    if not hints or sec is None:
        raise RuntimeError(
            "could not parse the cost model out of src/msc_lib.py -- "
            f"hints={len(hints)} entries, seconds={sec}. Estimates would be "
            "wrong, so refusing to generate notebooks.")
    return hints, sec, measured


COST_HINT, SEC_PER_UNIT, MEASURED = _cost_model()
TRANSFORMER_LIKE = {"convnext_femto", "vit_tiny", "mixer_nano"}


def est_hours(arch: str, epochs: int | None = None) -> float:
    ep = epochs or (300 if arch in TRANSFORMER_LIKE else 240)
    return COST_HINT.get(arch, 3.0) * ep * SEC_PER_UNIT / 3600.0


def scope_panel(archs, seeds=(1, 2, 3), epochs=None, kind="training",
                extra_note="") -> str:
    """The 'what this notebook runs' table, computed at build time."""
    rows, total = [], 0.0
    for a in sorted(archs):
        ep = epochs or (300 if a in TRANSFORMER_LIKE else 240)
        h = est_hours(a, ep)
        rows.append((a, len(seeds), ep, h, h * len(seeds),
                     "measured" if a in MEASURED else "estimate"))
        total += h * len(seeds)
    n_runs = len(archs) * len(seeds)

    def wall(nw):
        # cost-balanced packing: sort desc, give each to the least-loaded worker
        jobs = sorted((r[3] for r in rows for _ in range(len(seeds))),
                      reverse=True)
        load = [0.0] * nw
        for j in jobs:
            i = load.index(min(load))
            load[i] += j
        return max(load)

    lines = [
        "## What this notebook runs",
        "",
        f"| | |",
        f"|---|---|",
        f"| **Architectures** | {len(archs)} |",
        f"| **Seeds each** | {len(seeds)} |",
        f"| **Total {kind} runs** | **{n_runs}** |",
        f"| **Estimated GPU time** | **~{total:.0f} GPU-hours** |",
        "",
        "### Wall-clock, by how many accounts you run",
        "",
        "| NUM_WORKERS | Wall-clock | Kaggle sessions each |",
        "|---|---|---|",
    ]
    for nw in (1, 2, 4, 6):
        if nw > n_runs:
            continue
        w = wall(nw)
        lines.append(f"| {nw} | ~{w:.1f} h | {max(1, -(-w // SESSION_LIMIT_H)):.0f} |")
    lines += [
        "",
        "### Per architecture",
        "",
        "| Architecture | Epochs | Est. per run | × seeds | Basis |",
        "|---|---|---|---|---|",
    ]
    for a, ns, ep, h, tot, basis in rows:
        lines.append(f"| `{a}` | {ep} | {h:.2f} h | {tot:.1f} h | {basis} |")
    lines += [
        "",
        f"> Estimates are calibrated against measured Phase 0 timings on a "
        f"Kaggle T4 ({SEC_PER_UNIT:.2f} s per cost-unit-epoch). "
        f"{len([r for r in rows if r[5] == 'measured'])} of {len(rows)} "
        f"architectures here are measured; the rest are priced from a relative "
        f"cost model that self-corrects as runs finish.",
        "",
        f"> A Kaggle session lasts ~9–12 h and this pipeline pauses cleanly at "
        f"{SESSION_LIMIT_H} h, so a run needing more than one session resumes "
        f"automatically — just start a fresh session and re-run.",
    ]
    if extra_note:
        lines += ["", extra_note]
    # Match the surrounding template's indent so md()'s dedent works uniformly.
    # The first line is substituted at an already-indented position, so it is
    # left bare.
    out = [lines[0]] + [("        " + l) if l else "" for l in lines[1:]]
    return "\n".join(out)
KAGGLE_DATASET = "shanmuk4622/dataset-cifar100-python"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def _chunks(b64: str, width: int = 96):
    return [b64[i:i + width] for i in range(0, len(b64), width)]


def bootstrap_cell() -> str:
    lib_b64 = base64.b64encode(LIB.read_bytes()).decode("ascii")
    core_b64 = base64.b64encode(CORE.read_bytes()).decode("ascii")
    lib_sha = hashlib.sha256(LIB.read_bytes()).hexdigest()[:12]
    core_sha = hashlib.sha256(CORE.read_bytes()).hexdigest()[:12]

    def emit(name, chunks):
        body = ",\n".join(f"    '{c}'" for c in chunks)
        return f"_{name} = (\n{body},\n)"

    return f"""\
# === CELL 1 of every notebook: unpack the library ==========================
# This writes two Python files into the session and imports them. Nothing here
# touches the GPU or the network beyond installing three small packages.
#
#   msc_lib   {lib_sha}   the pipeline: HuggingFace sync, model zoo,
#                              measurement, training, the method
#   msc_core  {core_sha}   the reference maths: the MSC definition and
#                              every statistic in the paper
#
# Both are generated from KD/src by build_notebooks.py. Editing them HERE does
# nothing useful -- the next rebuild overwrites it. Edit the source instead.
import base64, os, subprocess, sys
from pathlib import Path

WORK = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()

# Kaggle images already ship torch, pandas and sklearn. These three vary by
# image version, so we check rather than assume.
#   pyarrow  writes the per-image measurement tables (Parquet)
#   pynvml   reads GPU power/temperature/utilisation directly
#   fvcore   counts FLOPs, which is how compute cost is defined
for _pkg in ('pyarrow', 'pynvml', 'fvcore', 'psutil'):
    try:
        __import__(_pkg)
    except ImportError:
        print(f'[BOOT] installing {{_pkg}} ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pkg,
                        '--break-system-packages'], check=False)

{emit("LIB", _chunks(lib_b64))}

{emit("CORE", _chunks(core_b64))}

for _name, _blob in (('msc_lib', _LIB), ('msc_core', _CORE)):
    (WORK / f'{{_name}}.py').write_bytes(base64.b64decode(''.join(_blob)))
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
for _m in [m for m in list(sys.modules) if m in ('msc_lib', 'msc_core')]:
    del sys.modules[_m]

import msc_lib as msc
import msc_core

print(f'[BOOT] msc_lib v{{msc.__version__}} ready  (torch available: {{msc._TORCH_OK}})')
print(f'[BOOT] artifact space: {{msc.WORK_ROOT}}   scratch space: {{msc.SCRATCH_ROOT}}')
"""


# ---------------------------------------------------------------------------
# Shared cells
# ---------------------------------------------------------------------------
def worker_cell(phase, workers=1, note="", n_runs=None, hours=None) -> str:
    extra = f"\n# {note}" if note else ""
    scope = ""
    if n_runs is not None:
        scope = (f"#\n"
                 f"# THIS NOTEBOOK: {n_runs} run(s), ~{hours:.0f} GPU-hours total.\n"
                 f"# At NUM_WORKERS = 1 that is all of it on this account.\n"
                 f"# Raise NUM_WORKERS and run the same notebook on each account\n"
                 f"# with a different WORKER_ID to divide it.\n")
    return f"""\
# === Who am I? =============================================================
{scope}#
# ACCOUNT   labels this Kaggle account in the run log. Two accounts calling
#           themselves the same thing makes the log useless.
# WORKER_ID splits the work. Every account runs THE SAME notebook; the only
#           thing that differs is this number. Account 1 -> 0, account 2 -> 1,
#           and so on. Each account then works out, by pure arithmetic, which
#           jobs belong to it -- no communication needed, no chance of two
#           accounts training the same model, no chance of a job being missed.
#
# DEFAULT IS 1: this account does everything in this notebook. That is the
# simplest thing that works. Change it only when you actually have several
# accounts running at once.{extra}
ACCOUNT     = 'acct1'      # <<< CHANGE ME
NUM_WORKERS = {workers}          # <<< how many accounts you are running in parallel
WORKER_ID   = 0            # <<< CHANGE ME: 0, 1, 2, ... up to NUM_WORKERS-1

sess = msc.Session(account=ACCOUNT, phase='{phase}', dataset='cifar100',
                   worker_id=WORKER_ID, num_workers=NUM_WORKERS,
                   shard_mode='cost',        # balance GPU-hours, not job counts
                   enable_hf=True,
                   session_limit_h=8.5,      # push + pause before Kaggle kills us
                   commits_per_hour_limit=20,# 6 accounts x 20 = 120 < HF's 128/hr
                   batch_interval_sec=1800)  # the 30-minute push policy
"""


DATA_CELL = """\
# === Get CIFAR-100 =========================================================
# Looks in this order: attached Kaggle dataset (instant) -> earlier extraction
# -> Kaggle CLI download -> torchvision as a last resort.
# Everything lands in /kaggle/temp (~1 TB scratch), never in /kaggle/working
# (20 GB, and that is the space your results need).
DATA_ROOT = sess.prepare_data()
print('dataset:', DATA_ROOT)
"""

ESTIMATE_CELL = """\
# === How long will this take? ==============================================
# Recomputed live, using measured per-epoch times from any runs that have
# already finished. The more of the project you have completed, the more
# accurate this gets -- the built-in estimates are only a starting prior.
est = msc.estimate_phase(run_ids, num_workers=NUM_WORKERS)
print(f"runs in this notebook : {est['n_runs']}")
print(f"total GPU time        : ~{est['total_gpu_hours']:.1f} GPU-hours")
print(f"at NUM_WORKERS={NUM_WORKERS:<2d}      : ~{est['wall_clock_hours']:.1f} h wall-clock "
      f"({est['sessions_needed']} Kaggle session(s) on this account)")
print(f"cost model            : {est['frac_measured']:.0%} of these architectures "
      f"have measured timings; the rest are estimates")
if NUM_WORKERS > 1:
    print(f"per-worker hours      : "
          f"{[round(h,1) for h in est['per_worker_hours']]}")
print()
for nw in (1, 2, 4, 6):
    if nw > est['n_runs']:
        break
    e = msc.estimate_phase(run_ids, num_workers=nw)
    mark = '  <-- you' if nw == NUM_WORKERS else ''
    print(f"  NUM_WORKERS={nw}: ~{e['wall_clock_hours']:5.1f} h wall-clock{mark}")
"""

SYNC_CELL = """\
# === Catch up with what has already been done ==============================
# Downloads only what this notebook needs -- never the whole repo, which would
# fill the 20 GB disk instantly.
#
# It also rebuilds the progress record from the actual training logs instead of
# trusting the status file. If a session died between writing a log and pushing
# its status, those two disagree, and the log is the one that tells the truth.
# Runs marked "finished" that clearly are not get reset so they resume.
sess.sync_state(verbose=True)
sess.status()
"""

FINISH_CELL = """\
# === Push everything and stop ==============================================
# Blocks until HuggingFace confirms. Safe to re-run.
sess.finish()

# D-19: draining the upload queue is NOT the same as the files being on
# HuggingFace, and "[SESSION] done" reads like a confirmation it is not.
# Ask the repository before you close this tab.
#
# D-20: three states, not two. FINISHED and RESUMABLE are both safe -- a run
# paused at epoch 120 whose ckpt_last.pt is on HF loses nothing when you close
# the tab. Only AT RISK (no summary.json AND no checkpoint) needs action.
try:
    _ids = [c['run_id'] for c in (all_cfgs if 'all_cfgs' in dir() else cfgs)]
except NameError:
    _ids = []
if _ids:
    sess.confirm_on_hf(_ids)
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.rstrip().splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.13",
                              "mimetype": "text/x-python",
                              "codemirror_mode": {"name": "ipython", "version": 3},
                              "pygments_lexer": "ipython3",
                              "nbconvert_exporter": "python",
                              "file_extension": ".py"},
            "kaggle": {"accelerator": "nvidiaTeslaT4",
                       "dataSources": [{"sourceType": "datasetVersion",
                                        "datasetId": 0, "sourceId": 0,
                                        "reference": KAGGLE_DATASET}],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


HEADER_NOTE = """
> **New here?** Read `05_PLAIN_ENGLISH_GUIDE.md` first — it explains what this
> project is measuring and why, without jargon. This notebook assumes you have.
"""


# ===========================================================================
# NB00 -- setup and verification
# ===========================================================================
def nb00():
    return notebook([
        md(f"""
        # NB00 — Setup & Verification

        **Run this first, on every account. ~15 minutes. Nothing here is wasted
        if it passes, and everything downstream is wasted if it doesn't.**

        {HEADER_NOTE}

        ## What this notebook is for

        Every check below corresponds to a specific failure that would otherwise
        surface hours into a real run:

        | Check | What it saves you from |
        |---|---|
        | HuggingFace reachable **and writable** | Nine hours of training with nowhere to put it |
        | CIFAR-100 found | Silently falling back to a slow download |
        | All 15 model types build and train | A Vision Transformer whose internals don't fit our measurement code |
        | FLOPs measured correctly | A compute scale that doesn't actually reach 100% |
        | **Kill-and-resume works** | See below — this is the important one |
        | Worker splitting is balanced | One account working 33 hours while another idles |

        ## Why the resume test matters more than it looks

        We kill a training run halfway and restart it, then check the result is
        *identical* to an uninterrupted run.

        The subtle failure it catches: a resume that reloads the model but not
        the random-number state. Training looks fine. But the resumed run now
        sees images in a different order than it would have. That breaks the
        meaning of "same model, different random seed" — which is exactly the
        comparison Q1 relies on, and Q1 is the denominator of every number in
        the paper.

        So we don't assume it works. We test it.

        ## Before you press Run

        1. `HF_TOKEN` in **Add-ons → Secrets**, with **WRITE** permission
        2. Internet **ON**
        3. Accelerator: **GPU T4 × 2**
        4. Dataset attached: `{KAGGLE_DATASET}`

        Everything is saved to **`{HF_REPO}`** — one repository, one folder per run.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Start a session"),
        code(worker_cell("test")),
        md("## Step 2 — Get the dataset"),
        code(DATA_CELL),
        md("""
        ## Step 3 — Check everything

        Builds all 15 architectures, runs data through them, attaches an early-exit
        head, and backpropagates. `quick=False` also measures each one's FLOPs
        table and checks the compute scale is sane (increasing, and ending at
        exactly 100%).

        This cell `assert`s. If it stops, fix what it names before continuing.
        """),
        code("""\
report = msc.preflight(sess, archs=list(msc.ZOO.keys()), quick=False)

p = sess.data_dir / 'analysis' / f'preflight_{sess.account}.json'
msc.atomic_write_json(p, report)
if sess.hub.enabled:
    sess.hub.hub.enqueue(p, f'analysis/preflight_{sess.account}.json')

assert report['all_passed'], 'Preflight failed. Fix the FAILs above before training.'
"""),
        md("""
        ## Step 4 — Measure the compute cost of every setting

        This builds the table that defines what "50% compute" *means* for each
        architecture.

        `rho` is the fraction-of-full-cost for each setting. It's the number
        that lets us compare a ResNet to a Transformer at all — without it,
        "needed 12 layers" and "needed 6 blocks" aren't comparable quantities.

        Two columns worth understanding:

        **`K`** is how many distinct depth settings that architecture supports.
        Usually 5. But `resnet8x4` has only 3 blocks in total, so it physically
        cannot have five distinct early-exit points — it gets K=3. That's
        correct, not a bug: MSC is a cost *fraction*, not an exit index, so
        architectures may carry different K and still be comparable. (Forcing 5
        would produce duplicate budgets, which makes "the smallest sufficient
        one" ill-defined and would crash the measurement later.)

        **`res_native`** is whether the architecture can run on a smaller image
        at all. MLP-Mixer cannot — its token-mixing layer is a linear map whose
        input size *is* the patch count — so it shows `False` and its resolution
        axis uses the shrink-then-restore proxy. Documented limitation, not a
        failure.

        Measured once and frozen. A cost table that drifts between sessions
        would make measurements from different sessions incomparable.
        """),
        code("""\
import pandas as pd
rows = []
for arch in msc.ZOO:
    b = sess.budgets(arch, 100)
    d, r, q = b['axes']['depth'], b['axes']['resolution'], b['axes']['precision']
    rows.append({'arch': arch, 'family': msc.ZOO[arch]['family'],
                 'params_M': round(b['params'] / 1e6, 2),
                 'full_GFLOPs': round(b['full_flops'] / 1e9, 3),
                 'K': d['K'], 'blocks': d['n_blocks'],
                 'depth_rho': [round(x, 3) for x in d['rho']],
                 'res_native': r['native_supported'],
                 'res_rho': [round(x, 3) for x in r['rho']],
                 'prec_rho': [round(x, 3) for x in q['rho']]})
bt = pd.DataFrame(rows)
display(bt)

# Three invariants the measurement code depends on.
for _r in rows:
    rho = _r['depth_rho']
    assert abs(rho[-1] - 1.0) <= 0.02, \\
        f"{_r['arch']}: deepest exit does not cost the full model ({rho})"
    assert all(rho[i] < rho[i + 1] for i in range(len(rho) - 1)), \\
        f"{_r['arch']}: depth costs not strictly ascending ({rho})"
    assert len(set(rho)) == len(rho), \\
        f"{_r['arch']}: DUPLICATE depth budgets ({rho}) -- the oracle cannot use these"
print('\\nAll depth cost curves are strictly ascending and reach 1.0.')

odd = bt[bt.K < 5]
if len(odd):
    print(f'\\n{len(odd)} architecture(s) carry fewer than 5 depth settings '
          f'(too few blocks). Expected and handled:')
    display(odd[['arch', 'blocks', 'K', 'depth_rho']])
nonat = bt[~bt.res_native]
if len(nonat):
    print(f'\\n{len(nonat)} architecture(s) cannot run at non-32px input; '
          f'their resolution axis uses the proxy:')
    display(nonat[['arch', 'res_native']])
print('\\nprofiler used:', sess.budgets('resnet20')['profiler'])
"""),
        md("""
        ## Step 5 — The kill-and-resume test

        Two runs of the **same config**:

        1. **Reference** — 4 epochs straight through.
        2. **Interrupted** — killed by a *real* `KeyboardInterrupt` after epoch 2,
           exercising the actual emergency-flush and paused-state path, then
           resumed in a fresh call.

        Then it compares them. Passing requires:

        - the interrupt actually fired
        - the resumed run reaches all 4 epochs
        - no duplicated epoch rows in the log
        - **per-epoch training loss after the seam matches the reference within 5%**

        That last one is the point. It's where a lost random-number state shows
        up: if the image order diverges on resume, the post-seam losses drift
        even though nothing looks broken. A resumed run that isn't equivalent to
        an uninterrupted one makes "same model, different seed" meaningless —
        and that comparison is the denominator of every number in the paper.

        You'll see the two loss curves printed side by side so you can check it
        yourself rather than trusting a boolean.
        """),
        code("""\
res = msc.resume_acceptance_test(sess, arch='resnet20', epochs=4, kill_at=2)
assert res['ok'], (f'RESUME TEST FAILED: {res}\\n'
                   'Do not start the atlas until this passes — every long run '
                   'depends on it.')
res
"""),
        md("""
        ## Step 6 — Check the work splits evenly

        Shows how the 45 Phase-1 runs would be divided among N accounts.

        The number to look at is **imbalance**. The phase isn't finished until
        the *slowest* worker finishes, so a 3× imbalance means the phase takes
        3× longer than it needs to.

        Simple hashing (what a straightforward implementation does) gives about
        4.9× here, because 45 jobs of very unequal size don't hash evenly. Our
        cost-balanced scheduler gets it to about 1.02×.
        """),
        code("""\
run_ids = [c['run_id'] for c in msc.phase1_configs()]
print(f'Phase 1 is {len(run_ids)} training runs.\\n')
for mode in ('hash', 'balanced', 'cost'):
    print(f'--- mode = {mode} ---')
    display(msc.shard_report(run_ids, NUM_WORKERS if NUM_WORKERS > 1 else 6, mode=mode))
"""),
        md("""
        ## Step 7 — Prove we can actually write to HuggingFace

        Uploads a small file, then **re-lists the repository to confirm it
        arrived**. An upload that didn't raise an error is not evidence that
        anything was written — a read-only token fails exactly this way.
        """),
        code("""\
probe = sess.data_dir / 'analysis' / f'smoketest_{sess.account}.json'
msc.atomic_write_json(probe, {'account': sess.account, 'worker': WORKER_ID,
                              'utc': msc.now_iso(), 'env': msc.environment_report()})
if sess.hub.enabled:
    sess.hub.hub.enqueue(probe, f'analysis/smoketest_{sess.account}.json')
    sess.hub.flush(timeout=300)
    files = sess.hub.hub.list_repo_files()
    ok = f'analysis/smoketest_{sess.account}.json' in files
    print(f'probe arrived on HF: {ok}   ({len(files)} files in the data repo)')
    assert ok, 'Upload did not arrive. Your HF_TOKEN probably lacks WRITE scope.'
    print(f'{len(files)} files in {sess.hub.repo_id}')
else:
    print('HF is disabled -- fix this before running anything else.')
sess.hub.print_stats()
"""),
        md("## Step 8 — Finish"),
        code(FINISH_CELL),
        md("""
        ---
        ### Everything passed?

        Go to **NB01**. Do not skip Phase 0 — it's 12 GPU-hours and it decides
        whether the remaining ~1,180 are worth spending.
        """),
    ])


# ===========================================================================
# NB01 -- Phase 0 training
# ===========================================================================
def nb01():
    return notebook([
        md(f"""
        # NB01 — Phase 0: train the four pilot models

        **4 runs · ~12 GPU-hours · shardable across up to 4 accounts**

        {HEADER_NOTE}

        ## What we're doing and why

        Before spending ~1,200 GPU-hours, we spend 12 to find out whether the
        thing we want to measure is even measurable.

        Four models:

        | Model | Seed | Why |
        |---|---|---|
        | resnet32x4 | 1 | reference model A |
        | resnet32x4 | 2 | **the same model, different random start** |
        | wrn-40-2 | 1 | a different architecture, B |
        | wrn-40-2 | 2 | **same again, different random start** |

        The duplicate seeds are not redundancy. They answer: *when we train the
        exact same thing twice, do the two copies agree about which images are
        hard?*

        That agreement is our **noise ceiling** — the reference point that makes
        every later number interpretable. If two copies of the same model only
        agree 60% of the time, then a ResNet and a ViT agreeing 55% is actually
        near-perfect. Without the ceiling, 55% is just a number.

        ## What to expect

        These are standard, published architectures on a standard recipe, so we
        know what accuracy they should reach: **resnet32x4 ≈ 79.4%**,
        **wrn-40-2 ≈ 75.6%**.

        The notebook checks this and warns loudly if a run lands more than 1
        point low. That matters because measuring "how much compute does this
        image need" on a badly-trained model gives meaningless answers — and a
        badly-trained model is otherwise easy to miss.

        ## Timing

        ~3 hours per model. With `NUM_WORKERS = 4`, one per account, about
        3 hours total. With 1 worker, about 12 hours across two sessions
        (it pauses and resumes automatically at the 8.5-hour mark).
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session (set your WORKER_ID)"),
        code(worker_cell("p0", workers=1, n_runs=4,
                         hours=2*est_hours('resnet32x4') + 2*est_hours('wrn_40_2'),
                         note="Phase 0 is 4 runs, so up to 4 accounts can help.")),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up with prior progress"),
        code(SYNC_CELL),
        md("""
        ## Step 4 — Train

        The cell below plans this worker's share, then trains it.

        **You can stop this at any time.** Press the stop button, close the tab,
        or let Kaggle time out — everything is pushed to HuggingFace first. To
        continue, open a fresh session and run all cells again; it resumes from
        the exact epoch it stopped at.
        """),
        code("""\
EPOCHS = 240        # the standard published recipe. Do not shorten for Phase 0 --
                    # under-trained models give meaningless measurements.

cfgs = [sess.config(a, seed=s, num_epochs=EPOCHS)
        for a in ('resnet32x4', 'wrn_40_2') for s in (1, 2)]
run_ids = [c['run_id'] for c in cfgs]

est = msc.estimate_phase(run_ids, num_workers=NUM_WORKERS)
print(f"{est['n_runs']} runs, ~{est['total_gpu_hours']:.1f} GPU-hours total")
print(f"at NUM_WORKERS={NUM_WORKERS}: ~{est['wall_clock_hours']:.1f} h wall-clock, "
      f"{est['sessions_needed']} session(s)")
for r in run_ids:
    print(f"  {r:34s} ~{msc.estimate_run_hours(r):.2f} h")
print()

summaries = sess.run_all(cfgs, title='Phase 0 training')

import pandas as pd
pd.DataFrame([{k: s.get(k) for k in
               ('run_id', 'best_accuracy', 'reference_accuracy',
                'accuracy_gap_vs_reference', 'recipe_ok', 'num_epochs_run',
                'total_time_sec', 'total_energy_kwh')}
              for s in summaries if s.get('status') == 'completed'])
"""),
        md("""
        ## Step 5 — Did the training actually work?

        Compares against the published numbers. Anything more than 1 point below
        means the recipe is wrong, and everything measured from that model would
        be worthless.
        """),
        code("""\
import pandas as pd
rows = []
for rid, st in sess.registry.latest().items():
    if st.get('state') != 'completed' or not rid.startswith('p0-'):
        continue
    ref, acc = msc.REFERENCE_ACC.get(st.get('arch')), st.get('best_accuracy')
    if ref and acc:
        rows.append({'run_id': rid, 'accuracy_%': round(acc * 100, 2),
                     'published_%': ref, 'gap': round(ref - acc * 100, 2),
                     'ok': (ref - acc * 100) <= 1.0})
audit = pd.DataFrame(rows)
if len(audit):
    display(audit)
if len(audit) and not audit.ok.all():
    print('\\nSOME RUNS ARE UNDER-TRAINED. Fix the recipe before NB02.')
elif len(audit):
    print('\\nAll runs match their published accuracy. Proceed to NB02.')
else:
    print('\\nNo completed Phase 0 runs yet on this account (others may have them).')
"""),
        md("""
        ## Step 6 — Audit HuggingFace

        Lists what is **actually** in both repositories, per run, so you can see
        at a glance whether anything is half-pushed or missing.

        Two things to look at:

        - **`ledger shards`** should equal the number of worker sessions that
          have run. If it says 0, you're on an older build of the library —
          re-upload the notebooks.
        - **`NOT STARTED`** lists runs nobody has picked up. With
          `NUM_WORKERS = 4` and 4 runs, every worker owns exactly one, so a run
          appearing here means that worker's session hasn't been started.
        - **`FOREIGN DATA`** flags runs that don't match any architecture in the
          current zoo — usually leftovers from an earlier version of the
          project. Harmless to the analysis, but worth clearing out.
        """),
        code("""\
audit = sess.audit_repos(expected_run_ids=[c['run_id'] for c in cfgs])
"""),
        md("## Step 7 — Finish"),
        code(FINISH_CELL),
        md("---\n**Next: NB02** — measure how much compute each image needs."),
    ])


# ===========================================================================
# NB02 -- Phase 0 measurement
# ===========================================================================
def nb02():
    return notebook([
        md(f"""
        # NB02 — Phase 0: measure compute-need per image

        **~2 hours · shardable across up to 4 accounts · inference only, no training**

        {HEADER_NOTE}

        ## What we're doing

        We now have four trained models. For each one we ask, **for every one of
        the 10,000 test images**: how little computation is enough?

        Three steps per model:

        **1. Attach early-exit points.** A trained network is one long pipeline.
        We attach five small "read-off points" at 20%, 40%, 60%, 80% and 100% of
        the way through, so we can see what the network would answer if it
        stopped early.

        These read-off points need a few minutes of training themselves — but
        **the main network is frozen solid** while that happens. This is not an
        optimisation, it's the definition: if the main network were allowed to
        adapt, each exit would be reading a *different* network, and "the same
        model with less compute" would stop being true.

        **2. Run every setting on every image.** Five depth settings, five
        resolutions (measured two ways), five numeric precisions — 20 settings ×
        10,000 images per model. We record what the model answered and how
        confident it was, every time.

        No shortcuts here. We can't stop at the first setting that gets the right
        answer, because our definition requires the answer to be *settled* —
        correct at that budget and every larger one. Checking that requires
        actually looking at all of them.

        **3. Also measure the "usual" difficulty scores.** Seven of them, so we
        can later test whether our new measurement is genuinely new or just an
        old idea renamed (that's Q4).

        ## The output

        Two Parquet tables per model: one for the test set, one for a 5,000-image
        slice of training data. **These tables are the actual scientific product
        of this project.** The model weights matter much less — we could throw
        them away and still write the paper.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session"),
        code(worker_cell("p0", n_runs=4, hours=4*0.6,
                         note="Measurement is inference-only: ~30 min per model.")),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up"),
        code(SYNC_CELL),
        md("""
        ## Step 4 — Measure

        **~30 minutes per model.** Downloads each trained model from HuggingFace
        as it needs it, measures, uploads the tables, moves on.

        > **If this cell finishes in under a minute, it did nothing.** The
        > printed status table before and after tells you directly. A genuine
        > run prints exit-head training progress and a sweep progress bar per
        > model.

        Already measured? It skips instantly — that is the one legitimate way
        this is fast. Safe to re-run.
        """),
        code("""\
cfgs = [sess.config(a, seed=s) for a in ('resnet32x4', 'wrn_40_2') for s in (1, 2)]

# Only measure models that actually finished TRAINING.
ready   = [c for c in cfgs if sess.trained(c['run_id'])]
missing = [c['run_id'] for c in cfgs if not sess.trained(c['run_id'])]
if missing:
    print(f'Not yet trained (run NB01 first): {missing}')
    print()

# State BEFORE we start, so it is obvious whether this cell had work to do.
print('measurement status before this run:')
for c in cfgs:
    print(f"  {c['run_id']:34s} trained={sess.trained(c['run_id'])!s:5s} "
          f"measured={sess.measured(c['run_id'])}")
print()

# stage='measure' matters. The ledger already says 'completed' for every run --
# training set that. Without a measurement-specific completion test, this cell
# plans zero work and exits in seconds looking like a success.
results = sess.run_all(ready, fn=sess.oracle, title='Phase 0 measurement',
                       done_fn=sess.measured, stage='measure')
for r in results:
    print(f"{r.get('run_id')}: {r.get('status')}")

print()
print('measurement status after:')
for c in cfgs:
    print(f"  {c['run_id']:34s} measured={sess.measured(c['run_id'])}")
"""),
        md("""
        ## Step 5 — Are the tables lined up?

        Every table must describe the same 10,000 images **in the same order**.
        We store a fingerprint of that order in each table and compare.

        This check matters enormously. If two tables' rows don't correspond to
        the same images, correlating them produces numbers that look completely
        plausible and are completely fictional. It's the single easiest way to
        accidentally invent a result.
        """),
        code("""\
import pandas as pd
rows = []
for d in sorted(sess.runs_dir.iterdir()) if sess.runs_dir.exists() else []:
    ps = d / 'per_sample'
    if not ps.is_dir():
        continue
    meta = msc.read_json(ps / 'meta.json', default={}) or {}
    for split in ('test', 'train_holdout'):
        f = ps / f'{split}.parquet'
        if f.exists():
            df = pd.read_parquet(f, columns=['sample_order_hash'])
            rows.append({'run_id': d.name, 'split': split, 'rows': len(df),
                         'order_fingerprint': df['sample_order_hash'].iloc[0][:16],
                         'arch': meta.get('arch')})
align = pd.DataFrame(rows)
if not len(align):
    print('No per-sample tables yet.')
    print()
    print('Expected if Step 4 has not produced anything, which happens when the')
    print('backbones are still training. NB02 only measures runs the registry')
    print('marks "completed" -- a half-trained model would give measurements')
    print('that look valid and are not.')
    state = sess.registry.latest()
    prog = [{'run_id': c['run_id'],
             'state': state.get(c['run_id'], {}).get('state', 'not started'),
             'epoch': state.get(c['run_id'], {}).get('epoch', '-')}
            for c in cfgs]
    display(pd.DataFrame(prog))
    print()
    print('Finish NB01 first, then re-run this notebook.')
else:
    display(align)
    for split, g in align.groupby('split'):
        k = g.order_fingerprint.nunique()
        print(f"{split}: {k} distinct ordering(s) -> "
              f"{'OK' if k == 1 else 'MISALIGNED -- DO NOT ANALYSE THESE'}")
    msc.save_analysis(sess.data_dir, 'per_sample_alignment_phase0', align, sess.hub)
"""),
        md("## Step 6 — Finish"),
        code(FINISH_CELL),
        md("---\n**Next: NB03** — the go/no-go decision."),
    ])


# ===========================================================================
# NB03 -- Phase 0 decision
# ===========================================================================
def nb03():
    return notebook([
        md(f"""
        # NB03 — Phase 0: the decision

        **CPU only — turn the accelerator OFF. ~10 minutes. Run on one account.**

        {HEADER_NOTE}

        ## Prerequisites

        **NB01 and NB02 must both be finished** for all four Phase 0 runs.
        NB01 trains; NB02 produces the per-sample tables every statistic here
        reads. Step 1 checks and stops with a clear message if anything is
        missing.

        ## This is the most important notebook in the project

        It answers: **is there a project here?**

        We compute three numbers from the Phase 0 measurements and read off a
        verdict.

        | | What it measures | Pass |
        |---|---|---|
        | **ρ_seed** | Two copies of the same model — do they agree about which images are hard? | ≥ 0.6 |
        | **T** | Two *different* architectures — do they agree, after correcting for the noise above? | ≥ 0.7 |
        | **ΔR²** | Does our measurement say anything that existing "difficulty" scores don't? | ≥ 0.05 |

        ## The five possible verdicts

        | Verdict | What it means | What you do |
        |---|---|---|
        | **FULL-PROGRAM** | Everything works | Build the whole thing |
        | **REFRAME** | Real and transfers, but it's just "difficulty" renamed | Still a paper: "cheap scores are enough" |
        | **PIVOT-STRONG-NEGATIVE** | Real, but does NOT transfer | **Still a paper — arguably the best one.** The field's assumption is wrong |
        | **MARGINAL** | Borderline | Use coarser settings, re-analyse, no retraining |
        | **FAIL** | It's noise | Retry once, then switch direction |

        **Three of five lead to a paper.** That was the entire point of designing
        the project this way — its value isn't contingent on one method winning.

        ## One thing to check before believing any of it

        Step 4 runs a **deliberately scrambled** comparison. We shuffle one
        model's answers and re-measure agreement. It must come out at
        approximately zero.

        If it doesn't, we have a bug — not a discovery. The cell stops if so.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session (no GPU, no dataset needed)"),
        code("""\
ACCOUNT = 'acct1'      # <<< CHANGE ME

sess = msc.Session(account=ACCOUNT, phase='p0', dataset='cifar100', enable_hf=True)
# Metrics and per-sample tables only -- checkpoints are excluded by
# allow_patterns, so this downloads in seconds even late in the project.
sess.sync_state(include_checkpoints=False, verbose=True)

import pandas as pd, numpy as np, matplotlib.pyplot as plt

A1, A2 = 'p0-resnet32x4-cifar100-base-s1', 'p0-resnet32x4-cifar100-base-s2'
B1, B2 = 'p0-wrn_40_2-cifar100-base-s1',   'p0-wrn_40_2-cifar100-base-s2'
PHASE0_RUNS = [A1, A2, B1, B2]

# Stop here, with a readable explanation, if the measurement step has not run.
# Every statistic below reads the per-sample tables, and a missing one would
# otherwise surface as a FileNotFoundError six frames deep inside a correlation.
msc.require_inputs(sess.data_dir, PHASE0_RUNS)

bud_a, bud_b = sess.budgets('resnet32x4'), sess.budgets('wrn_40_2')
print('inputs present -- proceeding')
"""),
        md("""
        ## Step 2 — Q1: the noise ceiling

        Two copies of the same architecture, different random starts. How much
        do they agree about which images need more compute?

        Reported as a curve over **τ** (tau), which controls how confident the
        model has to be before we call an answer "settled". We never pick a
        single τ — if a conclusion only holds at one value, it isn't a
        conclusion.
        """),
        code("""\
q1 = pd.concat([
    msc.analyse_q1_seed_ceiling(sess.data_dir, A1, A2, bud_a, axis='depth'),
    msc.analyse_q1_seed_ceiling(sess.data_dir, B1, B2, bud_b, axis='depth'),
], ignore_index=True)
msc.save_analysis(sess.data_dir, 'q1_seed_ceiling', q1, sess.hub)
display(q1[['run_a', 'tau', 'rho_seed', 'jaccard_top10',
            'frac_irreducible_a', 'mean_msc_a']].round(4))

ceil_a = float(q1[q1.run_a == A1].set_index('tau').loc[0.1, 'rho_seed'])
ceil_b = float(q1[q1.run_a == B1].set_index('tau').loc[0.1, 'rho_seed'])
print(f'\\nNOISE CEILINGS (tau=0.1):')
print(f'  resnet32x4: {ceil_a:.3f}')
print(f'  wrn_40_2:   {ceil_b:.3f}')
print(f'\\n  >= 0.6 is a pass. Below 0.4 means the measurement is mostly noise.')
"""),
        md("""
        ## Step 3 — What fraction of images are "too hard for anyone"?

        Images where even the full model isn't confident. Our measure is
        meaningless for those, so they're excluded from every correlation.

        If we quietly left them in, every model would "agree" on them (they'd all
        be 100%), which would inflate every agreement number in the paper.
        """),
        code("""\
print(q1[['run_a', 'tau', 'frac_irreducible_a']].round(4).to_string(index=False))
print('\\nThese are excluded from all correlations. Reported separately in the paper.')
"""),
        md("""
        ## Step 4 — The sanity check

        Scramble one model's answers on purpose. Agreement must collapse to ~0.

        If it doesn't, the two tables aren't lined up and every number above is
        fiction.
        """),
        code("""\
ctrl = msc.analyse_q3_shuffled_control(
    sess.data_dir, A1, B1, ceilings={A1: ceil_a, B1: ceil_b},
    budgets_by_run={A1: bud_a, B1: bud_b}, tau=0.1)
print(f"shuffled rho = {ctrl['spearman_raw']:+.4f}   z = {ctrl['z']:+.2f}   "
      f"n = {ctrl['n']}   (null SD = {ctrl['null_sd']:.4f})")
print(f"bug threshold: |z| > {ctrl['z_max']:.0f} AND |rho| > {ctrl['rho_floor']:.2f}")
assert ctrl['passed'], ('Scrambled control FAILED. This is a bug, not a finding. '
                        'The per-image tables are not row-aligned.')
print('\\nControl passed -- the numbers above are trustworthy.')
"""),
        md("""
        ## Step 5 — Q3: does it transfer between architectures?

        **T** is the raw agreement divided by the noise ceiling. T ≈ 1 means
        transfer is as complete as our measurement precision allows.

        We also report **top-decile overlap**: do the two models agree on which
        images are the *hardest*? For a system that routes compute, that matters
        more than agreeing about the easy majority.
        """),
        code("""\
ceilings, buds = {A1: ceil_a, B1: ceil_b}, {A1: bud_a, B1: bud_b}
q3 = msc.analyse_q3_transfer(sess.data_dir, [(A1, B1)], ceilings, buds, axis='depth')
msc.save_analysis(sess.data_dir, 'q3_transfer', q3, sess.hub)
display(q3[['tau', 'spearman_raw', 'T', 'T_lo', 'T_hi', 'jaccard_top10']].round(4))
print('\\n  T >= 0.7 -> transfer works, build the method')
print('  T <  0.5 -> transfer fails, and THAT IS THE STRONGER PAPER')
"""),
        md("""
        ## Step 6 — Q2 preview: are the three dials measuring the same thing?

        Depth, resolution and precision on the same model, same images. How much
        of the variation does a single shared factor explain?

        **A note on which resolution measurement we use.** There are two honest
        ways to reduce resolution: actually run the network on a smaller image
        (`res_native`), or shrink-then-restore so the network shape is unchanged
        and only information content drops (`res_proxy`).

        Native is cleaner in principle. But MLP-Mixer *cannot* run at another
        resolution — its token-mixing layer is a linear map whose input
        dimension is literally the patch count. So if we made native primary, 14
        architectures would be measured one way and one another way, and any
        cross-architecture claim on this axis would compare two different
        quantities.

        So **the proxy is primary** (uniform across all 15) and native is
        reported as a robustness check for the 14 that support it. The protocol
        says to decide this in Phase 0 and stay consistent — this is that
        decision. Step 6b checks the two agree.

        Nobody has asked this question before. The full version is NB10.
        """),
        code("""\
q2 = msc.analyse_q2_axis_structure(sess.data_dir, A1, bud_a,
                                   axes=('depth', 'res_proxy', 'precision'))
msc.save_analysis(sess.data_dir, 'q2_axis_structure_phase0', q2, sess.hub)
display(q2.round(4))
print('\\n  PC1 >= 0.60 -> one shared "compute need" factor. A single router is justified.')
print('  PC1 <  0.60 -> the dials are different things, and depth-only results')
print('                  do not license claims about the others.')

# 6b: do the two resolution measurements agree? If they do, the idealised cost
# model of the proxy is empirically harmless, and the methodological caveat is
# something we measured rather than merely argued about.
core = msc._import_msc_core()
dfa = msc.load_per_sample(sess.data_dir, A1)
print(f'\\naxes available for {A1}: {msc.available_axes(dfa)}')
if 'res_native' in msc.available_axes(dfa):
    for t in (0.0, 0.1, 0.3):
        n = msc.msc_for_run(dfa, bud_a, 'res_native', t).clean()
        p = msc.msc_for_run(dfa, bud_a, 'res_proxy', t).clean()
        print(f'  tau={t}: native vs proxy agreement = {core.spearman(n, p):.3f}  '
              f'(mean native {np.nanmean(n):.3f} vs proxy {np.nanmean(p):.3f})')
    print('\\n  High agreement -> the proxy is a fair stand-in and using it')
    print('  uniformly across all 15 architectures costs us nothing.')
"""),
        md("""
        ## Step 7 — Q4: is this actually a new idea?

        The threat to the whole project: maybe "how much compute does this image
        need" is just "how hard is this image", which people already measure.

        We test it directly — fit a model predicting compute-need from seven
        existing difficulty scores, then check whether adding our measurement
        helps.

        **If it doesn't help, that's a real finding and we publish it**, not a
        failure to hide. "Cheap existing scores are sufficient" saves the
        community effort and is a better engineering result than a complicated
        oracle.
        """),
        code("""\
# Runs on train_holdout: EL2N and forgetting events are TRAINING-set scores, so
# the test split carries only 5 of the 7. The full battery is the harder, fairer
# test of whether MSC survives controlling for classical difficulty.
q4 = msc.analyse_q4_irreducibility(sess.data_dir, A1, B1, buds, axis='depth',
                                   split='train_holdout')
msc.save_analysis(sess.data_dir, 'q4_irreducibility', q4, sess.hub)
display(q4[['tau', 'n_battery_scores', 'partial_spearman', 'r2_difficulty_only',
            'r2_difficulty_plus_msc', 'delta_r2', 'delta_r2_lo',
            'delta_r2_hi', 'battery']].round(4))

# Test split as a robustness check -- fewer scores available, so an easier test.
q4t = msc.analyse_q4_irreducibility(sess.data_dir, A1, B1, buds, axis='depth',
                                    split='test')
msc.save_analysis(sess.data_dir, 'q4_irreducibility_testsplit', q4t, sess.hub)
print()
print(f'robustness -- test split, '
      f'{int(q4t.n_battery_scores.iloc[0])} of 7 scores available there:')
display(q4t[['tau', 'n_battery_scores', 'delta_r2', 'partial_spearman']].round(4))

print('\\n  delta_R2 >= 0.05 -> genuinely new information')
print('  delta_R2 <  0.02 -> it is difficulty renamed; reframe the paper')
print('  The train_holdout number (full battery) is the one for the paper.')
"""),
        md("## Step 8 — **The verdict**"),
        code("""\
def _at_tau(df, col, tau=0.1):
    if df is None or not len(df):
        raise RuntimeError(
            f'no {col} computed -- the earlier steps produced nothing. '
            'Run NB01 and NB02 to completion first.')
    sub = df[df.tau == tau]
    if not len(sub):
        raise RuntimeError(f'{col} has no row at tau={tau}')
    return float(sub[col].iloc[0])

rho_seed = float(min(ceil_a, ceil_b))          # the weaker ceiling governs
T_val    = _at_tau(q3, 'T')
dR2      = _at_tau(q4, 'delta_r2')

decision = msc.phase0_decision(rho_seed, T_val, dR2)
decision.update({'ceiling_resnet32x4': ceil_a, 'ceiling_wrn_40_2': ceil_b,
                 'shuffled_control': ctrl, 'tau_reported': 0.1,
                 'note': 'Check the tau-curves in Step 9 before acting on this.'})
msc.write_gate_decision(sess.data_dir, decision, sess.hub)
"""),
        md("""
        ## Step 9 — Does the verdict hold at every τ?

        The verdict above was read at τ=0.1. Confirm it doesn't flip elsewhere.
        Dashed lines are pass thresholds; dotted lines are fail thresholds.
        """),
        code("""\
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for r, g in q1.groupby('run_a'):
    ax[0].plot(g.tau, g.rho_seed, 'o-', label=r.split('-')[1])
ax[0].axhline(0.6, ls='--', c='k', lw=1); ax[0].axhline(0.4, ls=':', c='r', lw=1)
ax[0].set_xlabel('tau'); ax[0].set_ylabel('agreement')
ax[0].set_title('Q1: noise ceiling'); ax[0].legend(); ax[0].grid(alpha=.3)

ax[1].plot(q3.tau, q3['T'], 'o-')
ax[1].fill_between(q3.tau, q3['T_lo'], q3['T_hi'], alpha=.2)
ax[1].axhline(0.7, ls='--', c='k', lw=1); ax[1].axhline(0.5, ls=':', c='r', lw=1)
ax[1].set_xlabel('tau'); ax[1].set_ylabel('T')
ax[1].set_title('Q3: transfer'); ax[1].grid(alpha=.3)

ax[2].plot(q4.tau, q4.delta_r2, 'o-')
ax[2].fill_between(q4.tau, q4.delta_r2_lo, q4.delta_r2_hi, alpha=.2)
ax[2].axhline(0.05, ls='--', c='k', lw=1); ax[2].axhline(0.02, ls=':', c='r', lw=1)
ax[2].set_xlabel('tau'); ax[2].set_ylabel('delta R^2')
ax[2].set_title('Q4: is it new?'); ax[2].grid(alpha=.3)
plt.tight_layout()
msc.save_figure(fig, sess.data_dir, 'phase0_tau_curves', sess.hub)
plt.show()
"""),
        md("## Step 10 — Finish"),
        code(FINISH_CELL),
        md("""
        ---
        ### Now stop and have the conversation

        Write the verdict and your reasoning into the repository before running
        anything else. If it said FULL-PROGRAM or REFRAME, go to NB04. If it said
        PIVOT-STRONG-NEGATIVE, still go to NB04 — you need the full atlas for
        that paper too, you just skip NB13/NB14 at the end.
        """),
    ])


# ===========================================================================
# Atlas training notebooks (NB04 - NB07)
# ===========================================================================
def atlas_nb(num, title, archs, hours, why, workers=6, extra_md=""):
    panel = scope_panel(archs)
    return notebook([
        md(f"""
        # NB{num:02d} — Atlas: {title}

        {HEADER_NOTE}

        {panel}

        ## Why this group is its own notebook

        {why}

        ## Why 3 seeds

        A single training run's accuracy is partly luck. Reporting one number
        without a spread isn't publishable — and more importantly here, the
        seed-to-seed comparison IS the noise ceiling that every transfer number
        in the paper gets divided by.

        ## Architectures in this notebook

        `{'`, `'.join(archs)}`

        ## Running this across accounts

        Set `NUM_WORKERS` to how many accounts you have and give each a different
        `WORKER_ID`. The work splits by estimated GPU-hours, not by run count, so
        every account should finish at roughly the same time even though the
        models differ in cost.

        **This is long-running.** It pushes to HuggingFace every 30 minutes,
        pauses cleanly at 8.5 hours, and resumes exactly where it stopped when
        you start a fresh session and re-run.
        {extra_md}
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session"),
        code(worker_cell("p1", n_runs=len(archs)*3,
                         hours=sum(est_hours(a) for a in archs)*3)),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up"),
        code(SYNC_CELL),
        md("""
        ## Step 4 — See the plan and the time before committing to it

        Prints how long this will actually take, using measured per-epoch times
        from any runs already finished. If the split looks badly unbalanced, or
        the hours are more than you want on one account, change `NUM_WORKERS`
        now rather than discovering it on day three.
        """),
        code(f"""\
ARCHS = {archs!r}
SEEDS = (1, 2, 3)

cfgs = [sess.config(a, seed=s) for a in ARCHS for s in SEEDS]
run_ids = [c['run_id'] for c in cfgs]
"""),
        code(ESTIMATE_CELL),
        code(f"""\
display(msc.shard_report(run_ids, NUM_WORKERS, mode='cost'))
plan = sess.plan(run_ids, title='NB{num:02d} plan')
"""),
        md("""
        ## Step 5 — Train

        Safe to stop at any moment. Re-run in a fresh session to continue.
        """),
        code("""\
summaries = sess.run_all(cfgs, title='atlas training')

import pandas as pd
pd.DataFrame([{k: s.get(k) for k in
               ('run_id', 'arch', 'seed', 'best_accuracy', 'reference_accuracy',
                'accuracy_gap_vs_reference', 'recipe_ok', 'num_epochs_run',
                'total_energy_kwh')}
              for s in summaries if s.get('status') == 'completed'])
"""),
        md("""
        ## Step 6 — Progress across ALL accounts

        Not just this one — this reads the shared record on HuggingFace, so you
        can see how the whole team is doing.
        """),
        code(f"""\
import pandas as pd
want = set(run_ids)
state = sess.registry.latest()
rows = [{{'run_id': r, 'state': state.get(r, {{}}).get('state', 'not started'),
         'accuracy': state.get(r, {{}}).get('best_accuracy'),
         'by': state.get(r, {{}}).get('account'),
         'mine': r in plan.mine}} for r in sorted(want)]
prog = pd.DataFrame(rows)
display(prog)
n_done = int((prog.state == 'completed').sum())
print(f'\\n{{n_done}} of {{len(prog)}} runs finished across all accounts '
      f'({{n_done/max(1,len(prog))*100:.0f}}%)')
"""),
        md("## Step 7 — Finish"),
        code(FINISH_CELL),
    ])


# ===========================================================================
# NB08 -- atlas measurement
# ===========================================================================
def nb08():
    return notebook([
        md(f"""
        # NB08 — Atlas: measure every trained model

        **Up to 45 models · ~25 GPU-hours · shardable across 6 accounts ·
        inference only**

        {HEADER_NOTE}

        ## What this does

        Exactly what NB02 did, but for the whole atlas instead of four pilot
        models. Per model: attach early-exit points (main network frozen), run
        all 20 compute settings over all 10,000 test images plus a 5,000-image
        training slice, compute the difficulty scores, write the tables.

        ~30–40 minutes per model. Nothing is trained except the small read-off
        heads.

        ## Why it's separate from the training notebooks

        Two reasons. It's cheap and re-runnable, so separating it means a bug in
        the measurement code costs 30 minutes rather than 3 hours. And it can
        start as soon as *any* model finishes training — you don't have to wait
        for the whole atlas.

        ## Safe to re-run

        Models already measured are skipped instantly.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session"),
        code(worker_cell("p1", n_runs=45, hours=45*0.6,
                         note="Measurement is inference-only: ~30-40 min per model.")),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up"),
        code(SYNC_CELL),
        md("""
        ## Step 4 — Which models are ready?

        Only fully-trained models. A truncated run would produce measurements
        from an under-trained network, which are worse than no measurements
        because they look valid.
        """),
        code("""\
import pandas as pd
# Identity comes from the run_id, not from ledger fields. Not every event
# carries arch/seed -- repair_ledger reconstructs a completion from history.csv
# and knows only the id -- so reading them from the ledger yields None.
ready = sess.completed_runs(phase='p1')
rdf = pd.DataFrame(ready)
display(rdf)
print(f'\\n{len(ready)} trained models available to measure')
if len(rdf):
    print(f"{int(rdf.measured.sum())} already measured, "
          f"{int((~rdf.measured).sum())} remaining")
"""),
        md("## Step 5 — Measure this worker's share"),
        code("""\
cfgs = []
for r in ready:
    c = sess.config(r['arch'], seed=int(r['seed']))
    c['run_id'] = r['run_id']
    cfgs.append(c)

# stage='measure': the ledger says 'completed' because TRAINING finished, so
# completion for this stage is decided by whether the per-sample tables exist.
results = sess.run_all(cfgs, fn=sess.oracle, title='atlas measurement',
                       done_fn=sess.measured, stage='measure')
for x in results:
    print(f"{x.get('run_id')}: {x.get('status')}")

n_done = sum(1 for c in cfgs if sess.measured(c['run_id']))
print(f'\\n{n_done}/{len(cfgs)} of this account\\'s runs now measured')
"""),
        md("""
        ## Step 6 — Row-alignment audit

        Every table must describe the same images in the same order. Two tables
        that disagree cannot be compared — and comparing them anyway produces
        believable nonsense.
        """),
        code("""\
import pandas as pd
rows = []
for d in sorted(sess.runs_dir.iterdir()) if sess.runs_dir.exists() else []:
    ps = d / 'per_sample'
    meta = msc.read_json(ps / 'meta.json', default={}) or {}
    f = ps / 'test.parquet'
    if f.exists():
        df = pd.read_parquet(f, columns=['sample_order_hash'])
        rows.append({'run_id': d.name, 'arch': meta.get('arch'),
                     'rows': len(df),
                     'order_fingerprint': df['sample_order_hash'].iloc[0][:16]})
align = pd.DataFrame(rows)
if not len(align):
    print('No per-sample tables yet -- nothing measured on this account.')
    print('Either the atlas is still training, or another worker owns these runs.')
else:
    display(align)
    k = align.order_fingerprint.nunique()
    print(f'\\n{k} distinct ordering(s) across {len(align)} tables -> '
          f"{'OK' if k == 1 else 'MISALIGNED -- STOP AND INVESTIGATE'}")
    msc.save_analysis(sess.data_dir, 'per_sample_alignment', align, sess.hub)
"""),
        md("## Step 7 — Finish"),
        code(FINISH_CELL),
        md("---\n**Next: NB09–NB12**, the analysis. Turn the GPU off for those."),
    ])


# ===========================================================================
# Analysis notebooks NB09-NB12
# ===========================================================================
def analysis_header(num, title, question, plain, why, minutes):
    return md(f"""
        # NB{num:02d} — {title}

        **CPU only — turn the accelerator OFF. ~{minutes} minutes. One account.**

        {HEADER_NOTE}

        ## The question

        > {question}

        ## In plain English

        {plain}

        ## Why it matters

        {why}
        """)


ANALYSIS_SESSION = """\
ACCOUNT = 'acct1'      # <<< CHANGE ME

sess = msc.Session(account=ACCOUNT, phase='analysis', dataset='cifar100',
                   enable_hf=True)
# Metrics and per-sample tables only -- checkpoints excluded. Fast.
sess.sync_state(include_checkpoints=False, verbose=True)

import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Inventory: what do we actually have to work with?
runs = {}
for d in sorted(sess.runs_dir.iterdir()) if sess.runs_dir.exists() else []:
    ps = d / 'per_sample'
    m = msc.read_json(ps / 'meta.json', default=None)
    if m and (ps / 'test.parquet').exists():
        runs[d.name] = m
budgets = {r: sess.budgets(m['arch']) for r, m in runs.items()}

inv = pd.DataFrame([{'run_id': k, 'arch': v['arch'], 'family': v['family'],
                     'seed': v['seed'],
                     'order': v['sample_order_hash'][:10]} for k, v in runs.items()])
if len(inv):
    inv = inv.sort_values(['family', 'arch', 'seed'])
    display(inv)
    print(f"\\n{len(runs)} measured models   "
          f"{inv.order.nunique()} distinct image orderings (must be 1)")
else:
    # Not an error -- it means the measurement notebook has not run yet on any
    # model this account can see.
    trained = [d.name for d in sorted(sess.runs_dir.iterdir())
               if (d / 'summary.json').exists()] if sess.runs_dir.exists() else []
    print('No measured models found.')
    print()
    if trained:
        print(f'{len(trained)} run(s) have finished TRAINING but not MEASUREMENT:')
        for t in trained[:10]:
            print(f'  {t}')
        print()
        print('-> Run NB02 (Phase 0) or NB08 (atlas) to produce the per-sample')
        print('   tables, then re-run this notebook.')
    else:
        print('-> No completed runs at all. Run sess.sync_state(), or finish')
        print('   the training notebooks first.')
"""


def nb09():
    return notebook([
        analysis_header(9, "Q1 — The noise ceiling",
                        "When we train the same model twice, do the two copies "
                        "agree about which images are hard?",
                        "This is our reality check and our measuring stick. Two "
                        "identical architectures trained from different random "
                        "starting points should, if compute-need is real, mostly "
                        "agree. How much they agree tells us how precise our "
                        "instrument is.",
                        "Every later number gets divided by this. Suppose a ResNet "
                        "and a ViT agree 60% of the time — is that a lot? If two "
                        "copies of the same ResNet agree 95%, then 60% is a big "
                        "drop and architecture matters. If two copies only agree "
                        "62%, then 60% is essentially perfect and architecture "
                        "barely matters. **Same number, opposite conclusions.** "
                        "The adjacent literature reports raw correlations without "
                        "this, which is why their numbers are hard to read.",
                        5),
        code(bootstrap_cell()),
        md("## Step 1 — Load"),
        code(ANALYSIS_SESSION),
        md("""
        ## Step 2 — Compute a ceiling per architecture

        Needs at least 2 seeds. Architectures with only one are excluded from the
        transfer analysis entirely — without a ceiling their numbers can't be
        interpreted.
        """),
        code("""\
by_arch = {}
for r, m in runs.items():
    by_arch.setdefault(m['arch'], []).append((m['seed'], r))

q1_all, ceilings = [], {}
for arch, lst in sorted(by_arch.items()):
    lst.sort()
    if len(lst) < 2:
        print(f'  {arch}: only {len(lst)} seed -- no ceiling, excluded from Q3')
        continue
    (s1, r1), (s2, r2) = lst[0], lst[1]
    d = msc.analyse_q1_seed_ceiling(sess.data_dir, r1, r2, budgets[r1], axis='depth')
    d['arch'] = arch
    q1_all.append(d)
    c = float(d.set_index('tau').loc[0.1, 'rho_seed'])
    for _, rr in lst:
        ceilings[rr] = c
    print(f'  {arch:16s} ceiling = {c:.3f}')

q1 = pd.concat(q1_all, ignore_index=True) if q1_all else pd.DataFrame()
if len(q1):
    msc.save_analysis(sess.data_dir, 'q1_seed_ceilings_all', q1, sess.hub)
    msc.atomic_write_json(sess.data_dir / 'analysis' / 'ceilings.json', ceilings)
    sess.hub.hub.enqueue(sess.data_dir / 'analysis' / 'ceilings.json',
                         'analysis/ceilings.json') if sess.hub.enabled else None
"""),
        md("## Step 3 — The ceiling table, across all τ"),
        code("""\
if not len(q1):
    print('No noise ceilings computed. Every architecture needs at least TWO')
    print('seeds measured before a ceiling exists -- run NB08 on more runs.')
elif len(q1):
    display(q1.pivot_table(index='arch', columns='tau', values='rho_seed').round(3))
    print('\\n  >= 0.6 pass | 0.4-0.6 marginal | < 0.4 noise-dominated')
    weak = q1[(q1.tau == 0.1) & (q1.rho_seed < 0.4)].arch.tolist()
    if weak:
        print(f'\\n  WARNING: noise-dominated architectures: {weak}')
        print('  Their transfer numbers cannot be interpreted.')
"""),
        md("""
        ## Step 4 — Plot

        Flat, high lines are good: the measurement is stable regardless of how
        strict we are about confidence.
        """),
        code("""\
if len(q1):
    fig, ax = plt.subplots(figsize=(9, 5))
    for arch, g in q1.groupby('arch'):
        ax.plot(g.tau, g.rho_seed, 'o-', label=arch)
    ax.axhline(0.6, ls='--', c='k', lw=1, label='pass threshold')
    ax.axhline(0.4, ls=':', c='r', lw=1, label='fail threshold')
    ax.set_xlabel('tau (how confident before an answer counts as settled)')
    ax.set_ylabel('seed-to-seed agreement')
    ax.set_title('Q1: how precise is our measurement?')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3)
    plt.tight_layout()
    msc.save_figure(fig, sess.data_dir, 'q1_noise_ceilings', sess.hub)
    plt.show()
"""),
        md("""
        ## Step 5 — The "too hard for anyone" population

        Images where the full model itself isn't confident. Excluded from all
        correlations. Worth reporting on its own: *do different architectures
        agree about which images are genuinely ambiguous?* That's a real
        sub-question and it comes free.
        """),
        code("""\
rows = []
for r, m in runs.items():
    df = msc.load_per_sample(sess.data_dir, r)
    for t in msc.TAU_GRID:
        res = msc.msc_for_run(df, budgets[r], 'depth', t)
        rows.append({'run_id': r, 'arch': m['arch'], 'tau': t,
                     'frac_irreducible': res.frac_irreducible,
                     'mean_msc': float(np.nanmean(res.clean()))})
irr = pd.DataFrame(rows)
if len(irr):
    msc.save_analysis(sess.data_dir, 'irreducible_subpopulation', irr, sess.hub)
    display(irr.pivot_table(index='arch', columns='tau',
                            values='frac_irreducible').round(3))
else:
    print('No measured runs yet.')
"""),
        md("## Step 6 — Finish"),
        code(FINISH_CELL),
    ])


def nb10():
    return notebook([
        analysis_header(10, "Q2 — Are the three dials the same thing?",
                        "If an image needs more depth, does it also need more "
                        "resolution and more numeric precision?",
                        "We can reduce compute three ways: fewer layers, smaller "
                        "images, fewer bits. We measured all three on the same "
                        "models and the same images. Now we ask whether they're "
                        "measuring one underlying thing or three separate things.",
                        "**Nobody has ever asked this.** Every adaptive-inference "
                        "paper picks one dial — nearly always depth — and treats "
                        "it as *the* compute axis. If one shared factor explains "
                        "most of the variation, that assumption is validated and a "
                        "single compute-need number is justified. If it doesn't, "
                        "then results about early-exit depth say nothing about "
                        "precision-adaptive inference, and a lot of published "
                        "generalisation is unwarranted. Either answer is a "
                        "contribution, and the data comes free once the atlas "
                        "exists — the best novelty-per-GPU-hour in the project.",
                        5),
        code(bootstrap_cell()),
        md("## Step 1 — Load"),
        code(ANALYSIS_SESSION),
        md("""
        ## Step 2 — Principal component analysis across the dials

        `pc1_variance` is the fraction of variation explained by a single shared
        factor. Our pre-registered prediction (H2) is ≥ 0.60.

        **Which resolution measurement:** we use `res_proxy` (shrink-then-restore)
        as the primary, because it is defined for **all 15 architectures**.
        MLP-Mixer cannot run at another resolution at all — its token-mixing
        layer is a linear map whose input dimension is the patch count — so
        making native primary would mean measuring one architecture differently
        from the other fourteen, and any cross-architecture claim on this axis
        would then compare two different quantities.

        Native is reported as a robustness check in Step 5, for the 14 that
        support it.
        """),
        code("""\
q2_all = []
for r, m in runs.items():
    if m['seed'] != 1:
        continue                     # one seed per architecture is enough here
    try:
        d = msc.analyse_q2_axis_structure(sess.data_dir, r, budgets[r],
                                          axes=('depth', 'res_proxy', 'precision'))
        d['arch'] = m['arch']; d['family'] = m['family']
        q2_all.append(d)
    except Exception as e:
        print(f'  {r}: {e}')
q2 = pd.concat(q2_all, ignore_index=True) if q2_all else pd.DataFrame()
if not len(q2):
    print('No axis structure computed -- no measured runs found (run NB08).')
if len(q2):
    msc.save_analysis(sess.data_dir, 'q2_axis_structure_all', q2, sess.hub)
    display(q2.pivot_table(index='arch', columns='tau',
                           values='pc1_variance').round(3))
    frac = (q2.pc1_variance >= 0.6).mean()
    print(f'\\nH2 predicts PC1 >= 0.60.')
    print(f'Cells clearing it: {frac:.0%}')
    print('\\n  Mostly above  -> one shared "compute need". Single router justified.')
    print('  Mostly below  -> the dials are different. Depth-only results do not')
    print('                   generalise, and that is a finding worth reporting.')
"""),
        md("""
        ## Step 3 — Which dials agree with which?

        Pairwise correlations. If depth and resolution correlate strongly but
        precision doesn't, that's a more interesting story than a single number.
        """),
        code("""\
if len(q2):
    cols = [c for c in q2.columns if c.startswith('rho_')]
    if cols:
        display(q2[q2.tau == 0.1][['arch'] + cols].round(3))
        long = q2[q2.tau == 0.1][cols].melt(var_name='pair', value_name='rho')
        display(long.groupby('pair').rho.agg(['mean', 'std', 'min', 'max']).round(3))
"""),
        md("## Step 4 — Plot"),
        code("""\
if len(q2):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for arch, g in q2.groupby('arch'):
        ax[0].plot(g.tau, g.pc1_variance, 'o-', label=arch)
    ax[0].axhline(0.6, ls='--', c='k', lw=1, label='H2 threshold')
    ax[0].set_xlabel('tau'); ax[0].set_ylabel('variance explained by PC1')
    ax[0].set_title('Q2: is compute-need one-dimensional?')
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)

    load_cols = [c for c in q2.columns if c.startswith('loading_')]
    if load_cols:
        sub = q2[q2.tau == 0.1].set_index('arch')[load_cols]
        sub.plot(kind='bar', ax=ax[1])
        ax[1].set_ylabel('PC1 loading'); ax[1].set_title('How each dial loads on PC1')
        ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    plt.tight_layout()
    msc.save_figure(fig, sess.data_dir, 'q2_axis_structure', sess.hub)
    plt.show()
"""),
        md("""
        ## Step 5 — Robustness check: native resolution vs the proxy

        For the 14 architectures that *can* run at a smaller input, we measured
        the resolution axis both ways. If the two agree, then using the proxy
        uniformly (Step 2) costs us nothing, and the methodological caveat is
        something we **measured** rather than merely argued about.

        The `native_supported` column records which architectures could do both.
        MLP-Mixer will show `False` — that's the documented limitation, not a
        failure.
        """),
        code("""\
rows = []
core = msc._import_msc_core()
for r, m in runs.items():
    if m['seed'] != 1:
        continue
    df = msc.load_per_sample(sess.data_dir, r)
    axes_here = msc.available_axes(df)
    native_ok = 'res_native' in axes_here
    rec = {'arch': m['arch'], 'native_supported': native_ok,
           'axes_available': ' '.join(axes_here)}
    if native_ok:
        for t in (0.0, 0.1, 0.3):
            a = msc.msc_for_run(df, budgets[r], 'res_native', t).clean()
            b = msc.msc_for_run(df, budgets[r], 'res_proxy', t).clean()
            rec[f'agree_tau{t}'] = core.spearman(a, b)
            if t == 0.1:
                rec['mean_native'] = float(np.nanmean(a))
                rec['mean_proxy'] = float(np.nanmean(b))
    rows.append(rec)
rp = pd.DataFrame(rows)
msc.save_analysis(sess.data_dir, 'q2_resolution_native_vs_proxy', rp, sess.hub)
display(rp.round(3))

ok = rp[rp.native_supported]
if len(ok) and 'agree_tau0.1' in ok:
    m_ = ok['agree_tau0.1'].median()
    print(f'\\nMedian native-vs-proxy agreement: {m_:.3f} '
          f'across {len(ok)} architectures')
    if m_ > 0.9:
        print('  -> The two are near-interchangeable. Using the proxy uniformly')
        print('     is well justified, and we can say so with a number.')
    else:
        print('  -> They differ materially. Report BOTH in the paper and discuss;')
        print('     do not present either as if it were the other.')
n_no = int((~rp.native_supported).sum())
if n_no:
    print(f'\\n{n_no} architecture(s) cannot run at native resolution by')
    print('construction. Stated as a limitation in the model card.')
"""),
        md("## Step 6 — Finish"),
        code(FINISH_CELL),
    ])


def nb11():
    return notebook([
        analysis_header(11, "Q3 — Does compute-need transfer between architectures?",
                        "Does a ResNet agree with a Vision Transformer about which "
                        "images need more computation?",
                        "**This is the main question of the project.** We take "
                        "every pair of architectures and measure how much they "
                        "agree — then divide by the noise ceiling from NB09, so "
                        "the number means 'how much of the achievable agreement "
                        "did we actually get' rather than a raw correlation of "
                        "unknown scale.",
                        "If compute-need transfers, a big teacher model can tell a "
                        "small student how much effort each image deserves — and "
                        "that's a useful method. If it doesn't transfer, a growing "
                        "line of teacher-guided adaptive-inference work rests on a "
                        "false premise, and demonstrating that clearly is the "
                        "stronger paper.",
                        15),
        code(bootstrap_cell()),
        md("## Step 1 — Load"),
        code(ANALYSIS_SESSION),
        md("""
        ## Step 2 — Load the ceilings from NB09

        Run NB09 first. Without ceilings, transfer numbers can't be interpreted.
        """),
        code("""\
ceilings = msc.read_json(sess.data_dir / 'analysis' / 'ceilings.json', default=None)
assert ceilings, 'No ceilings found. Run NB09 first.'
print(f'{len(ceilings)} runs have a noise ceiling')

seed1 = msc.representative_runs(runs, require=ceilings)
_missing = sorted(set(msc.ZOO) - set(seed1))
if _missing:
    print(f'[NOTE] {len(_missing)} architecture(s) absent from the transfer '
          f'matrix (no measured seed pair): {", ".join(_missing)}')
    print('       Run NB08 for those runs to include them. See D-15/D-18.')
fam = {m['arch']: m['family'] for m in runs.values()}
print(f'{len(seed1)} architectures available for the transfer matrix')
"""),
        md("""
        ## Step 3 — The sanity check, on every pair

        Scramble one side, re-measure. Agreement must collapse.

        Run this **before** looking at the real numbers. Misaligned tables
        produce entirely believable results.

        **"Collapse" needs a number, and eyeballing one is how you build an
        alarm that cries wolf.** A shuffle doesn't leave exactly zero — the
        leftover correlation wobbles, with standard deviation `1/√n` ≈ 0.013
        here. Across ~78 pairs you should *expect* two or three draws sitting
        2–3 wobbles out. That is arithmetic, not a bug.

        So a pair is flagged only when the leftover correlation is both
        **impossible under shuffling** (|z| > 5) **and large enough to matter**
        (|ρ| > 0.10). A real alignment bug doesn't squeak past that pair of
        conditions — it lands near ρ ≈ 0.6, roughly 45σ out. The earlier
        version of this check used a flat `|T| < 0.05` and failed a healthy
        pair at 2.6σ; see **D-17**.
        """),
        code("""\
import itertools
pairs = list(itertools.combinations(sorted(seed1), 2))
print(f'{len(pairs)} architecture pairs -- testing every one\\n')

ctrl = []
for a, b in pairs:
    c = msc.analyse_q3_shuffled_control(sess.data_dir, seed1[a], seed1[b],
                                        ceilings, budgets, tau=0.1)
    c.update({'arch_a': a, 'arch_b': b})
    ctrl.append(c)
ctrl = pd.DataFrame(ctrl)
if not len(ctrl):
    print('No architecture pairs available yet. Q3 needs at least two')
    print('architectures measured AND a noise ceiling for each (NB09).')
else:
    # Save BEFORE asserting. If the control ever does fail we want the evidence
    # sitting on HuggingFace to diagnose from, not an exception and an empty
    # analysis folder.
    msc.save_analysis(sess.data_dir, 'q3_shuffled_control', ctrl, sess.hub)

    worst = ctrl.reindex(ctrl.z.abs().sort_values(ascending=False).index)
    print('Ten largest deviations (everything else is smaller):')
    display(worst[['arch_a', 'arch_b', 'spearman_raw', 'z', 'n', 'passed']]
            .head(10).round(4))
    zmax, rfloor = ctrl.z_max.iloc[0], ctrl.rho_floor.iloc[0]
    print(f'\\n{ctrl.passed.sum()}/{len(ctrl)} pairs pass')
    print(f'largest |rho| = {ctrl.spearman_raw.abs().max():.4f} '
          f'(|z| = {ctrl.z.abs().max():.1f})')
    print(f'bug threshold = |z| > {zmax:.0f} AND |rho| > {rfloor:.2f}')
    print(f'null SD at this n is ~{ctrl.null_sd.mean():.4f}, so |z| of 2-3 on a '
          f'few of {len(ctrl)} pairs is expected and is NOT a defect.')
    assert ctrl.passed.all(), (
        'Scrambled control FAILED -- shuffling did not destroy the correlation, '
        'so the per-image tables are not paired by sample_idx. Bug, not a finding.')
    print('\\nControl passed -- the transfer numbers below are trustworthy.')
"""),
        md("""
        ## Step 4 — The transfer matrix

        `pair_type` is the grouping that makes our prediction testable. H3 says
        the ordering should be:

        **within-family > across-CNN-family > CNN→Transformer**

        because architectures with similar "thinking styles" should agree more.
        """),
        code("""\
TOKEN = {'vit', 'mixer'}
q3 = msc.analyse_q3_transfer(
    sess.data_dir, [(seed1[a], seed1[b]) for a, b in pairs],
    ceilings, budgets, axis='depth', taus=(0.1,), n_boot=1000)

inv_arch = {v: k for k, v in seed1.items()}
q3['arch_a'] = q3.run_a.map(inv_arch); q3['arch_b'] = q3.run_b.map(inv_arch)
q3['fam_a'] = q3.arch_a.map(fam);      q3['fam_b'] = q3.arch_b.map(fam)

def pair_type(r):
    if r.fam_a == r.fam_b:
        return '1_within-family'
    if (r.fam_a in TOKEN) != (r.fam_b in TOKEN):
        return '3_CNN->transformer'
    if r.fam_a in TOKEN and r.fam_b in TOKEN:
        return '4_transformer->transformer'
    return '2_across-CNN-family'
q3['pair_type'] = q3.apply(pair_type, axis=1)

msc.save_analysis(sess.data_dir, 'q3_transfer_matrix', q3, sess.hub)
if len(q3):
    display(q3.groupby('pair_type')[['T', 'spearman_raw', 'jaccard_top10']]
              .agg(['mean', 'std', 'count']).round(3))
else:
    print('No pairs to compare yet.')
print('\\nH3 predicts T decreases down this table.')
print('  T ~ 1.0 -> transfer as complete as measurement allows')
print('  T < 0.5 -> architecture-specific; the field assumption is wrong')
"""),
        md("## Step 5 — The heatmap (a main figure of the paper)"),
        code("""\
archs = sorted(seed1)
M = pd.DataFrame(np.nan, index=archs, columns=archs)
for _, r in q3.iterrows():
    M.loc[r.arch_a, r.arch_b] = r['T']
    M.loc[r.arch_b, r.arch_a] = r['T']
np.fill_diagonal(M.values, 1.0)

fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(M.values.astype(float), vmin=0, vmax=1.1, cmap='viridis')
ax.set_xticks(range(len(archs))); ax.set_xticklabels(archs, rotation=90)
ax.set_yticks(range(len(archs))); ax.set_yticklabels(archs)
for i in range(len(archs)):
    for j in range(len(archs)):
        v = M.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                    color='w' if v < 0.7 else 'k')
ax.set_title('Q3: transfer T(A,B) — 1.0 means "as much agreement as our\\n'
             'measurement precision permits"')
plt.colorbar(im, ax=ax, shrink=.8)
plt.tight_layout()
msc.save_figure(fig, sess.data_dir, 'q3_transfer_heatmap', sess.hub)
plt.show()
"""),
        md("""
        ## Step 6 — Is the predicted ordering there?

        A box plot by pair type. If H3 holds, the boxes step downward.
        """),
        code("""\
fig, ax = plt.subplots(figsize=(9, 5))
order = sorted(q3.pair_type.unique())
ax.boxplot([q3[q3.pair_type == p]['T'].dropna() for p in order],
           labels=[p.split('_', 1)[1] for p in order])
ax.axhline(0.8, ls='--', c='g', lw=1, label='H3: within-family > 0.8')
ax.axhline(0.6, ls='--', c='r', lw=1, label='H3: CNN->transformer < 0.6')
ax.set_ylabel('T'); ax.set_title('Q3: does transfer depend on architectural similarity?')
ax.legend(); ax.grid(alpha=.3)
plt.xticks(rotation=15)
plt.tight_layout()
msc.save_figure(fig, sess.data_dir, 'q3_transfer_by_pair_type', sess.hub)
plt.show()
"""),
        md("""
        ## Step 7 — Does it hold at every τ?

        A conclusion that survives only one confidence threshold is not a
        conclusion. Slower (bootstraps over the full τ grid).
        """),
        code("""\
def _kind(ab):
    fa, fb = fam[ab[0]], fam[ab[1]]
    if fa == fb:
        return '1_within-family'
    if (fa in TOKEN) != (fb in TOKEN):
        return '3_CNN->transformer'
    if fa in TOKEN and fb in TOKEN:
        return '4_transformer->transformer'
    return '2_across-CNN-family'

# Up to 3 pairs from EACH pair type. The old `pairs[:8]` took the alphabetical
# head, which was eight convnext_femto pairs -- so the "does it hold at every
# tau" check only ever tested one architecture, and the most atypical one at
# that. See D-18.
sample_pairs = msc.stratified_pairs(pairs, _kind, per_kind=3)
print(f'tau curves on {len(sample_pairs)} pairs, stratified by pair type:')
for _k in sorted({_kind(p) for p in sample_pairs}):
    print(f'   {_k}: {sum(1 for p in sample_pairs if _kind(p) == _k)}')
q3t = msc.analyse_q3_transfer(
    sess.data_dir, [(seed1[a], seed1[b]) for a, b in sample_pairs],
    ceilings, budgets, axis='depth', taus=msc.TAU_GRID, n_boot=300)
q3t['pair'] = [f'{inv_arch[a]}->{inv_arch[b]}'
               for a, b in zip(q3t.run_a, q3t.run_b)]
msc.save_analysis(sess.data_dir, 'q3_transfer_tau_curves', q3t, sess.hub)

fig, ax = plt.subplots(figsize=(9, 5))
for p, g in q3t.groupby('pair'):
    ax.plot(g.tau, g['T'], 'o-', label=p)
ax.axhline(0.7, ls='--', c='k', lw=1)
ax.set_xlabel('tau'); ax.set_ylabel('T'); ax.set_title('Q3: stability across tau')
ax.legend(fontsize=7); ax.grid(alpha=.3)
plt.tight_layout()
msc.save_figure(fig, sess.data_dir, 'q3_tau_stability', sess.hub)
plt.show()
"""),
        md("## Step 8 — Finish"),
        code(FINISH_CELL),
    ])


def nb12():
    return notebook([
        analysis_header(12, "Q4 — Is this a new idea, or an old one renamed?",
                        "Is 'how much compute does this image need' just 'how hard "
                        "is this image', which people already measure?",
                        "We fit a predictor using seven existing difficulty scores "
                        "(model confidence, entropy, loss, how often the model "
                        "forgot the image during training, and so on). Then we add "
                        "our measurement and see whether the prediction improves. "
                        "If it doesn't improve, our measurement carries no new "
                        "information.",
                        "**This is the biggest threat to the project and we test it "
                        "head-on rather than hoping nobody asks.** If it turns out "
                        "MSC is fully explained by existing scores, that is still "
                        "publishable and we must not hide it: 'per-sample compute "
                        "requirements are fully explained by classical difficulty "
                        "scores' saves the community effort, and the engineering "
                        "result that follows — use a cheap score instead of an "
                        "expensive oracle — is arguably more useful than the "
                        "method paper.",
                        20),
        code(bootstrap_cell()),
        md("## Step 1 — Load"),
        code(ANALYSIS_SESSION),
        md("""
        ## Step 2 — Is the difficulty battery complete?

        Seven scores. Two of them — **EL2N** and **forgetting events** — are
        *training-set* quantities: they index training images, so they exist on
        the `train_holdout` split and are genuinely undefined on the test set.

        **Q4 therefore runs on `train_holdout`**, where all seven are available.
        Running it on the test split would use 5 of 7, which is an *easier* test
        for MSC and would overstate its irreducibility.

        Phase 0 measured ΔR² = 0.254 on the test split with 5 scores. The
        full-battery number is the one that belongs in the paper.
        """),
        code("""\
battery = ('msp', 'margin', 'entropy', 'ce_loss', 'el2n', 'forget_events',
           'pred_depth')
rows = []
for r in list(runs)[:6]:
    try:
        df = msc.load_per_sample(sess.data_dir, r, 'train_holdout')
    except FileNotFoundError:
        continue
    rows.append({'run_id': r,
                 **{c: (int(df[c].notna().sum()) if c in df.columns else 0)
                    for c in battery}})
cov = pd.DataFrame(rows)
if not len(cov):
    print('No per-sample tables found. Run NB08 (or NB02 for Phase 0) first.')
    missing = list(battery)
else:
    display(cov)
    missing = [c for c in battery if cov[c].sum() == 0]
if missing:
    print(f'\\nMISSING: {missing}')
    print('Q4 will be weaker. These come from training-time instrumentation;')
    print('re-run NB08 with train_dynamics present to recover them.')
else:
    print('\\nAll seven difficulty scores present.')
"""),
        md("""
        ## Step 3 — The test

        Two things reported:

        - **partial correlation** — do two models still agree about compute-need
          *after* removing everything the difficulty scores explain?
        - **ΔR²** — how much better can we predict one model's compute-need when
          we add another model's, on top of the difficulty scores?

        Prediction (H4): ΔR² ≥ 0.05 and partial correlation ≥ 0.3.
        """),
        code("""\
import itertools
ceilings = msc.read_json(sess.data_dir / 'analysis' / 'ceilings.json', default={}) or {}
seed1 = msc.representative_runs(runs, require=ceilings)
pairs = list(itertools.combinations(sorted(seed1), 2))
print(f'{len(seed1)} architectures -> {len(pairs)} pairs (ALL of them; the old '
      f'pairs[:15] cap took the alphabetical head, which was 12 convnext_femto '
      f'pairs + 3 mixer_nano pairs -- see D-18)')

q4_all = []
for a, b in pairs:
    try:
        d = msc.analyse_q4_irreducibility(sess.data_dir, seed1[a], seed1[b],
                                          budgets, axis='depth', taus=(0.1,),
                                          n_boot=500, split='train_holdout')
        d['arch_a'] = a; d['arch_b'] = b
        q4_all.append(d)
    except Exception as e:
        print(f'  {a} -> {b}: {e}')
q4 = pd.concat(q4_all, ignore_index=True) if q4_all else pd.DataFrame()
if not len(q4):
    print('No pairs analysed. Q4 needs at least two measured architectures.')
if len(q4):
    TOKEN_ARCH = {'vit_tiny', 'mixer_nano'}
    q4['has_transformer'] = (q4.arch_a.isin(TOKEN_ARCH)
                             | q4.arch_b.isin(TOKEN_ARCH))
    msc.save_analysis(sess.data_dir, 'q4_irreducibility_all', q4, sess.hub)
    display(q4[['arch_a', 'arch_b', 'partial_spearman', 'r2_difficulty_only',
                'r2_difficulty_plus_msc', 'delta_r2', 'delta_r2_lo',
                'delta_r2_hi']].round(4))
    print(f"\\n  n pairs               = {len(q4)}")
    print(f"  median delta R2       = {q4.delta_r2.median():.4f}   (H4 wants >= 0.05)")
    print(f"  median partial corr   = {q4.partial_spearman.median():.4f}   (H4 wants >= 0.30)")
    print(f"  pairs clearing dR2    = {(q4.delta_r2 >= 0.05).mean():.0%}")
    print(f"  pairs clearing partial= {(q4.partial_spearman >= 0.30).mean():.0%}")
    print('\\n  split by whether the pair involves a non-convolutional model:')
    display(q4.groupby('has_transformer')[['delta_r2', 'partial_spearman']]
              .agg(['median', 'min', 'max', 'count']).round(4))
    print('  Q1 found MSC is measured less reliably in ViT/Mixer, so a lower')
    print('  delta R2 for those pairs is expected -- a noisier measurement')
    print('  carries less unique information. Report the split, not just the')
    print('  pooled median, or the two effects get confounded.')
"""),
        md("## Step 4 — Plot"),
        code("""\
if len(q4):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    lbl = q4.arch_a + '->' + q4.arch_b
    ax[0].barh(lbl, q4.delta_r2,
               xerr=[q4.delta_r2 - q4.delta_r2_lo, q4.delta_r2_hi - q4.delta_r2])
    ax[0].axvline(0.05, ls='--', c='g', lw=1, label='H4 pass')
    ax[0].axvline(0.02, ls=':', c='r', lw=1, label='reframe below this')
    ax[0].set_xlabel('delta R^2 (information beyond difficulty scores)')
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].tick_params(labelsize=7)

    ax[1].scatter(q4.r2_difficulty_only, q4.r2_difficulty_plus_msc)
    lo = float(min(q4.r2_difficulty_only.min(), q4.r2_difficulty_plus_msc.min()))
    hi = float(max(q4.r2_difficulty_only.max(), q4.r2_difficulty_plus_msc.max()))
    ax[1].plot([lo, hi], [lo, hi], 'k--', lw=1, label='no improvement')
    ax[1].set_xlabel('R^2 using difficulty scores alone')
    ax[1].set_ylabel('R^2 adding our measurement')
    ax[1].set_title('Points above the line = we add information')
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    msc.save_figure(fig, sess.data_dir, 'q4_irreducibility', sess.hub)
    plt.show()
"""),
        md("""
        ## Step 5 — Read the verdict honestly

        Whichever way this comes out, it determines the shape of the paper.
        """),
        code("""\
if len(q4):
    med = float(q4.delta_r2.median())
    if med >= 0.05:
        print('VERDICT: MSC carries information beyond classical difficulty scores.')
        print('  -> The construct is new. Keep the multi-axis oracle. Full paper.')
    elif med >= 0.02:
        print('VERDICT: Marginal. Some information beyond difficulty, not much.')
        print('  -> Report honestly, temper the novelty claim.')
    else:
        print('VERDICT: MSC is largely explained by existing difficulty scores.')
        print('  -> REFRAME. This is a real, useful, citable finding: cheap scores')
        print('     suffice for compute routing. Do NOT hide it. The method section')
        print('     becomes "use a difficulty score", which is a BETTER engineering')
        print('     result than an expensive multi-axis oracle.')
"""),
        md("## Step 6 — Finish"),
        code(FINISH_CELL),
    ])


# ===========================================================================
# NB13 / NB14 -- the method
# ===========================================================================
def nb13():
    return notebook([
        md(f"""
        # NB13 — The method: MSC-KD

        **~120 GPU-hours · shardable across 6 accounts**

        {HEADER_NOTE}

        ## Only run this if NB11 said transfer works

        If transfer failed, the paper is the negative result and this is wasted
        compute. Check `analysis/q3_transfer_matrix.csv` first.

        ## What we're building

        A small "student" network that, besides doing its job, predicts **how
        much computation each image needs** — learned from a big "teacher"
        network's measurements.

        At deployment: the student looks at an image briefly, decides how much
        more thinking it deserves, and spends accordingly.

        ## Three ideas that make it defensible

        **1. Three loss terms, two knobs.** Do the task, imitate the teacher's
        answers, imitate the teacher's compute assessment. An earlier version of
        this project had seven terms and six knobs, which no realistic experiment
        budget can justify and which reads to a reviewer as "we tried everything".

        **2. The prediction is forced to make sense by construction.** "Enough
        compute at 80%" must imply "enough at 100%". Rather than adding a penalty
        that discourages violations, we use a head shape that makes them
        *impossible*. It can't be violated, adds no knob, and can't be traded off
        against the other terms.

        **3. The decision is made early and cheaply.** The compute-need predictor
        reads the *earliest* features. A router that needs deep features to decide
        not to compute deep features saves nothing.

        ## Run the scrambled version FIRST

        Set `SHUFFLE_ABLATION = True`, run it, then set it to `False` and run
        again.

        The scrambled version gives the student **deliberately wrong** compute
        targets. If it performs just as well, then the compute signal isn't doing
        anything — the loss term is just acting as a generic regulariser, and our
        central claim is wrong.

        Better to find that out now than after writing the paper.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session"),
        code(worker_cell("p3")),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up"),
        code(SYNC_CELL),
        md("""
        ## Step 4 — Choose teacher and students

        Default pairs are the standard benchmark pairs from the distillation
        literature, so published baseline numbers can be cited rather than
        re-run.

        Sweeping student size tests our stated mechanism. We claim small models
        are badly calibrated, so their own confidence is a poor guide — which
        predicts the advantage should **grow** as students get smaller. If it
        doesn't, the mechanism claim is wrong even if the method wins, and we
        have to say so.
        """),
        code("""\
TEACHER_ARCH = 'resnet32x4'
TEACHER_SEED = 1
STUDENTS     = ['resnet8x4', 'resnet20', 'vgg8']   # decreasing capacity
SEEDS        = (1, 2, 3)
ALPHA, BETA, TEMPERATURE = 1.0, 1.0, 4.0
TAU          = 0.1
EPOCHS       = 240

# Both arms, in one run. The control (scrambled targets) goes first so that if
# it trains as well as the real thing you find out before spending the rest.
# This used to be a single flag you had to flip and re-run by hand, which is
# why the real arm kept not existing when NB14 asked for it.
ARMS = [True, False]        # True = scrambled control, False = the method

teacher_run = msc.make_run_id('p1', TEACHER_ARCH, 'cifar100', 'base', TEACHER_SEED)
tstate = sess.registry.latest().get(teacher_run, {})
print(f'teacher {teacher_run}: {tstate.get("state")} acc={tstate.get("best_accuracy")}')
assert tstate.get('state') == 'completed', 'Train the teacher first (NB04).'

sess.sync_state(run_ids=[teacher_run], include_checkpoints=True, verbose=True)

def arm_cfgs(shuffled):
    m = 'mscKDshuf' if shuffled else 'mscKD'
    return [sess.config(a, seed=s, method=f'{m}-from-{TEACHER_ARCH}',
                        num_epochs=EPOCHS)
            for a in STUDENTS for s in SEEDS]

for _arm in ARMS:
    _c = arm_cfgs(_arm)
    _done = sum(1 for c in _c if sess.trained(c['run_id']))
    print(f"{'SCRAMBLED control' if _arm else 'REAL method   '}: "
          f"{_done}/{len(_c)} already done")
all_cfgs = [c for arm in ARMS for c in arm_cfgs(arm)]
display(msc.shard_report([c['run_id'] for c in all_cfgs], NUM_WORKERS,
                         mode='cost'))
"""),
        md("""
        ## Step 5 — Train

        Each run first measures the teacher's compute-need on the training set
        (with augmentation off — the compute-need of a randomly-cropped view
        isn't the compute-need of the image), then trains the student.
        """),
        code("""\
import pandas as pd
results = []

# Whether a run is scrambled is a property of its OWN run_id, not of a global
# flag -- so both arms can train in one pass without the two ever mixing.
for arm in ARMS:
    label = 'SCRAMBLED control' if arm else 'REAL method'
    cfgs = arm_cfgs(arm)

    def _train(cfg, _shuf=arm):
        return msc.train_msc_kd(cfg, sess.hub, sess.registry, teacher_run,
                                TEACHER_ARCH, work_root=sess.work,
                                data_root_out=sess.data_dir,
                                alpha=ALPHA, beta=BETA,
                                temperature=TEMPERATURE, tau=TAU,
                                shuffle_targets=_shuf)

    print(f'\\n{"#"*74}\\n### {label}\\n{"#"*74}')
    # D-31: "done" for this stage means trained AND still compatible. The
    # default predicate (sess.trained) only asks whether a summary exists, so
    # students invalidated by D-28 were filtered out as finished before
    # train_msc_kd -- and its validity check -- could ever run.
    results += sess.run_all(cfgs, fn=_train, done_fn=sess.msckd_valid,
                            title=f'MSC-KD training ({label})')

done = pd.DataFrame([{k: r.get(k) for k in
                      ('run_id', 'arch', 'seed', 'best_accuracy',
                       'shuffled_targets', 'num_epochs_run')}
                     for r in results if r.get('status') == 'completed'])
n_real = sum(1 for c in arm_cfgs(False) if sess.trained(c['run_id']))
print(f'\\n{n_real}/{len(SEEDS)*len(STUDENTS)} REAL-method students trained '
      f'-- NB14 needs all of them')
display(done)
"""),
        md("""
        ## Step 6 — Real vs scrambled

        Only meaningful once you've run both. If the two columns are close, the
        compute signal isn't carrying the benefit.
        """),
        code("""\
import pandas as pd
rows = [{'run_id': r['run_id'], 'arch': r['arch'], 'seed': r['seed'],
         'scrambled': 'shuf' in r['run_id'], 'accuracy': r['accuracy']}
        for r in sess.completed_runs() if 'mscKD' in r['run_id']]
ab = pd.DataFrame(rows)
if len(ab) and ab.scrambled.nunique() == 2:
    piv = ab.pivot_table(index=['arch', 'seed'], columns='scrambled',
                         values='accuracy')
    piv.columns = ['real_targets', 'scrambled_targets']
    piv['gain_from_real_signal'] = (piv.real_targets -
                                    piv.scrambled_targets) * 100
    display(piv.round(4))
    m = piv.gain_from_real_signal.mean()
    print(f'\\nMean gain from real compute targets: {m:+.2f} points')
    if abs(m) < 0.3:
        print('  WARNING: real and scrambled perform the same. The MSC loss is')
        print('  acting as a regulariser, not carrying compute information.')
        print('  The central mechanism claim would be wrong. Report this.')
    else:
        print('  The compute signal is doing real work.')
else:
    print('Run this notebook with BOTH SHUFFLE_ABLATION=True and False to compare.')
    display(ab)
"""),
        md("## Step 7 — Finish"),
        code(FINISH_CELL),
    ])


def nb14():
    return notebook([
        md(f"""
        # NB14 — Head-to-head at equal compute

        **~5 GPU-hours · one or more accounts**

        {HEADER_NOTE}

        ## The comparison that decides Q5

        Three things on the same axes:

        | | What it is | Why it's here |
        |---|---|---|
        | **B2** | Route by the student's own confidence | **What the field actually deploys.** The real rival |
        | **B10** | Route by our teacher-distilled compute prediction | Ours |
        | **B11** | Route by the student's *true* compute-need, measured after the fact | The ceiling — the best any router could do |

        **The result is not "B10 beats B1".** Beating a non-adaptive baseline is
        trivial. The result is **what fraction of the B2→B11 gap does B10
        close?** B2 is where we are, B11 is where we could be, and the fraction
        in between is the contribution.

        ## Equal compute or it means nothing

        Every comparison is at **matched average FLOPs**. A method that's more
        accurate while using more compute hasn't demonstrated anything. We sweep
        the full accuracy-vs-compute curve for each and compare at equal cost.

        ## An honest limitation we state up front

        Per-image routing gives **no wall-clock speedup under batched inference**
        unless you split the batch by route. A reviewer will raise this, so we
        raise it first: the deployment claim is scoped to batch-size-1, edge and
        streaming settings, where it genuinely holds.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session"),
        code(worker_cell("p3")),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up"),
        code(SYNC_CELL),
        md("## Step 4 — Evaluate every trained student"),
        code("""\
import torch, numpy as np, pandas as pd

TEACHER_ARCH = 'resnet32x4'
TAU = 0.1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

students = [(r['run_id'], r['arch'], r['seed'])
            for r in sess.completed_runs()
            if 'mscKD' in r['run_id'] and 'shuf' not in r['run_id']]
shuffled = [r['run_id'] for r in sess.completed_runs()
            if 'mscKDshuf' in r['run_id']]
print(f'{len(students)} REAL-target students to evaluate')
print(f'{len(shuffled)} scrambled-ablation runs present (not used here)')

if not students:
    print('\\n' + '='*70)
    print('NOTHING TO COMPARE -- and this is almost certainly not a bug.')
    print('='*70)
    if shuffled:
        print(f'You have {len(shuffled)} runs of the SCRAMBLED ablation')
        print('(`mscKDshuf`), which is the control, not the method.')
        print('')
        print('NB13 defaults to SHUFFLE_ABLATION = True so the control runs')
        print('first -- if scrambled targets train as well as real ones, the')
        print('mechanism claim is wrong and there is no point comparing.')
        print('')
        print('TO PROCEED: open NB13, set  SHUFFLE_ABLATION = False,  and run')
        print('it again. That produces the `mscKD` runs this notebook needs.')
    else:
        print('No MSC-KD runs of any kind found. Run NB13 first.')
    print('='*70)

comparisons = []
stale = []
for rid, arch, seed in students:
    print(f'\\n>>> {rid}')
    # D-23/D-25: checkpoints live in runs/{rid}/checkpoints/, not the run root.
    ck = msc.run_layout(sess.work, rid)['checkpoints'] / 'ckpt_best.pt'
    if not ck.exists() and sess.hub.enabled:
        # D-25: repo paths are `runs/{rid}/...` and downloads land in
        # sess.work, not sess.runs_dir. Both were wrong here, so the fallback
        # pull could never have recovered a checkpoint.
        sess.hub.hub.download(sess.work, allow_patterns=[f'runs/{rid}/**'])
    if not ck.exists():
        print('  checkpoint unavailable'); continue

    blob = torch.load(ck, map_location=device, weights_only=False)
    rho = blob['rho']
    student = msc.MSCStudent(msc.build_model(arch, 100), 100, len(rho)).to(device)
    student.load_state_dict(blob['model'])

    cfg = sess.config(arch, seed=seed)
    _, val_loader, _, _, _ = msc.build_loaders(cfg)
    sb = sess.budgets(arch)

    oracle_msc = None
    try:
        df = msc.load_per_sample(sess.data_dir, rid)
        oracle_msc = msc.msc_for_run(df, sb, 'depth', TAU).msc
    except Exception:
        print('  no per-image table for this student -- B11 ceiling unavailable')

    # D-29: skip students whose router predates the D-28 fix, and say so once
    # at the end, instead of aborting the whole notebook on the first one.
    ok, why = msc.msckd_router_ok(sess.work, rid, cfg, sess.data_dir, sess.hub)
    if not ok:
        print(f'  SKIPPED -- {why}')
        stale.append(rid)
        continue

    out = msc.evaluate_routing_methods(student, val_loader, device, rho,
                                       sb['full_flops'], oracle_msc)
    cmp_ = out.get('matched_flops_comparison', {})
    comparisons.append({'run_id': rid, 'student': arch, 'seed': seed,
                        'full_accuracy': out['full_accuracy'], **cmp_,
                        **({'B11_accuracy': out['B11_oracle']['accuracy'],
                            'B11_avg_rho': out['B11_oracle']['avg_rho']}
                           if 'B11_oracle' in out else {})})
    for name, curve in out['curves'].items():
        curve.insert(0, 'run_id', rid); curve.insert(1, 'method', name)
        msc.save_analysis(sess.data_dir, f'curve_{rid}_{name}', curve, sess.hub)

if stale:
    print('\\n' + '='*70)
    print(f'{len(stale)} of {len(students)} students SKIPPED as invalid (D-28).')
    print('Their router was sized from the teacher\\'s budget grid, so the')
    print('weights cannot be reused.')
    print('')
    print('FIX: re-run NB13 with the current library. It detects these')
    print('automatically (D-29) and retrains only the affected students --')
    print('nothing to delete by hand.')
    print('='*70)

cmp_df = pd.DataFrame(comparisons)
msc.save_analysis(sess.data_dir, 'q5_matched_flops', cmp_df, sess.hub)
display(cmp_df.round(4))
"""),
        md("""
        ## Step 5 — Does the advantage grow as students shrink?

        Our stated mechanism predicts it should: smaller models are worse
        calibrated, so their own confidence is a worse guide, so external advice
        helps more.

        If the gap doesn't widen, the mechanism claim is wrong **even if the
        method wins**, and we have to say so in the paper.
        """),
        code("""\
if len(cmp_df) and 'gap_points' in cmp_df.columns:
    order = ['resnet8x4', 'resnet20', 'vgg8']
    g = (cmp_df.groupby('student').gap_points.agg(['mean', 'std', 'count'])
         .reindex([s for s in order if s in set(cmp_df.student)]))
    display(g.round(3))
    params = {}
    for s in g.index:
        params[s] = msc.count_parameters(msc.build_model(s, 100)) / 1e6
    print('\\nstudent size (M params):', {k: round(v, 2) for k, v in params.items()})
    if len(g) >= 2:
        xs = [params[s] for s in g.index]
        ys = list(g['mean'])
        trend = np.polyfit(xs, ys, 1)[0]
        print(f'trend: {trend:+.3f} accuracy points per million parameters')
        print('  negative -> advantage grows as students shrink -> mechanism supported')
        print('  positive -> mechanism claim is NOT supported; say so in the paper')
"""),
        md("## Step 6 — The central figure"),
        code("""\
import matplotlib.pyplot as plt

if not students:
    print('No real-target students yet -- the figure below will be empty.')
    print('See Step 4: run NB13 with SHUFFLE_ABLATION = False first.')

fig, ax = plt.subplots(figsize=(9, 6))
for rid, arch, seed in students:
    for name, style, lbl in (('B2_confidence', '--', 'B2 confidence (field)'),
                             ('B10_msc_kd', '-', 'B10 MSC-KD (ours)')):
        p = sess.data_dir / 'analysis' / f'curve_{rid}_{name}.csv'
        if not p.exists():
            continue
        d = pd.read_csv(p).sort_values('avg_rho')
        ax.plot(d.avg_rho, d.accuracy, style, label=f'{arch} s{seed} — {lbl}')
if len(cmp_df) and 'B11_accuracy' in cmp_df:
    for _, r in cmp_df.iterrows():
        ax.scatter([r.B11_avg_rho], [r.B11_accuracy], marker='*', s=200, zorder=5,
                   label=f'B11 ceiling — {r.student}')
ax.set_xlabel('average compute used (fraction of full)')
ax.set_ylabel('top-1 accuracy')
ax.set_title('Q5: accuracy vs compute\\n'
             'the gap B10 closes between B2 and B11 is the result')
ax.grid(alpha=.3)
if ax.get_legend_handles_labels()[0]:
    ax.legend(fontsize=7)          # only when something was actually plotted
plt.tight_layout()
msc.save_figure(fig, sess.data_dir, 'q5_central_figure', sess.hub)
plt.show()

if len(cmp_df) and 'fraction_of_B2_to_B11_gap_closed' in cmp_df:
    f = cmp_df.fraction_of_B2_to_B11_gap_closed.mean()
    print(f'\\nHEADLINE: MSC-KD closes {f:.0%} of the gap between what the field')
    print('          currently does and the best any router could do.')
"""),
        md("""
        ## Step 7 — Risk-controlled operating point

        Pick a threshold with a *provable* guarantee: "accuracy drops by no more
        than ε, with confidence 1−δ".

        **Read the numbers printed below carefully.** The statistics are
        unforgiving: guaranteeing a 1% drop at 95% confidence needs ~14,979
        calibration images, and CIFAR-100's test set has 10,000. So we calibrate
        on the 5,000-image training holdout at ε=0.03 and say so in the paper.

        This is a design decision, not a bug — and it's much cheaper to make now
        than to discover after the runs.
        """),
        code("""\
print('Calibration images needed for a Hoeffding guarantee:')
for eps in (0.01, 0.02, 0.03, 0.05):
    print(f'   epsilon={eps:.2f}, delta=0.05  ->  {msc.ltt_min_calibration_n(eps, 0.05):,} images')
print('   CIFAR-100 test set: 10,000    our training holdout: 5,000\\n')

EPSILON, DELTA = 0.03, 0.05
rows = []
for rid, arch, seed in students:
    # D-23/D-25: checkpoints live in runs/{rid}/checkpoints/, not the run root.
    ck = msc.run_layout(sess.work, rid)['checkpoints'] / 'ckpt_best.pt'
    if not ck.exists():
        continue
    cfg = sess.config(arch, seed=seed)
    # D-34: Step 5 skips students whose router predates D-28; Step 7 did not,
    # so a stale checkpoint reached learn_then_test_threshold and died there
    # with a bare IndexError. Same guard, same place in the loop.
    ok, why = msc.msckd_router_ok(sess.work, rid, cfg, sess.data_dir, sess.hub)
    if not ok:
        print(f'  {rid}: SKIPPED -- {why}')
        continue
    blob = torch.load(ck, map_location=device, weights_only=False)
    rho = blob['rho']
    student = msc.MSCStudent(msc.build_model(arch, 100), 100, len(rho)).to(device)
    student.load_state_dict(blob['model'])
    _, _, hold_loader, _, _ = msc.build_loaders(cfg)

    S, C = [], []
    student.eval()
    with torch.no_grad():
        for b in hold_loader:
            x, y = b[0].to(device), b[1].to(device)
            lg, sf, _ = student(x)
            S.append(sf.float().cpu().numpy())
            C.append(torch.stack([(l.argmax(1) == y).float() for l in lg], 1)
                     .cpu().numpy())
    S, C = np.concatenate(S), np.concatenate(C)
    gamma = msc.learn_then_test_threshold(S, C, full_accuracy=float(C[:, -1].mean()),
                                          epsilon=EPSILON, delta=DELTA)
    hit = S >= gamma
    route = np.where(hit.any(1), hit.argmax(1), S.shape[1] - 1)
    rows.append({'run_id': rid, 'student': arch, 'seed': seed, 'gamma': gamma,
                 'epsilon': EPSILON, 'delta': DELTA, 'calibration_n': len(S),
                 'accuracy_at_gamma': float(C[np.arange(len(S)), route].mean()),
                 'full_accuracy': float(C[:, -1].mean()),
                 'avg_compute_used': float(np.asarray(rho)[route].mean())})
ltt = pd.DataFrame(rows)
msc.save_analysis(sess.data_dir, 'q5_risk_calibration', ltt, sess.hub)
display(ltt.round(4))
"""),
        md("## Step 8 — Finish"),
        code(FINISH_CELL),
    ])


# ===========================================================================
# NB15 -- paper outputs
# ===========================================================================
def nb15():
    return notebook([
        md(f"""
        # NB15 — Paper outputs

        **CPU only. ~10 minutes.**

        {HEADER_NOTE}

        Assembles every table and figure, totals the energy, and — the part that
        matters for credibility — builds a manifest mapping **every number to the
        run that produced it**.

        The engineering spec's first reproducibility requirement is that every
        number in the paper maps to a run ID. This notebook makes that checkable
        rather than aspirational.
        """),
        code(bootstrap_cell()),
        md("## Step 1 — Session"),
        code("""\
ACCOUNT = 'acct1'      # <<< CHANGE ME
sess = msc.Session(account=ACCOUNT, phase='paper', dataset='cifar100', enable_hf=True)
sess.sync_state(include_checkpoints=False, verbose=True)
import pandas as pd, numpy as np, matplotlib.pyplot as plt
"""),
        md("## Step 2 — Everything that ran"),
        code("""\
ledger = sess.status()
display(ledger)
if len(ledger):
    msc.save_analysis(sess.data_dir, 'run_ledger', ledger, sess.hub)
    print(f"\\n{len(ledger)} runs   "
          f"{int((ledger.state == 'completed').sum())} completed")
"""),
        md("""
        ## Step 3 — Table 1: the atlas

        Accuracy against published references. The `gap` column is the audit —
        anything above 1.0 means that model is under-trained and every
        measurement from it is suspect.
        """),
        code("""\
# D-35: Table 1 is the ATLAS table -- backbone runs only. Iterating every run
# directory also swept in the 18 p3 MSC-KD runs, whose summaries have a
# different schema (no num_parameters, full_flops, reference_accuracy or
# family), so those columns arrived as None and `num_parameters / 1e6` failed.
# Two run types share one directory tree; the reader has to say which it wants.
rows, msckd_rows = [], []
for d in sorted(sess.runs_dir.iterdir()) if sess.runs_dir.exists() else []:
    s = msc.read_json(d / 'summary.json', default=None)
    if not s or s.get('status') != 'completed':
        continue
    if str(s.get('phase') or msc.parse_run_id(d.name).get('phase')) == 'p3':
        msckd_rows.append({k: s.get(k) for k in
                           ('run_id', 'arch', 'teacher', 'method', 'seed',
                            'best_accuracy', 'shuffled_targets', 'alpha',
                            'beta', 'temperature', 'tau', 'num_epochs_run',
                            'total_time_sec', 'config_hash')})
        continue
    rows.append({k: s.get(k) for k in
                 ('run_id', 'arch', 'family', 'seed', 'best_accuracy',
                  'final_accuracy_top5', 'reference_accuracy',
                  'accuracy_gap_vs_reference', 'num_parameters', 'full_flops',
                  'total_time_sec', 'total_energy_kwh', 'total_co2_kg',
                  'num_epochs_run', 'config_hash')})
t1 = pd.DataFrame(rows)
t_msckd = pd.DataFrame(msckd_rows)
print(f'{len(t1)} backbone runs, {len(t_msckd)} MSC-KD runs')
if not len(t1):
    print('No completed backbone runs found. Nothing to tabulate yet.')
if len(t1):
    # Coerce rather than assume: one malformed summary should not take the
    # whole table down.
    for _c in ('num_parameters', 'full_flops', 'best_accuracy'):
        t1[_c] = pd.to_numeric(t1[_c], errors='coerce')
    t1['params_M'] = (t1.num_parameters / 1e6).round(2)
    t1['GFLOPs'] = (t1.full_flops / 1e9).round(3)
    t1['acc_pct'] = (t1.best_accuracy * 100).round(2)
if len(t_msckd):
    msc.save_analysis(sess.data_dir, 'table_msckd_runs', t_msckd, sess.hub)
    display(t_msckd)
    t1 = t1.sort_values(['family', 'arch', 'seed'])
    display(t1[['run_id', 'arch', 'seed', 'acc_pct', 'reference_accuracy',
                'accuracy_gap_vs_reference', 'params_M', 'GFLOPs',
                'total_energy_kwh']])
    msc.save_analysis(sess.data_dir, 'table1_atlas', t1, sess.hub)

    agg = (t1.groupby('arch').agg(acc_mean=('acc_pct', 'mean'),
                                  acc_std=('acc_pct', 'std'),
                                  n_seeds=('seed', 'count'),
                                  params_M=('params_M', 'first'),
                                  GFLOPs=('GFLOPs', 'first')).round(3))
    print('\\nmean +/- std across seeds (single-seed numbers do not go in a paper):')
    display(agg)
    msc.save_analysis(sess.data_dir, 'table1_atlas_aggregated',
                      agg.reset_index(), sess.hub)
"""),
        md("""
        ## Step 4 — Training telemetry summary

        Everything we recorded per epoch, summarised. This is what lets you
        answer "why was that architecture slow?" months later.
        """),
        code("""\
import re
rows = []
for d in sorted(sess.runs_dir.iterdir()) if sess.runs_dir.exists() else []:
    h = d / 'metrics' / 'epochs.csv'
    if not h.exists():
        continue
    try:
        df = pd.read_csv(h)
    except Exception:
        continue
    if df.empty:
        continue
    rec = {'run_id': d.name, 'epochs': len(df)}
    # D-36: these must be REAL HISTORY_FIELDS names. Three were invented --
    # `throughput_img_s`, `gpu_util_mean_pct`, `gpu_temp_max_c`. The schema has
    # `throughput_train_img_s`, and GPU columns are PER DEVICE (`gpu0_*`,
    # `gpu1_*`) because a dual-T4 session has two. The build loop silently
    # skipped them (`if c in df.columns`) and the display below then asked for
    # one by name and raised KeyError.
    for c, agg in (('epoch_time_sec', 'median'),
                   ('throughput_train_img_s', 'median'),
                   ('dataload_frac', 'median'),
                   ('peak_vram_mb', 'max'),
                   ('grad_norm_mean', 'median'),
                   ('update_to_weight_ratio', 'median'),
                   ('step_time_p99_ms', 'median'),
                   ('nan_or_inf_batches', 'sum'),
                   ('cumulative_energy_kwh', 'max')):
        if c in df.columns:
            rec[c] = float(getattr(df[c], agg)())
    # Average utilisation and peak temperature across whatever devices this
    # session actually had, rather than assuming a single GPU.
    for stem, agg, out in (('util_mean_pct', 'mean', 'gpu_util_mean_pct'),
                           ('temp_max_c', 'max', 'gpu_temp_max_c')):
        cols = [c for c in df.columns
                if re.fullmatch(rf'gpu\\d+_{stem}', c) and df[c].notna().any()]
        if cols:
            rec[out] = float(getattr(df[cols].mean(axis=1), agg)())
    rows.append(rec)
tel = pd.DataFrame(rows)
if len(tel):
    display(tel.round(3))
    msc.save_analysis(sess.data_dir, 'table_training_telemetry', tel, sess.hub)
    if 'dataload_frac' in tel:
        starved = tel[tel.dataload_frac > 0.3]
        if len(starved):
            print('\\nThese runs spent >30% of their time waiting for data --')
            print('the GPU was idle. Worth knowing before scaling up:')
            # Ask only for columns that are actually present.
            cols = [c for c in ('run_id', 'dataload_frac', 'gpu_util_mean_pct')
                    if c in starved.columns]
            display(starved[cols])
    if 'nan_or_inf_batches' in tel:
        bad = tel[tel.nan_or_inf_batches > 0]
        if len(bad):
            print('\\nRuns with NaN/Inf batches (silent AMP failures):')
            display(bad[['run_id', 'nan_or_inf_batches']])
"""),
        md("""
        ## Step 5 — Energy and carbon

        Reported as measurement methodology, never claimed as a contribution.
        FLOPs is our primary efficiency metric; FLOP-based proxies underestimate
        real energy by 2–6× because of memory traffic and kernel-launch overhead,
        which is exactly why we sample power directly.
        """),
        code("""\
if len(t1):
    e = t1.groupby('family').agg(
        total_kwh=('total_energy_kwh', 'sum'),
        total_co2_kg=('total_co2_kg', 'sum'),
        total_gpu_h=('total_time_sec', lambda s: s.sum() / 3600.0),
        runs=('run_id', 'count')).round(4)
    display(e)
    print(f"\\nPROJECT TOTAL: {t1.total_energy_kwh.sum():.3f} kWh | "
          f"{t1.total_co2_kg.sum():.3f} kg CO2 | "
          f"{t1.total_time_sec.sum()/3600:.1f} T4-hours")
    msc.save_analysis(sess.data_dir, 'table_energy_accounting',
                      e.reset_index(), sess.hub)
"""),
        md("""
        ## Step 5b — Combined tables across every run

        Concatenates every run's `metrics/epochs.csv` and `metrics/final.csv`
        into two repo-level files. This is the "all combined" view — one CSV you
        can open and see the entire project in, without walking folders.

        - `tables/all_epochs.csv` — every epoch of every run
        - `tables/all_final.csv` — one row per run
        - `tables/atlas_summary.csv` — mean ± std across seeds
        """),
        code("""\
ep_frames, fi_frames = [], []
for d in sorted(sess.runs_dir.iterdir()) if sess.runs_dir.exists() else []:
    e, f = d / 'metrics' / 'epochs.csv', d / 'metrics' / 'final.csv'
    if e.exists():
        try:
            df = pd.read_csv(e)
            df['run_id'] = df.get('run_id', d.name)
            ep_frames.append(df)
        except Exception as ex:
            print(f'  {d.name} epochs: {ex}')
    if f.exists():
        try:
            fi_frames.append(pd.read_csv(f))
        except Exception as ex:
            print(f'  {d.name} final: {ex}')

tdir = msc.ensure_dir(sess.data_dir / 'tables')
if ep_frames:
    all_ep = pd.concat(ep_frames, ignore_index=True, sort=False)
    all_ep.to_csv(tdir / 'all_epochs.csv', index=False)
    print(f'tables/all_epochs.csv : {len(all_ep):,} rows x {len(all_ep.columns)} cols '
          f'from {len(ep_frames)} runs')
    display(all_ep.head(3))
if fi_frames:
    all_fi = pd.concat(fi_frames, ignore_index=True, sort=False)
    all_fi.to_csv(tdir / 'all_final.csv', index=False)
    print(f'tables/all_final.csv  : {len(all_fi)} runs x {len(all_fi.columns)} cols')
    display(all_fi)

    num = all_fi.select_dtypes('number').columns
    summ = (all_fi.groupby('arch')[list(num)].agg(['mean', 'std', 'count'])
            if ('arch' in all_fi.columns and len(all_fi)) else pd.DataFrame())
    if len(summ):
        summ.to_csv(tdir / 'atlas_summary.csv')
        print(f'tables/atlas_summary.csv : mean +/- std across seeds')

if not ep_frames and not fi_frames:
    print('Nothing to combine yet -- no run has written metrics/ on this account.')
    print('Run sess.sync_state() first, or wait for training to produce epochs.')

if sess.hub.enabled and (ep_frames or fi_frames):
    sess.hub.hub.enqueue_dir(tdir, 'tables')
    sess.hub.flush(timeout=600)
"""),
        md("## Step 6 — Every computed statistic"),
        code("""\
adir = sess.data_dir / 'analysis'
found = sorted(p.name for p in adir.glob('*.csv')) if adir.exists() else []
print(f'{len(found)} analysis tables saved:')
for f in found:
    print('  ', f)
figs = sorted(p.name for p in (sess.data_dir / 'paper' / 'figures').glob('*.png')) \\
    if (sess.data_dir / 'paper' / 'figures').exists() else []
print(f'\\n{len(figs)} figures:')
for f in figs:
    print('  ', f)
"""),
        md("""
        ## Step 7 — Provenance manifest

        Every artifact, its size, its checksum, and the run that made it.
        """),
        code("""\
prov = msc.provenance_manifest(sess.data_dir, sess.hub)
print(f'{len(prov)} artifacts tracked')
display(prov.head(30))
"""),
        md("## Step 8 — Model card"),
        code("""\
lines = [
    '# MSC — Minimum Sufficient Compute', '',
    'Artifacts for *Is Compute Difficulty Architecture-Agnostic? Measuring and',
    'Distilling Per-Sample Minimum Sufficient Computation*.', '',
    f'Generated {msc.now_iso()} by msc_lib v{msc.__version__}.', '',
    '## Repositories', '',
    f'- `{msc.HF_REPO}` — everything, one folder per run', '',
    '## What MSC is', '',
    'The smallest cost-normalised configuration at which a network\\'s decision has',
    '*stably settled* to its full-compute decision, defined uniformly over depth,',
    'resolution and precision reduction. Stability means the decision agrees at that',
    'budget **and every larger one** — predictions under compute reduction are not',
    'monotone, so a naive minimum records an accident rather than a property.', '',
    '## Compute grid', '',
    f'- depth: exits at {list(msc.DEPTH_FRACTIONS)} of network depth',
    f'- resolution: {list(msc.RESOLUTIONS)} px, measured natively AND via a',
    '  downsample-upsample proxy (the proxy cost model is labelled idealised)',
    f'- precision: {list(msc.PRECISIONS)}, simulated by fake quantisation;',
    '  cost priced analytically as bits/32, never as measured latency',
    f'- confidence thresholds: tau in {list(msc.TAU_GRID)} — all results are tau-curves',
    '', '## Per-image table schema', '', '```',
    'sample_idx, label,',
    'pred_d1..d5    top1p_d1..d5    top2p_d1..d5     depth',
    'pred_rn1..rn5  top1p_rn1..rn5  top2p_rn1..rn5   resolution (native)',
    'pred_rp1..rp5  top1p_rp1..rp5  top2p_rp1..rp5   resolution (proxy)',
    'pred_q1..q5    top1p_q1..q5    top2p_q1..q5     precision',
    'msp, margin, entropy, ce_loss, el2n, forget_events, pred_depth',
    'sample_order_hash, run_id, split', '```', '',
    'Every table carries `sample_order_hash`. Tables whose hashes differ are not',
    'row-aligned and must not be correlated.', '',
    '## Telemetry recorded per epoch', '',
    'Losses and accuracies; learning rate per group; gradient norm mean/max/p95;',
    'gradient-clip hit rate; weight norm; update-to-weight ratio; AMP scale;',
    'NaN/Inf batch count; epoch/train/eval time; dataload vs compute split;',
    'step-time p50/p90/p99; throughput; VRAM allocated/reserved/peak; GPU',
    'utilisation and temperature; CPU and RAM; free disk; energy in J/kWh and CO2',
    'per epoch and cumulative. Plus raw power samples at 10 Hz, system samples at',
    '1 Hz, and a downsampled per-step trace.', '',
    '## Reproducibility', '',
    '- `config.yaml` frozen at run start, sha256-hashed, asserted on resume',
    '- checkpoints carry optimizer, scheduler, AMP scaler and all four RNG streams',
    '- 3 seeds per headline number, mean +/- std',
    '- every artifact mapped to a run_id in `paper/provenance.csv`',
    '- work split across accounts by a deterministic cost-balanced scheduler;',
    '  each run records which worker produced it', '',
]
if len(t1):
    lines += ['## Atlas results', '',
              t1[['arch', 'seed', 'acc_pct', 'reference_accuracy',
                  'params_M', 'GFLOPs']].to_markdown(index=False), '']
lines += ['## Limitations', '',
          '- Per-image routing gives **no wall-clock speedup under batched',
          '  inference** unless the batch is split by route. The deployment claim',
          '  is scoped to batch-1 / edge / streaming.',
          '- INT4 and INT6 are simulated; no T4 kernel exists to time them.',
          '- The resolution proxy runs at 32 px; its cost is an idealised model.',
          '- Risk control is calibrated at epsilon=0.03 on a 5,000-image holdout,',
          '  because epsilon=0.01 would need ~14,979 calibration images and the',
          '  CIFAR-100 test set has 10,000.',
          '- Energy is measurement methodology, not a contribution.',
          '- T4-only hardware; CIFAR-100 scale.', '']

card = sess.data_dir / 'README.md'
msc.atomic_write_text(card, '\\n'.join(lines))
if sess.hub.enabled:
    sess.hub.hub.enqueue(card, 'README.md')
print('\\n'.join(lines[:45]))
"""),
        md("## Step 9 — Finish"),
        code(FINISH_CELL),
    ])


# ---------------------------------------------------------------------------
def nb16():
    return notebook([
        md(f"""
        # NB16 — Fix: finish the atlas

        {HEADER_NOTE}

        ## What this notebook is for

        The atlas is **one training run and a handful of measurements short of
        complete**, and the shortfall is not random. The work scheduler packs
        runs by estimated cost and places the *cheapest* ones last, so when a
        Kaggle session hits its time limit the small architectures are what
        gets dropped. `wrn_16_2` (cost 1.70) and `vgg8` (cost 1.34) are the two
        least expensive models in the zoo, which is exactly why they are the
        ones missing. See defect **D-15**.

        ## Why it is worth ~3.5 GPU-hours

        `wrn_16_2` currently has **one** measured seed. A noise ceiling needs
        two, so it contributes to **nothing** — not Q1, not Q2, not Q3, not Q4.
        The atlas is being described as fifteen architectures and analysed as
        thirteen.

        | run | state | what finishing it buys |
        |---|---|---|
        | `wrn_16_2` s1 | **not trained** | third seed |
        | `wrn_16_2` s2 | trained, unmeasured | **the ceiling — this is the one that matters** |
        | `wrn_40_1` s3 | trained, unmeasured | 2 seeds → 3, tighter ceiling |
        | `vgg8` s1 | trained, unmeasured | 2 seeds → 3, tighter ceiling |
        | `mixer_nano` s3 | trained, unmeasured | 2 → 3 on a **low-ceiling** model where the extra seed is worth most |

        Measuring `wrn_16_2` s2 alone takes the atlas from 13 architectures to
        14. Training s1 as well takes it to the full 15 with three seeds each.

        ## It finds the gaps itself

        Nothing below is hard-coded. The notebook reads the ledger and the
        per-sample tables and works out what is missing, so it stays correct as
        you run it — and it is safe to re-run at any time. If everything is
        already done it says so and stops.

        ## Cost

        | stage | runs | estimate |
        |---|---|---|
        | train | ≤ 1 | ~0.9 GPU-h |
        | measure | ≤ 5 | ~0.5 GPU-h each |
        | **total** | | **~3.5 GPU-h — one session** |

        Order matters: training comes first, because a run trained here must
        also be measured here.
        """),
        md("## Step 0 — Library"),
        code(bootstrap_cell()),
        md("""
        ## Step 1 — Session

        `NUM_WORKERS = 1` by default: one account does all of it. This is a
        small job — splitting it across accounts costs more in coordination
        than it saves.
        """),
        code(worker_cell("p1", n_runs=6, hours=3.5,
                         note="Gap-filling. Tiny job -- one account is fine.")),
        md("## Step 2 — Dataset"),
        code(DATA_CELL),
        md("## Step 3 — Catch up with HuggingFace"),
        code(SYNC_CELL),
        md("""
        ## Step 4 — Work out what is actually missing

        Two questions per run, and they have different answers:

        - **trained?** — does `summary.json` exist with a full epoch count
        - **measured?** — does `per_sample/test.parquet` exist

        A run can be trained and unmeasured, which is the common case here.
        `train_dynamics.parquet` is written *during training*, so a `per_sample`
        directory that contains only that file is **not** a measured run — that
        is precisely how these five hid.
        """),
        code("""\
import pandas as pd

ZOO_ARCHS = list(msc.ZOO)
SEEDS = (1, 2, 3)
expected = [(a, s) for a in ZOO_ARCHS for s in SEEDS]

rows = []
for arch, seed in expected:
    rid = msc.make_run_id('p1', arch, 'cifar100', 'base', seed)
    rows.append({'arch': arch, 'seed': seed, 'run_id': rid,
                 'trained': sess.trained(rid),
                 'measured': sess.measured(rid)})
state = pd.DataFrame(rows)

to_train   = state[~state.trained]
to_measure = state[state.trained & ~state.measured]

print(f'atlas: {int(state.trained.sum())}/{len(state)} trained, '
      f'{int(state.measured.sum())}/{len(state)} measured\\n')

per_arch = (state.groupby('arch')['measured'].sum()
            .sort_values().rename('measured_seeds').to_frame())
per_arch['in_analysis'] = per_arch.measured_seeds >= 2
print('measured seeds per architecture (>=2 needed for a noise ceiling):')
display(per_arch)
starved = per_arch[~per_arch.in_analysis]
if len(starved):
    print(f'>>> {len(starved)} architecture(s) contribute to NO analysis: '
          f'{", ".join(starved.index)}')

print(f'\\nTO TRAIN   ({len(to_train)}): '
      f'{", ".join(to_train.run_id) if len(to_train) else "none"}')
print(f'TO MEASURE ({len(to_measure)}): '
      f'{", ".join(to_measure.run_id) if len(to_measure) else "none"}')

if not len(to_train) and not len(to_measure):
    print('\\n*** Nothing to do -- the atlas is complete. '
          'Re-run NB09-NB12 to pick up any new seeds. ***')
"""),
        md("""
        ## Step 5 — Train whatever is missing

        Resumable and idempotent, like every other training cell. If a run was
        part-trained by an earlier session it continues from its checkpoint
        rather than restarting (D-19).
        """),
        code("""\
if len(to_train):
    cfgs_t = [sess.config(r.arch, seed=int(r.seed))
              for r in to_train.itertuples()]
    res_t = sess.run_all(cfgs_t, title='gap-fill training')
    for x in res_t:
        print(f"{x.get('run_id')}: {x.get('status')}  "
              f"acc={x.get('best_accuracy')}")
else:
    print('nothing to train')
"""),
        md("""
        ## Step 6 — Measure everything outstanding

        Recomputed from scratch rather than reusing the list from Step 4,
        because Step 5 may have just produced a newly-trained run that also
        needs measuring. Reusing the stale list is how the newly-trained model
        would end up trained-but-unmeasured all over again.
        """),
        code("""\
pending = [r for r in
           (msc.make_run_id('p1', a, 'cifar100', 'base', s)
            for a, s in expected)
           if sess.trained(r) and not sess.measured(r)]
print(f'{len(pending)} run(s) to measure: {pending}\\n')

if pending:
    cfgs_m = []
    for rid in pending:
        meta = msc.parse_run_id(rid)
        c = sess.config(meta['arch'], seed=int(meta['seed']))
        c['run_id'] = rid
        cfgs_m.append(c)
    # stage='measure': the ledger says 'completed' because TRAINING finished,
    # so completion here is decided by whether the tables exist.
    res_m = sess.run_all(cfgs_m, fn=sess.oracle, title='gap-fill measurement',
                         done_fn=sess.measured, stage='measure')
    for x in res_m:
        print(f"{x.get('run_id')}: {x.get('status')}")
else:
    print('nothing to measure')
"""),
        md("""
        ## Step 7 — Did it actually close the gap?

        The question that matters is not "did the cells run" but **"does every
        architecture now have two measured seeds"**, because that is the
        condition for entering the analysis at all.
        """),
        code("""\
final = []
for arch, seed in expected:
    rid = msc.make_run_id('p1', arch, 'cifar100', 'base', seed)
    final.append({'arch': arch, 'seed': seed,
                  'trained': sess.trained(rid),
                  'measured': sess.measured(rid)})
fdf = pd.DataFrame(final)
summary = (fdf.groupby('arch')[['trained', 'measured']].sum()
           .rename(columns={'trained': 'trained_seeds',
                            'measured': 'measured_seeds'}))
summary['has_ceiling'] = summary.measured_seeds >= 2
display(summary)

n_arch = int(summary.has_ceiling.sum())
print(f'\\n{int(fdf.trained.sum())}/{len(fdf)} trained, '
      f'{int(fdf.measured.sum())}/{len(fdf)} measured')
print(f'{n_arch}/{len(summary)} architectures have a noise ceiling '
      f'(>=2 measured seeds)')

if n_arch < len(summary):
    missing = list(summary[~summary.has_ceiling].index)
    msc.log(f'{len(missing)} architecture(s) still contribute to no analysis: '
            f'{missing}. Re-run this notebook in a fresh session -- the most '
            f'likely cause is simply that the 8.5 h limit was reached.', 'ALARM')
else:
    print('\\n*** All 15 architectures now have a noise ceiling. ***')
    print('Next: re-run NB09 -> NB10 -> NB11 -> NB12 (CPU only, ~20 min) so')
    print('the analysis picks up the new seeds, then NB15 for the tables.')
"""),
        md("""
        ## Step 8 — Push and verify

        `finish()` drains the upload queue; the verify cell asks HuggingFace
        whether the files actually landed. Draining is not confirmation — see
        **D-19** and **D-20**.
        """),
        code(FINISH_CELL),
    ])


NOTEBOOKS = {
    "NB00_Setup_And_Verify.ipynb": nb00,
    "NB01_Phase0_Train.ipynb": nb01,
    "NB02_Phase0_Measure.ipynb": nb02,
    "NB03_Phase0_Decision.ipynb": nb03,
    "NB04_Atlas_Train_ResNets.ipynb": lambda: atlas_nb(
        4, "ResNet family",
        ["resnet20", "resnet56", "resnet110", "resnet8x4", "resnet32x4"], 25,
        "These five are the backbone of the whole study. They're the standard "
        "pairs used across the distillation literature, which means our accuracy "
        "numbers can be checked directly against published ones — the cheapest "
        "available proof that our training recipe is correct. They also give us "
        "*within-family* transfer: the case where we expect agreement to be "
        "highest, and therefore the top of the ordering H3 predicts."),
    "NB05_Atlas_Train_WRN_VGG.ipynb": lambda: atlas_nb(
        5, "WideResNets & VGGs",
        ["wrn_40_2", "wrn_16_2", "wrn_40_1", "vgg13", "vgg8"], 25,
        "WideResNets are a different width regime — same family idea, different "
        "shape. VGGs are the interesting ones: convolutional networks **without "
        "skip connections**. They sit between 'same family' and 'completely "
        "different architecture', which is exactly the middle rung of the "
        "ordering H3 predicts. Without them we'd only have two points and no "
        "ordering to test."),
    "NB06_Atlas_Train_Mobile.ipynb": lambda: atlas_nb(
        6, "Mobile architectures", ["mobilenetv2", "shufflenetv2"], 15,
        "Depthwise-separable convolutions — a genuinely different way of "
        "processing an image, and the architectures people actually deploy on "
        "constrained hardware. If compute-need transfers to these, the practical "
        "case for the method is much stronger, because these are the models that "
        "would benefit most from spending less."),
    "NB07_Atlas_Train_Modern.ipynb": lambda: atlas_nb(
        7, "ConvNeXt, ViT & MLP-Mixer",
        ["convnext_femto", "vit_tiny", "mixer_nano"], 45,
        "**These are the most important architectures in the project and the "
        "reason Q3 is interesting at all.** A Vision Transformer processes an "
        "image in a fundamentally different way from a ResNet, and an MLP-Mixer "
        "has almost no built-in assumption about images being spatial. Our "
        "central prediction is that agreement drops when we cross that boundary. "
        "Without these three, the transfer study is 'do ResNets agree with other "
        "ResNets', which is a much weaker paper.\n\n"
        "        They're in a separate notebook because they need a **completely "
        "different training recipe** — AdamW, long warmup, strong augmentation, "
        "label smoothing. Under plain SGD they don't learn at all; accuracy sits "
        "near 1% forever. The library selects the right recipe automatically, but "
        "keeping them separate means a failure here doesn't take the CNNs with it.",
        6),
    "NB08_Atlas_Measure.ipynb": nb08,
    "NB09_Analysis_Q1_NoiseCeiling.ipynb": nb09,
    "NB10_Analysis_Q2_AxisStructure.ipynb": nb10,
    "NB11_Analysis_Q3_Transfer.ipynb": nb11,
    "NB12_Analysis_Q4_Irreducibility.ipynb": nb12,
    "NB13_Method_MSCKD_Train.ipynb": nb13,
    "NB14_Method_Comparison.ipynb": nb14,
    "NB15_Paper_Outputs.ipynb": nb15,
    "NB16_Fix_Gaps.ipynb": nb16,
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
            ok = cur == text
            stale += (not ok)
            print(f"  {name:42s} {'current' if ok else 'STALE — rebuild'}")
        else:
            p.write_text(text, encoding="utf-8")
            nc = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
            nm = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
            print(f"  {name:42s} {nc:2d} code + {nm:2d} md   "
                  f"{p.stat().st_size/1024:.0f} KB")
    if not check:
        print(f"\n  library {LIB.stat().st_size/1024:.0f} KB · "
              f"core {CORE.stat().st_size/1024:.0f} KB · "
              f"{len(NOTEBOOKS)} notebooks → {OUT}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
