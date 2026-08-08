# ImageNet-100 Runbook

**Read [`20_IN100_PORT_PLAN.md`](20_IN100_PORT_PLAN.md) first.** This is the
operating procedure: what to run, in what order, and — more importantly — what
to check after each step and what a wrong answer looks like.

Target machine: **single RTX 4000 Ada (20 GB), 24 cores, 63 GB RAM.**
**Local disk only — no HuggingFace, and no network at run time.**

---

## Step −1 — Install, once, with internet

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The CUDA index URL matters: the RTX 4000 Ada is sm_89 and needs cu121 or newer.
**A CPU-only torch imports fine, passes every self-test, and then trains at
about 1/200th of the speed while reporting entirely plausible numbers.** Check:

```
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then record the environment and fingerprint the zoo:

```
python tools/fetch_assets.py
```

**Nothing is downloaded, and that is correct.** `resnet50(weights=None)` is
Python source inside torchvision — training from scratch fetches no weights.
What this command actually does is pin the thing that *can* silently break you:
it records every package version and fingerprints all eight architectures
(parameter count, FLOPs, K, stage cuts, feature dims). A torchvision upgrade
can change how a backbone decomposes into blocks, which changes the budget
table, which changes ρ — and ρ is a ratio, so every resulting MSC value still
looks reasonable. `--check` compares against the manifest later.

Then **prove** you can run offline, rather than assuming it:

```
python tools/fetch_assets.py --verify-offline
```

This blocks the socket layer outright and then builds every architecture,
prices every budget table, and runs both dry runs. Anything that reaches for
the network raises with the host it wanted. Installing a package is not
offline-readiness, the same way draining an upload queue is not confirmation.

After this, unplug it.

---

## Status: what is ready and what is not

| | |
|---|---|
| ✅ | Library ported. **352** offline self-checks pass, exit code verified. |
| ✅ | Local-only + offline. Nothing uploaded, nothing fetched, nothing deleted. |
| ✅ | Packing tool, verified against the real data (dry run only — not yet packed). |
| ✅ | Zoo registered, dry runs written and wired in. |
| ✅ | **Five notebooks generated** in `notebooks_in100/`, validated clean. |
| ✅ | Throughput measured for 6 of 8 architectures (`benchmark/results/`). |
| ⬜ | **Dataset not packed yet** — NB1 step 3 does it (`RUN_PACKER = True`). |
| ⬜ | **Nothing has trained.** No architecture in this zoo has completed a run. |
| ⬜ | `resnet50` and `vgg16` need re-measuring after D-43; `vit`/`deit` after D-42. |

Every step below is runnable now.

---

## Step 0 — Prove the library is sound, offline

```
python src/msc_lib.py --selftest
echo $?          # must be 0
```

Expect the last three lines to read:

```
  352 checks run, 0 failed

ALL CHECKS PASSED
```

**What to actually look at.** Not the last line — the *counts*. Per **D-37**,
this suite spent the entire CIFAR project unable to fail: a stray tuple unpack
rebound the verdict scalar and roughly 80% of checks could not affect the exit
code. It now runs a canary that deliberately fails, and asserts the failure
registered. So:

- If `checks run` is below 250, a section exited early and the run is void
  regardless of what the verdict says.
- If you see `*** THE HARNESS ITSELF IS BROKEN`, stop. Every line above it is
  meaningless.
- One `[FAIL] D-37: the harness registers a failure  canary -- expected FAIL`
  is **correct and required**. Its absence is the problem.

This takes about 20 seconds and needs no GPU, no network and no dataset.

---

## Step 1 — Pack the dataset

Once. ~20–40 minutes on 24 cores. Needs **25 GB** free.

**Easiest: do it from NB1.** Step 3 of the notebook runs the packer itself,
as a subprocess with live output, using the `DATA_DIR` cell 2 already resolved
and a `SRC_DIR` it auto-detects. Set `RUN_PACKER = True` and re-run that cell.
It defaults to `False` so a 40-minute job never starts by accident.

Or from a terminal, using the `DATA_DIR` cell 2 printed:

```
python tools/pack_imagenet100.py ^
    --src "C:\Users\Administrator\Desktop\New folder" ^
    --out "<the DATA_DIR that cell 2 printed>"
```

**Labels are folder-order.** Sorted WNIDs map to class 0–99 (`n01440764` → 0,
`n01855672` → 99). Arbitrary and deterministic; nothing downstream depends on
matching official ImageNet indices, and the mapping is published in
`manifest.json`.

**Do not hardcode a drive letter.** That was D-44: a default naming `D:\` on a
machine with no D: drive, failing forty lines deep in `pathlib` with a message
that named neither the setting nor the file to change. Cell 2 now chooses the
roomiest existing root and proves it writable by writing and reading back a
probe file.

**What to check.** The tool ends by verifying itself and exits non-zero on any
problem, so a zero exit is meaningful. Read these lines anyway:

```
[DRY]    32 images -> memmap -> read back OK (32 non-zero, mean 115.8, 0.2s)
[VERIFY] n=129395  classes=100  val=10000 train=119395 holdout=15000
[VERIFY] decode failures recorded: <should be 0 or a small number>
[VERIFY] fingerprint 2b6269ef51ff87b2c9e00fa17c44326ce634a67892c9eb550ec518a6dd2d2b6c
```

- **The fingerprint must be `2b6269ef…`.** It is a pure function of the file
  names and the split seeds, so it is reproducible on any machine. A different
  value means the source tree differs from the one this port was designed
  against, and every downstream comparison would be against different images.
- **Decode failures are recorded, not skipped.** Failed images are packed as
  zeros and excluded from every split. A nonzero count is fine; an *unrecorded*
  all-zero image is what the verify step is looking for.
- Interrupted? Just re-run it. Resumable at chunk granularity.

Re-check an existing pack at any time:

```
python tools/pack_imagenet100.py --src ... --out ... --verify
```

---

## Step 2 — Preflight on the real GPU

**This is the step that earns its keep.** On CIFAR the preflight caught D-01a,
D-01b and D-02 before a single GPU-hour was spent — an architecture that could
not run at a resolution the sweep assumed, and a budget table with duplicate
entries that would have made MSC undefined.

It must check, for **every one of the eight architectures**:

| check | what it catches |
|---|---|
| builds, forwards, backprops at 224px | a torchvision decomposition that mis-orders blocks |
| `forward_prefix(x, k)` genuinely stops at stage k | an "early exit" costing full compute, which would make every FLOPs saving fictional |
| an `ExitHead` attaches to every feature | a rank or layout mismatch — Swin's blocks speak NHWC |
| runs natively at **96, 128, 160, 192, 224** | **the expected partial failure**: Swin-T reduces by 32, so its last stage is 3×3 at 96px, smaller than its own attention window |
| depth ρ strictly ascending, distinct, ending at 1.0 | D-01b — MSC is undefined when two budgets cost the same |
| the results root is **writable**, verified by writing and reading back | ten hours of training with nowhere to put it. This replaced the HF write-scope check, and it is the same check against a different store |
| free disk ≥ the run matrix's estimated footprint | running out at run 19 of 24 |
| kill-and-resume equivalence, post-seam per-epoch loss | D-06 |
| **no network is reached** — sockets blocked for the whole preflight | a dependency that quietly fetches on first use, discovered on the day the machine is offline |

**Expect `swin_tiny` to fail native resolution at 96px, possibly 128px.** That
is a recorded fact about the architecture, not a bug — the budget table probes
per resolution and falls back to the analytic cost model for the values that
fail, and per **DC-3** the downsample-upsample **proxy is primary for all eight
architectures anyway**. What would be a bug is that failure going unrecorded.

---

## Step 3 — Phase 0

**4 runs, ~33 GPU-h.** `resnet50` × 2 seeds, `vit_small_p16` × 2 seeds.

Do not skip this and do not treat it as a formality. It is 8% of the programme
and it answers the only question that matters before committing the other 92%.

**Then stop and read the gate**, which is pre-registered in
[`20_IN100_PORT_PLAN.md`](20_IN100_PORT_PLAN.md) §7:

| ρ_seed outcome | action |
|---|---|
| ViT below CNN by > 0.05 | CIFAR finding reproducing → build the full atlas |
| within ±0.05 | **small-data artifact** → retract the CIFAR headline, shrink to the 2×2, write a different paper |
| ViT *above* CNN by > 0.05 | audit the measurement before spending anything |
| either below 0.40 | noise-dominated → coarsen the grid, re-gate |

**Rule 12 applies here specifically.** Row 1 is the outcome that flatters the
existing paper. If you get it, scrutinise it harder than you would row 2 —
check that both architectures cleared the §4.4 acceptance thresholds, that
seed spread is under 2 points, that `nan_or_inf_batches` is 0, and that the two
ceilings were computed on the same number of surviving samples after the τ mask.

---

## Step 4 — The atlas  *(after a Phase 0 verdict)*

24 runs, ~235 GPU-h at 100 epochs, ~10 days continuous.

`EPOCHS` is one named constant (`IN100_EPOCHS`). If the budget binds:

- **Lowering epochs costs accuracy, not validity.** ρ_seed is a rank
  correlation between two seeds of the same recipe; it stays well-defined. What
  weakens is the "these are converged models" claim, which §4.4's replacement
  acceptance test then has to carry alone.
- **Dropping `vgg16` saves 28% of the training budget** for one across-CNN
  data point. It weakens Q3's family ordering and leaves the Q1 headline
  untouched. That is usually the better cut.

---

## Every session, before you stop

```python
sess.finish()
sess.confirm_on_disk(my_run_ids)          # measured=True after NB08
```

`finish()` prints `[SESSION] done`. **That is not confirmation** — it says this
process finished tidying up, which is a fact about this process, not about what
is on disk (rule 10).

`confirm_on_disk` **opens every required artifact.** This is stronger than the
HuggingFace check ever was: `confirm_on_hf` established that a file arrived, and
never looked inside it. Three states:

```
[VERIFY] 3 run(s) on local disk: 1 complete, 2 resumable, 0 at risk  (14.2 GiB)
```

- **complete** — every required artifact present, non-empty and parseable
- **resumable** — `ckpt_last.pt` is there. **Safe to stop.** It resumes at its
  epoch. Being unfinished is the normal state of a paused run (D-20)
- **at risk** — missing, zero-byte, or corrupt

Read the reason, not just the count. Three failure classes, three meanings:

| reported | means |
|---|---|
| `MISSING` | the step never ran, or crashed before writing |
| `EMPTY` | the file was created and the write failed — what an interrupted non-atomic write produces routinely |
| `UNREADABLE` | present, non-empty and **corrupt**. Only found by opening it. A presence check calls this run healthy, and you find out during analysis, weeks later |

### Nothing is deleted, ever

With HuggingFace off there is **no code path that removes a run directory** —
the confirm-then-delete branch is gated on `hub.enabled`, and
`cleanup_local_after_complete` is `False` in the ImageNet recipe. The only thing
that wipes a run is an explicit `force_rerun`.

`[SESSION] LOCAL-ONLY` is **not** a warning. On Kaggle, HF-off meant the work
evaporated at session end and the alarm was correct. Here the disk is the
permanent store.

### Audit everything at any time

```python
for r in sess.completed_runs():
    rep = verify_run_artifacts(sess.work, r["run_id"], measured=True)
    if not rep["ok"]:
        print(r["run_id"], rep["missing_required"], rep["empty"], rep["unreadable"])
```

Worth running before any analysis notebook. D-15 was six runs that were trained
and never measured, discovered late, and it cost one of the fifteen
architectures its entire contribution to the atlas.

---

## When something breaks

**A dry run failed.** Good — that cost a second. The message names the stage
(`at stage 'forward/loss/backward'`). Nothing was claimed and no GPU time was
spent. Fix and re-run.

**`data fingerprint mismatch`.** The run was configured against a different
pack or different splits. Do not override it. Correlating per-sample tables
across two packs aligns them by index and compares different images, and the
result would look entirely plausible.

**`cached budget table for X is INVALID`.** Expected after any change to
resolution, class count or the grid. It rebuilds in seconds. The alternative —
trusting it — yields well-formed numbers describing a network nobody trained.

**`depth costs are not strictly ascending`.** The stage partition is wrong for
that architecture. This is D-01b and it is fatal by design: "the smallest
sufficient budget" is undefined when two budgets cost the same.

**Kaggle only: your fix isn't running.** Kaggle does not re-read `notebooks/`
by itself. The rebuilt `.ipynb` has to be re-uploaded. "I fixed it" and "the fix
is running" are different claims and only the second one matters — this cost a
session on CIFAR (D-26, operational note).

---

## Defect index — where to look when something fails

Every defect found during this port, with the symptom you would actually see.
Full write-ups in [`22_IN100_LAB_NOTEBOOK.md`](22_IN100_LAB_NOTEBOOK.md) §2.

| # | symptom | cause | now |
|---|---|---|---|
| **D-48** | `TypeError: resume_acceptance_test() got an unexpected keyword argument` | a notebook call named a parameter that does not exist; the existence check could not see it | validator now checks notebook call signatures, suggesting the closest real parameter |
| **D-47** | `TypeError: load_checkpoint() missing 2 required positional arguments` / `MSCResult has no len()` | calls named real functions but passed the wrong things; existence guards cannot see arity | AST arity check over every internal call |
| **D-46** | preflight reports `FAILED: HF token` / `imagenet100 present` | HF checks ran in a local-only session; "not packed yet" counted as a failure | HF checks skipped when local; three states (pass/fail/todo); dry runs no longer need the pack |
| **D-45** | `[FLOP] profiler fvcore failed ... using analytic fallback` | fvcore priced the CNNs, the analytic counter priced the transformers — and it cannot see attention | torch's flop_counter preferred; a fallback now RAISES. **Delete `budgets/` and rebuild if you saw this** |
| **D-44** | `FileNotFoundError: [WinError 3] ... 'D:\'` in NB1, in the **bootstrap** cell | a default path named a drive that does not exist; `enforce_offline` made *import* depend on a writable dir | `DATA_DIR`/`MSC_ROOT` default to `None`; `resolve_storage()` picks the roomiest existing root and proves it writable |
| **D-43** | `resnet50` 82 img/s vs `resnet18` 413 — a 5× gap for 2.3× the FLOPs | the benchmark ran with `cudnn.benchmark=False`; every real run has it True | one `set_perf_flags()` serves both |
| **D-42** | `TypeError: build_vit_small() got an unexpected keyword argument` | `build_model` injects `probe_res`; that one builder lacked it | signature guard over every ImageNet builder |
| **D-41** | machine hangs / display freezes during the benchmark | no VRAM cap on a GPU that drives the display → Windows TDR reset | batch ladder deleted, 50% VRAM cap, per-config subprocess, timeout |
| **D-40** | `dataload_frac` high while the loader is idle | GPU-side augmentation counted as data wait | `augment_time_sec` split out and subtracted |
| **D-39** | `AttributeError: module 'msc_lib' has no attribute 'analyse_q1_all'` | six invented function names | the validator checks library names at build time |
| **D-38** | `AttributeError: 'BatchNorm2d' object has no attribute 'out_channels'` | builders guessed at torchvision internals | `feature_dims` derived from a forward probe |
| **D-37** | self-test prints `[FAIL]` then `ALL CHECKS PASSED`, exits 0 | a tuple unpack rebound the verdict scalar | lists + canary + floor |

### Reading the self-test after any change

```
python src/msc_lib.py --selftest
```

Look at the **counts**, not the last line:

```
  352 checks run, 0 failed

ALL CHECKS PASSED
```

- below **250** checks run → a section exited early; the verdict is void
- `*** THE HARNESS ITSELF IS BROKEN` → everything above it is meaningless
- one `[FAIL] D-37: the harness registers a failure  canary` is **required**

### Rebuilding the notebooks after any library change

```
python build_notebooks_in100.py
```

Refuses to write if any notebook names a column, a library function, or a drive
letter that does not exist. All five share one bootstrap cell and one paths
cell, so a fix to either lands in five places from one edit — which is why they
are generated rather than hand-written.
