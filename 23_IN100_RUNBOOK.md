# ImageNet-100 Runbook

**Read [`20_IN100_PORT_PLAN.md`](20_IN100_PORT_PLAN.md) first.** This is the
operating procedure: what to run, in what order, and — more importantly — what
to check after each step and what a wrong answer looks like.

Target machine: **single RTX 4000 Ada (20 GB), 24 cores, 63 GB RAM.**
Notebooks stay Kaggle-compatible; nothing below changes if you move them there
except the paths, which the library detects.

---

## Status: what is ready and what is not

| | |
|---|---|
| ✅ | Library ported. 278 offline self-checks pass, exit code verified. |
| ✅ | Packing tool, verified against the real data (dry run only — not yet packed). |
| ✅ | Zoo registered, dry runs written and wired in. |
| ⬜ | **Notebooks not yet regenerated** (O-23). `build_notebooks.py` still emits the CIFAR set. |
| ⬜ | **Nothing has run on a GPU.** No architecture in this zoo has been built on hardware. |

Steps 0–2 below are runnable now. Step 3 onward waits on O-23.

---

## Step 0 — Prove the library is sound, offline

```
python src/msc_lib.py --selftest
echo $?          # must be 0
```

Expect the last three lines to read:

```
  278 checks run, 0 failed

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

```
python tools/pack_imagenet100.py ^
    --src "C:\Users\Administrator\Desktop\New folder" ^
    --out "D:\msc_data\in100"
```

Then tell the library where it is, either by setting `MSC_IN100_DIR` or by
placing it at the default scratch path. The error message names both if it
cannot find it.

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

## Step 2 — Preflight on the real GPU  *(needs O-23)*

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
| HF token has **write** scope, verified by pushing and re-listing | nine hours of training with nowhere to save it |
| kill-and-resume equivalence, post-seam per-epoch loss | D-06 |

**Expect `swin_tiny` to fail native resolution at 96px, possibly 128px.** That
is a recorded fact about the architecture, not a bug — the budget table probes
per resolution and falls back to the analytic cost model for the values that
fail, and per **DC-3** the downsample-upsample **proxy is primary for all eight
architectures anyway**. What would be a bug is that failure going unrecorded.

---

## Step 3 — Phase 0  *(needs O-23)*

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

## Step 4 — The atlas  *(needs O-23, and a Phase 0 verdict)*

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

## Every session, before you close the tab

```python
sess.finish()
sess.confirm_on_hf(my_run_ids)
```

`finish()` drains the upload queue and prints `[SESSION] done`. **That is not
confirmation** — it says the queue emptied, which is a fact about this process,
not about the repository (rule 10). `confirm_on_hf` asks the repo, per file,
through `resolve`.

Read the three-state summary, not the word "done":

```
[VERIFY] 3 run(s): 1 finished, 2 resumable, 0 at risk
```

- **finished** — `summary.json` is there, nothing left to do
- **resumable** — `ckpt_last.pt` is there. **Safe to close.** It resumes at its
  epoch. Being unfinished is the normal state of a paused run (D-20)
- **at risk** — neither. This alone is worth stopping for

If it prints `UNCONFIRMED`, that means the lookup failed, not that the work is
lost. Retry before concluding anything — per the retracted 2026-08-02 audit, a
negative finding deserves the same verification standard as a positive one.

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
