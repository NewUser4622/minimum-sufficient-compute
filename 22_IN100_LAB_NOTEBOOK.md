# Lab Notebook — ImageNet-100 Port

**Running record of every defect and every decision that changed.**

Append-only, newest first, same contract as
[`09_LAB_NOTEBOOK.md`](09_LAB_NOTEBOOK.md): when the paper is written, three
questions must be answerable from this one file.

1. What did we measure, and exactly where does that number live?
2. Was any reported number produced by code that later turned out to be wrong?
3. What changed in the design after the plan was frozen, and why?

Defect numbering continues from the CIFAR log, which ended at **D-36**.

---

## 0. Status board

| | |
|---|---|
| **Phase** | infrastructure port — **no GPU-hour spent yet, by design** |
| **Question** | does the ViT/Mixer ρ_seed gap (0.547 vs 0.62–0.73) survive at ImageNet scale? |
| **Design** | frozen — [`20_IN100_PORT_PLAN.md`](20_IN100_PORT_PLAN.md) |
| **Data** | **packed** — [`25_IN100_DATA_CARD.md`](25_IN100_DATA_CARD.md) |
| **Dataset** | 129,395 images · 100 classes · fingerprint `2b6269ef…` |
| **Zoo** | 8 architectures registered · all 8 build, price and dry-run on the GPU |
| **Storage** | **local disk only** — no HuggingFace, no network at run time |
| **Self-checks** | **381** offline + 6 build-time name layers, all passing |
| **Telemetry** | **160** per-epoch + **91** final columns · parity with CIFAR confirmed (§D-40) |
| **Notebooks** | **5** (`notebooks_in100/`) · columns, library names, call signatures, result keys and drive letters all validated at build time |
| **Defects found this port** | **17** (D-37 … D-53) · 17 fixed · 0 open |
| **Runs trained** | 0 / 24 |
| **Artifacts** | local, under `MSC_ROOT/runs/` — nothing uploaded, nothing deleted |

> **Nothing in this document is a scientific result.** No model has been
> trained. Every number here is about the infrastructure.

---

## 1. Results ledger

Empty. Phase 0 has not run.

The four Phase 0 runs (`resnet50` × 2 seeds, `vit_small_p16` × 2 seeds, ~33
GPU-h) and their pre-registered gate are specified in
[`20_IN100_PORT_PLAN.md`](20_IN100_PORT_PLAN.md) §7. **The gate is written down
before the numbers exist, and it is written so that "the CIFAR finding was a
small-data artifact" is a recorded, publishable outcome rather than a
disappointment.** Rule 12 cuts hardest here: the outcome that flatters us is
the one that reproduces, and that is the one to distrust.

---

## 2. Defect log

### D-53 · Swapped arguments — right count, right names, wrong order

**Severity:** **NB2, NB3 and NB5 all failed on their training cell**
**Status:** **fixed** · **Found:** 2026-08-08 by the user running NB2

```
TypeError: 'function' object is not iterable
  by_id = {c["run_id"]: c for c in cfgs}
```

```python
results = sess.run_all(M.train_backbone, cfgs)   # signature: run_all(cfgs, fn)
```

Arguments reversed. `cfgs` received a function, so the dict comprehension tried
to iterate it. **The same line appeared in all three training notebooks** — NB2
(`train_backbone`), NB3 (`run_oracle`), NB5 (`train_msc_kd`) — because they were
written from the same template.

**Why the arity check missed it, and this is the point.** D-47/D-48 count
positional arguments and name keywords. Both were **correct**: two positional
arguments, `run_all` takes two, no keywords involved. Only the *order* was
wrong, and a count cannot see order.

Sixth defect in this family, and each one has needed a different lens:

| | what was wrong | what could see it |
|---|---|---|
| D-39 | function did not exist | name existence |
| D-47 | 6 args where 8 required | positional count |
| D-48 | keyword `interrupt_after` | keyword names |
| D-51 | key `passed` where `ok` | declared result keys |
| D-52 | column `passes` where `passed` | declared result keys |
| **D-53** | **right count, wrong order** | **argument *kind*** |

**Contamination analysis.** None. `TypeError` on the first line of the training
cell, before the plan was built and before anything was claimed. Loud and
immediate — the benign shape again.

**Fix.** All three call sites corrected, and a guard that compares argument
*kind* against parameter *role*:

> `M.train_backbone` used as a **value** is unambiguous evidence of a function
> reference. A parameter named `fn` / `done_fn` / `criterion` unambiguously
> wants one. Comparing those two facts costs nothing.

The check needed one addition to the signature extractor — positional **order**,
which it had not been recording because nothing until now needed it.

Verified against the exact line and three valid calls:

```
D-53 sess.run_all(M.train_backbone, cfgs)
  -> argument 1 is the function `M.train_backbone` but parameter 1 is `cfgs`.
     Arguments look swapped -- the signature is (cfgs, fn, steal_stale, ...)
VALID sess.run_all(cfgs, M.train_backbone)   -> clean
VALID sess.run_all(cfgs, M.run_oracle)       -> clean
VALID M.build_model('resnet50', 100, ...)    -> clean
```

---

### Note · Phase 0 is 98 GPU-h, not the 33 the plan claimed

Visible in NB2's estimate cell and worth stating plainly. The plan's §7 figure
of ~33 GPU-h for Phase 0 was built on the pre-measurement estimates. Against
your measured numbers it is **98 GPU-h ≈ 4.1 days**, and the full atlas is
**457 GPU-h ≈ 19 days** — the plan was optimistic by 94%, not 66%.

Two of those numbers are still provisional: `resnet50` at 82 img/s and `vgg16`
at 56 are the **D-43** measurements taken with `cudnn.benchmark = False`.
`resnet50` alone is 27% of the atlas at that figure and should roughly halve on
re-measurement. Until then the honest reading is *"457 GPU-h, of which ~65% rests
on two numbers known to be understated."*

`vgg16` is 39% of the budget for one across-CNN data point. Dropping it →
**280 GPU-h ≈ 11.7 days**. That trade is in NB2 as a commented line.

---

### D-52 · A gate column that could never exist — found by auditing the other four notebooks

**Severity:** **`KeyError` in the analysis phase, after every GPU-hour is spent**
**Status:** **fixed** · **Found:** 2026-08-08 auditing NB2–NB5 for the D-51 pattern

`analyse_q3_shuffled_control` returns **`passed`**. Its atlas-wide wrapper did:

```python
if len(df) and "passes" not in df.columns and "ok" in df.columns:
    df["passes"] = df["ok"]
```

There is no `ok` column, so **`passes` was never created** — and NB4 reads
`ctrl[~ctrl['passes']]`.

**Where it would have surfaced is what makes it serious.** The shuffled control
is the alignment gate for Q3: it proves the per-sample tables are genuinely
row-aligned across architectures, and Q3 is uninterpretable without it. This
`KeyError` fires in NB4, **after** the atlas has trained (~10 days) and been
measured. Not a lost result — but the last place you want to discover a typo.

Three names for one concept — `ok`, `passed`, `passes` — and a renaming layer
in between. The wrapper now uses the primitive's name and **raises** if the
column is absent, rather than quietly returning a frame without the gate.

**Contamination analysis.** None. Nothing has been analysed.

---

### The guard: declared result keys

D-51 and D-52 are the same defect at two ends of the pipeline, and neither was
catchable by anything already in place:

| existing guard | catches | misses this |
|---|---|---|
| D-39 | a function that does not exist | `resume_acceptance_test` exists |
| D-47/D-48 | a call whose arguments do not fit | the calls are fine |
| D-22/D-36 | a column absent from `HISTORY_FIELDS` | these are result keys, not schema |

**A key read off a returned dict or frame had no owner.** `RESULT_KEYS` now
declares, per function, what a caller may read — 17 functions — and
`build_notebooks_in100.py` **refuses to generate** a notebook that reads
anything else, suggesting the closest declared name.

Declaring the set is what makes a guess *detectable*. A guess against an
undeclared contract is indistinguishable from a correct read until it runs.

Two deliberate limits, both to avoid the 73-false-positive mistake:

- **Undeclared functions are not policed.** Opt-in, not inferred.
- **Q1's τ-suffixed columns match by shape** (`rho_seed_tau0.1`, `j10_tau0.3`),
  because the τ grid is a parameter and the columns cannot be enumerated.

Verified against both real lines and three valid ones:

```
D-51 res.get('passed')  -> resume_acceptance_test declares no such key
D-52 ctrl['passes']     -> analyse_q3_shuffled_control_all declares no such key
VALID res['ok']         -> clean
VALID ctrl['passed']    -> clean
VALID q['rho_seed_tau0.1'] -> clean
```

**The audit that found it.** Every `x = M.f(...)` followed by `x[k]` across all
five notebooks was extracted and checked against the function's real return.
41 reads, one wrong. The other 40 were already subscripts rather than `.get()`,
so a typo among them would at least have raised.

Self-checks 373 → **381**.

---

### D-51 · A passing test reported as failed, because of one wrong key

**Severity:** **NO-GO on a green result; ~40 minutes of the user's GPU time**
**Status:** **fixed** · **Found:** 2026-08-08 by the user

The test said:

```
resume is equivalent to an uninterrupted run
interrupt actually fired : True
epochs  reference=4  resumed=4   (want 4)
duplicated epoch rows    : 0
max post-seam loss drift : 0.3077%   (want < 5%)
RESUME TEST: PASS
ok: True
```

The notebook then printed **`RESUME FAILED -- do not train`** and the gate said
**NO-GO**.

`resume_acceptance_test` returns `ok`. The notebook read **`res.get('passed')`**.
`.get()` returned `None`, `None` is falsy, and a passing test was reported as a
failure — twice, in two different cells.

**`.get()` on a key you require turns a typo into a wrong answer. A subscript
turns it into an error.** That is the whole lesson, and it is the same family as
D-39, D-47 and D-48: a name that does not exist, in a place nothing compared it
against anything. Fourth time. The previous three guards check that *functions*
exist and that *calls* match signatures; none of them can see a **key** read
from a returned dict.

**Contamination analysis.** None to the science — **resume works, and this
result proves it**: post-seam loss drift 0.31% against a 5% tolerance, four
epochs both legs, zero duplicate rows, final accuracy 0.5072 vs 0.5027. The
cost was 40 minutes of GPU time and a NO-GO that should have been a GO.

**Fix.**

- The notebook reads `res['ok']`, so a wrong key raises rather than evaluating
  to `None`.
- `RESUME_TEST_KEYS` pins the returned key set, with a self-check that `passed`
  is **not** among them — pinning the set is what makes a guess detectable —
  and that every declared key is actually written by the function.

---

### The resume test now runs on 5% of the data

Asked for, and correct: three legs × 4 epochs × 119,395 images is ~40 minutes
for a **smoke test of the resume machinery**. It exercises checkpoint,
interrupt, resume and the post-seam comparison — none of which cares how much
data there is.

`subset_frac=0.05` by default, so it takes about two minutes and tests exactly
the same code. `_subset_train` applies to the **train split only**: `val` and
`train_holdout` are what results are measured on, and a test that shrinks them
is testing something else. Self-checked.

The subset preserves `index_space` and leaves `sample_idx` global — renumbering
alongside the data would quietly reintroduce **D-49**.

---

### D-50 · "No session limit" was read as "a limit of zero"

**Severity:** **every run would pause after epoch 1, for the whole programme**
**Status:** **fixed** · **Found:** 2026-08-08 by the user, in the resume test

```
ep   1/4  train 16.57%  val 24.32%  ...  *BEST*
[LIFE] session limit reached at 0.1 h -- pausing cleanly at epoch 1
```

The watchdog exists for Kaggle, where a session dies at 8–12 hours without
warning, so stopping cleanly first is the civilised move. A local machine has
no such deadline, and the ImageNet-100 profile sets `session_limit_h = 0.0` to
say so.

`LifecycleGuard` read that as **a limit of zero hours**. `session_expiring()`
was true on the first call, so every run paused at epoch 1.

**What this would have cost.** The atlas is ~10 days of training. A run that
pauses after every epoch needs a manual restart every three to eight minutes,
around the clock, for the entire programme. It would have been noticed
immediately — but only *after* Phase 0 had been started and babysat.

**It also broke the test that found it, in a way that pointed elsewhere.** The
resume test reported:

```
interrupt actually fired : False
epochs  reference=1  resumed=2   (want 4)
RESUME TEST: FAIL
```

Every leg paused at epoch 1, so the debug interrupt never reached `kill_at=2`.
**The test failed for a reason with nothing to do with resume**, and its report
named the symptom (`interrupt_fired: False`) rather than the cause. That is the
**D-06 shape** — a test that can fail for the wrong reason — and it cost a
round trip.

**Contamination analysis.** None. No completed run exists. Any run that *had*
completed would be valid: pausing is a clean, resumable stop, not corruption.
The cost was entirely diagnostic time.

**Fix.**

- `session_limit_h <= 0`, `None`, or negative → **unbounded**, explicitly, with
  `self.unlimited` as a named property rather than an `inf` comparison buried in
  arithmetic.
- The armed message now reads `session limit NONE -- runs to completion)`
  instead of `0.0 h`, so the state is visible rather than inferred.
- The config comment says what `0` means at the place someone reads it.
- `resume_acceptance_test` sets `session_limit_h=0.0` explicitly: a test whose
  purpose is one stop reason must not be interruptible by a different one.

**And the report now names the failure MODE.** `interrupt_fired: False` is true
of both "resume is broken" and "something else stopped the run first", and
those need completely different responses. The test now prints one of six
diagnoses, e.g.:

> *the REFERENCE leg stopped at epoch 1 of 4 without being asked to. Nothing
> about resume has been tested. Check the session watchdog (`session_limit_h <=
> 0` means no limit)…*

versus the one it exists to catch:

> *post-seam loss drifted 8.3% — RNG or optimiser state did not survive the
> seam.*

**Guards added:** 7 self-checks — zero, negative and `None` are unbounded; 8.5 h
is still honoured; **a real limit that has elapsed still fires** (a watchdog
that can never say yes is decoration); and each recipe asks for the right one.

Self-checks 359 → **366**.

---

### Note · a status line that printed a dict

`[DRY] backbone dry run ok (0.27s, {'start_epoch': 1, ...}px, 100 classes)`

The D-47 fix introduced `res = load_checkpoint(...)`, shadowing the `res` that
held the input resolution. Harmless to the run, and fixed — but a status line
that prints a dict where a number belongs is a status line nobody reads
carefully afterwards, and this project has now paid twice for output that was
technically correct and practically unreadable (D-46, D-47).

---

### D-49 · `sample_idx` became global; the dynamics arrays did not

**Severity:** **every training run would crash in epoch 0** · **Status:** **fixed**
**Found:** 2026-08-08 by the user, in the kill-and-resume test

```
IndexError: index 121978 is out of bounds for axis 0 with size 119395
  TrainingDynamics.observe_batch -> self._epoch_correct[i] = corr
```

`TrainingDynamics` allocates its arrays with `len(train_set)` = **119,395** and
indexes them by `sample_idx`. On the packed backend `sample_idx` is the
**global pack index**, 0…129,394. The first training image whose global index
exceeded the split length blew up.

**Both halves of this were deliberate, and that is the problem.**
`PackedImageDataset` says so in its own docstring:

> *"`sample_idx` is the global pack index, not the position in this split…
> That makes every per-sample table self-describing, lets val and
> train_holdout tables coexist without ambiguity, and means an accidental
> split mismatch shows up as non-overlapping indices rather than as a
> plausible correlation."*

All true, and worth keeping. But it **changed what an index means**, and
`TrainingDynamics` was written against the old meaning — where `sample_idx` was
a dense position within one split.

**This is D-40 again, on a different quantity.** There, GPU-side augmentation
changed what `dataload_frac` measured while its name stayed the same. Here a
backend change moved the definition of an index while its name stayed the same.
Twice now: **when a quantity's meaning moves, every consumer written against
the old meaning is silently wrong** — and the only reason this one was loud is
that arrays have bounds. `dataload_frac` had none, which is why it needed an
audit to find rather than a crash.

**Contamination analysis.** None. `IndexError` in epoch 0 of a 4-epoch test
run, before anything was written. The benign shape again: loud and immediate.
The kill-and-resume test found it, which is what that test is for — five CIFAR
defects were about resume, and this is the first thing it caught here.

**Fix.** Both datasets now declare `index_space` — the size of the space
`sample_idx` values live in — and `train_backbone` asks the dataset instead of
assuming:

| backend | `len(dataset)` | `index_space` |
|---|---|---|
| `CIFARTensor` | 50,000 | 50,000 (positions within the split) |
| `PackedImageDataset` (train) | 119,395 | **129,395** (the whole pack) |

`GPUBatchLoader` forwards it, since callers see the wrapper. `_check_space()`
raises with the cause and the remedy named, rather than an `IndexError` four
frames deep that mentions neither.

**And a second bug the fix exposed.** `to_frame()` emitted one row per index in
the space. With a global space that is 129,395 rows including **val and holdout
positions this run never trained on**, whose `forget_events` would be 0 and
`el2n` NaN — entering the Q4 difficulty battery as if they were measurements.
It now emits only indices actually seen.

**Guards added:** 7 self-checks, including the exact out-of-bounds case and an
assertion that `to_frame` returns 3 rows when 3 indices were seen.

Self-checks 352 → **359**.

---

### Note · `RUN_PACKER` is one-time

Asked directly, so recorded. **Once.** The cell checks `data_present` first and
prints `already packed -- nothing to do` without touching anything. The packer
itself is idempotent and resumable, so leaving the flag `True` costs a
directory listing.

---

### Note · HuggingFace notices silenced in an offline run

`[HF] no token: add 'HF_TOKEN'...` was still printing inside the resume test.
This programme is local-only **by design**, so that is advice for a
configuration the operator deliberately is not in. Both notices are now gated on
`MSC_OFFLINE`. Same reasoning as D-46: a message that fires on the intended
setup is noise, and noise is what makes a real line get skimmed past.

---

### D-48 · The same defect, one layer out — a notebook call with a wrong keyword

**Severity:** NB1 step 6 crashed · **Status:** **fixed**
**Found:** 2026-08-08 by the user running NB1

```
TypeError: resume_acceptance_test() got an unexpected keyword argument 'interrupt_after'
```

The parameter is **`kill_at`**. Every name in that line is real — `M` exists,
`resume_acceptance_test` exists, `sess` exists. The call is still wrong.

**This is D-47 exactly, one layer out, and I fixed only the inner layer.**
D-47's arity check covers calls *inside* `msc_lib`. The notebook validator
checked that `M.x` **exists** and stopped there. So the same class of defect
survived in the place I had just finished writing.

That is the pattern D-32 warned about verbatim: *"when work can be skipped at N
points, an invalidation must be understood at ALL N. Fixing the first one
relocates the symptom and looks like progress."* Here N = 2 — library-internal
calls and notebook calls — and I fixed one.

**Contamination analysis.** None. A `TypeError` at the top of a cell, before
anything ran.

**Fix.** The notebook validator now extracts **signatures** from `msc_lib.py`
(module functions and `Session` methods) and checks every `M.x(...)` and
`sess.x(...)` call for positional count and keyword names, suggesting the
closest real parameter. Verified against the actual line:

```
OLD -> msc_lib.resume_acceptance_test() has no parameter 'interrupt_after'
NEW -> clean
```

and end-to-end, by feeding the validator a notebook containing the bug —
generation refused, exit 1.

---

### Note · packing no longer needs a terminal round-trip

Not a defect. The `[TODO] imagenet100 packed` in steps 3 and 4 was **correct**
— the pack had not been built and the D-46 fix reports that as a prerequisite
rather than a failure. But it required leaving the notebook to run a command
with two paths pasted in, which is friction with nothing to recommend it.

NB1 now runs the packer itself, as a subprocess with live output, using the
`DATA_DIR` cell 2 already resolved and a `SRC_DIR` it auto-detects. `RUN_PACKER
= False` by default, so a 40-minute job never starts by accident.

**On the labelling, since it was asked about:** folders sorted by WNID map to
class indices 0–99 (`n01440764` → 0, `n01855672` → 99). Arbitrary, and that is
fine — nothing downstream depends on matching official ImageNet indices. What
would matter is the mapping *changing* between runs, and the fingerprint makes
that impossible. It is published in `manifest.json` as `classes` and
`class_names` either way.

---

### D-47 · Names that exist, calls that don't — 16 dry-run failures

**Severity:** every dry run failed; NB1 could not reach GO · **Status:** **fixed**
**Found:** 2026-08-08 by the user running NB1 on real hardware

```
[FAIL] resnet50  backbone  at stage 'checkpoint round trip':
       TypeError: load_checkpoint() missing 2 required positional arguments:
       'dynamics' and 'device'
[FAIL] resnet50  oracle    at stage 'compute_msc':
       TypeError: object of type 'MSCResult' has no len()
```

Both dry runs, all eight architectures, 16 failures — **two bugs, both mine.**

| call | I wrote | it is |
|---|---|---|
| `load_checkpoint` | 6 positional args, unpacked as a 5-tuple | **8** positional args, returns a **dict** |
| `msc_for_run` | treated the return as an array, `len(msc)` | returns an **`MSCResult` dataclass**; the vector is `.msc` |

**Why the existing guards missed it, and this is the point.** D-38 added a check
that every free name in the dry runs **resolves**. It passes here: `load_checkpoint`
exists, `msc_for_run` exists, every name is real. D-39 added a check that every
`M.x` a notebook calls exists. Also passes.

**Names being real is not the same as calls being right.** Two guards, both
about existence, and neither could see a call that names the right function and
hands it the wrong things. The guard needed was arity — and arity is
mechanically checkable from the same source the names came from.

**Contamination analysis.** None. Both are hard `TypeError`s in a synthetic dry
run, which is the benign shape: loud, immediate, before any GPU time and before
the run is claimed. The dry runs did exactly their job — this is a *defect in
the check*, found by the check.

**Also fixed: the output was unreadable.** Eight architectures × two dry runs
printed sixteen paragraphs of sklearn's `y_pred contains classes not in y_true`
(2 samples against 100 classes) and torch's `lr_scheduler.step() before
optimizer.step()` (the AMP scaler legitimately skips the first step while it
finds a loss scale). Both are guaranteed on a synthetic batch and mean nothing.
They are now suppressed **inside the dry runs only**, because a report nobody
can read is a report nobody reads — the D-17 cost, in a new place.

**Guard added.** An AST arity check: for every call to a module-level function
from within the dry runs, the analysis wrappers, `verify_run_artifacts`,
`resolve_storage` and `in100_estimate`, the positional count must fall within
the callee's range and every keyword must name a real parameter.

**Verified against the actual bug**, not just asserted:

```
old call passed 6 positional; load_checkpoint needs >= 8   ->  CAUGHT
```

The oracle dry run now also asserts MSC lies in (0, 1], since ρ is a fraction
and a value outside that range means the budget table is wrong rather than the
sweep.

Self-checks 339 → **352**.

---

### D-45 · One atlas, two FLOPs profilers — and a transformer with no attention counted

**Severity:** **would have invalidated every cross-architecture number**
**Status:** **fixed** · **Found:** 2026-08-08 in the user's NB1 preflight output

```
  [PASS] budgets resnet50   -- K=5 depth rho=[0.194, 0.392, 0.643, 0.803, 1.0]
[FLOP] profiler fvcore failed (type Tensor doesn't define __round__ method); using analytic fallback
  [PASS] budgets vit_small_p16 -- K=5 depth rho=[0.178, 0.425, 0.589, 0.836, 1.0]
```

Every convolutional backbone was priced with **fvcore**. `vit_small_p16`,
`deit_small` and `swin_tiny` fell back to the **analytic** counter. Both lines
say `PASS`.

**This module's own comment forbids exactly this:**

> *"The SAME profiler and the SAME accounting convention must be used for every
> architecture and every axis. A budget table built with fvcore for one model
> and thop for another silently corrupts every transfer number."*

The rule was written down and then not enforced, which is rule 7 in miniature —
an invariant living in a comment rather than a mechanism.

**Why the fallback is worse than a different-but-valid profiler.** The analytic
counter hooks `Conv2d` and `Linear` only. For a transformer that **omits the
attention matmuls entirely** — QK^T and AV. Those scale with tokens² while the
linear parts scale with tokens, so the omission is *not* a constant factor that
cancels in a ratio:

- **depth axis:** ViT blocks are identical, so the undercount is roughly
  proportional across prefixes and ρ_depth survives. Swin's stages change
  resolution, so it does not.
- **resolution axis:** attention is quadratic in tokens, linear parts are
  linear. ρ_res is distorted for precisely the architectures the study is
  about. The measured ViT ρ_res was `[0.126, 0.221, 0.343, 0.493, 1.0]` against
  the CNNs' clean `[0.184, 0.327, 0.510, 0.735, 1.0]` = (r/224)².

**ρ is DEFINED in FLOPs** (protocol §2.1). It is the normalisation that makes
"did MSC transfer from ResNet to ViT?" a well-posed question at all. Pricing
half the zoo with a counter that cannot see attention makes that question
ill-posed while every individual table still looks reasonable.

**Cause of the fvcore failure.** fvcore traces with `torch.jit`. Tracing a
positional-embedding resample applies Python `round()` to what has become a
tensor → `type Tensor doesn't define __round__ method`.

**Contamination analysis.** No measurement affected — nothing has trained, and
`budgets/*.json` from this preflight must be deleted before Phase 0 because the
transformer tables were built by the wrong counter. **Every budget table
produced before this fix is void.**

**Fix, in two parts.**

1. **`torch.utils.flop_counter.FlopCounterMode` is now the preferred
   profiler.** It works by `__torch_dispatch__` rather than tracing, so there
   is nothing to trip over, and it counts matmul and scaled-dot-product
   attention natively. It reports true FLOPs, so no MAC doubling is applied.
2. **A fallback now RAISES.** Silently switching profilers mid-zoo is the
   defect; the escape hatch is `MSC_ALLOW_MIXED_PROFILER=1`, which has to be
   asked for and which logs an ALARM naming the consequence.

**Guards added:** 5 self-checks, plus NB1 now prints `M.profilers_used()` and
**fails the GO gate** if more than one profiler priced the zoo. The first
version of one check compared `.index()` over the whole source and matched the
docstring explaining why fvcore is no longer first — the same
prose-instead-of-code mistake the notebook validator already made twice. It now
compares import statements.

Self-checks 334 → **339**.

---

### D-46 · A preflight that fails on the intended configuration

**Severity:** two red lines beside a real one · **Status:** **fixed**
**Found:** 2026-08-08 from the user's NB1 output

```
39/42 checks passed
  FAILED: HF token  -- from Kaggle Secrets or env
  FAILED: HF repo reachable  -- Shanmuk4622/msc-cifar100
  FAILED: imagenet100 present  -- packed ImageNet-100 not found...
```

Three failures, and **none of them is a defect**:

| reported | actually |
|---|---|
| `HF token` | this programme is **deliberately local-only and offline**. There is no token by design |
| `HF repo reachable` | ...and it names the **CIFAR** repo, a stale default |
| `imagenet100 present` | the pack has not been built yet. That is the **next step**, not a fault |

**Why this matters beyond tidiness.** A preflight that fails on the intended
configuration teaches the operator to skim past red lines — which is the D-17
and D-20 cost, arriving for the third time. Here it did real damage: two false
failures sat beside a genuine one (**D-45**, the mixed profiler, which printed
`PASS`), so the summary line pointed at the wrong things entirely.

**And a fourth problem the same output revealed.** The dry-run cell raised:

```
RuntimeError: packed ImageNet-100 not found
```

The dry runs are **synthetic** — they push noise through the whole path and
never open the dataset. But `Session.config()` called `prepare_data()`, which
raised. So the cheapest and earliest check in the notebook could not run until
after the most expensive prerequisite was complete. Exactly backwards: a
config-level bug should surface *before* a 40-minute packing job, not after it.

**Fix.**

- HF checks are **skipped** when the session is local-only, and replaced by the
  equivalents that matter here: results root writable (probe written and read
  back) and enough free space for the atlas.
- `HF_REPO` default corrected to `msc-imagenet100`, and it reads
  `MSC_HF_REPO` from the environment.
- **Three states, not two.** `preflight_summary()` separates `passed`,
  `failed` and `todo`; a prerequisite that has not been done prints `[TODO]`
  with the exact command, and does not count against the gate.
- `Session.config(require_data=False)` and `prepare_data(required=False)`, so
  the synthetic dry runs run before the pack exists.

---

### D-44 · A default path named a drive that does not exist

**Severity:** NB1 could not run at all · **Status:** **fixed**
**Found:** 2026-08-08 from the user's NB1 outputs

```
FileNotFoundError: [WinError 3] The system cannot find the path specified: 'D:\'
```

Twice — once in the **bootstrap cell**, once in the paths cell.

**Cause.** I wrote `DATA_DIR = r'D:\msc_data\in100'` and
`MSC_ROOT = r'D:\msc_results'` as placeholder defaults. **There is no D: drive
on this machine.** A default that names a drive letter is wrong on any machine
without that letter, which is most of them.

**Two separate faults, and the second is worse.**

1. **The default was unusable.** Fine on its own — the operator edits two lines.
2. **The failure named neither the setting nor the file that had to change.**
   Forty lines of `pathlib.mkdir` traceback ending in `WinError 3`. Nothing in
   it says "edit `DATA_DIR` at the top of the notebook", which is the entire
   remedy.
3. **Import itself failed.** `enforce_offline` called
   `ensure_dir(TORCH_HOME)` unconditionally at module import, and `TORCH_HOME`
   derives from `MSC_SCRATCH`. So after the paths cell set `MSC_SCRATCH=D:\...`,
   re-running the **bootstrap** cell crashed — before the operator ever reached
   the cell that sets the path. An import that requires a writable directory
   turns a one-line fix into a traceback with no visible cause.

**Contamination analysis.** None. Nothing ran. The cost was one user round trip.

**Fix — remove the guess, and make the failure legible.**

| change | why |
|---|---|
| `DATA_DIR = None`, `MSC_ROOT = None` by default | there is no placeholder left to be wrong |
| `resolve_storage()` picks the roomiest **existing** root | `storage_candidates()` probes drive letters for existence, so a machine without D: simply never reports one |
| writability proved by **writing a probe file and reading it back** | `os.access` lies on Windows shares and on permission-inherited folders. Same discipline as `verify_run_artifacts`: presence is not usability |
| free space checked against ~26 GB (pack) and ~120 GB (results) | running out at run 19 of 24 is the alternative |
| `ensure_dir` names the **first missing level** and the remedy | `WinError 3` becomes "the first missing level is `D:\` — if that is a drive letter it does not exist on this machine; set DATA_DIR / MSC_ROOT, or leave them None" |
| `enforce_offline` falls back to a temp dir | importing the library can no longer fail because a cache path is absent |

**Replicated everywhere by construction.** All five notebooks share one
`paths_cell()` in the generator, so the fix landed in five places from one
edit. That is why the cell is generated rather than written per notebook.

**Guards added.** Ten self-checks over `storage_candidates` / `resolve_storage`
/ `ensure_dir` / `enforce_offline`, plus a **build-time** rule: any string
literal matching `^[A-Za-z]:[\\/]` in any notebook **refuses generation**.
Verified by feeding the checker a synthetic notebook containing exactly the old
line — it fails, with the remedy named.

Self-checks 324 → **334**.

---

### D-43 · The benchmark measured a machine the pipeline never uses

**Severity:** every throughput number was wrong, one badly · **Status:** **fixed**
**Found:** 2026-08-08 reading the user's benchmark results

```
resnet18   413.0 img/s   (11.2M params, 1.8 GFLOPs)
resnet50    82.3 img/s   (23.7M params, 4.1 GFLOPs)
```

A **5× gap for 2.3× the FLOPs.** `vgg16` at 56.3 img/s is, by contrast,
*consistent* with its 8.6× FLOPs ratio against `resnet18` — so only ResNet-50
was anomalous, which is what made it diagnosable.

**Cause.** `set_seed(deterministic=False)` sets `cudnn.benchmark = True`, so
every real training run has cuDNN autotuning on. **The benchmark never called
`set_seed`**, so it ran with torch's default of `False`. With autotuning off
cuDNN selects convolution algorithms by heuristic, and for ResNet-50's many
distinct 1×1 and 3×3 shapes in `channels_last` that heuristic is poor.
ResNet-18 has four 3×3 shapes and heuristics handle it fine — which is exactly
why the two diverged.

**A benchmark whose entire purpose is to predict the real run, configured
differently from the real run, produces a number that is precise and about
nothing.** It would have driven the run matrix, the epoch budget and the
decision about which architectures to drop.

**Contamination analysis.** No training affected — none has happened. But the
re-plan built on these numbers is wrong in a specific direction: convolutional
throughput is **understated**, so the atlas is cheaper than the 391 GPU-hours
currently recorded. `resnet50` and `vgg16` are marked `pending` in
`IN100_MEASURED_IMG_S` and must be re-measured.

**Fix.** `set_perf_flags(deterministic)` — **one** function, called by
`set_seed` and by the benchmark child, so the two cannot configure different
machines. This is the D-16 lesson: the writer and the reader must not be two
independent spellings of the same setting. TF32 is enabled there too.

Recorded rather than glossed: `cudnn.benchmark = True` makes algorithm
selection non-deterministic, which changes floating-point summation order. This
project measures seed-to-seed reliability, so anything adding within-seed
variance is relevant. The effect is far below the seed variation being measured
— AMP alone already forfeits bitwise reproducibility — and `deterministic: True`
turns it off. **Guard:** a self-check asserts the benchmark goes through
`set_perf_flags` and sets no cuDNN flag of its own.

---

### D-42 · Two of eight architectures could not be built at all

**Severity:** `vit_small_p16` and `deit_small` raised `TypeError`
**Status:** **fixed** · **Found:** 2026-08-08 by the user running the benchmark

```
vit_small_p16: TypeError: build_vit_small() got an unexpected keyword argument
deit_small:    TypeError: build_vit_small() got an unexpected keyword argument
```

`build_model` **injects** `probe_res` into every ImageNet builder, because the
D-38 fix has them derive `feature_dims` from a forward pass and the probe needs
a resolution. Five of the six builders were given the parameter.
`build_vit_small` was not.

**The pair it broke is the one that matters most.** `vit_small_p16` and
`deit_small` are the recipe-versus-architecture control — identical geometry,
one builder, one argument set, differing only in augmentation. Without them the
2×2 loses a corner and the strongest answer to the accuracy confound goes with
it.

**Why the existing guard missed it.** D-38 added a check that no builder
introspects foreign module internals. It never checked that builders **accept
what the caller passes**. Signatures are a contract, and contracts are
checkable.

**Contamination analysis.** None — the failure is a hard `TypeError` at build
time, so nothing was trained on a wrong model. This is the benign shape of a
defect: loud, immediate, impossible to mistake for a result.

**Guard added.** For every ImageNet architecture, the registry's declared
kwargs *and* `probe_res` must appear in the builder's signature — read from the
**source**, not from `globals()`, because every builder lives under
`if _TORCH_OK:` and a torch-free checking machine would otherwise report all
eight as missing. That is the **third** time this session a checker's notion of
"what exists" omitted the torch-gated half of the file.

Self-checks 310 → **324**.

---

### D-41 · The benchmark crashed the workstation

**Severity:** **took the machine down; ~2 hours of the user's time to recover**
**Status:** **fixed** · **Found:** 2026-08-08 by the user running it
**This is the worst defect of the port, and it is entirely mine.**

The throughput benchmark swept batch sizes up to **256 at 224px on a 20 GB
card**, in-process, with no cap on how much VRAM it could take and no timeout.
The machine hung and required a second person with administrator access to
recover.

**Cause.** Three compounding mistakes, and the first is the one that matters:

1. **No VRAM ceiling, on a GPU that is almost certainly also driving the
   display.** Filling the card starves the Windows desktop compositor. When the
   driver stops responding for more than ~2 seconds, Windows fires a TDR
   (Timeout Detection & Recovery) reset. On a workstation that is a hang or a
   `VIDEO_TDR_FAILURE` bugcheck. `resnet50` at batch 256 and 224px is well over
   20 GB for activations alone; `vgg16` and `swin_tiny` worse.
2. **No isolation.** Every configuration ran in the benchmark's own process, so
   a CUDA error poisoned the context for everything after it, and dataloader
   workers — each mapping the 24 GiB pack — accumulated across a sweep that
   created up to 40 of them.
3. **No timeout.** A kernel that stopped making progress stopped the sweep, and
   there was no mechanism to notice or intervene.

**What I got wrong in reasoning, not just in code.** I wrote a tool whose entire
purpose is to *find the limits of the hardware* and did not treat approaching
those limits as dangerous. Catching `torch.cuda.OutOfMemoryError` felt like
sufficient handling — but an OOM you catch in Python is the *benign* case. The
damaging case is the allocation that succeeds and leaves the display driver
with nothing, and no exception handler sees that at all. **The safety mechanism
had to prevent the condition, not react to it.**

I also assumed a dedicated compute GPU without checking, on a machine described
to me as a Windows 10 workstation with one GPU. That was the assumption to
question first.

**Contamination analysis.** No data lost — nothing had been trained and the
benchmark writes no experimental artifacts. The cost was entirely the user's
time, which is not a category this log has had to account for before and should
have been.

**Fix — prevent, isolate, bound.**

| mechanism | why this one |
|---|---|
| `set_per_process_memory_fraction(0.70)` | caps the process at ~14 of 20 GiB, so an oversized batch raises a **catchable Python OOM** long before the driver is starved. Prevention, not reaction |
| **every configuration in its own subprocess** | a child that OOMs, hangs or dies takes nothing with it; the OS reclaims its CUDA context and its workers on exit |
| **hard timeout, killing the process *tree*** | a hung config is killed in seconds. The tree matters: an orphaned worker holding the 24 GiB memmap is how the *next* config runs out of RAM |
| **the batch ladder deleted outright** (second pass) | it was the dangerous stage and a useless one — see below |
| `torch.compile` **off by default** | Triton on Windows is unreliable and compilation can hang for minutes with no output |
| results written **after every configuration** | an interruption costs one data point, not the run |
| `--plan-only`, `--vram-frac` | the operator can see what it will do, and lower the ceiling further, before it does anything |

**Verified, not asserted.** The three mechanisms were each exercised: a child
that fails emits a structured result and the parent survives; a deliberately
hung child is killed at the timeout with the parent alive; the results file
exists and is valid after every append with no `.tmp` left behind.

**Second pass, after the user asked for the risky parts to be removed
rather than guarded.** They were right, and the reason is sharper than
"be careful":

**The batch-size ladder was the dangerous stage AND a scientifically useless
one.** All eight architectures must share one batch size, or batch joins
accuracy and family as a confounded variable — learning rate is scaled linearly
from a reference batch, so eight per-architecture "winners" would be eight
different recipes. There was never anything to *do* with the ladder's output. It
risked the machine to produce a number that could not be used.

So it is **gone**, not tuned. Batch is fixed at 64 for every architecture, and
headroom for larger batches is **predicted from measured peak VRAM** instead:

```
peak 4.2 GB at batch 64  ->  largest estimated batch 182
peak 9.1 GB at batch 64  ->  largest estimated batch  84
```

Activation memory is close to linear in batch, so measuring 64 says what 128
would cost **without allocating it** — and allocating it is what took the
machine down. `torch.compile` removed too, as the likeliest thing to hang on
Windows. 80 configurations → 29, VRAM cap 70% → **50%**, timeout 240 s → 120 s.

**The lesson, in two parts.** The playbook's dry-run rule is "never spend an
hour discovering something findable in a second". Its neighbour: **a tool that
probes for a limit must be built so that reaching the limit is survivable.**

And the part I missed until told: **before making a dangerous measurement safe,
check whether the measurement was needed at all.** I spent a round hardening a
sweep I should have deleted. The safest configuration is the one that is never
run.

---

### D-40 · GPU-side augmentation silently changed what `dataload_frac` means

**Severity:** a recorded column would have answered a different question than
its name · **Status:** **fixed** · **Found:** 2026-08-08 auditing telemetry
parity against the CIFAR run

The training loop measures data cost as *time until the next batch arrives*:

```python
_t_batch = time.time()
for step, batch in enumerate(it):
    load_t = time.time() - _t_batch        # "waiting for data"
```

On CIFAR that was exactly right. `CIFARTensor` did augmentation on the CPU
inside `__getitem__`, so the gap between batches genuinely was data preparation.

**On the packed ImageNet backend it is not.** `GPUBatchLoader` performs the
H2D copy, the `grid_sample` crop/resize and the normalisation *inside* that gap
— all device work. So `dataload_frac` would have measured "CPU wait **plus**
GPU augmentation", `compute_time_sec` would have been correspondingly short,
and both numbers would have looked entirely reasonable.

**Why this matters more than it sounds.** `dataload_frac` is one of the five
columns the playbook singles out as impossible to recover after the fact, and
its entire purpose is one decision: *high means the GPU is starving and the fix
is the loader, not the model.* A version of it that also counts GPU work says
"the loader is the bottleneck" when the loader is idle — which is the opposite
of the truth, and would have sent tuning effort at the wrong thing for the whole
programme.

**Contamination analysis.** None — no run has produced a row. But this is
squarely the shape the lab notebook exists to catch: **a quantity whose meaning
changed without its name changing.** Nothing would have errored. The CIFAR
numbers remain valid; they were measured under the CPU-augmentation design where
the column meant what it says.

**Fix.** `GPUBatchLoader` reports its own split. `wait_s` — the genuine block on
the worker pool — is free to measure. `augment_s` needs a device synchronise,
which costs throughput, so it is **sampled every 50 batches and extrapolated**
rather than measured per batch: a per-batch sync would slow the run it is
measuring, and an estimate that is labelled as one is better than a precise
number bought with the thing being observed.

Two new columns, `augment_time_sec` and `augment_frac`, and
`dataload_time_sec`/`dataload_frac` now have the device time subtracted out —
so `dataload_frac` still answers exactly the question it answered on CIFAR.
Zero on the CIFAR backend, where augmentation *is* CPU work and belongs in
dataload.

`HISTORY_FIELDS` 158 → **160**.

**A note on 158 vs CIFAR's 171.** The difference is the second T4's thirteen
per-device columns, correctly absent on a single-GPU machine — `N_GPU_COLUMNS`
is derived, not assumed. Every other column CIFAR collected is collected here.
The consequence to remember: **the schema width depends on the hardware**, so
history CSVs from machines with different GPU counts cannot be naively
concatenated. That is why the floor is 1 rather than 0.

---

### D-38 · Five offline-verify failures, three of which needed no hardware

**Severity:** blocked the first real run · **Status:** **fixed** · **Found:** 2026-08-07 by the user running `--verify-offline`

```
5 offline failure(s). The pipeline is NOT self-contained:
  - shufflenetv2_in: AttributeError: 'BatchNorm2d' object has no attribute 'out_channels'
  - backbone dry run resnet50: ValueError: too many values to unpack (expected 2)
  - oracle dry run resnet50: NameError: name 'MultiExit' is not defined
```

All five were mine. **Three needed no GPU, no dataset and no device to find** —
they needed somebody to compare a name against what exists.

| symptom | cause |
|---|---|
| `NameError: MultiExit` | the class is `MultiExitModel` |
| `too many values to unpack` | `optimisation_health` returns **4** values, unpacked as 2 |
| `BatchNorm2d has no out_channels` | guessed at another package's module internals |

**The third is the one worth recording.** `build_shufflenetv2_imagenet` read
`b.branch2[-2].out_channels`; in torchvision's ShuffleNetV2 that index is a
`BatchNorm2d`. Correcting the index would have been the *wrong fix* — three
sibling builders made the same kind of guess (`b.conv3.out_channels`,
`m.reduction.out_features`, `b.norm1.normalized_shape[0]`) and happened to be
right. **A guess that is right for three of four is precisely what rule 2 is
about**, and D-33 is the precedent: a literal `5` that was correct for most of
the CIFAR zoo, which is exactly why it survived.

**Fix — stop guessing.** `StagedBackbone` now derives `feature_dims` from a real
forward pass. Ask the model. All four introspection sites are gone, and the
answer is definitive by construction rather than dependent on torchvision's
internal layout. `build_model` passes the dataset's resolution to the probe,
because probing a 224px model at 32px gives the wrong spatial size and Swin
would not run at all.

**Contamination analysis.** None. No run had started; `--verify-offline` is the
gate that exists to catch this, and it did. The cost was one user round trip.

**Guards added, all torch-free so they run in the offline self-test:**

- every free name in the dry runs, `_imagenet_config`, `build_budget_table` and
  `verify_run_artifacts` must **resolve**. This would have caught `MultiExit`.
- every tuple-unpack of `optimisation_health` must be 4 wide, checked by AST.
- no ImageNet builder may reference `out_channels`, `normalized_shape`,
  `out_features`, `num_features`, `branch2`, `conv3` or `reduction`.

Building the first of those needed the lesson twice more. Resolving against
`globals()` reported five real names as missing, because half the file lives
under `if _TORCH_OK:` and torch was absent on the checking machine. **A checker
producing exactly the false negative it exists to prevent** is worse than no
checker (D-17). It now parses the source for module-level names.

Self-checks 292 → 310.

---

### D-39 · Six invented library names, in one sitting

**Severity:** would have failed mid-notebook after GPU time was spent
**Status:** **fixed** · **Found:** 2026-08-07 while writing NB4 and NB5

`analyse_q1_all`, `analyse_q2_all`, `analyse_q3_all`,
`analyse_q3_shuffled_control_all`, `analyse_q4_all`, `compare_routing_methods`.
**None of them existed.** Every one would have surfaced only when the user ran
the cell — the same defect as `MultiExit` (D-38), in a new place, an hour later.

**Why it happened.** On CIFAR this assembly lived in *notebook cells*, so there
were no library functions to call and nothing established what they should be
named. Writing five new notebooks against a half-remembered API produced six
plausible names in one sitting.

**Two fixes, and the second is the one that lasts.**

1. **The functions now exist, in the library.** Not in cells. That placement is
   itself a defect fix: **D-18 came from this assembly living in a cell.**
   `pairs[:15]` over a sorted list looked like cost control and was a biased
   sample of the two most atypical architectures in the zoo; a dict
   comprehension silently dropped an architecture whose seed 1 was unmeasured,
   so the analysis covered 13 architectures while calling itself the atlas.
   Neither was catchable, because nothing tests a notebook cell (rule 8). The
   new functions **report what they excluded**, use **every** seed pair rather
   than just (s1, s2), and run **every** pair rather than the first N.

2. **The validator now checks library names as well as column names.** Rule 3
   says column names are data and must be validated at build time. Function
   names are data in exactly the same way, and the failure is *worse*: a wrong
   column yields a `KeyError` with a suggestion, a wrong function name yields an
   `AttributeError` several cells into a run that may already have spent
   GPU-hours. Generation is now refused on either.

Building *that* check needed the D-38 lesson twice more: `dir(M)` omits
everything under `if _TORCH_OK:`, and `dir(Session)` omits instance attributes
set in `__init__`. Both made the checker report real names as missing. It now
parses the source for module-level names and for `self.X = ...`.

---

### D-37 · The self-test harness could not fail

**Severity:** **every self-check in the project was advisory, not binding**
**Status:** **fixed** · **Found:** 2026-08-07 · **Pre-existing — inherited from the CIFAR pipeline**

```
  [FAIL] every 15.1 requirement has a column  {'gpu utilization (per gpu)': ['gpu1_util_mean_pct'], ...}
  [FAIL] per-GPU columns exist for both T4s

ALL CHECKS PASSED
$ echo $?
0
```

Two checks failed. The suite reported success and exited 0.

**Cause.** `_selftest` accumulated its verdict in a scalar:

```python
def _selftest() -> bool:
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
    ...
    # 960 lines later:
    ok, z, sd = shuffled_control_verdict(-0.0341, 5872)   # <-- REBINDS ok
```

That line is a perfectly ordinary tuple unpack. It also **destroys the entire
accumulated verdict** and replaces it with the outcome of one unrelated
shuffled-control test — which passes. So every check before it was discarded,
and the function's return value described only the checks that came after.

**How much was affected.** The rebinding sits at roughly the 80% mark of the
suite. Of the 232 checks the CIFAR project reported, **the great majority could
not influence the exit code.** The `[FAIL]` lines still printed, so a human
reading the output would have seen a failure — but any scripted use of the exit
code, and any claim of the form "the self-checks pass", was worth less than it
appeared.

**Contamination analysis.**

- **No published CIFAR number is affected.** This is a defect in the test
  harness, not in any statistic. Nothing it failed to catch is known to have
  been wrong; the point is that we cannot say it *would* have caught anything.
- **Every "self-checks N → M, all passing" line in
  [`09_LAB_NOTEBOOK.md`](09_LAB_NOTEBOOK.md) overstates its evidence.** Those
  entries record checks *written*, which is real work; they do not record checks
  *enforced*, which is what the wording implies. D-17's 11 regression checks,
  D-22's 12, D-24's 6, D-26's 4 all sit before the rebinding.
- Because the failures still *printed*, and because each defect's guards were
  added while its fix was being verified by eye, the practical exposure is
  smaller than the structural exposure. That is luck, not design.
- **The two checks it was hiding here were real** — see the GPU column count
  below — and they were found only because a platform change made them fail.

**Fix — structural, not a rename.** Renaming the shadowing variable would fix
this instance and nothing else; the next accidental `ok = ...` is one edit away.

- The verdict accumulates in **lists** (`_ran`, `_failed`). Appending mutates,
  so a stray rebinding of the name cannot silently destroy the contents, and if
  one did the count would stop growing.
- A **canary**: the harness deliberately fails a check at the end and asserts
  that the failure registered. If it does not, it says so in the loudest terms
  available and every result above it is declared meaningless.
- A **floor on checks run** (250), so a suite that exits a section early — the
  other way to lose results silently — is caught.
- Failures are re-printed by name at the end, so the verdict is legible without
  scrolling.

**Guard added:** 3 self-checks, one of which is the canary itself. This is rule
8 at its most literal: `check` is the function this entire file is organised
around, and nothing tested it.

**The lesson is D-06's, one level up.** D-06 was "a test that cannot fail for
the right reason manufactures confidence." This is a *harness* that could not
fail for any reason. Every guard added across 36 defects was written into it in
good faith and, for most of the file, could not do its job.

---

### Note · the notebook validator had to be built three times

Not a defect — it never shipped wrong. Recorded because the failure mode it
kept producing is the one D-17 and D-20 were about, and because the third
version found something.

`tools/validate_notebooks.py` enforces rules 3 and 4 at build time. First
version: flag every subscript string literal absent from the schema.

| version | rule | false positives on the 17 CIFAR notebooks |
|---|---|---|
| 1 | not in `HISTORY_FIELDS ∪ FINAL_FIELDS` | **73** |
| 2 | ...and not assigned in the notebook | **61** |
| 3 | ...and not created anywhere in the project's source | **0** |

Versions 1 and 2 were useless. `n_runs`, `wall_clock_hours`, `fam_a`,
`delta_r2_lo` are columns the notebooks and the library compute for
themselves. **A check that fires on healthy data teaches you to ignore it, and
the next alarm is the real one** — which is exactly what D-17 cost when the
shuffled control halted NB11 on a perfectly aligned pair.

Version 3 harvests every string key created by `msc_lib.py`, `msc_core.py` and
`msc_torch.py`. Going from 2 to 3 was forced by a real finding:
`partial_spearman`, `r2_difficulty_only` and `r2_difficulty_plus_msc` are
produced by **`msc_core.py`**, the reference implementation, which version 2
never read. A checker whose notion of "everything this project defines" omits
a source file will keep crying wolf until someone switches it off.

**The validator has its own self-test, and it changed the design.** Run the
five real D-22 names and the three real D-36 names through it; assert it
catches them; assert it accepts their correct counterparts. That test failed,
and the reason is worth keeping:

> `grad_norm` and `f1_score` are **legitimate internal dict keys** inside
> `msc_lib` — `EpochTelemetry` returns `{"grad_norm": ...}`. They are simply
> not CSV column names; those are `grad_norm_mean` and `f1_macro`. D-22 was
> using the internal key where the column name was needed. **No static check
> over string literals can tell those apart, because they are the same
> string.**

So the two defects have different owners, and saying otherwise would have been
the kind of "it's handled" that D-16 was closed with:

| defect | shape | owner |
|---|---|---|
| **D-22** | *writing* a row with keys the schema lacks | `append_history_row(strict=True)` — raises and names the column you meant. Verified against all five. |
| **D-36** | *reading* a column that exists nowhere | this validator. Verified against both un-suffixed GPU names. |

Rule 12 in miniature: the first version's "73 problems found" looked like the
tool working. It was the tool being wrong 73 times.

---

### The two failures D-37 was hiding

Not numbered as defects because they are the port doing its job — a platform
change surfaced two literals that were correct on the old platform. Recorded
because they are exactly rule 2, and because they are how D-37 was found.

```python
"gpu utilization (per gpu)": ["gpu0_util_mean_pct", "gpu1_util_mean_pct"],
check("per-GPU columns exist for both T4s",
      all(f"gpu{i}_{k}" in H for i in range(2) ...))
```

`N_GPU_COLUMNS` was a literal `2`, because dual T4 was the only platform. The
port target has **one** RTX 4000 Ada.

**This is D-36 read from the other end.** There, NB15 asked for an un-suffixed
`gpu_util_mean_pct` that never existed because the fields are per device. Here,
a test demanded a `gpu1_*` that should not exist on a single-GPU machine. Same
root: a device count written down instead of asked for.

Fixed by deriving `N_GPU_COLUMNS` from `torch.cuda.device_count()`, with a
floor of 1 so the schema does not change shape on a CPU-only analysis session —
two runs whose history CSVs have different column sets cannot be concatenated,
and that failure would surface in NB15 rather than where it was caused.

---

## 3. Design decisions changed after the plan was frozen

| | Decision | CIFAR-100 | ImageNet-100 | Why |
|---|---|---|---|---|
| **IN-1** | Data delivery | JPEG/pickle at load time | **256px uint8 packed memmap** | `dataload_frac` is a recorded column that must measure the model, not NTFS. Also makes the artifact mountable on Kaggle unchanged. |
| **IN-2** | Augmentation | CPU, per sample, in the Dataset | **GPU, batched, in the LOADER** | Eleven sites consume batches. Rule 6: a step that can be skipped at N points will be forgotten at one, and the symptom is a silent wrong answer. The loader yields what every consumer already expects, so nothing *can* forget. |
| **IN-3** | Native-resolution support | one boolean per architecture | **probed per resolution** | On CIFAR one `False` took the whole axis down (D-02). At 224px failures are partial: Swin-T's last stage is 3×3 at 96px, smaller than its own window. |
| **IN-4** | Resolution grid | 0.5–1.0 × native | **96/128/160/192/224** | Every value must divide by 32 — ViT-S/16's patch grid *and* Swin-T's four-stage reduction. 224 × the CIFAR fractions gives 140 and 196, which satisfy neither. |
| **IN-5** | Split provenance | shipped with the dataset | **carved, published, and fingerprinted into `config_hash`** | No val split exists in the source. Two runs disagreeing about which 10,000 images are `val` produce tables that align by index and compare different pictures — a failure that looks exactly like a result. |
| **IN-6** | `train_holdout` size | 5,000 | **15,000** | Learn-then-Test needs ≈14,979 samples for ε=0.01 at δ=0.05. CIFAR had to settle for ε=0.03 and say so in its limitations. |
| **IN-7** | Published references | 12 of 15 architectures | **none, for any** | No from-scratch reference exists for this subset at this recipe. D-14 is the cautionary case: a reference without a matching parameter count is unfalsifiable, and it produced the largest apparent win in the CIFAR atlas. The acceptance test is replaced (plan §4.4). |
| **IN-8** | Classifier head | per-architecture | **GAP + Linear, all eight** | Stock VGG-16's 124M-parameter FC head would make the depth-axis ρ measure the head rather than the backbone — and ρ is what the whole project normalises by. |
| **IN-9** | Epoch budget | 240 CNN / 300 modern | **100, all eight** | Removes schedule length as a third confounded variable. Does **not** fix the accuracy confound, which is reported (plan §1). |
| **IN-10** | GPU column count | literal 2 | **`device_count()`** | See D-37's sequel above. |
| **IN-11** | Verdict accumulation in the test harness | scalar | **lists + canary + floor** | D-37. |
| **IN-12** | Artifact store | HuggingFace, per-user rate limited | **local disk only** | Single machine, no network at run time. Everything HF guaranteed is now guaranteed locally — and one thing it never guaranteed is added (IN-14). |
| **IN-13** | Network | required (HF, Kaggle CLI) | **forbidden, and proven** | `enforce_offline()` sets the guards; `no_network()` replaces `socket.socket` so a fetch raises with the host it wanted. Environment variables are a request; blocking the socket layer is a guarantee. |
| **IN-14** | "Is my work safe?" | file present on HF | **present, non-empty, and parseable** | `confirm_on_hf` established that a file arrived. It never opened it. A run with a zero-byte `epochs.csv` or a truncated `summary.json` was indistinguishable from a healthy one until analysis. |

### IN-12 note — what "no HuggingFace" actually removes, and what it doesn't

HF was doing four jobs. Three of them were about *multiple machines*, and a
single local box does not need them:

| HF's job | local replacement |
|---|---|
| permanent store across ephemeral Kaggle sessions | the disk **is** permanent. `cleanup_local_after_complete` is `False`, and the confirm-then-delete branch is gated on `hub.enabled`, so **no code path removes a run directory** except an explicit `force_rerun` |
| coordination between accounts | not needed — one worker |
| rate limiting | not needed |
| **evidence that the work landed** | **`verify_run_artifacts` / `confirm_on_disk`** |

Only the fourth needed replacing, and the replacement is stronger than the
original. `[SESSION] LOCAL-ONLY` is deliberately **not** phrased as an alarm:
on Kaggle, HF-off genuinely meant the work evaporated at session end, and that
warning was correct there. Repeating it here would be false, and a warning that
is false teaches the operator to ignore the line — D-20's actual cost.

### IN-13 note — nothing is downloaded, and that surprised me too

**Training from scratch downloads no model weights.**
`torchvision.models.resnet50(weights=None)` is Python source inside the
installed package. So is `swin_t`. There is no architecture to fetch.

What needs one-time internet is the **pip packages**, and what needs pinning is
their **versions** — because this pipeline decomposes every backbone into an
ordered block list so `forward_prefix(x, k)` genuinely stops at stage k. A
torchvision upgrade that changes that structure changes `n_blocks`,
`feature_dims`, the budget table and therefore ρ — and ρ is a ratio, so every
resulting MSC value looks entirely reasonable.

`tools/fetch_assets.py` therefore fingerprints all eight architectures
(parameter count, FLOPs, K, stage cuts, feature dims, per-resolution native
support) into `assets/environment_manifest.json`, and `--check` compares
against it. That pin is the useful artifact; a directory of downloaded weights
would have been theatre.

---

## 4. Open items

| | Item | Blocks | Cost | Priority |
|---|---|---|---|---|
| **O-22** | **Run the packing tool and verify.** Nothing downstream exists until the fingerprint is real | everything | ~30 min CPU | **highest** |
| ~~O-23~~ | ~~Regenerate the notebooks~~ — **DONE.** Five, in `notebooks_in100/`, validated clean | — | — | shipped |
| ~~O-24~~ | ~~Build-time schema and path validation~~ — **DONE.** `tools/validate_notebooks.py`, gating `build_notebooks.py`. Self-tested against the real D-22 and D-36 names. 0 false positives on the 17 CIFAR notebooks; 2 genuine rule-4 path warnings | — | — | shipped |
| **O-25** | **Run NB00 preflight on the real GPU.** Every architecture, every resolution, kill-and-resume. This is where a Swin-T that cannot do 96px, or a torchvision decomposition that mis-orders blocks, will surface | Phase 0 | ~30 min GPU | **highest** |
| **O-26** | Re-anchor `ARCH_COST_HINT` on measured epochs. Current values are estimates and D-10 showed the first guess was 40% low. Per DC-11 they refine the *display* only | honest ETAs | ~1 h | medium |
| **O-18** | *(inherited, still open)* cross-session resume test. Five CIFAR defects were about resume and every test runs inside one session — the case that works. The new checkpoint round trip in `backbone_dry_run` does **not** close this; it proves the contract round-trips, not that a fresh process resumes | the next D-19 | ~1 h | **high** |
| **O-9** | *(inherited)* break the family/accuracy confound | the Q1 headline | — | **high, deliberately not attempted** — the equal-epoch decision leaves this open on purpose; the 2×2 is the mitigation |
| **O-27** | Decide whether `p3`/MSC-KD students get their own per-sample measurement pass, so B11 is computable. CIFAR's O-21 was never closed and it cost the Q5 headline | the Q5 result | ~10 GPU-h | medium |

---

## 5. Timeline

| Date | Event |
|---|---|
| 2026-08-08 | **D-53 — swapped arguments in all three training notebooks.** `sess.run_all(M.train_backbone, cfgs)`; the signature is `(cfgs, fn)`. Right count, right names, wrong order — the arity check counts and names, and neither sees order. Sixth defect in the family, sixth lens |
| 2026-08-08 | **D-52 — a gate column that could never exist.** `analyse_q3_shuffled_control` returns `passed`; the wrapper synthesised `passes` from a nonexistent `ok`, and NB4 read `passes`. KeyError in ANALYSIS, after ten days of training. Found by auditing NB2-NB5 for the D-51 pattern |
| 2026-08-08 | **`RESULT_KEYS` — result keys are now declared and checked at build time.** The fifth name-checking guard, and the first that can see a key read off a returned dict or frame |
| 2026-08-08 | **RESUME VERIFIED ON REAL HARDWARE.** `resnet18`, 4 epochs, real interrupt at 2, resumed in a fresh call: post-seam loss drift **0.31%** against a 5% tolerance, 0 duplicate rows, 4/4 epochs both legs. Five CIFAR defects were about resume; it works here |
| 2026-08-08 | **D-51 — a passing test reported as failed.** `res.get('passed')`; the key is `ok`. `.get()` on a required key turns a typo into a wrong answer, a subscript turns it into an error. Key set now pinned. Resume test cut to 5% of the data: two minutes, not forty |
| 2026-08-08 | **D-50 — "no session limit" read as "a limit of zero".** Every run paused after epoch 1; over a ten-day atlas that is a manual restart every few minutes. It also broke the resume test in a way that pointed at resume rather than at the watchdog — the D-06 shape. The test now names the failure MODE, not just the verdict |
| 2026-08-08 | **D-49 — `sample_idx` became global; the dynamics arrays did not.** `IndexError: index 121978 out of bounds for size 119395`. Both halves were deliberate; the index's MEANING moved while its name did not — D-40's shape on a different quantity. Datasets now declare `index_space`. Found by the kill-and-resume test, which is what it is for |
| 2026-08-08 | **Dataset packed.** The pipeline is now reading real data |
| 2026-08-08 | **D-48 — the same defect one layer out.** `M.resume_acceptance_test(..., interrupt_after=2)`; the parameter is `kill_at`. D-47's arity check covered library-internal calls only, so the class survived in the notebooks. D-32's lesson verbatim: fixing one of N skip points relocates the symptom. The validator now checks notebook call signatures too |
| 2026-08-08 | **NB1 packs the dataset itself** — subprocess with live output, `RUN_PACKER=False` by default. The TODO was correct; the terminal round-trip was friction |
| 2026-08-08 | **D-47 — names that exist, calls that don't.** `load_checkpoint` called with 6 of 8 positional args; `msc_for_run`'s `MSCResult` treated as an array. Two existence guards passed both. Arity is now checked by AST, verified against the real bug |
| 2026-08-08 | **D-45 — one atlas, two FLOPs profilers.** fvcore priced the CNNs and failed on ViT/DeiT/Swin, whose tables were then built by a counter that cannot see attention. rho is DEFINED in FLOPs. torch's flop_counter is now preferred and a fallback RAISES. **Every budget table built before this fix is void** |
| 2026-08-08 | **D-46 — the preflight failed on the intended configuration.** HF checks in a local-only run, plus "not packed yet" reported as a failure — two false reds beside the genuine D-45, which printed PASS. Three states now, and the synthetic dry runs no longer require the pack |
| 2026-08-08 | **D-44 — a default path named a drive that does not exist.** NB1 died twice on `WinError 3` for `D:\`, once in the bootstrap cell because `enforce_offline` made *import* depend on a writable directory. Defaults are now `None` and `resolve_storage()` picks the roomiest existing root, proving it by writing and reading back a probe file. Build-time guard: no notebook may contain a drive-letter literal |
| 2026-08-08 | **First real measurement.** 6 of 8 architectures benchmarked. The plan's 235 GPU-hours was optimistic by 66%: the atlas is **391 GPU-h** and `vgg16` alone is **45%** of it. Cost model re-anchored on measurement (D-10's lesson) |
| 2026-08-08 | **D-43 — the benchmark measured a machine the pipeline never uses.** `cudnn.benchmark=False` while every real run has it True; ResNet-50 read 82 img/s where ~180 is expected. One `set_perf_flags` now serves both |
| 2026-08-08 | **D-42 — `build_vit_small` did not accept `probe_res`**, so two of eight architectures raised TypeError — and they are the recipe-vs-architecture control pair. Guard: every builder's signature must accept what `build_model` injects |
| 2026-08-08 | **Live training display.** Per-epoch bar with loss/acc/img-s/lr/VRAM updating beside it, and an epoch line carrying the silent-by-default columns as inline warnings |
| 2026-08-08 | **D-41 — the benchmark crashed the workstation.** No VRAM ceiling on a GPU that also drives the display, no isolation, no timeout. Rebuilt: memory fraction capped, every config in its own subprocess, process trees killed on timeout, results written after each one. The worst defect of the port |
| 2026-08-08 | **Throughput benchmark written** (`benchmark/`). Staged coordinate descent over dtype x layout, batch size, workers, torch.compile, then a cold confirmation run. The plan's 235 GPU-hours is an estimate anchored on one guess; D-10 was that estimate being 40% low |
| 2026-08-08 | **Paper artifacts wired.** Six tables, three figures, and `verify_paper_artifacts` — each of the protocol's six contributions now has a named artifact, and the notebook checks the list rather than trusting it |
| 2026-08-08 | **D-40 — GPU-side augmentation changed what `dataload_frac` measures.** A recorded column that would have answered a different question than its name. Split into `dataload_time_sec` (CPU wait) and `augment_time_sec` (device work). Schema 158 → 160 |
| 2026-08-07 | **Five notebooks generated** (`notebooks_in100/`), down from seventeen. The CIFAR split followed Kaggle's failure modes; this one follows the stages of the experiment |
| 2026-08-07 | **D-39 — six invented library names in one sitting.** The validator now checks library names as well as column names: a wrong column yields a KeyError with a suggestion, a wrong function name yields an AttributeError several cells into a run that has already spent GPU-hours. The six functions moved into the library, which also fixes D-18's root cause |
| 2026-08-07 | **D-38 — five offline-verify failures, three findable without hardware.** The zoo stops guessing at other packages' module internals and derives feature dims from a forward pass. Guards: every free name must resolve, every unpack arity is checked, no builder may name a foreign attribute |
| 2026-08-07 | **Local-only and offline.** HuggingFace removed; `verify_run_artifacts` opens every required artifact rather than stat-ing it, so a zero-byte or truncated file is caught where a presence check called the run healthy. `no_network()` proves offline operation by blocking the socket layer. `requirements.txt` + `tools/fetch_assets.py`. Self-checks 278 → 292 |
| 2026-08-07 | **D-37 — the self-test harness could not fail.** A tuple unpack 960 lines below the definition rebound the verdict scalar; ~80% of checks could not affect the exit code. Fixed structurally with lists, a canary and a floor |
| 2026-08-07 | Rule 1: `backbone_dry_run` and `oracle_dry_run` written **and wired in**, with build-time checks asserting the call *and its position*. The oracle dry run covers every axis at every resolution and reads its parquet back |
| 2026-08-07 | Rules 9/10: HF verification moved off `list_repo_files` (tree endpoint) onto per-file `resolve`. The source-inspection checks initially matched their own docstrings — now they parse the AST, because a check that reads prose is checking the wrong artifact |
| 2026-08-07 | Zoo: 8 ImageNet architectures registered. `vit_small_p16` and `deit_small` are one builder with one argument set, differing only in recipe |
| 2026-08-07 | Library parameterised by dataset. `measure_flops` refuses a missing shape rather than defaulting to `(1,3,32,32)` |
| 2026-08-07 | Data verified: 129,395 images, 100 classes, splits deterministic, fingerprint `2b6269ef…`. Packing tool dry-runs the full path on 32 images in 0.2 s |
| 2026-08-07 | Design frozen: [`20_IN100_PORT_PLAN.md`](20_IN100_PORT_PLAN.md), [`25_IN100_DATA_CARD.md`](25_IN100_DATA_CARD.md) |

---

*Append new entries at the top of each log. Every claim here should be
traceable to a file in the HuggingFace repository or a commit in this one.*

---

## D-54 — `run_all` was handed a callable it cannot call

**Symptom.** All four Phase-0 runs, identically:

```
[ERROR] p0-resnet50-imagenet100-base-s1 failed: TypeError:
    train_backbone() missing 2 required positional arguments: 'hub' and 'registry'
    -- continuing
File "...msc_lib.py", line 10392, in run_all
    s = fn(by_id[rid], **kw)
```

**Cause.** `run_all` invokes `fn(cfg, **kw)` — one positional argument.
`train_backbone(cfg, hub, registry, ...)` requires three. `Session.train` and
`Session.oracle` exist for exactly this reason: they are thin bound wrappers
that supply `hub`, `registry`, `work_root` and `data_root_out` from the session.
The notebook passed the raw library function, so the two the session owns were
never supplied.

**This is D-53 one layer out, and that is the interesting part.** D-53 was
`sess.run_all(M.train_backbone, cfgs)` — arguments swapped. I fixed the *order*
and shipped, because the order was what the error named. But the same line was
wrong in a second, independent way that the order fix could not touch: even in
the right slot, that callable cannot be invoked the way `run_all` invokes it.
Every existing check passed it. The argument is a function (D-53), the name
exists (D-39), the count and keywords are right (D-47/D-48), the column names
are clean (D-22). Nothing asked whether the callee could *call* it.

Fixing what the traceback names is not the same as fixing the line.

**Contamination.** Three notebooks, one line each — NB2 (train), NB3
(measure), NB5 (MSC-KD). Zero runs completed, so no artifact is affected. Cost
is wall-clock only: the failure is per-run inside the try/except, so the work
plan computed and printed normally and four identical tracebacks scrolled past
as "continuing", which reads like four problems rather than one.

**Fix — two mechanisms, because a fixed call site protects one line.**

1. *Runtime, in `run_all`, before the plan is built.* Arity is knowable
   before any work: if `fn` requires more than one positional argument and has
   no `*args`, raise immediately, naming the missing parameters and the bound
   wrapper to use instead. Fails once, before the plan, instead of N times
   inside it.
2. *Build time, in `validate_notebooks.py`.* `_callback_arity_problems`
   rejects any `M.*` function passed as `fn`/`done_fn` to `run_all` whose
   minimum arity exceeds 1 — checked against `LIB_SIGS`, the signature table
   already parsed from source. Its self-test asserts it catches the literal
   D-54 line **and** accepts all three valid forms, so it cannot pass by being
   permissive.

**Correct forms (from the CIFAR generator, which had this right):**

```python
sess.run_all(cfgs, title='...')                  # fn=None -> sess.train
sess.run_all(cfgs, fn=sess.oracle, title='...')  # bound wrapper
sess.run_all(cfgs, fn=_closure, done_fn=sess.msckd_valid, ...)
```

---

## D-54b — an invented config key that silently disabled the control arm

Found by the column checker while fixing D-54, in the closure written *for*
D-54. NB5 read `cfg.get('shuffle_msc_targets')` to decide the ablation arm.

`shuffle_msc_targets` is a **library function name**, not a config key. It was
set via `sess.config(..., shuffle_msc_targets=bool(shuffled))`, and
`config(**overrides)` accepts any key without complaint. The key the library
actually reads is the `shuffle_targets` **parameter** of `train_msc_kd`.

**What that would have produced.** The invented key enters `config_hash`, so
the two arms hash differently and both run. The run_id says `mscKDshuff...`.
The teacher targets are never permuted. The shuffled-target control — the test
that proves MSC-KD transfers *MSC structure* rather than generic distillation
— would have returned a healthy-looking result under a name asserting it was
the null, and healthy is exactly the direction rule 12 says to distrust. It
would have been read as evidence *for* the method.

**Fix.** Delete the key. Derive the arm from `method`, which is already in the
run_id and is what the status line two cells later already keys on — one
source of truth for the arm, and it is the one that names the artifact.

**Standing note.** `Session.config(**overrides)` silently accepts unknown
keys. It absorbed a typo here; it will absorb the next one. The column checker
caught this one only because the key was *read* in a notebook — a key that is
set and never read is still invisible.

---

## D-55 — the model and its input disagreed on memory format, for 69 epochs

**Symptom.** ResNet-50 trained at a flat **80 img/s** on an RTX 4000 Ada.
1,491 s per epoch, 41 h projected for one run of 100 epochs. Three days of wall
clock produced two thirds of one run out of a planned atlas of twenty-four.

Nothing looked broken. Loss fell monotonically, val top-1 reached **80.6% by
epoch 66**, `*BEST*` kept appearing. The training was correct. It was slow.

**The tell was the flatness.** Across 69 epochs the reading was `80` on 159
lines and `81` on 9. Thermal throttling drifts. Disk contention spikes. Other
processes come and go. A number that never moves is a *fixed tax per batch*,
and that narrows the search enormously — it points at layout, batch size, or a
serial per-batch cost, and away from anything environmental.

**Cause.** `GPUBatchLoader` ends every batch with

```python
x = x.contiguous(memory_format=torch.channels_last)
```

unconditionally, and `base_config` declares `channels_last: True`. Of the
**sixteen** places this library constructs a model, **one** applied that
format: `backbone_dry_run`. Every path that actually trains or measures —
`train_backbone`, `run_oracle`, `train_exit_heads`, `train_msc_kd` — built an
NCHW model and fed it NHWC activations.

cuDNN cannot convolve an input and a weight that disagree on layout. It
converts one of them: per convolution, per batch, forward and backward, across
all 53 convolutions of ResNet-50. The GPU sits near 100% utilisation the entire
time, which is why every "is the GPU busy" check passed. It was busy
transposing.

**Two rules failed, and the second is why it survived three days.**

> **Rule 7 — an invariant in a comment is not a mechanism.** `channels_last:
> True` sat in the config as a declaration that nothing enforced. It was read
> by exactly one function out of sixteen.

> **Rule 8 — test the thing you WROTE, not the things you imported.** The dry
> run applied the memory format. The trainer did not. So `[DRY] backbone dry
> run ok` certified a configuration the real run never executed. The dry run
> was not merely insufficient here — it was *actively misleading*, because it
> was the artifact that authorised starting a three-day run.

**And D-43, which I got backwards.** The throughput benchmark measured
resnet50 at **82 img/s**. I recorded that as "understated, RE-MEASURE pending
(D-43)" and told the user to re-measure before trusting the atlas budget — and
that annotation was still sitting in the NB2 estimate table while the real run
produced **80**. The benchmark had been right to within 2.5%. The prose
dismissing it was wrong.

I dismissed it because I had a prior that a 4000 Ada "should" do ~180, and I
attributed the gap to a known defect in the benchmark rather than to the
pipeline. That is rule 12 inverted: I scrutinised the *unfavourable*
measurement until it went away, and left the favourable estimate standing.
457 GPU-h was the honest number all along, and the plan's 235 was never the
thing that needed explaining.

**Contamination.**

- Every IN-100 training and measurement run so far. All are numerically valid
  — memory format changes arithmetic only through summation order, which AMP
  already forfeits, far below seed-to-seed variation. Nothing needs discarding.
- `IN100_MEASURED_IMG_S` and every budget derived from it describe the broken
  configuration. The whole atlas estimate must be re-derived after measuring.
- The `RE-MEASURE pending (D-43)` annotations on `resnet50` and `vgg16` are
  wrong in the direction they claim and must be removed once re-measured.
- The completed 69 epochs are **not lost**. Memory format is a stride, not a
  parameter: `state_dict` is identical either way, so the checkpoint resumes
  into a channels_last model and simply runs faster from epoch 69 on.

**Fix — one accessor, one assertion, one source check.**

1. `place_model(model, device, cfg, tag)` is now the only sanctioned way to
   put a model on a device. Twelve construction sites route through it.
2. `assert_layout_match(model, x)` runs on the **first batch of the run** and
   *raises*. It compares input format against conv-weight format and fails if
   they disagree. Microseconds, once. This is the mechanism rule 7 asked for:
   the failure it guards produces correct numbers and therefore never
   announces itself.
3. A self-test parses this module's own source and fails if any function in
   the compute set builds a model with a bare `.to(device)`. It ships with a
   canary proving it can fail (D-37).

**The source check earned itself immediately.** On first run it failed on
`train_msc_kd:9815`, `MSCStudent(...).to(device)` — a thirteenth site my
manual grep had missed because I had searched for `build_model` and
`MultiExitModel` and not for the third constructor. A hand audit of sixteen
sites missed one; the mechanism did not.

**Not yet answered.** Whether layout is the *whole* story. `tools/verify_d55.py`
measures the two arms in isolated, VRAM-capped subprocesses;
`tools/diagnose_epochs.py` reads `gpu0_util_mean_pct` and `dataload_frac` out
of the 69 epochs already on disk. No new budget gets written from an estimate.

---

## D-56 — the GPU was idle; the bottleneck was one random read per image

**Symptom.** After the D-55 memory-format fix, unchanged: `1.19 b/s`,
`img/s=58`, `vram=4.3G`. 0.84 s per batch of 64.

**The D-55 fix buying nothing IS the finding.** A compute-path fix that
changes throughput by zero means compute was never the constraint. I should
have concluded that from the flatness alone before writing a line of
`place_model` — a fixed per-batch tax is *equally* consistent with a fixed
per-batch **wait**, and I only considered the first. D-55 was a real defect and
worth fixing; it was not this defect.

`vram=4.3G` of 20 GiB said the same thing twice and I did not read it either.
A GPU that is the bottleneck is usually full. This one was 21% occupied.

**Cause.** `PackedImageDataset.__getitem__`:

```python
img = np.asarray(self._mmap()[g])       # ONE random 192 KiB read
return torch.from_numpy(img), int(self.labels[i]), g
```

Per batch that is 64 independent random reads scattered across a 24 GiB file,
64 Python round trips, a `default_collate` stack of 64 tensors, and 12.6 MiB
pickled through a Windows pipe to the parent process. Measured end to end:
about **15 MiB/s**. NVMe does 1-2 GiB/s at this block size. 15 MiB/s is what a
spinning disk gives on random access — the pack is not being served from cache
and 24 GiB of random reads is the worst possible access pattern for it.

Meanwhile the arithmetic ResNet-50 needs is ~0.07 s of the 0.84 s. The card
spent **92% of every batch waiting.**

**Fix — remove the disk, the per-sample gather, and the IPC together.**

`load_pack_to_ram` reads the pack once, sequentially, into one resident uint8
array; `RAMBatchLoader` gathers a whole batch with a single `arr[idx]` and
hands `GPUBatchLoader` byte-identical `(uint8 NHWC, labels, GLOBAL idx)`. One
prefetch **thread** keeps the gather off the critical path — a thread and not a
process, because Windows spawn would copy 23.5 GiB into every worker, which is
the OOM this design exists to avoid. `num_workers` goes to 0 and there is
nothing left to serialise.

Augmentation stays in `GPUBatchLoader`, untouched. That is the D-40 rule: the
augmentation lives in the loader so that eleven consumer sites cannot each
forget it, and a faster loader must not become a twelfth place it can go wrong.

**Two things this nearly broke, both caught before shipping.**

1. **`.indices` means two different things.** `PackedImageDataset.indices` are
   GLOBAL pack indices; `torch.utils.data.Subset.indices` are POSITIONS in the
   parent. My first `RAMBatchLoader` read `ds.indices` directly, so on the
   subset path it would have trained on images `[1, 3]` where it meant
   `[200, 400]` — numerically valid, silently wrong, images and labels sheared
   apart. This is D-49 through a side door, and the quiet version is worse than
   the one that raised. `pack_view_of` walks the wrapper chain and indexes
   through at each level; the self-test asserts the naive and resolved answers
   *differ*, so it cannot pass against a broken implementation.

2. **`ram_cache` must not enter `config_hash`.** It very nearly did, by being
   an ordinary config key. Hashing it would have made every checkpoint on disk
   unresumable the moment it flipped — 69 epochs of ResNet-50 discarded to
   change a buffering strategy. It is in `_HASH_EXCLUDE` with
   `ram_headroom_gb`, `num_workers` and `prefetch_batches`, and a check proves
   the hash is unchanged. `batch_size` is deliberately NOT excluded: it scales
   the learning rate and is the recipe, not a knob.

**Safety.** `ram_budget_ok` is asked *before* allocating and refuses unless the
pack plus 6 GiB of headroom fits in available RAM, falling back to the memmap
with a reason. Slow is survivable; swapping a Windows box is what cost two
hours and someone else's admin password in D-41.

**Still open.** Whether the resident loader is fast enough to make the model
the bottleneck, and what batch size to run once it is. `tools/verify_loader.py`
measures both arms with no model attached. VRAM at 4.3 of 20 GiB says batch
size has room — but `batch_size` scales the LR, so raising it restarts the
resnet50 run, and that trade is only worth making once a full run costs hours
instead of days. Measure first.

---

## D-57 — two wrong diagnoses from aggregate numbers, next to a file with the answer

**Not a defect in the pipeline. A defect in how I have been debugging it**, and
it has now cost more of the user's time than any code defect in this project.

**The record.**

| | claim | fix | result |
|---|---|---|---|
| D-55 | model is in the wrong memory format | `place_model` everywhere | **0% change** |
| D-56 | loader is reading from a slow disk | resident RAM pack | **0% change** |

D-56 disproved its own premise on the way past: the RAM cache read 23.7 GiB at
**3.40 GiB/s**. That is NVMe. The "~15 MiB/s, spinning-disk territory" figure I
built the whole diagnosis on was not measured — it was inferred from throughput
by assuming the answer.

**The number that was there the whole time.** `0.045 kWh` per `1491 s` epoch is
**108 W on a 130 W card** — 83% of TDP, logged on every epoch line since epoch
1. A starved GPU idles at 20-30 W. This one was working flat out and returning
80 img/s, which is the signature of *memory-bound work*, not of a stall. Both of
my diagnoses were stall theories. Both were excluded by a number printed on
every line I had already read.

**Why it kept happening.** Aggregate throughput cannot distinguish waiting from
augmenting from computing. Every argument built on it is a guess dressed as an
inference. `dataload_frac` and `augment_frac` have been computed since D-40 —
but written only to `epochs.csv`, which nobody opens during a run. I twice
proposed a tool to read them and twice moved on to a fix before the number came
back, because a fix feels like progress and waiting does not.

Rule 1 says dry-run the whole path before expensive work. The debugging
equivalent -- measure the whole path before an expensive fix -- is the same
rule, and I broke it twice.

**Fix.**

1. **The split is on the progress bar**, refreshed with everything else:
   `wait%` (blocked on the next batch), `aug%` (GPU augmentation), `step` (ms
   of forward+backward+optimizer). Whichever dominates is the answer. No tool,
   no file, no theory. Backed by two real accessors — `EpochTelemetry.
   load_seconds()` and `GPUBatchLoader.augment_seconds()` — because my first
   version of this called `tel.load_time_sec()` and `loader.augment_share()`,
   neither of which exists. That is D-39 exactly, in the patch written to stop
   guessing.
   `augment_seconds()` returns `None` rather than `0.0` before its first
   sample: a confident zero is how you conclude augmentation is free when you
   have merely not measured it.

2. **`tools/bisect_speed.py`** adds one thing at a time — compute alone,
   +augmentation, +real data, +instrumentation — in isolated VRAM-capped
   subprocesses. The stage that drops throughput is the cause. Stage 1 also
   answers the question neither D-55 nor D-56 asked: *what can this card do at
   all?* If a 130 W Ada card at batch 64 and 224px tops out near 150 img/s
   with no loader attached, then no amount of plumbing was ever going to help
   and the levers are batch size, resolution and architecture.

**Open.** Everything. The cause is not yet known and this entry does not claim
one. The next entry gets written from the bisection output, not before it.
