# Lab Notebook — ImageNet-100 Port

**Running record of every defect and every decision that changed.**

Append-only, newest first, same contract as
[`09_LAB_NOTEBOOK.md`](../cifar100/09_LAB_NOTEBOOK.md): when the paper is written, three
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
  [`09_LAB_NOTEBOOK.md`](../cifar100/09_LAB_NOTEBOOK.md) overstates its evidence.** Those
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

---

## D-59 — channels_last was 6.7x SLOWER, and D-55 enforced it

**Measured**, `tools/conv_sweep.py`, ResNet-50 @224 bs64, RTX 4000 Ada /
cuDNN 9.1 / driver 581.42:

| configuration | img/s | ms/batch |
|---|---|---|
| current (channels_last + AMP + benchmark) | 81.6 | 784 |
| **no channels_last** | **550.3** | **116** |
| no cudnn.benchmark | 81.7 | 783 |
| no AMP (fp32) | 56.5 | 1133 |
| plain torchvision, no `StagedBackbone` | 81.1 | 790 |
| batch 128 | 66.9 | 1913 |
| batch 256 | 59.0 | 4341 |

One variable. **6.7x.** ResNet-50 for three seeds goes from 122 h to 18 h.

The controls matter as much as the answer. `plain torchvision` at 81.1 clears
`StagedBackbone` — had that row been fast, every cuDNN theory would have been
irrelevant and I would have kept tuning the wrong layer. `no cudnn.benchmark`
at 81.7 retires the D-43 hypothesis I had been carrying since the first
benchmark. And batch 128/256 getting *worse* under the bad layout shows how a
sweep on one axis misleads when a different axis is broken: a batch-size study
run yesterday would have concluded "bigger batches hurt on this card", which is
false.

**Root cause.** `GPUBatchLoader` ended every batch with

```python
x = x.contiguous(memory_format=torch.channels_last)
```

unconditionally — since the first ImageNet-100 commit. The config carried a
`channels_last` flag that **only the model ever read**, so the two could never
visibly disagree and the flag could never turn the behaviour off.

**And this is where D-55 goes badly wrong.** I found sixteen model sites, one
of which applied the format, and concluded the fifteen were wrong. The
alternative reading — that the *one* was wrong, and the loader with it — never
got considered, because channels_last is standard advice for convnets under
AMP and I treated "usually true" as "true here". So I wrote `place_model`,
routed thirteen sites through it, and added `assert_layout_match` to *raise* if
anything ever disagreed. I built a mechanism to enforce a choice nobody had
measured, and made the slow path harder to escape.

Rule 2 says never hardcode a value; ask. I asked the config, and the config was
repeating my assumption back to me.

The mechanism was still right. `place_model` and `assert_layout_match` are what
make `channels_last: False` land at all thirteen sites at once and stay
consistent. Only the direction was wrong, and it took a measurement — not an
argument — to find that out.

**Fix.**

1. `GPUBatchLoader` takes `channels_last` and honours it. The flag now reaches
   the line that was ignoring it.
2. `base_config` defaults ImageNet-100 to `channels_last: False`, with the
   measurement in the comment and an instruction to re-run `conv_sweep.py` on
   any new machine rather than inherit the number.
3. `channels_last` joins `_HASH_EXCLUDE`. Memory format changes floating-point
   summation order and nothing else — the forfeit AMP already makes, far below
   seed-to-seed variance. Hashing it would have orphaned resnet50 s1+s2 (100
   epochs each) and vit s2 (73) the instant the measurement said to flip:
   **90 hours discarded over a stride.**
4. `IN100_MEASURED_IMG_S` is corrected. Every convolutional entry was taken
   under the slow layout and is now marked STALE with `IN100_PENDING_REMEASURE`
   widened from two architectures to five. The numbers were real; the
   configuration was wrong, which is a worse failure than a missing number
   because it looks like data.

**Contamination.** None of the finished runs is invalidated — resnet50 s1
(82.62%) and s2 (82.12%) are numerically valid and stay comparable to seed 3
run under the new layout. What was lost is time: roughly 35 h per ResNet-50
seed, twice.

---

## D-60 — the fix that protected 90 hours is what orphaned them

**Symptom.**

```
RuntimeError: config_hash mismatch for p0-vit_small_p16-imagenet100-base-s2:
  checkpoint 971a9a257b42 != config 7f5ee93dcca1
```

73 epochs, refused.

**Cause.** `config_hash` hashes everything **except** `_HASH_EXCLUDE`. So
*adding* a key to that set changes the hash of every config that has ever
existed — the key leaves the hashed space entirely. In D-59 I added
`channels_last` to the exclusion **specifically to stop the flip from orphaning
finished runs**, and that addition is what orphaned them.

**The test passed.** That is the part worth keeping.

```python
check("D-59: flipping channels_last does not change config_hash",
      config_hash(dict(c, channels_last=True))
      == config_hash(dict(c, channels_last=False)))
```

Both sides are computed under the **new** rule, where the key is excluded from
both. It is `hash(x) == hash(x)`. The assertion cannot fail — for any key, in
any project, forever — and I wrote it, watched it pass, and reported that the
runs were safe.

D-37 was a self-test that could not fail. This is the same defect in a single
line, written 400 checks later, in the check whose entire job was to prove a
change was safe. The lesson does not generalise by having been learned once.
**A test of an invariant across versions must compare across versions.** Both
sides being green is not the same as the invariant holding.

**Fix.** `_HASH_EXCLUDE_HISTORY` records every rule this project has hashed
under. `hash_compatible(cfg, stored)` re-includes the keys excluded *now* but
hashed *then*, tries each plausible past value, and asks whether any assignment
reproduces the stored hash. If one does, everything else in the hash is
byte-identical and the difference is confined to keys since declared
performance-only.

Verified against the real artifact rather than a fixture: reconstructing with
`channels_last=True` under rule v1 reproduces `971a9a257b42` exactly.

**It cannot launder a real change.** `lr`, `batch_size`, `num_epochs`, `arch`
and `seed` are never excluded, so no substitution of a performance key can
reproduce a hash differing in a recipe key. Three checks assert precisely that,
each with a changed recipe key that must still be refused, plus a canary
asserting the old and new hashes genuinely differ — without which the whole
test would prove nothing again.

**Contamination.** One run blocked, none lost. But the same mismatch would have
hit `resnet50-s1`, `s2` and `vit-s1` the moment anything touched them, and the
documented remedy — `force_rerun=True` — would have silently discarded 90 hours
while looking like the correct instruction.

**Standing.** Every future addition to `_HASH_EXCLUDE` must append the previous
set to `_HASH_EXCLUDE_HISTORY` in the same commit. The two are one operation.

---

## D-61 — a defensive default that does not defend

**Symptom.** NB2 raises after `run_all` returns. Training had already
succeeded.

```
TypeError: unsupported format string passed to NoneType.__format__
```

**Cause.** The summary loop:

```python
f"top1={r.get('best_accuracy', float('nan')):.2f}"
```

`dict.get`'s default fires only when the key is **absent**. A key that is
present and `None` goes straight to `format()`. Paused, failed and skipped runs
all report `best_accuracy: None` — present, and null.

So the line crashed on exactly the runs whose status the operator most needed
to read, and it crashed *after* the training finished. A successful epoch looks
like a broken notebook. `vit_small_p16-s2` sat at `paused`, epoch 73, having
resumed correctly through the D-60 fix, and the notebook still reported an
error.

The `float('nan')` is what makes it worse than a plain oversight: it is a
visible act of care that does nothing. Reviewing that line, the eye stops at
the default and moves on.

**Fix.** `M.fmt_metric(value, spec)` — None- and NaN-safe, in the library
rather than spelled into a cell (D-39), and a build-time check that refuses any
notebook applying a numeric format spec to a `.get()`.

**And the check itself needed narrowing, which is the second lesson.** The
first version flagged subscripts too, and reported **9 problems for 1 real
defect**: `est['total_gpu_hours']`, `g[col].mean()` and six more, every one of
them healthy. A subscript that misses raises `KeyError` — loud, immediate,
impossible to miss. `.get` is the one that quietly hands back a `None`.

`validate_notebooks.py` already says this in its own docstring, about its own
first draft:

> A check that fires on healthy data teaches you to ignore it, and the next
> alarm is the real one -- which is precisely what D-17 and D-20 cost.

I wrote that sentence and then shipped a checker that did it. The narrowed
version flags the trap and nothing else, and its self-test now asserts that two
healthy subscript lines are **accepted**, not merely that the bad line is
caught.

---

## D-62 — the fix was correct, and the code that ran was not the code on disk

**Symptom.** Byte-identical to D-60, after D-60 was fixed, verified and
regenerated:

```
RuntimeError: config_hash mismatch for p0-vit_small_p16-imagenet100-base-s2:
  checkpoint 971a9a257b42 != config 7f5ee93dcca1
```

**What the evidence said, and why it was so confusing.** Everything checked out:

- `notebooks_in100/msc_lib.py` was current — 667,507 bytes, matching `src/`.
- It contained `hash_compatible` and `_HASH_EXCLUDE_HISTORY`.
- The traceback's line 7381 landed exactly on the `raise` in the **new** code,
  four lines *after* the `hash_compatible` call.
- Running that same file against the run's real `config.yaml` returned
  `True (rule v1, channels_last=True)` and reconstructed `971a9a257b42` exactly.
- A faithfully rebuilt runtime config returned `True` too.

Every artifact said the fix was present and working. The run said otherwise.

**Cause.** The code executing was not the code on disk. Jupyter holds an
imported module until something removes it, and — the part that makes this
survive a bootstrap re-run — **any object built from the old module keeps the
old functions**. `sess` is a `Session` whose `.train` closes over the *previous*
`train_backbone`, which calls the *previous* `load_checkpoint`, which has no
D-60 in it. Re-running the bootstrap cell replaces `sys.modules['msc_lib']` and
cannot reach inside an object that already exists.

So the fixed library sat on disk, unused, while its predecessor produced the
old error. Nothing in the system could tell the two apart, and the evidence
therefore read as "the fix does not work" when it was "the fix never ran". That
misreading cost two rounds.

**Rule 5, exactly:** *a completion cache must answer "is what I have still
VALID?", not just "do I have something?"* An imported module is a cache. The
notebook asked whether `msc_lib` was importable. It never asked whether it was
**this** `msc_lib`.

**Fix — the build stamp is written into the bytes, so it cannot drift.**

1. `bootstrap()` appends `__MSC_BUILD__ = "<sha12>"` to the library bytes
   *before* base64-encoding them, and the cell asserts the imported module
   reports that exact stamp. A stale module fails immediately, by name, with
   "restart the kernel" — instead of running.
2. `run_all` compares `Session.run_all.__globals__["__MSC_BUILD__"]` — the
   module that *defined* the method, which is the one that will actually
   execute — against the live `sys.modules['msc_lib']`. This catches the case
   the bootstrap cannot: a fresh module and a **stale object**.

**Verified both directions.** The first version of this test was worthless:
`M.__MSC_BUILD__` and `Session.run_all.__globals__` are *the same dict*, so
setting one set the other and the guard never fired. The real scenario has two
distinct module objects, and the test now builds two. It asserts the guard
fires on mismatch **and** stays silent on agreement — a guard that always fires
would break every run.

**Contamination.** D-60 and D-61 were both correct when written and both
appeared to fail. Any conclusion drawn from a run in this window is suspect,
including my own reasoning about which fix was needed.

**Standing note.** `src/msc_lib.py` carries no `__MSC_BUILD__`; it is appended
at build time. So `_mine` is `None` when running from source and the guard is
inert there by design — it exists for generated notebooks, which is where the
staleness lives.

---

## D-63 — the tests agreed with me instead of with the program

**Symptom.** The D-60 error again, third time, with D-60 verified present and
D-62 confirming the right build was loaded.

**What made it hard.** Every check I could run said the fix worked:

- `hash_compatible(config.yaml, stored)` → `True`, reconstructing
  `971a9a257b427f20...` in full, read out of the checkpoint's own pickle.
- A reconstructed runtime config → `True`.
- Library on disk current, build stamp `87c16f6986cc` matching the notebook.

**Cause, and it was in my own function.** `hash_compatible` recomputed
`config_hash(cfg)` while everything around it used the *stored*
`cfg["config_hash"]`. By the time `load_checkpoint` runs, `cfg` has gained keys
that were not in the dict whose hash was taken, so those two are different
numbers — and every probe built on the drifted dict misses.

Each of my tests passed a **clean** config, which is the one shape the runtime
never has. So the function returned `True` for me and `False` on the machine,
every time, reproducibly. That is the most expensive shape a bug can have: the
tests confirm the author's mental model rather than exercising the program's
actual inputs. It is D-37 and D-60's flaw a third time — not "a test that
cannot fail", but "a test of a situation that never occurs".

**Fix — probe the RECORD, not a reconstruction.** `runs/<id>/config.yaml` is
written at claim time and is what this run *is*. Now:

1. probe the live config (fast path, clean resume);
2. probe the record; if it reproduces `stored`, the checkpoint provably
   belongs to this run;
3. require the live config not to **change** any key the record has. Keys it
   merely **adds** were in no hash and cannot alter a result; a changed value
   is a genuine edit and is still refused, by name and by value.

**Verified against the real artifacts**, not fixtures: the true 64-character
hash extracted from `ckpt_last.pt`'s pickle, and the run's own `config.yaml`.
Six cases — clean, two drifted, and `batch_size` / `num_epochs` / `seed`
changes — all correct. The self-test now builds a drifted config on a real
temporary record, and its canary asserts that **without** the record the
drifted config still fails, which is precisely what happened on your machine.

**Two things fixed on the way.**

- `read_yaml` did not exist. `atomic_write_yaml` had been writing `config.yaml`
  for the whole project and nothing had ever read one back — a writer with no
  reader, which is why the record was never consulted.
- The D-62 build stamp broke the base64 round-trip check, which compares the
  embedded blob to the source file; the stamp is appended after. It failed on a
  correct build. Fixed to compare against the stamped bytes, and to report
  *where* the first differing byte is rather than only that one exists.

**The error message now names what changed.** Three rounds were spent guessing
at a dict the program was holding and could have printed. It says
`batch_size: 64 -> 128`, or that only runtime keys were added.

---

## D-64 — the artifact spec disagreed with the code that writes

`verify_run_artifacts(measured=False)` reported **all four** completed Phase-0
runs as incomplete: `missing_required: ['metrics/final.csv']`.

`metrics/final.csv` sat in `RUN_ARTIFACTS_REQUIRED`, which is checked after
**training**. But `final_evaluation()` is the only thing that writes it, and it
is called from exactly one place — inside `run_oracle`, the **measurement**
stage. So the file cannot exist until NB3 runs, and every healthy training run
verified as broken.

Nothing was lost; the file arrives with NB3. The cost is the one this project
keeps paying: a verifier that flags healthy runs trains you to skim its output,
and the next alarm is the real one. That is D-17 and D-20's lesson, and D-61's,
in a third place.

**Fix.** `final.csv` moved to `RUN_ARTIFACTS_MEASURED`. And because the list
and the writers are two spellings of one truth (D-16), a self-test now parses
this module's own source, maps every artifact filename to the functions that
write it, and fails if anything in the train-stage REQUIRED list is written
only by `run_oracle`. Its canary asserts the map can see `run_oracle`'s outputs
at all, without which the check would pass by seeing nothing.

---

## D-65 — three notebooks hardcoded a phase the pilot never writes

NB2 trains `p0`. NB3, NB4 and NB5 each opened with `PHASE = 'p1'`.

Run them in the documented order, unedited, and NB3 finds zero `p1` runs,
prints `0 trained run(s), 0 still to measure`, calls `run_all([])`, and **exits
successfully**. Nothing failed. Nothing happened. NB4 then has nothing to
analyse, for a reason three notebooks upstream, and the only symptom anywhere
is a line saying zero — which reads like "already done".

A default that is wrong for the documented order is not a default, it is a
trap. Silence is the worst way to spring one: every other defect in this log
announced itself with a traceback.

**Fix.** `detect_phase(work, prefer=)` reads the phases actually on disk and
returns the one with completed runs, logging when it overrides the preference
and **raising** — listing what is present — rather than returning a phase with
no work in it. Notebooks that *create* runs still name their phase; notebooks
that *consume* runs detect it, via the shared `paths_cell`, so the three cannot
drift apart again. `phases_present()` is filesystem-only: no Session, no
ledger, no data directory, because its job is to tell you what to configure.

Verified against the real results root: `prefer='p1'` resolves to `p0`, with
`[PHASE] phase 'p1' has no completed runs; using 'p0' (4 completed)`.

**Contamination.** None — NB3 had not been run yet. Had it been, it would have
appeared to succeed.

---

## D-66 / D-67 — NB3 reported success and measured nothing; NB4 died on the empty table

**Symptom.** `KeyError: 'rho_seed_tau0.1'` in NB4, on
`q1.sort_values('rho_seed_tau0.1')`.

The column name is **correct** — `analyse_q1_all` builds exactly that key. The
frame was **empty**: no rows, therefore no columns. And it was empty because
NB3, which the user reported as having "run successfully", had produced nothing
at all. Every oracle artifact was missing on all four runs:
`test.parquet`, `train_holdout.parquet`, `exit_heads.pt`, `final.csv`,
`confusion_matrix.csv`, `per_class.csv`, `exit_metrics.csv`.

**D-67 — the cause.** NB3 called

```python
sess.run_all(cfgs, fn=sess.oracle, title='measurement')
```

`plan_work` filters out runs that are already "done" **before** `fn` is ever
called, and "done" means whatever `stage` and `done_fn` say. The default is
`stage='train'`. All four runs *were* trained, so all four were filtered as
complete, `MY REMAINING WORK: 0`, and the notebook exited successfully.

The CIFAR generator has had this right all along:

```python
sess.run_all(ready, fn=sess.oracle, title='...',
             done_fn=sess.measured, stage='measure')
```

**This is D-31, verbatim.** D-31 was a validity check placed downstream of the
predicate that decides whether to do the work, so it could never fire. Its
lesson is written in the `msckd_valid` docstring **three screens above the line
I wrote**: *"A compatibility test has to live in the predicate that decides
whether to do the work, not in the code that does it."* I read that docstring
while fixing D-54 and still shipped this. Documenting a trap is not removing
it.

**D-66 — why the error pointed somewhere else.** Every `analyse_*_all` defaulted
to `phase="p1"` internally. NB4 called them with no phase argument, so even
with the notebook-level phase detection from D-65, the *library* still indexed
`p1` and found nothing. D-65 fixed this default in the notebooks; the same
default was sitting one layer down where a notebook fix could not reach it.

So two independent causes produced one misleading error, and it named neither:
a column, in a display line, three notebooks from the phase and two from the
measurement.

**Fixes.**

1. `run_all` **raises** if `fn` is `sess.oracle` and `stage != 'measure'`,
   naming the correct call. It also defaults `done_fn` to `sess.measured` when
   the stage is measurement. Silence is not an option this path gets any more.
2. `resolve_analysis_phase` — the five `analyse_*_all` entry points take
   `phase=None` and resolve through `_run_index`, the single choke point they
   all pass through.
3. `_require_runs` refuses to analyse an empty index and says **which** of the
   three causes it is: no completed runs, runs present but unmeasured (naming
   them), or present but unusable. An empty DataFrame no longer travels two
   lines to become a `KeyError` about a column.
4. Build-time: `_stage_problems` rejects any `run_all(fn=sess.oracle)` without
   `stage='measure'` and `done_fn`; `_column_arg_names` extends the column
   check to `sort_values`/`groupby`/`set_index` arguments, which
   `_defined_and_read` never looked at because it only walked subscripts.
   Both ship with self-tests asserting they catch the exact failing line and
   accept the correct one.

**Contamination.** No data lost — nothing had been computed. `analysis/
q1_seed_ceilings_all.csv` on disk is an empty artifact written from the empty
frame before the KeyError, and is overwritten on the next run.

---

## D-68 — Jupyter saved the stale notebook over the fixed one

**Symptom.** NB3 was regenerated with `done_fn=sess.measured, stage='measure'`
(D-67). It was re-run. It measured nothing again, and the plan on disk said
`n_todo: 0`.

**Not a code defect.** The file timestamps tell the whole story:

```
13:43:53   registry/plans/...json   NB3 ran -> todo=0   (no done_fn, no stage)
13:46:22   notebooks_in100/msc_lib.py   rewritten, current
13:50:15   notebooks_in100/NB3_Measure.ipynb   SAVED
```

The notebook was **saved after it ran**. Jupyter writes an open notebook to
disk on execution, so the copy sitting in the tab — opened before the
regeneration — overwrote the fixed file. NB3 on disk then embedded build
`7624d306be9c` while `src/msc_lib.py` had moved to `4a0d5f870b7b`.

Regenerating a notebook that is open in an editor does not change what runs.
The editor's copy wins, silently, at the moment of running.

**Why D-62 could not catch it.** That check compares the imported module
against the notebook's own embedded stamp. Both sides come from the same
`.ipynb`, so they always agree — and can be arbitrarily old together. It
verifies internal consistency, which is not currency.

That is rule 5 one level up. D-62 asked "is the module I loaded the module this
notebook ships?" It never asked "is this notebook the current notebook?"

**Fix.** Cell 1 now hashes `../src/msc_lib.py` — the repository source, the
actual authority — and refuses to run if it differs from the embedded build,
naming both hashes and the remedy: close without saving, rebuild, reopen.

**Operationally.** Regenerating a notebook the user has open is not enough, and
"re-run it" is the wrong instruction. The notebook must be **closed without
saving and reopened**. Every previous "still failing" report in this session is
consistent with this mechanism, and D-62 was only half of it.

---

## D-69 — the checkpoint was one directory away, and the right path was in dead code

**Symptom.** NB3 planned correctly at last (`stage: measure`, 4 runs), then
failed on all four:

```
FileNotFoundError: no ckpt_best.pt for p0-resnet50-imagenet100-base-s1.
  Train the backbone first (notebook 02).
```

The backbone *had* been trained. `ckpt_best.pt` was 91 MB, on disk, one
directory away.

**Cause.**

```python
ckpt = run_dir / "ckpt_best.pt"          # the run ROOT
```

Checkpoints live in `checkpoints/`. And the function knew it — three lines
below, the HuggingFace fallback read

```python
alt = L["checkpoints"] / "ckpt_best.pt"  # correct
```

**The correct spelling was already there, in unreachable code.** With HF
removed for local-only operation, `hub.enabled` is False, that branch never
runs, and the only surviving spelling was the wrong one. Removing HuggingFace
did not create this bug; it removed the thing that had been hiding it.

That is a specific hazard worth naming: when a fallback is deleted or disabled,
any correctness that lived *only* in the fallback goes with it. The primary
path had presumably never been exercised on a machine where the fallback could
not save it.

**This is D-16 and D-23 for the third time.** D-16 was a path spelled two ways
and closed as "cosmetic". D-23 was the same file, `exit_heads.pt`, where writer
and readers disagreed and every MSC-KD run silently retrained the teacher's
heads. `exit_heads_path()` was created *specifically* so that path would have
one spelling — and thirty lines from where it is defined, `run_oracle` was
still spelling it by hand as `run_dir / "exit_heads.pt"`. Both are now routed
through the accessors.

Rule 4 says never spell a repo path as a literal; go through one accessor. The
accessor existed. The rule was written down. Neither is a mechanism.

**Fix.**

1. `ckpt = L["checkpoints"] / "ckpt_best.pt"`; `heads_path =
   exit_heads_path(work, run_id)`.
2. The error message now prints the path it looked at, whether `ckpt_last.pt`
   is present, and that `MSC_ROOT` may be pointing elsewhere — instead of
   asserting a false cause ("train the backbone first") about a run that had
   trained for 41 hours.
3. A self-test walks this module's AST for `run_dir / "<name>"` and fails if
   `<name>` appears in the artifact lists under a subdirectory. The lists
   already record where each file belongs, so this compares two existing
   statements rather than inventing a third.

**The check needed rewriting too.** Its first version used a regex and matched
its own explanatory comment and its own pattern string — 2 reported problems
for 1 real one. Rewritten over the AST, with canaries proving it catches
`run_dir / "ckpt_best.pt"` and accepts both `L["checkpoints"] / "ckpt_best.pt"`
and `run_dir / "summary.json"`, which legitimately lives at the run root.

**Contamination.** No data lost. Four measurement attempts wasted, and the
error text actively misdirected — it told the user to retrain models that were
already trained.

---

## D-70 — one loader yields CPU labels, the other device labels

**Symptom.** NB3 got further than ever: exit heads trained
(`d1=0.2504 … d5=0.8257`), final evaluation written
(`top1=0.8262`, matching the trained value exactly), sweep started — then, at
the first batch of the depth axis:

```
TypeError: can't convert cuda:0 device type tensor to numpy.
           Use Tensor.cpu() to copy the tensor to host memory first.
  chunks_l.append(np.asarray(y).astype(np.int64))
```

Roughly 40 minutes in, after two expensive stages had succeeded. The most
costly place a one-line conversion bug can sit.

**Cause.** `sweep_all_axes._collect` does `np.asarray(y)` on the label tensor.

- **CIFAR**: `build_loaders` returns raw `DataLoader`s → `y` is on the host →
  works.
- **ImageNet-100**: the batch comes through `GPUBatchLoader`, which ends with
  `yb = y.to(self.device)` → `y` is on cuda:0 → numpy refuses.

**This is the port's central premise failing at a seam.** The design was one
library parameterised by dataset rather than forked — good, and it is why the
zoo, the training loop, the resume machinery and the analysis are all shared.
But it holds only where the two datasets present the **same interface**, and
here they do not: one loader hands back host labels, the other device labels.
Three call sites had quietly encoded the CIFAR shape.

A parameterised library needs its parameterisation to be *total*. Where a
difference remains, it has to be absorbed at one boundary — which is exactly
what `GPUBatchLoader` was supposed to be, and where the augmentation D-40
correctly put. The labels were left half-converted.

**Fix.** `to_numpy(v, dtype=None)` — accepts a tensor on any device or anything
array-like, and is the single conversion all three sites now go through. A
self-test asserts a `np.asarray` on a value named `y`/`idx`/`yb` cannot
reappear anywhere in the module, and its canary confirms that bare
`np.asarray` *does* work on a CPU tensor — which is precisely why the CIFAR
path never exposed this.

**Also visible in that output, and expected:**
`[WARN] checkpoint config_hash differs from the current config` is the D-59
`channels_last` flip being correctly recognised as performance-only by D-60's
`hash_compatible`, and recorded rather than ignored.

**Contamination.** None — the sweep writes nothing until it completes. Two
prior stages (exit heads, final evaluation) had already written correctly and
are reused on the next run.

---

## D-71 — a membership test across two identifier spaces emptied Q3 and Q4

**Symptom.** NB4: `KeyError: 'passed'` on `ctrl[~ctrl['passed']]`, with
`q3_shuffled_control.csv` **2 bytes** on disk. Q1 and Q2 had produced real
results in the same run.

**Cause.**

```python
if require is not None and rid not in require:      # rid is a RUN ID
```

`require` is only ever given `_ceilings(...)`, which is keyed by
**architecture** (`resnet50`). `rid` is a **run id**
(`p0-resnet50-imagenet100-base-s1`). No run id is ever a member, so every run
was skipped, `cand` stayed empty, `reps` came back `{}` — and every caller that
passed `require` got nothing.

Three analyses at once: Q3 axis structure, Q3 shuffled control, Q4 difficulty.
Q2 was unaffected **only because it passes no `require`**, which is why the
notebook produced two good tables and then failed — the most confusing possible
presentation of a single upstream fault.

The docstring was correct: *"an ARCHITECTURE is only represented by a run that
appears in it"*. The prose described one identifier space and the code tested
the other. This is D-49 and D-56b again — `sample_idx` vs split position,
`PackedImageDataset.indices` vs `Subset.indices` — a third pair of identifier
spaces where one is silently accepted in place of the other.

**The existing self-test asserted the bug.** D-18's fixture built

```python
_ceil = {"p1-vgg8-cifar100-base-s2", ...}     # a set of RUN IDS
```

so it exercised the buggy semantics and passed, for as long as that test has
existed. Every real caller passes an arch-keyed dict. That is D-63's lesson a
second time: **the fixture had the shape I imagined, not the shape the program
receives.** A test whose fixture is wrong in the same direction as the code
cannot fail.

**Fix.**

1. Test `arch not in require`, after `arch` is resolved.
2. `representative_runs` **raises** if `require` excludes every run while runs
   exist — printing both key spaces so the mismatch is visible in the message.
   Silently returning `{}` is what let this reach the notebook.
3. `analyse_q3_all` and `analyse_q3_shuffled_control_all` raise on zero pairs
   instead of returning an empty frame, naming how many architectures have a
   ceiling and that two are needed.
4. The D-18 fixture is now arch-keyed, plus three D-71 checks: a run-id-keyed
   `require` must raise, the arch-keyed one must still return both
   architectures, and an genuinely empty `runs` must not be mistaken for a
   key-space error.

**Contamination.** Q1 and Q2 are unaffected and their published numbers stand.
Q3 and Q4 produced nothing, so nothing wrong was recorded. `q3_shuffled_control
.csv` on disk is a 2-byte artifact of the failed run and is overwritten.

---

## D-72 — NB5 could not read the gate that decides whether to run it

NB4 produced all four analyses. The pre-registered gates:

| gate | measured | verdict |
|---|---|---|
| ρ_seed ≥ 0.60 | resnet50 **0.822**, vit **0.649** | PASS |
| shuffled control, \|z\| < 5 | z = **2.30** (T_shuffled 0.037 vs T 0.640) | PASS |
| partial ρ ≥ 0.30 | **0.282** | **MISS by 0.018** |

Q4's ΔR² is 0.0411 with CI [0.0352, 0.0468] — excludes zero, so MSC does carry
information beyond the seven-score difficulty battery. But the partial Spearman
falls just under its pre-registered threshold.

**That is exactly the number NB5 exists downstream of.** MSC-KD's premise is
Q4's: that MSC is worth distilling because it is not already available from
`msp`, `margin`, `entropy`. And NB5 had no way to consult it. `save_analysis`
had **no reader** — the third writer in this library without one, after
`atomic_write_yaml` (D-63) and the config record. So every gate in the protocol
was a thing a human had to remember to eyeball, on the way into an 18-run
commitment.

**Fix.** `load_analysis()` and `gate_report()` read the gates as data. NB5 now
opens with a gate table, a cost estimate that flags architectures whose
throughput is STALE (D-59), and `CONFIRM = False` — it raises rather than
training until the numbers have been looked at. Missing analyses report as
`None`, never as a pass: a gate that has not been evaluated is not a gate met.

**A near-miss reported as a near-miss.** 0.282 against 0.30 is the kind of
result rule 12 exists for — it would be easy to call it "essentially 0.30". It
is not 0.30. It is recorded as a MISS, and NB5 says so before it spends
anything.

---

## D-73 — six validation layers, none of which checked the code parses

The build shipped five notebooks whose **first cell was a syntax error**:

```python
f"module reports {_got}.
"
```

A real newline inside a double-quoted f-string. `\n` written where `\\n` was
needed — trivially easy when a generator emits code through an f-string of its
own, and I introduced several while writing the D-62 and D-68 guards.

**Every check passed.** Column names, repo paths, library names, call arity,
result keys, stage predicates — six layers, and the notebook could not run at
all. They could not catch it, and the reason is worth stating plainly: **every
layer begins with `ast.parse` inside a `try`, and a cell that fails to parse is
silently skipped rather than reported.** A malformed cell was invisible to the
entire validator *by construction*. The more broken the cell, the less it was
checked.

Ten broken lines across four notebooks, including the cost estimate I had just
added to NB5 for D-72.

**Fix.** The build now parses every emitted code cell with
`ast.parse(..., feature_version=(3, 10))` and refuses to generate on failure.
Pinned to 3.10 because that is what the machine runs; syntax accepted by the
build host and rejected by the target is the same defect with a longer feedback
loop.

The check found all seven failing cells on its first run, and the last two only
after the first fixes moved the error line — which is itself the argument for
having it in the build rather than running it by hand.

---

## D-74 — a correct stop that looked like a crash, and a sweep nobody read

The D-72 gate cell worked: it printed the gate table, showed 18 runs at 112.2 h
= 4.7 days, and stopped before training. Two things were wrong with *how*.

**1. `SystemExit` in a notebook is indistinguishable from a failure.**

```
An exception has occurred, use %tb to see the full traceback.
SystemExit: Set CONFIRM = True ...
UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
```

A red traceback and a warning about how to quit Python, for a deliberate,
correct, expected stop. In this project specifically that is a bad choice:
distinguishing "it failed" from "it did the right thing" has already cost
several rounds, and I added a mechanism that makes them look the same.

The gate cell now prints a plain banner, and the training cell simply does
nothing while `CONFIRM` is `False`. No exception for an expected control flow.

**2. The cost table said STALE at a user who had been told to fix it.**

`tools/conv_sweep.py` writes a corrected throughput to
`benchmark/convsweep_<arch>_*.json`. Nothing read it. So running the sweep — as
the notebook instructs, in the very next line — changed nothing the notebook
showed. **A fourth writer with no reader**, after `atomic_write_yaml` (D-63),
`config.yaml` as a record (D-63), and `save_analysis` (D-72).

That is now a pattern rather than three coincidences: this library has
repeatedly produced artifacts nothing consumes. The write is the visible half
of the work and the read is the half that makes it matter, and I have shipped
the first without the second four times.

`measured_img_s(arch)` returns `(img_s, basis)` and prefers a `conv_sweep`
result on this machine over the stale table. Verified: with a sweep file
present, `resnet18` moves from 413 img/s "STALE" to the swept figure and the
budget for 3 seeds × 2 arms drops accordingly — 48.2 h to 10.8 h on a
simulated 1840 img/s. The notebook picks it up with no edit.

NB5 also now prints the exact commands for whichever students are still stale,
and says plainly that the true cost is **lower** than shown, so the number
cannot be read as pessimism about the method.

**Not a defect.** `metrics/exit_metrics.csv` is 269 bytes and correct — exit,
depth_fraction, rho, flops, stage_cut, feature_dim for all five exits. It was
reported as "0K" by my own status check, which rounded 269 bytes to zero KB. A
display bug in the reporting, not a gap in the data. Corrected in
`24_IN100_STATUS.md`.

---

## D-75 — "control first, so a null result stops you early" did the opposite

NB5 built its 18 configs arm-major:

```python
for shuffled in ARMS:          # [True, False]
    for a in STUDENTS:
        for s in SEEDS:
```

All nine SHUFFLED runs, then all nine real ones. The comment above `ARMS` read
*"control FIRST, so a null result stops you early"* — and I wrote D-54b's fix
directly beneath it without noticing it is wrong.

**A control arm alone cannot produce a null result.** The claim is
real-versus-control; neither arm means anything without the other. So the order
that was supposed to enable early stopping guaranteed that **no comparison was
possible until run 10**, after roughly half the budget. The stated intent and
the actual effect were opposites, and the intent was written down, which is
what made it invisible.

**Fix — seed-major, arms adjacent:**

```python
for s in SEEDS:
    for a in STUDENTS:
        for shuffled in ARMS:      # control first WITHIN each pair
```

After **6 runs** you have all three students in both arms at seed 1 — a
complete n=1 comparison at a third of the cost. If the shuffled arm matches the
real one there, `L_MSC` is a regulariser, the mechanism claim is dead, and the
remaining twelve runs cannot rescue it. The notebook now prints that checkpoint
explicitly.

Nothing about the runs changes — same eighteen `run_id`s, same configs, same
hashes. Only the order in which they are executed, and therefore how early the
experiment can be abandoned.

**The general shape.** This is the third time in this log that a comment
asserted a property the code did not have (rule 7: an invariant in a comment is
not a mechanism). D-55 was `channels_last: True` enforced at one site of
sixteen; D-71 was a docstring describing architecture-keyed membership over
code testing run ids. Here the comment described a stopping rule the loop order
made impossible. In each case the prose was the *thing I checked against*,
which is exactly why it did not catch the code.

---

## D-76 — the MSC-KD teacher sweep rebuilt its loader and dropped the conversion layer

**Symptom.** All 18 MSC-KD runs failed identically, ~16 s in:

```
RuntimeError: Given groups=1, weight of size [64, 3, 7, 7],
  expected input[256, 256, 256, 3] to have 3 channels, but got 256 channels
```

`[256, 256, 256, 3]` is `(B, H, W, C)` uint8 at 256 px — raw pack output. No
permute, no float cast, no normalisation, no 256→224 crop.

**Cause.** `train_msc_kd`, sweeping the teacher to build MSC targets:

```python
train_eval = DataLoader(train_loader.dataset, batch_size=..., ...)
train_eval.dataset.augment = False
```

`train_loader` is a `GPUBatchLoader`, and `.dataset` delegates through to the
raw `PackedImageDataset`. Rebuilding a `DataLoader` from it **discards the
conversion layer**, which is where every transformation lives.

**Both lines are correct on CIFAR and wrong on ImageNet-100**, for the same
reason:

- `CIFARTensor.__getitem__` returns finished NCHW float tensors; the packed
  dataset returns raw NHWC uint8 and `GPUBatchLoader` finishes it.
- `CIFARTensor` owns a real `augment` flag; `PackedImageDataset` has none, so
  `train_eval.dataset.augment = False` created an attribute nothing reads —
  inside a bare `except: pass`. The stated intent, *"augmentation off while
  measuring: MSC of an augmented view is not MSC of the sample"*, silently did
  nothing. Had the shape error not fired first, the MSC targets — the entire
  input to the method — would have been measured through an unspecified view.

That second failure is the more dangerous one. The first stops the run; the
second would have produced numbers.

This is D-70's seam a second time, and the wider lesson stands: a library
parameterised by dataset holds only where both datasets present the **same
interface**. Two places now assumed the CIFAR shape.

**Fix.**

1. `eval_view_of(loader, cfg)` returns an ordered, augmentation-free view built
   the way the backend requires — a dataset flag on CIFAR, `train=False` on
   `GPUBatchLoader` for ImageNet-100 — so no caller needs to know which backend
   it holds. It **raises** rather than guessing if it can turn augmentation off
   by neither route.
2. `_assert_model_ready` runs on the first batch of every sweep and rejects a
   batch that is not model input, naming what is missing: NHWC, wrong
   resolution, non-float. The torch error blamed a convolution's channel count;
   the fault was three layers up in loader construction, and nothing in that
   message pointed there.

**The guard's own logic is tested without torch.** `_model_input_problems` is a
pure function over `(shape, is_float, want_res)`, so the sandbox can exercise
it — including the exact `(256, 256, 256, 3)` uint8 batch that failed here, and
three canaries proving a **correct** batch is not refused. A guard that raises
is only as safe as its false-positive rate, and one I could only exercise on
the user's GPU would be a guard shipped unverified. That is precisely the shape
D-63 punished.

**Contamination.** None of consequence, and I checked rather than assumed:
all 18 `p3-*` directories exist, each holding exactly two files — `config.yaml`
and `config_hash.txt`, written at claim time — for **288 KB total**. No
checkpoint, no `epochs.csv`, no parquet, no `STATUS.json` state. Every run
failed during the teacher sweep, before the first student step. They are reused
on the next attempt rather than needing cleanup.

Phase 0 is untouched. The cost was ~5 minutes, not 4.7 days: the failure was
fast and total rather than slow and partial, which is the good kind. Had D-75's
reordering not landed first, this would still have been the first run to fail —
but the point of that change was that the *cheapest* informative thing happens
first, and a total failure at run 1 is exactly that.

---

## D-77 — a global index into a split-sized array killed the kernel

**Symptom.** The kernel died before epoch 1. No Python traceback, only:

```
IndexKernel.cu:93: Assertion `-sizes[i] <= index && index < sizes[i]
                              && "index out of bounds"` failed.
ExitCode: 3221226505
```

**Cause.** `train_msc_kd` built the teacher's MSC targets positionally:

```python
order = np.argsort(sweep["sample_idx"])
msc_train = r.msc[order]              # length 119,395 -- the TRAIN split
...
targets = sufficiency_targets(msc_t[idx], rho_t)   # idx is the GLOBAL index
```

`sample_idx` is the **global pack index**, 0…129,394. The array has 119,395
entries. Every index past the end is out of bounds.

**This is D-49, in a function D-49 never touched.** D-49 was the identical
confusion in `TrainingDynamics` — `IndexError: index 121978 out of bounds for
size 119395` — and its fix introduced `index_space` so that anything indexed
*by* `sample_idx` is sized for the whole space. `train_msc_kd` has carried the
same defect since the port and only fires now because it is the one place that
gathers a dense array by `sample_idx` **on the GPU**.

**And that is why it was so much worse.** On CPU this is an `IndexError` with a
traceback naming the line. On CUDA it is a device-side assert: the process
aborts, the kernel dies, and the user gets `ExitCode: 3221226505` and a wall of
identical thread messages. Same defect, no diagnosis. The lesson is not about
indexing — it is that **the same bug is far more expensive on the device**, so
bounds that depend on a data convention must be checked on the host.

**Fix.**

1. Scatter by `sample_idx` into an `index_space`-sized array, so position *is*
   the global index and `msc_t[idx]` is correct by construction rather than by
   a sort that has to stay in step with a convention.
2. The shuffled-target ablation permutes the **compact** vector *before* the
   scatter. Permuting the padded array would move NaN padding into real
   samples and silently weaken the control — the arm whose whole job is to be
   a fair null.
3. A host-side bounds check on the first batch, raising a readable `IndexError`
   instead of letting CUDA abort.

Reproduced in the self-test at the real sizes (129,395 / 119,395): the old
positional build is provably too short, the scattered build puts every sample
at its own global index, padding stays NaN, and the shuffle-before-scatter
leaves no NaN in a real sample.

**And the fix's first draft had a `NameError`.** The host-side check said
`if step == 0` — `step` exists in `train_backbone`'s loop, not this one. It
parses, it imports, the self-test passes, and it raises on the first batch of
an 18-run job. Python resolves names at call time, so a typo in an unexercised
branch survives every check this project had.

`tools/check_names.py` now reports any name loaded in a function and bound
nowhere, and the build refuses on failure. Narrow on purpose: its first draft
flagged `__file__` and nine conditional module-level definitions, and a checker
that fires on healthy code is one this project has already paid for three times.
Verified against both a file containing the exact `step` mistake and one using
comprehensions, `with`, `except as`, conditional module definitions and
dunders — caught the first, silent on the second.

**And then it caught me twice more, immediately.** Wiring it into the build
introduced `NameError: name 'subprocess' is not defined` — my guarded insert
(`if "\nimport subprocess" not in s`) matched an `import subprocess` that lives
inside a *generated notebook cell*, three hundred lines down, so the
module-level import was never added. The build failed on its own new check.
The checker also only inspected `src/msc_lib.py`; it now inspects the generator
as well, which is where that defect was.

Two rounds of the same mistake in the space of adding one guard is a fair
measure of how easy this class is, and of why it needed a mechanism rather than
more care.

**Contamination.** None. The abort happened during target construction, before
any optimiser step. The 18 `p3-*` directories still hold only `config.yaml` and
`config_hash.txt`.

---

## D-78 — `shufflenetv2_in` contains "shuff"

Found while inspecting a live NB5 run. The arms were being split by substring:

```python
real = [r for r in results if 'shuff' not in r['run_id']]
```

The architecture is named **`shufflenetv2_in`**. Every run_id for it contains
`shuff`, so both arms classified as the control and the real arm was
undercounted by a third.

```
shufflenetv2_in  mscKDshuffromresnet50   'shuff' in run_id=True   truth=True
shufflenetv2_in  mscKDfromresnet50       'shuff' in run_id=True   truth=False  <-- WRONG
resnet18         mscKDfromresnet50       'shuff' in run_id=False  truth=False
```

Rule 2 names this exactly: *a literal that is right for 13 of 15 architectures
is the worst kind.* Two of the three MSC-KD students classify correctly by
accident, and the third is indistinguishable from them at a glance.

**The training was never wrong.** Both training call sites test
`cfg['method']`, where `mscKDfromresnet50` genuinely lacks the substring. Only
the *reporting* was wrong — which is its own hazard, and arguably a worse one:
the numbers on disk are correct while the label attached to them is not, so the
error survives inspection of the data.

**Fix.** `is_control_arm(run_id_or_cfg)` decides on `method`, accepts either a
run_id or a config, and **raises** on anything it cannot parse rather than
falling back to a guess. Both sites use it. The self-test asserts all five arms
classify correctly *and* — the canary that matters — that the naive substring
test really is wrong on `shufflenetv2_in`, without which the test would prove
nothing.

**Contamination.** None to the runs. `analyse_msckd`'s `arm` column already
tested `method` and was correct. The only wrong output was NB5's printed
`N/M REAL-method students trained` line, which is regenerated on the next run.

**The guard's first version did not guard.** It wrapped `parse_run_id` in a
`try/except` and raised in the handler — but `parse_run_id` does not raise on a
malformed id, it returns `method: None`. So `str(None).startswith(...)` gave
`False`, and a function whose docstring says it *refuses to guess* quietly
guessed "real". Relying on an exception that never comes is exactly how that
happens. The `None` is now checked directly, and the self-test that caught it
was the one asserting a bad id raises — which failed on first run, as it
should have.

---

## D-79 — 18 students trained, and the number they exist to produce was never computed

**All 18 MSC-KD runs completed 100 epochs correctly.** Then NB5's comparison
table printed `None` in every result column, and `verify_run_artifacts`
reported `0 complete, 18 resumable`.

**D-79a — the reader with no writer.** `compare_routing_methods` reads
`b1_static`, `b2_confidence`, `b10_msckd`, `b11_oracle`, `avg_flops_ratio` from
each `summary.json`. `train_msc_kd`'s summary dict contains none of them.

`evaluate_routing_methods` — whose own docstring says *"B2 vs B10 vs B11 is the
paper's central figure … the fraction of the B2→B11 gap that B10 closes IS the
result"* — is called from exactly one place in the library:
**`msckd_dry_run`**. The dry run measures the paper's central quantity. The
real run does not.

That is D-55's shape again (the dry run exercised a configuration the trainer
never used), now on the output side. And it is the fourth writer/reader
mismatch in this log — `atomic_write_yaml` with no `read_yaml` (D-63),
`save_analysis` with no loader (D-72), `conv_sweep` results nothing consulted
(D-74), and now a reader whose writer was never wired. **Four, in both
directions.** The write and the read are separate acts and I have repeatedly
shipped one of them.

**D-79b — `config_hash.txt`.** `train_backbone` writes it; `train_msc_kd` wrote
only `config.yaml`. It is in `RUN_ARTIFACTS_REQUIRED`, so all 18 runs verified
as incomplete on a file that costs nothing to write.

**Recoverable without retraining.** Everything B1/B2/B10/B11 need comes from
one forward pass of the saved student over the val set, and the B11 ceiling —
the student's own post-hoc MSC — is computed from that same pass's exit
predictions. `evaluate_msckd_routing(session, run_id)` loads `ckpt_best.pt`,
runs it, and merges the results into `summary.json`. **Minutes per run against
~79 GPU-hours of training already spent.** NB5 gains a backfill cell that finds
runs missing `b10_msckd` and fills them.

`train_msc_kd` now calls it at the end, so the number exists when the run
finishes rather than being discovered missing afterwards. The failure is caught
and logged rather than losing the run — a routing evaluation that fails should
not discard a trained student.

**Three checks, and each one earned its place.**

1. Every routing column declared in `RESULT_KEYS["compare_routing_methods"]`
   that comes from `summary.json` must have a writer in the source.
2. `train_msc_kd` must call the evaluator.
3. `train_msc_kd` must write `config_hash.txt`.

**Checks 2 and 3 failed on correct code first.** They located the function by
`_msckd_src.split("def train_msc_kd")[-1]` — and that string appears *in the
check itself*, so `[-1]` returned the self-test's own source. Rewritten over
the AST with `ast.get_source_segment`, plus a canary asserting the segment was
actually found (23,167 chars) rather than silently empty — because an empty
string makes `"x" in src` False and every such check fail closed, which looks
identical to a real defect.

---

## D-79c — the backfill contradicted its own docstring, and I could not run it

The D-79 backfill failed on all 18 runs:

```
AttributeError: 'list' object has no attribute 'float'
```

`evaluate_msckd_routing` called `sweep_all_axes(cfg, student, ...)`.
`sweep_all_axes` calls `multi_exit(x)` and expects a **list of exit logits**.
`MSCStudent.forward` returns **`(logits, suff, feats)`**. The tuple was
iterated, `l` became the logits *list*, and `l.float()` failed.

**The docstring I wrote one hour earlier said:**

> the B11 ceiling — the student's own post-hoc MSC — is computed from that
> same pass's exit predictions rather than a separate sweep

The code did a separate sweep. I wrote the correct design in prose and the
wrong one in Python, in the same function, in the same edit. That is rule 7 —
an invariant in a comment is not a mechanism — turned on its author.

**Fix.** `evaluate_routing_methods` gains `oracle_from_self=True` and derives
the B11 ceiling from the `L` tensor it already has:

```python
_srt = np.sort(probs, axis=2)
oracle_msc = compute_msc(L.argmax(2), _srt[:, :, -1], _srt[:, :, -2],
                         rho, tau=tau, axis="depth").msc
```

One pass, no second model interface, no adapter. `sweep_all_axes` is no longer
involved and `_sweep_takes_axes` is deleted.

**Verified, against the real `msc_core.compute_msc`** — not asserted this time.
The sandbox has no torch, so I stubbed `scipy`/`sklearn` with an import hook,
built a synthetic `L` of the evaluator's exact shape `(500, 5, 100)`, and ran
the new lines verbatim: `oracle_msc` comes back `(500,)`, every value is a
valid ρ level, `top1p >= top2p` everywhere, and B11 route indices land in
`0..K-1`.

**The structural cause, stated plainly.** My environment has numpy but not
torch. Every torch-dependent line I have written in this project has been
shipped unexecuted, and the defects that reached the user — D-70, D-76, D-77,
D-79c — are all in that category, while the numpy-side logic has held up. The
answer is not more care; it is to keep pushing the decidable part of each fix
into pure functions I can actually run (`_model_input_problems`,
`_module_names`, and now this), and to say clearly which half of a change is
verified and which is not.

**Also fixed:** the backfill called `build_loaders(cfg)`, which builds the
train loader too and therefore tried to resident-cache the whole 23.7 GiB
pack — declined at 17 GiB free, falling back to memmap. It now passes
`ram_cache=False`; only the 10,000-image val set is needed.

**Not a defect:** `[VERIFY] 0 run(s) on local disk` was `confirm_on_disk(results)`
with `results == []`, because `CONFIRM` was False on that pass so nothing
trained. It verified an empty list because it was handed one. All 18 runs,
their 100-epoch histories and their checkpoints (4.4 GiB) are intact.

---

## D-80 — the gap ratio divided by noise, and NB5's tables read the wrong list

The backfill succeeded: all 18 students now carry B1/B2/B10/B11. Then it
printed `closed=26.0`, `closed=-47.9`, `closed=83.6`.

**Cause.** `fraction_of_B2_to_B11_gap_closed = (B10 - B2) / (B11 - B2)`, guarded
by `abs(gap_total) > 1e-9`. That is a formality, not a guard. Measured across
all 18 runs:

```
B11 - B2  =  +0.00007   (sd 0.00036)   <- the denominator
B10 - B2  =  -0.00880   (sd 0.00683)   <- the numerator
```

Dividing a real difference by a denominator four orders of magnitude smaller
than itself gives 26, -48, 84. The ratio is **undefined**, not large.

A ratio is only meaningful when its denominator exceeds the noise on the
quantities it is built from. At n = 10,000 the binomial 2 SE on an accuracy
difference is **0.0141**; the measured gap is 0.00007. It now reports `NaN`
plus `gap_verdict` naming the reason, and records `B2_to_B11_gap` and
`B2_to_B11_gap_noise_2se` so the reader sees the size of both.

**D-80b.** `compare_routing_methods` and `confirm_on_disk` were called with
`[r['run_id'] for r in results]`. On a re-run where everything is already
trained, `run_all` returns `[]`, so both got an empty list — the comparison
table printed `Empty DataFrame` and the verifier printed
`0 run(s) on local disk` beside 4.4 GiB of runs. That is D-65's silent-no-op
shape a third time. Both now read every completed run in the phase.

---

## Q5 RESULT — MSC-KD does not beat confidence routing, and the oracle says why

Not a defect. This is the finding, and it is worth stating precisely because
the control that makes it interpretable is the one thing that nearly did not
get computed.

At a matched operating point of **ρ = 0.806** (a real 19% compute reduction,
not a degenerate ρ≈1), across 18 students:

| router | mean accuracy vs B2 | in how many of 18 runs |
|---|---|---|
| **B1** full compute, no routing | ≈ B2 | — |
| **B2** confidence thresholding | baseline | — |
| **B11** oracle: the student's own true post-hoc MSC | **+0.00007** (sd 0.00036) | headroom in 0 |
| **B10** MSC-KD | **−0.00880** (sd 0.00683) | worse in **18 / 18** |

Three things follow, and the third is the one that matters:

1. **Confidence thresholding already reaches full-compute accuracy at 80%
   compute.** B1 ≈ B2 everywhere.
2. **MSC-KD is consistently below it** — about 0.9 accuracy points, in every
   single run, both arms.
3. **The oracle ceiling offers no headroom either.** B11 — routing by the
   student's *own true* post-hoc MSC, the best any MSC-based router could
   possibly do — matches B2 to within 0.00007.

Point 3 is the result. Without B11 this reads as *"our distillation failed"*.
With B11 it reads as **"MSC-based routing has nothing to offer over confidence
at this operating point, and the failure is in the premise rather than in the
student"**. That is a much stronger, much more useful negative, and it is
exactly the claim the B11 baseline exists to license. The method section can be
written honestly around it.

It also sits consistently with Q4, which missed its gate at partial ρ = 0.282:
MSC carries *some* information beyond a difficulty battery (ΔR² = 0.041, CI
excludes zero), but not enough to route better than confidence does.

**None of this touches Q1–Q3.** The seed-reliability result, the multi-axis
result and the disattenuated transfer result stand exactly as they were.

---

## D-81 — the headline compared architectures that do not appear in both studies

Not a code defect. A claim I wrote in `24_IN100_STATUS.md` and repeated to the
user: *"ViT's ρ_seed rose from 0.547 to 0.649, and the CNN/ViT gap widened from
~0.10 to 0.173 — the opposite of the small-data-artifact hypothesis."*

`table6_cifar_vs_imagenet.csv` had been saying otherwise the whole time:

```
arch,family,in100_rho_seed,cifar_rho_seed,delta,same_architecture
resnet50,resnet,0.822,,,False
vit_small_p16,vit,0.649,,,False
```

`same_architecture = False` on both rows, and an **empty** CIFAR column —
because there is nothing to join on. CIFAR ran `vit_tiny` and `mixer_nano`
against twelve small CNNs; ImageNet ran `vit_small_p16` against `resnet50`.
**No architecture appears in both studies.**

So every cross-study number folds architecture, resolution and dataset into one
difference. 0.547 → 0.649 is `vit_tiny`@32px versus `vit_small_p16`@224px.
The "gap widened" claim compares 12-vs-2 architectures against 1-vs-1, where
the ImageNet side has no error bar at all. And `vit_small_p16` at 0.649 lands
*inside* the CIFAR CNN range of 0.622–0.722, which the "ViT is unreliable"
framing hides.

**Rule 12 is explicit about this** — *scrutinise favourable results harder than
unfavourable ones* — and I did the opposite. I scrutinised the MSC-KD null hard
enough to check the operating point was not degenerate (ρ = 0.806, correctly),
and took the flattering seed-reliability comparison at face value for three
messages. The table that refutes it was generated by NB4, on disk, unread.

**What survives**: the *ordering* replicates. Within ImageNet-100 — one
dataset, one resolution, one recipe family, one pipeline — resnet50 0.822 >
vit_small_p16 0.649, the same direction CIFAR found across 14 architectures.
An independent replication on non-overlapping architectures is arguably
stronger evidence the effect is not architecture-specific. It is also a
strictly weaker claim than the one I made.

**To recover the quantitative claim** the study needs an architecture in both
zoos (`convnext` and `shufflenetv2` qualify) or ≥3 architectures per family on
ImageNet. That is training, not analysis, and it is now the top item in §8.

---

## Repository restructure + NB6_Publish

Not a defect. Recording what moved, so the next person is not surprised.

**Documents are now split by study.** Seventeen numbered files sat at the top
level with two studies interleaved. Now:

```
docs/cifar100/     00-10   protocol, spec, schema, playbook, results
docs/imagenet100/  20-25   port plan, delta, lab notebook, runbook, status, data card
```

`tools/check_links.py` verifies that every cross-reference resolves — the docs
reference each other about 120 times, and `20_IN100_PORT_PLAN.md` opens by
telling the reader to read four other files in order. That reading order is the
onboarding path and nothing was checking it.

**It found a break before the move:** `20_IN100_PORT_PLAN.md` referenced a
`24_IN100_RESULTS` document that has never existed — the file is
`24_IN100_STATUS.md`. Fixed.

(That sentence is deliberately not written with the dead filename in full: the
checker searches prose for `NN_NAME.md` mentions, so quoting a broken reference
verbatim re-creates it. It caught this entry on the first run, which is the
behaviour you want from it.) Moving seventeen files without that checker would have
broken all 120 references at once and nobody would have noticed until someone
followed one.

**Removed from version control:** `download minimax.ipynb` (unrelated), a
stray `Z:` directory, `scratch/`, the write-probe directories `msc_data/` and
`msc_results/` (they are storage roots, not source), and the `msc_lib.py` /
`msc_core.py` copies inside `notebooks/` and `notebooks_in100/` — those are
written at run time from the embedded blob and are build artifacts, not source.
Tracking them meant a stale copy could be committed and then loaded (D-62).

**NB6_Publish** mirrors the local tree to `Shanmuk4622/msc-imagenet100`,
matching the CIFAR layout under `Shanmuk4622/msc-cifar100` exactly — one
dataset repo, one folder per run. The two-repo split (models + data) was tried
during CIFAR and reverted because HuggingFace's write limit is per *user*, so
two uploaders doubled commit consumption for no benefit.

It defaults to `DRY_RUN = True`, reads `HF_TOKEN` from the environment rather
than a cell, uploads one commit per run (≈28 commits, not ≈400, against a
~120/hour limit), skips runs already present so a re-run resumes, generates the
dataset card **from the results on disk** so it cannot drift, and verifies by
re-reading the repo rather than trusting that the upload queue drained (D-9,
D-10).

**Two accessors added so NB6 obeys rule 4.** Remote paths are repo paths: a
literal `runs/{id}/...` in the publish notebook is the same hazard D-23 was.
`repo_rel_path(work, local)` computes the HuggingFace path from the local one —
which is what `run_layout` was designed to make possible — and
`publish_manifest(work)` derives the size report from `RUN_SUBDIRS` instead of
hand-written globs, so a new subdirectory appears automatically rather than
being silently omitted.

**The validator learned `run_layout` keys.** `L["per_sample"]` is a path, not a
column, and flagging it punished exactly the accessor rule 4 asks for.
`LAYOUT_KEYS` is accepted, with a canary asserting an invented column name is
still rejected — widening a checker without proving it can still fail is how a
checker quietly stops checking.

---

## D-82 — NB6 shipped without the two cells every notebook opens with

```
Cell In[2], line 3
----> 3 ROOT = Path(MSC_ROOT)
NameError: name 'MSC_ROOT' is not defined
```

`nb1` through `nb5` all begin with `code(bootstrap())` and
`code(paths_cell(...))` — the cells that unpack the library, bind `M`, resolve
storage and set `MSC_ROOT`. I wrote `nb6` starting from its own markdown and
never added them, so `M` and `MSC_ROOT` did not exist and the notebook failed
on its first real line.

**Every validation layer passed it.** Column names, repo paths, library names,
call arity, result keys, stage predicates, and — since D-73 — that every cell
parses as Python 3.10. Seven layers. None of them asked the most basic question
you can ask about a notebook: **does each cell only use names that something
earlier defines?**

`check_names.py` already answered exactly that question for `.py` files, and
had done since D-77. Extending it to notebooks was about thirty lines. I built
the tool, wired it to the library and the generator, and did not point it at
the artifacts the generator produces.

**Fix.** `check_names.py` now takes `.ipynb` files, walks cells in order with
bindings accumulating — which is what a kernel does on Run All — and reports
any load with no earlier binding. The build runs it over every emitted notebook
and refuses on failure.

Verified against the real artifacts: pointed at the broken NB6 from the previous
commit it reports `MSC_ROOT` and `M` at the exact cells and lines; pointed at
the fixed one it is silent. A checker that cannot demonstrate both is not
evidence of anything.

**The pattern this makes eight of.** D-55 (dry run vs trainer), D-63 (test used
a clean config the runtime never has), D-71 (fixture keyed by run id, callers
key by arch), D-76/D-77/D-79c (torch paths shipped unexecuted), D-79 (reader
with no writer), and now a checker never aimed at the thing it exists to check.
Every one is the same shape: **the verification and the artifact were not the
same object.** Writing the check is the easy half; pointing it at what actually
ships is the half I keep missing.

---

## D-83 — the offline guard is right for five notebooks and fatal for the sixth

```
OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/repos/create:
offline mode is enabled. To disable it, please unset the `HF_HUB_OFFLINE`
environment variable.
```

`msc_lib` calls `enforce_offline()` at import whenever `MSC_OFFLINE` is set,
and the notebook bootstrap sets it. That is exactly right for NB1–NB5: this
pipeline must be provably self-contained, and the guard is what proves it.

**NB6 is the one notebook whose entire job is to reach HuggingFace**, and it
inherits the guard from the shared bootstrap. I added NB6 without considering
that the thing every other notebook needs is the thing this one cannot have.

**The error's own advice is misleading here**, which cost the user a round:
`HF_HUB_OFFLINE` is set **inside the notebook process**, after the shell has
been left behind. `unset` in PowerShell (which is `Remove-Item Env:` anyway)
operates on a shell that is not the one with the problem.

**And popping the variable is not sufficient either.** `huggingface_hub` reads
`HF_HUB_OFFLINE` **once, at import**, into `huggingface_hub.constants`. If the
package is already imported — and importing it is how you find out you are
offline — clearing the environment changes nothing. The constant has to be
patched as well.

**Fix.** `allow_network()` clears the four variables *and* patches
`HF_HUB_OFFLINE` on every already-imported `huggingface_hub` module, returning
what it changed. `offline_state()` reports the guard for display. NB6 prints
the state before and after and calls `allow_network()` explicitly, so going
online is a visible, deliberate act in the one notebook that does it — rather
than something a later cell fails on.

**The fix's first version did not run at all.** I inserted `allow_network`
immediately above `def no_network`, which sits under a `@contextmanager`
decorator — so the decorator attached to the new function and
`allow_network()` returned a context manager. The self-test caught it:
`TypeError: '_GeneratorContextManager' object is not subscriptable`. Inserting
between a decorator and its function is D-69's ordering mistake in a new place;
the lesson is that "insert before `def X`" is not a safe operation when a
decorator can sit above it.

**Verified in both directions.** The self-test sets all four variables and
installs a fake `huggingface_hub.constants` with `HF_HUB_OFFLINE = True`,
asserts the guard is genuinely on first (the canary), then asserts
`allow_network()` clears all four *and* flips the module constant to `False`.
Without the canary the test would pass on a machine that was never offline.

---

## D-84 — a token problem presented as a forty-line traceback

```
403 Forbidden: You don't have the rights to create a dataset under the
namespace "Shanmuk4622". Make sure your token has the correct permissions.
```

**Not a code defect.** The notebook worked: it went online (D-83's fix held),
reached HuggingFace, authenticated, and HuggingFace declined. The token is
read-only, or belongs to another account, or is fine-grained without write
access to that namespace.

**But the presentation was mine.** `create_repo` was NB6's first network call,
and a token-permission failure is the single most likely thing to be wrong at
that point. The one accurate sentence arrived at the bottom of a stack running
`httpx.HTTPStatusError` → `HfHubHTTPError` → a deprecation wrapper → an
argument validator → `create_repo`. Everything above it is noise, and the
reader has to get to the last line to learn anything.

`whoami()` answers the same question in one call, before anything is created,
and can distinguish the three causes — which the 403 cannot.

**Fix.** `hf_token_check(token, repo_id, repo_type)` returns a verdict rather
than raising (a preflight that throws is just a different traceback) and names
which case applies:

- no token → where to create one;
- role `read` → "read-only … needs a WRITE token", with the settings URL;
- namespace mismatch → **both** names, and the two ways to reconcile them;
- fine-grained → passes, with a caution about the scope it needs, since its
  permissions cannot be read from `whoami`;
- `whoami` itself failing → reported, not raised.

NB6 prints user / role / namespace / verdict before the upload cell, and the
upload cell refuses on a bad verdict.

**Verified — all six branches, including the exact failure.** The self-test
installs a stub `huggingface_hub` whose `whoami()` returns each shape:
read-only, wrong namespace, write, org membership, and a raising call. The
canary asserts a **valid write token passes**, without which a preflight that
always says no would look like a working check.

This is the discipline the torch-side defects (D-70, D-76, D-77, D-79c) needed
and did not get: the environment here has no network, so the decidable part —
what to do with each `whoami` response — was pushed into a pure function and
exercised against every case that matters.

---

## D-85 — three hypotheses had results and no verdict, and I mis-scored a fourth

Documentation audit, prompted by "are we clear on what is what".

**The gap.** Five hypotheses were pre-registered in
`docs/cifar100/00_RESEARCH_PROTOCOL.md`. The ImageNet-100 numbers existed in
`analysis/*.csv` and were quoted in `24_IN100_STATUS.md`, but **no document
scored them against the predictions**. H2, H3 and H5 had measured outcomes and
no recorded verdict. `PAPER.md` contains zero mentions of ImageNet.

A result that is not scored against its own pre-registration is not a
pre-registered result. That is the whole point of registering it.

**And I mis-scored H4.** I reported it repeatedly as "partial ρ 0.282 misses
the 0.30 gate" while treating ΔR² = 0.0411 as the passing half. H4 requires
**both** ΔR² ≥ 0.05 **and** partial ρ ≥ 0.30. ΔR² 0.0411 is also below its bar.
Both criteria miss. Corrected in `24_IN100_STATUS.md` and scored correctly in
the new findings document.

**What the audit turned up that is genuinely good news.**
`docs/cifar100/10_FINAL_RESULTS.md` §6 names B11 — the oracle ceiling — as the
CIFAR study's *"one substantive gap remaining"*, tracked as **O-21**, because
computing it needs a measurement pass over the `p3` students that was never
done. **This study computed it, for all 18.** That reframes the Q5 negative
entirely: without B11 the result is "our distillation underperformed" (a claim
about an implementation); with it, "MSC-based routing has no headroom over
confidence" (a claim about the premise).

The scorecard also shows H2 and H3 **replicating** across a 49× change in
pixels on non-overlapping architectures, and H4 missing in both studies at
almost the same value (0.0425 vs 0.0411).

**Added.** `26_IN100_FINDINGS.md` — the missing scientific document: the five
hypotheses scored against both studies, what is novel and why, what is
explicitly not supported, the strongest honest framing, and the cost of the
three options that would strengthen it. `README.md` now leads with the science
and `PAPER.md` carries a scope note saying it is CIFAR-only and which of its
statements this study supersedes.

---

## D-86 — one dropped packet poisoned every upload after it

The publish reached run 12 of 22 and then produced **two different** errors:

```
[Errno 11001] getaddrinfo failed ... Retrying in 1s [Retry 1/5].
RuntimeError: Cannot send a request, as the client has been closed.
```

**The first is transient and handled.** DNS resolution failed for a moment;
`huggingface_hub` retried five times, correctly.

**The second is the defect.** Once those retries are exhausted, the underlying
httpx client is closed — and it stays closed **for the life of the `HfApi`
object**. Runs 13 through 22 then failed instantly with the same message, even
though the network had come back. One momentary blip cost ten uploads, and none
of the failures after the first were real.

So the fix is *not* more retries — `huggingface_hub` already retries, and doing
it again around a dead client changes nothing. It is to **rebuild the client**
instead of reusing a corpse, and to treat a failed item as one failed item
rather than the end of the run.

**Fix.** `hf_upload_resilient(token, repo_id, repo_type, items)`:

- constructs a **fresh `HfApi` per attempt** — that is the actual mechanism;
- retries each item with linear backoff;
- **never raises** — it returns `{"uploaded": [...], "failed": [(label, why)]}`,
  because a publish that dies on the first error has to be babysat, and the
  point is that it can be re-run;
- NB6 lists what is already on the hub first, so a re-run **resumes** and
  nothing is uploaded twice.

**Verified against the exact failure sequence.** The self-test installs a stub
`HfApi` whose third upload raises `getaddrinfo failed` *and* marks that client
dead, so every subsequent call on it raises `Cannot send a request, as the
client has been closed` — precisely what happened. Five assertions:

1. the drop does not poison the items after it (5/5 still upload);
2. a permanently unreachable item is reported and the rest continue;
3. a new client is built per upload — 5 clients for 5 items;
4. **canary**: with no failures, everything uploads exactly once;
5. total failure returns a report rather than raising.

Without (3) and (4) this would pass while quietly reusing one client, or while
uploading nothing at all.

**Your 12 uploaded runs are on the hub and are not re-sent.** The remaining 10
plus `budgets/registry/analysis/tables/paper` and the card go on the next run
of that cell.
