# Study 3 — live log

**Newest first.** One entry per session. Every entry answers four things: what
changed, what it cost, what it **settled**, what is next.

A phase is `done` only when its artifact exists on disk **and has been opened
and checked**. Study 1's D-79 was 18 runs that all reported success while the
number they existed to produce was never computed.

---

## Machine facts — the answers to "where is X?"

Recorded here because I asked twice and wasted a 169 MB download at 17 kB/s.
Anything in this table belongs in code, not in a chat message.

| | path | wired in as |
|---|---|---|
| CIFAR-100 (python format) | `C:\Users\Administrator\Desktop\New folder\cifar-100-python` | `CIFAR_DIR` at the top of every S3 notebook → `MSC_CIFAR_DIR` → `locate_cifar100()` |
| results root | `C:\msc_results` | `MSC_ROOT`, auto-resolved |
| ImageNet-100 pack | `C:\msc_data\in100` | `MSC_IN100_DIR` |
| GPU | RTX 4000 Ada, 20 GB | — |
| HuggingFace repo | `Shanmuk4622/msc-cifar100` | `MSC_HF_REPO`; the library default is the **ImageNet** repo |
| network | **intermittent** | Study 3 trains OFFLINE; `S3_NB5_Publish` uploads once, at the end |

`locate_cifar100()` now checks an explicit location **before** any download
path, which it previously could not do — ImageNet-100 had `MSC_IN100_DIR` since
the port and CIFAR-100 had no equivalent, so "the data is already here" was a
thing the caller had no way to say.

---

## Status board

| | what | cost | state | artifact |
|---|---|---|---|---|
| **P0** | exit quality → excess extrapolation | ~15 min CPU | **DONE** — predicts excess GROWS with exit quality | `analysis/s3_exit_quality.csv` |
| **P1** | Q1 joint exit training | ~10 GPU-h | **DONE** | 3 runs under `runs/p4-*-jointexit-s1` |
| **P2** | Q1 verdict, paired comparison | ~10 min CPU | **DONE — H1 SUPPORTED** | `analysis/s3_q1_comparison.csv` |
| **P3** | Q2 learned router | minutes, CPU | **DONE — H2 SUPPORTED** | `analysis/s3_router_capture.csv` |
| **P4** | Q3 pruning | ~6 GPU-h (5k pool) | **ready — run next** | `analysis/s3_pruning.csv` |
| **P5** | publish everything, once | minutes | ready — run LAST, with a network | `Shanmuk4622/msc-cifar100` |

**Next action: run `S3_NB4_Pruning` (Q3).** Q1 and Q2 are both settled.

---

## Pre-registered predictions — fill in as results arrive

Written **before** any run. Do not edit the prediction column.

| | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| **H1** | oracle excess survives joint exit training | ≥ 2.0 pt, 3 of 3 archs | **8.55 / 9.15 / 10.64 pt** | **SUPPORTED** (3 of 3) |
| **H2** | a learned router captures little of the gap | < 25 % (cross-seed) | **1.7 %** | **SUPPORTED** |
| **H3** | saturated-source pruning is worse | ≥ 1.0 pt at 30 % keep | _pending_ | _pending_ |
| **H3b** | saturated source ≈ random pruning | ± 0.5 pt at 30 % keep | _pending_ | _pending_ |

---

## 2026-08-20 (Q2 SETTLED) · H2 SUPPORTED — a deployable gate captures ~nothing

`analysis/s3_router_capture.csv`, ρ = 0.80, exit-local confidence features:

| arch | baseline | router | oracle | gap | **capture (cross-seed)** |
|---|---|---|---|---|---|
| `resnet20` | 60.58 | 60.85 | 76.22 | 15.64 | **1.7 %** |
| `resnet32x4` | 66.68 | 67.21 | 78.30 | 11.62 | **4.6 %** |
| `vgg8` | 70.07 | 70.05 | 79.25 | 9.18 | **−0.2 %** |

**H2 threshold was < 25 %. Measured median: 1.7 % — SUPPORTED.** A learned
per-exit gate, given everything the baseline sees plus the margin, recovers
essentially none of the 9–16 point oracle gap.

**And it is not overfitting.** In-seed capture (3.8 / 2.4 / −0.4 %) is barely
distinguishable from cross-seed (1.7 / 4.6 / −0.2 %). If the gate were
memorising seed noise, in-seed would be high and cross-seed near zero. Both are
near zero, which says something stronger: **there is nothing in exit-local
confidence to capture.** The gap is not merely non-transferable — it is not
expressible in the signal a deployed gate can read.

This tightens Study 2 rather than contradicting it. Study 2 showed a second
*seed* cannot reach the gap; Q2 shows a *learned gate on the deployable signal*
cannot either.

**Stated limitation, printed by the notebook itself:** these are exit-local
confidence features, so this is a **lower bound**. A gate with access to pooled
embeddings might do better; measuring that needs checkpoints that were
deliberately not downloaded.

---

## 2026-08-20 (Q3 blocked, twice) · a guard that grepped instead of testing

```
RuntimeError: build_loaders does not read subset_path.
```

**The guard was wrong, not the library.** `subset_path` handling lives in
`_subset_train`, which `build_loaders` *calls*. I grepped `build_loaders`'s
source for the string and raised when it was absent — so a correct
implementation was rejected.

**Grepping for a feature is not testing it.** The guard now *exercises* the
behaviour: it passes a real keep-list through `_subset_train`, checks the subset
length matches, and checks `index_space` was preserved so global `sample_idx`
stays valid (D-49). Verified against the real library: keep-list of 1500 → 1500
samples, `index_space` 50000.

### And a scope correction that matters more

`ce_loss` — the score whose reliability collapses — is written **only to
`train_holdout.parquet`, 5,000 samples**. `train_dynamics.parquet` covers more
but carries only `el2n` and `forget_events`, which are the *stable* scores and
so cannot test this hypothesis.

So Q3 is: **one fixed 5,000-sample pool, three selection rules, one target, two
seeds.** A real controlled comparison, and it does test H3 — but it is a
narrower claim than "pruning CIFAR-100", and the notebook now says so in its
markdown and in its output. At 30 % retention the target trains on ~1,500
images, so absolute accuracies will be low by design; **the comparison between
arms is what H3 tests, not the level.**

Scaling to the full 50,000 needs `ce_loss` recomputed across all of train, which
needs the source checkpoints — excluded from the fetch, so a separate ~2 GPU-h
job. Cost estimate for Q3 revised from ~18 GPU-h to **~6**.

**The kept-set gate passed:** 51.6 % overlap at 50 % retention, 33.5 % at 30 %.
The two sources genuinely disagree about which samples are hard, so a downstream
difference is possible.

---

## 2026-08-20 (Q1 SETTLED) · H1 SUPPORTED — Study 2 is not an artifact

**The blocker is cleared.** Jointly trained exits do not remove the oracle
excess. They make it **larger**.

| arch | frozen | **joint** | change | Δacc_full | exit_quality |
|---|---|---|---|---|---|
| `resnet20` | 6.69 | **10.64** | +3.95 | −1.53 | 0.413 → 0.756 |
| `resnet32x4` | 6.42 | **8.55** | +2.13 | +1.39 | 0.535 → 0.919 |
| `vgg8` | 7.95 | **9.15** | +1.20 | +1.08 | 0.634 → 0.753 |

**H1 threshold was ≥ 2.0 pt in 3 of 3. Measured: 8.55, 9.15, 10.64 — SUPPORTED**,
by more than four times the bar, and the excess **grew in 3 of 3**.

### The experiment was a real test

`exit_quality` rose in all three, by +0.119 to +0.384 (`resnet32x4` reaching
0.919). Joint training changed exactly what it was supposed to change, so H1
was genuinely at risk rather than untested.

### R-02 fired, and conditioning makes it stronger

Backbone accuracy moved by more than 1 pt in all three (−1.53, +1.39, +1.08),
so the conditioned number is primary. Using the slope of excess on `acc_full`
measured across the 45 frozen runs (−0.2559 pt per pt):

| arch | raw Δ | **conditioned Δ** |
|---|---|---|
| `resnet20` | +3.95 | +3.56 |
| `resnet32x4` | +2.13 | +2.49 |
| `vgg8` | +1.20 | +1.48 |
| **median** | +2.13 | **+2.49** |

The effect survives, slightly *larger* once accuracy is held fixed. `resnet20`
gained most partly because its accuracy fell; `resnet32x4` and `vgg8` gained
despite theirs rising.

### P0's free extrapolation was right

P0 predicted, from 45 frozen runs and no GPU time, that the excess would **grow**
with exit quality — reaching ~+10.98 pt at exit_quality 0.86. Measured: **8.55
to 10.64 pt at exit_quality 0.75–0.92.** Direction correct, magnitude within a
couple of points, and the prediction was written down before the runs.

That is the strongest argument yet for the cheap-gate-before-expensive-run rule.
It also means the +12.45 pt figure at exit_quality = 1.0 was an extrapolation
past the observed range and should not be quoted.

### What this means for the paper

`study2/PAPER.md`'s blocking limitation is **removed**. The oracle excess is not
a property of weak post-hoc exits — it is larger when the exits are trained the
way the field trains them. The paper should now report both numbers, with the
joint figures as the primary evidence and the frozen ones as the conservative
case.

---

## 2026-08-20 (Q2 fixed) · the checkpoints were never downloaded

```
FileNotFoundError: C:\msc_results\runs\p1-resnet20-cifar100-base-s1\checkpoints\ckpt_best.pt
```

**Checked the disk instead of guessing, and the picture is unambiguous:**

| | runs | parquet | `ckpt_best.pt` |
|---|---|---|---|
| Study 1 base (45) | all | **yes** | **no** |
| Study 3 joint (3) | all | yes | yes |

`S2_NB0_Fetch` **deliberately excluded checkpoints** — ~95 % of the bytes, and
nothing had needed them. So the embedding-based feature dump was impossible from
the moment it was written, for every multi-seed architecture.

**This is the root of the whole NB3 cascade.** Three failures in a row —
constructed run ids, empty frame, missing weights — were all the same mistake:
writing code against an assumed disk state. The earlier fixes addressed the
symptoms one at a time.

### The fix uses what is actually there

Q2 now runs on **exit-local confidence features** — `top1p_dk`, `top2p_dk`, the
margin between them, and two derived ratios — read straight from the parquets.

| gate features | needs | runs | cross-seed control |
|---|---|---|---|
| **exit-local confidence** | parquet only | **all 45** | **yes, 3 seeds** |
| pooled embeddings | `ckpt_best.pt` | 3, one seed each | no |

This is not a downgrade so much as a sharpening: it is exactly what a deployed
early-exit gate reads. The baseline thresholds `top1p_dk`; the gate sees the
same plus the margin, and learns a **per-exit** boundary instead of one global
threshold. **H2 becomes a lower bound**, and the notebook says so in its output.

Cost: **zero GPU, zero network.** Q2 dropped from ~5 GPU-h to minutes of CPU.

### Verified by execution, not by reading

`tools/s3_nb3_harness.py` runs NB3's real cells against synthetic data in two
worlds:

* **margin carries signal that raw confidence does not** → capture **56.9 %**
* **nothing carries signal** → capture **6.4 %**

The first is the one that matters. Had I only tested "confidence is
informative", the baseline would already be optimal, the gate would add nothing,
and a capture of ~0 would have looked like a finding instead of an inert
instrument. That is the same trap as Study 2's canary 9, in a new place.

### A preflight cell, because this keeps happening

NB3 now opens with a table of every run it intends to use and whether each
required artifact exists, then **picks its mode from what it finds**. Every
Study 3 failure so far has been an assumed input; this makes the assumption
visible in the first cell rather than fatal in the fifth.

69/69 canaries; selftest 461/461.

---

## 2026-08-20 (Q2 blocked) · NB3 constructed run ids instead of finding them

```
missing p4-resnet20-cifar100-base-s1, skipped     (x6)
0 feature dump(s)
...
KeyError: 'kind'
```

NB3 built `p4-{arch}-cifar100-base-s{seed}` from the **session** phase. The base
runs are phase **p1**. Every id missed, no features were dumped, and the empty
frame surfaced four cells later as `KeyError: 'kind'`.

Third time this exact shape has appeared: a run id assembled from parts instead
of looked up, plus an empty DataFrame indexed by column name. `measured_runs()`
was written for precisely this and NB3 was not using it.

**Fixes:** NB3 now **finds** its runs through `measured_runs()`, picks the
architectures that actually have ≥ 2 seeds, rebuilds each config against the
run's real phase, and refuses with a clear message if fewer than two dumps
exist. Two new canaries:

* no notebook may build a run id from an f-string phase prefix;
* every notebook that builds a DataFrame from a scan must guard it empty before
  indexing a column.

**The second canary immediately caught the same latent bug in `S3_NB4`**, which
would have failed the same way after 18 GPU-hours of pruning runs. That is the
canary paying for itself before the cost was incurred.

Q2 runs on the **frozen** base runs, because the cross-seed control needs ≥ 2
seeds and the joint runs have one apiece. Given Q1 showed the excess is *larger*
on joint runs, a router evaluated on frozen runs is a conservative test —
recorded as a limitation, not worked around.

69/69 canaries; selftest 461/461.

---

## 2026-08-20 (P1 measure) · the D-88 fix hit D-67 — the call needs all three

My D-88 fix passed `fn=sess.oracle` but left `stage` at its default, so **D-67
fired instead**:

```
ValueError: run_all(fn=sess.oracle) with stage='train' would ask 'is it
TRAINED?' to decide whether to MEASURE it, so every trained run is skipped.
  Use: sess.run_all(cfgs, fn=sess.oracle, done_fn=sess.measured, stage='measure')
```

The guard was right and its message contained the answer. **The correct call
names all three**, because they answer three different questions:

| | |
|---|---|
| `fn` | what to run — the oracle, not the trainer |
| `done_fn` | what *"already done"* MEANS here — measured, not trained |
| `stage` | the label the plan prints |

### Why the auto-inference never saved us

`run_all` does contain a shortcut for this:

```python
if fn is getattr(self, "oracle", None):
    done_fn, stage = self.measured, "measure"
```

**A bound method is a new object on every attribute lookup**, so
`sess.oracle is sess.oracle` is `False` and that branch can never be taken. It
is dead code for the oracle path — D-67 is what actually catches the mistake,
and D-67 compares `__func__`, which is why *it* works.

Left as-is deliberately: D-67's selftest encodes "the caller must be explicit"
as the intended contract, and for a call that decides whether 10 GPU-hours of
measurement happen, explicit is the right default. Recorded here so the next
person does not spend an hour wondering why the shortcut looks broken — it is.

**Verified rather than assumed.** Both failure shapes and the notebook's exact
call were exercised against the real guards:

```
stage='oracle' only (the D-88 bug)   -> REFUSED
fn=oracle only      (the D-67 bug)   -> REFUSED
fn + done_fn + stage (the notebook)  -> passes both guards
```

Canary tightened from "passes fn=sess.oracle" to "passes fn AND done_fn AND
stage". 59/59.

**Next: re-run the measurement cell in `S3_NB1`.** Third time; the call now
satisfies both guards.

---

## 2026-08-20 (P1 trained) · D-88 — the measurement stage silently did nothing

**Joint training finished.** All three runs completed their full schedules and
wrote checkpoints and jointly-trained exit heads:

```
p4-resnet20-cifar100-jointexit-s1      joint heads OK  scheme=uniform  epoch=236
p4-resnet32x4-cifar100-jointexit-s1    joint heads OK  scheme=uniform  epoch=182
p4-vgg8-cifar100-jointexit-s1          joint heads OK  scheme=uniform  epoch=219
```

**Then the measurement stage ran nothing**, and said so as if it were good news:

```
already finished (GLOBAL, from HF): 3   <- for the 'oracle' stage
MY REMAINING WORK                 : 0
(nothing to do -- either finished, or owned by other workers)
```

`confirm_on_disk` had the truth in it the whole time —
`per_sample/test.parquet: missing` for all three — but it is printed as a dict
among twenty other keys, so it read as noise. `S3_NB2` then failed with *"no
joint runs found"*, two notebooks from the cause.

### D-88 — `stage` is a label, `fn` selects the work

I wrote `sess.run_all(cfgs, stage='oracle', ...)`. But `run_all` infers what to
do from **`fn`**, not from `stage`:

```python
fn = fn or self.train                    # <- defaulted to TRAINING
if done_fn is None:
    if fn is getattr(self, "oracle", None):
        done_fn, stage = self.measured, "measure"
    else:
        done_fn = self.trained           # <- "is it trained?"  yes, x3
```

So it planned the *training* stage, asked "is it trained?", got yes three
times, filtered all three out, and reported success.

**This is D-67 from the other side.** D-67 already guarded `fn=sess.oracle`
planned as training, and its docstring describes this exact trap — but only in
that one direction. Documenting a trap is not removing it (rule 7), and I fell
into its mirror image.

**Fixes:**

1. **`run_all` now refuses** `stage='oracle'`/`'measure'` unless `fn` really is
   `sess.oracle`, with a message that says the label does not select the work.
   Selftest 459 → **461**, including a canary that the correct call is *not*
   refused.
2. **NB1 opens the artifact** after measuring instead of trusting the plan
   (rule 5, D-79): it checks `per_sample/test.parquet` exists and is non-empty
   for every run, and raises naming the file NB2 will read.
3. **NB2 distinguishes** "never trained" from "trained but not measured" and
   tells you which cell to re-run.

**Nothing is lost.** The 3 trained runs are intact — checkpoints, exit heads,
epoch histories, telemetry all on disk. Only the measurement pass is missing,
and it is inference-only, ~30–40 min.

**Next: re-run the measurement cell in `S3_NB1`.** It will now do the work.

---

## 2026-08-20 (P1 crashed) · D-87 — one flag, two defaults; and Study 3 goes offline

`S3_NB1` passed its dry run, claimed the run, started epoch 1, and died on
batch one:

```
[PERF] resnet20 joint multi-exit: channels_last on cuda:0
RuntimeError: [train resnet20] memory-format mismatch: input is contiguous
but conv weights are channels_last.  ... This is D-55
```

### D-87 — the same defect as D-55, from the other side

`place_model` defaulted `channels_last` to **True**. `build_loaders` defaulted
the same key to **False**. Only `_imagenet_config` ever set it explicitly, so
**every CIFAR-100 config omitted it** — and the two functions then gave
different answers to the same question. Model NHWC, batches NCHW.

D-55 was "the config said channels_last and nothing enforced it". D-87 is "two
places enforced it and disagreed". Same root: an invariant with more than one
home.

**Study 1's CIFAR runs predate `assert_layout_match`**, which is why this was
never seen. Those runs were most likely paying the conversion tax the whole
time — it costs throughput, not correctness, so the results stand, but the
timings in `01_PROTOCOL.md` may be pessimistic.

**Fix:** `place_model`'s default is now `False`, matching the loader; and the
CIFAR recipe **states** `channels_last: False` outright so nothing depends on a
default at all. D-59 measured channels_last as 6.7× slower on this GPU, so
False is also the fast answer. Canaries assert both halves and that the two
defaults agree.

### Study 3 now trains completely offline

```
[HF:hub] AUTH FAILURE -- check HF_TOKEN write scope
[HF:hub] BATCH FAILED after 8 attempts (17 files): 403 Forbidden
```

Two problems, and the second is the structural one.

1. **It was pushing to `msc-imagenet100`** during a CIFAR study, because
   `HF_REPO` defaults to the ImageNet repo. Every notebook now sets
   `MSC_HF_REPO=Shanmuk4622/msc-cifar100`.
2. **It was pushing at all.** A background uploader retrying every 30 minutes
   turns *"no network right now"* into *"the run failed"*. On a machine without
   a permanent connection that is the wrong default, and it killed a run that
   had already passed its dry run.

**Every Study 3 notebook now runs with `enable_hf=False`.** Nothing is uploaded
during training; everything is written to disk in full — configs, epoch
histories, telemetry, per-sample parquets, checkpoints, environment records.

**`S3_NB5_Publish` is new** and is the only notebook that touches the network:

* checks the token first with `hf_token_check` — valid, **write** scope, right
  namespace — so a 403 names its own cause instead of arriving under forty
  lines of traceback (D-84);
* lists and **sizes** everything before a byte moves, so a 40 GB surprise is
  visible while it is still cancellable;
* uploads folder-at-a-time through `hf_upload_resilient`, which survives a DNS
  drop mid-run (D-86);
* verifies with `resolve_meta` on specific paths — **a drained queue is not
  confirmation** (rules 9 and 10).

The local tree is the source of truth. HuggingFace is a copy.

Canaries: **55/55**, including "every training notebook has HF off", "the
publisher has it on", "token check precedes upload", and "verification is by
resolve".

---

## 2026-08-20 (P0 done) · NB0 ran — and predicts Study 2 SURVIVES

**P0 complete.** 45 CIFAR-100 base runs, 15 architectures,
`analysis/s3_exit_quality.csv`.

```
Spearman(exit_quality, excess) = +0.669   p < 0.0001   n = 45
```

**Positive — the opposite of what I expected.** Better early exits go with a
*larger* oracle excess, not a smaller one. In hindsight the mechanism is
obvious: a weak early exit is right on almost nothing, so it cannot rescue a
sample the final layer gets wrong. Rescues require competent early exits. The
pool grows with exit quality.

**The extrapolation, therefore:**

| exit_quality | predicted excess |
|---|---|
| 0.86 (best frozen run) | +10.98 pt |
| 1.00 (extrapolated) | **+12.45 pt** |

**GATE: expect Study 2 to SURVIVE joint training**, and the excess to be
*larger* than the +6.86 pt measured on frozen heads — not smaller. That is now
a pre-registered expectation `S3_NB1` can falsify.

### A confound I nearly published

`exit_quality = mean(acc_k / acc_full)` — **`acc_full` is its own denominator.**
A low-accuracy network scores high exit_quality with weak exits, *and* makes
more final-layer errors, which is exactly what `excess` counts. The raw
correlation could have been pure circularity:

```
Spearman(acc_full, excess)               = -0.672     <- the confound
PARTIAL (holding acc_full fixed)         = +0.617     <- what actually counts
```

The relationship survives, but **NB0 reported only the raw number**, so I would
have been quoting a confounded statistic. The partial correlation is now
computed and printed, with a message that says outright when the raw
relationship was mostly the confound. Fixed before P1, not after.

---

## 2026-08-20 (P1 blocked) · NB1 tried to download CIFAR-100 that was already on disk

```
[DATA] not found locally -- downloading shanmuk4622/dataset-cifar100-python
[DATA] falling back to torchvision auto-download
  1%|          | 983k/169M [00:18<2:42:44, 17.2kB/s]
```

**Two causes, both mine.**

1. **The notebooks called `M.base_config(...)` instead of `sess.config(...)`.**
   The bound method calls `prepare_data()` and fills in `data_root`; the raw
   function does not. So the loader fell through to the download path. This is
   D-54 again — the same frozen-vs-bound distinction, in a new notebook.
2. **`locate_cifar100()` could not be told where the data was.** It checked
   `/kaggle/input` and a scratch folder, then downloaded. ImageNet-100 has had
   `MSC_IN100_DIR` since the port; CIFAR-100 had no equivalent.

**Fixes:** `locate_cifar100()` gained an explicit-location step that runs before
anything that downloads, reading `MSC_CIFAR_DIR` plus a few common local paths.
It accepts either the parent folder or `cifar-100-python` itself, but only
unwraps the parent when the basename actually matches — checking the parent
unconditionally would make a typo resolve via whatever sits beside it, which is
a silent wrong answer rather than a visible miss. Every notebook now sets
`CIFAR_DIR` at the top and reports what it resolved to, before training.

Canaries: 41/41, including both spellings resolving, a wrong path *not*
resolving, the explicit check ordering before the download, and every notebook
being free of bare `M.base_config`.

---

## 2026-08-20 (earlier) · S3_NB0 failed on first run — root cause was the paths cell

```
KeyError: 'method'
  base = meta[(meta['method'] == 'base') & (meta['dataset'] == 'cifar100')]
```

**Root cause, two layers deep.** I built the session with
`build_notebooks.py`'s `worker_cell`, which constructs a `Session` **without**
`work_root`. The Session then chose its own default directory instead of the
one holding the runs, so the scan found nothing, `pd.DataFrame([])` had no
columns, and the first column access raised `KeyError` — a message pointing at
pandas, 40 lines from the real problem.

The alias cell's guard did not catch it because it asserted `runs/` **is a
directory**, and a Session *creates* that directory. The check proved nothing.

**Fixes:**

1. **`paths_cell()`** replaces `worker_cell` + `alias_cell` in all five
   notebooks. It resolves through `M.resolve_storage`, exactly as Study 2's
   notebooks do, and passes `work_root=MSC_ROOT` to the Session. `MSC_ROOT` is
   settable at the top if the runs live somewhere specific.
2. **It counts MEASURED RUNS, not directories**, and raises with the fix
   instructions if there are none.
3. **`runs_cell()` / `measured_runs()`** — one accessor for "which runs do I
   analyse", validating that every expected column exists and applying the
   pilot-replicate dedupe in a single place. NB0 and NB2 both use it instead of
   each re-typing the logic.

**A tooling defect, found because this was the fourth time.** An escaped `\n`
inside a generated f-string kept collapsing into a real newline, breaking a
cell. `check_names.py` silently skipped unparseable cells, so the symptom
appeared as *"undefined name three cells later"* rather than *"this cell does
not parse"*. Two changes:

* `check_names.py` now **reports** an unparseable cell instead of skipping it;
* the generator runs the **parse gate before the name check**, so a syntax
  error is reported as a syntax error.

It paid for itself within the hour: the very next mistake (a `**` on a
conditional expression) was reported at the right line instead of surfacing as
a phantom missing name.

**`tools/s3_nb0_harness.py`** added — executes S3_NB0's real cells against
synthetic frames built so that excess *falls* as exit quality rises, and
requires the notebook to recover that sign. It does (slope −0.32). The three
in-notebook canaries pass inside the harness too. NB0 no longer ships unrun.

---

## 2026-08-20 · Study 3 built

**Library changes** (`src/msc_lib.py`, selftest 459/459 still passing):

1. **`exit_loss_weights(K, scheme)`** — new, pure, fully canaried. Three
   schemes (`uniform` / `linear` / `final_heavy`), always summing to 1.0 so the
   joint loss is directly comparable in magnitude to a single-head loss.
   Without that normalisation "same LR" would silently mean a different
   effective step size and the frozen-vs-joint comparison would confound
   optimisation with architecture.
2. **`joint_exits` in `train_backbone`** — a *guarded branch*, not a parallel
   training loop. A second loop would duplicate the resume, push and registry
   machinery, which is what caused D-23 and D-49. Defaults to `False`, so every
   Study 1 run is bit-identical.
3. **`ckpt_best` stays in the established format.** A joint run's model is a
   `MultiExitModel` whose `state_dict` is prefixed `backbone.*` / `heads.*`,
   and `run_oracle` loads `ckpt_best` into a *plain* backbone with
   `strict=True`. Writing the wrapped dict would have broken every downstream
   consumer. The backbone is saved unwrapped and the heads go to
   `exit_heads.pt` via `exit_heads_path()` — **THE accessor** (D-23) — so
   measurement, budgets and all of Study 2's analysis work unchanged on joint
   runs. `ckpt_last` keeps the full wrapped state, because resume needs it.
4. **`evaluate()` unwraps per-exit logits**, taking the final exit, so accuracy,
   calibration and best-checkpoint selection keep the meaning they had.
5. **`subset_path` in `_subset_train`** — an explicit keep-list for Q3, distinct
   from `train_subset_frac` (a random smoke-test fraction). Preserves
   `index_space` so global `sample_idx` stays valid (D-49). **Refuses** a
   missing file, an empty list, or an out-of-range index rather than falling
   through to full-data training — which would make every pruning arm identical
   and return a null that looks like a finding.

Neither `joint_exits`, `exit_weight_scheme` nor `subset_path` is in
`_HASH_EXCLUDE`: all three change what is trained, so the runs get distinct
`config_hash` values and cannot collide with or resume from a Study 1
checkpoint.

**Notebooks** (`build_notebooks_study3.py` → `notebooks_study3/`), reusing the
CIFAR-100 generator's bootstrap, session and data cells:

| notebook | phase | cost |
|---|---|---|
| `S3_NB0_Extrapolate` | P0 | ~15 min CPU |
| `S3_NB1_JointTrain` | P1 | ~10 GPU-h |
| `S3_NB2_Compare` | P2 | ~10 min CPU |
| `S3_NB3_Router` | P3 | ~5 GPU-h |
| `S3_NB4_Pruning` | P4 | ~18 GPU-h |

**Verification:** `tools/s3_canaries.py` **32/32**; selftest 459/459;
`check_names` and the Python-3.10 parse gate pass on all five notebooks; every
`M.*` and `sess.*` the notebooks reference was checked to exist against the
library's AST.

**One naming decision worth recording.** The CIFAR generator imports the
library as `msc` and finds the results root via `sess.work`; Study 2's
notebooks say `M` and `MSC_ROOT`. Study 3 reads Study 2's outputs, so both have
to work. Rather than interleaving conventions, there is one visible
`alias_cell()` that declares `M = msc` and `MSC_ROOT = sess.work` — the
accessor rule applied to names.

### What is deliberately *not* built

No confirmation runs, no second dataset, no ImageNet. Every one of those is
downstream of Q1, and Q1 may falsify Study 2 outright — in which case most of
them would be work spent on a withdrawn claim. **Study 1 spent 79 GPU-hours on
exactly that mistake.**

---

## How to use this file

After each notebook: add a dated entry, fill the prediction table, and state
what the **next** action is. If a result contradicts an earlier entry, add a new
entry and mark the old one `SUPERSEDED` — never edit history. Study 2's log did
this and it is the reason the three wrong routing implementations are
recoverable rather than folklore.
