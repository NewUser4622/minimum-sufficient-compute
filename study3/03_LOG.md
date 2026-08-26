# Study 3 — live log

**Newest first.** One entry per session. Every entry answers four things: what
changed, what it cost, what it **settled**, what is next.

A phase is `done` only when its artifact exists on disk **and has been opened
and checked**. Study 1's D-79 was 18 runs that all reported success while the
number they existed to produce was never computed.

---

## Status board

| | what | cost | state | artifact |
|---|---|---|---|---|
| **P0** | exit quality → excess extrapolation | ~15 min CPU | **ready to run** | `analysis/s3_exit_quality.csv` |
| **P1** | Q1 joint exit training | ~10 GPU-h | ready — gated on P0 | 3 runs under `runs/p4-*-jointexit-s1` |
| **P2** | Q1 verdict, paired comparison | ~10 min CPU | ready — needs P1 | `analysis/s3_q1_comparison.csv` |
| **P3** | Q2 learned router | ~5 GPU-h | ready — gated on P2 | `analysis/s3_router_capture.csv` |
| **P4** | Q3 pruning | ~18 GPU-h | ready — independent | `analysis/s3_pruning.csv` |

**Next action: run `S3_NB0_Extrapolate`.** It is free and it sets the
expectation that `S3_NB1` will confirm or falsify.

---

## Pre-registered predictions — fill in as results arrive

Written **before** any run. Do not edit the prediction column.

| | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| **H1** | oracle excess survives joint exit training | ≥ 2.0 pt, 3 of 3 archs | _pending_ | _pending_ |
| **H2** | a learned router captures little of the gap | < 25 % (cross-seed) | _pending_ | _pending_ |
| **H3** | saturated-source pruning is worse | ≥ 1.0 pt at 30 % keep | _pending_ | _pending_ |
| **H3b** | saturated source ≈ random pruning | ± 0.5 pt at 30 % keep | _pending_ | _pending_ |

---

## 2026-08-20 (later) · S3_NB0 failed on first run — root cause was the paths cell

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
