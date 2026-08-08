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
| **Data** | verified, not yet packed — [`25_IN100_DATA_CARD.md`](25_IN100_DATA_CARD.md) |
| **Dataset** | 129,395 images · 100 classes · fingerprint `2b6269ef…` |
| **Zoo** | 8 architectures registered, **0 built on a GPU yet** |
| **Storage** | **local disk only** — no HuggingFace, no network at run time |
| **Self-checks** | **310** offline, all passing, exit code verified |
| **Notebooks** | **5** (`notebooks_in100/`), validated clean, base64 round-trips |
| **Defects found this port** | **3** (D-37, D-38, D-39) · 3 fixed · 0 open |
| **Runs trained** | 0 / 24 |
| **Artifacts** | `huggingface.co/datasets/Shanmuk4622/msc-imagenet100` — not yet created |

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
