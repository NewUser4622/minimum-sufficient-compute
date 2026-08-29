# Study 4 — risk register

Each risk has a **detector**, a **trigger**, and a **response**, fixed before
any result exists.

Study 3's register worked — R-02, R-05, R-06 and R-09 all fired, and because the
responses were written in advance none turned into an argument afterwards. This
file leads with the risk that actually materialised six times.

---

## R-01 · Code written against an assumed state · **HIGH — occurred 6× in Study 3**

Every Study 3 defect was this one shape:

| defect | the assumption |
|---|---|
| D-87 | that one flag had one default (it had two, in two files) |
| D-88 | that `stage=` selected the work (`fn` does) |
| NB3 ×3 | that run ids could be assembled; that a frame had columns; that checkpoints were on disk |
| NB4 | that grepping `build_loaders` tested `subset_path` (it lives in `_subset_train`) |
| HF ×2 | that a truncated API response was a complete listing |

**Detector.** Every phase opens with a **preflight cell**: a table of what it
needs and whether each item exists, printed before any work.
**Trigger.** Cell one, every notebook.
**Response.** Refuse and name the gap. Never fall through to a default that
produces a plausible-looking null.

**Specific preflights required:**

| phase | must verify before running |
|---|---|
| P0 | `s2_headroom_sweep.csv` spans ρ 0.40–0.95; joint runs have `test.parquet` |
| P1 | `top2p_dk` present for every *k* (margin); `pred_dk` for every *k* (patience) |
| P2 | ImageNet-100 pack resolves; `MSC_IN100_DIR` set; **throughput gate armed** |
| P3 | MSDNet builds; `forward_features` returns K tensors; FLOPs table is non-degenerate |

## R-02 · The layout regression returns and costs 35 GPU-hours · **HIGH**

`p0-resnet50-imagenet100-base-s1` took **41.5 GPU-h** against a documented ~6 h,
because the model was NHWC and the batches NCHW (D-55/D-59/D-87). Nothing looked
broken — the loss fell, the accuracy climbed, each epoch simply took four times
too long.

**Detector.** Measure img/s during **epoch 1** and compare against
`MEASURED_THROUGHPUT`. Also assert `assert_layout_match` fires on batch one.
**Trigger.** P2, first epoch, before epoch 2 starts.
**Response.** **Abort the run** if throughput is more than 2× below benchmark.
Losing one epoch beats losing 35 hours, and the flat-line signature is
diagnostic: a layout tax is a fixed cost, not a variable one.

## R-03 · Our MSDNet is not the published MSDNet · **HIGH for P3**

A re-implementation can differ from the paper in ways that matter — scale
counts, growth rate, classifier placement, the transition layers.

**Detector.** Compare final accuracy against the published CIFAR-100 number for
the same configuration. Record every hyperparameter and its source.
**Trigger.** After P3's first run completes.
**Response.** If accuracy is more than 3 pt below published, the architecture is
wrong and H5 is not tested — fix it before reporting. If it is close, state the
configuration explicitly and note that variant sensitivity is untested.

**This is why P3 is last.** It is the only phase needing genuinely new
architecture code, and this environment cannot execute torch, so it ships
unverified until it reaches the GPU.

## R-04 · A stronger baseline overturns the headroom numbers · **MEDIUM — and welcome**

P1 adds margin and patience. If either beats confidence thresholding, every
headroom figure in Studies 2–3 was computed against a weak comparator.

**Detector.** Baseline accuracy at matched cost, all three rules, all seven
budgets.
**Trigger.** P1, immediately, and **before** P2 spends anything.
**Response.** Recompute the headline against the **strongest** baseline and say
so. A limits paper that picks a weak comparator is worthless, so this risk
firing improves the paper. It is also cheap: pure re-analysis.

## R-05 · The transformer behaves differently from the CNNs · **MEDIUM**

Every result so far is convolutional. `vit_small_p16` is the first transformer,
and H4b isolates it deliberately.

**Detector.** H4b — the transformer alone must clear 2.0 pt.
**Trigger.** P2.
**Response.** If conv holds and the transformer does not, **the claim becomes
architecture-conditional and the title must say so.** Do not average the two and
report a mean that describes neither.

## R-06 · Bootstrap intervals are mistaken for seed intervals · **MEDIUM**

The interval resamples the 10,000 test images. It captures sampling noise, not
training variation, and a reader who conflates them will overstate the result.

**Detector.** Manuscript review; the notebook prints the caveat with the number.
**Trigger.** P0.
**Response.** Label every interval "95 % CI over test samples (n = 10,000);
**not** a seed interval". Where seed variation exists (Study 2's 90 pairs),
report it separately and distinctly.

## R-07 · P2/P3 find the effect and we stop looking · **MEDIUM**

Studies 2 and 3 both had a headline that matched the prediction and both were
wrong on first measurement — the +5.165 pt, the +22.41 pt mechanism story, the
raw P0 correlation. Each was corrected only because it was attacked after it
looked good.

**Detector.** Stopping rule 4.
**Trigger.** The moment a number matches its prediction.
**Response.** Before writing it up, spend one round trying to break it: what
confound would produce this? Study 3's conditioning on `acc_full` came from
exactly this habit and made the result stronger.

## R-08 · Hardware · **LOW, but it has bitten**

224px joint multi-exit training holds K heads and K losses at ImageNet
resolution — materially more memory than Study 3's 32px runs. A benchmark once
crashed this machine.

**Detector.** Dry run, then a memory probe before epoch 1.
**Trigger.** Before P2's first real batch.
**Response.** Reduce batch size rather than risk the machine.

---

## What success means

Not "the hypotheses are confirmed". Success is that **the three standing
objections to Paper A each have an answer with a number attached** — even if the
answer is "the claim is narrower than we thought".

By that standard every branch succeeds. The branch that wastes the compute is
running P2 and P3 and still not being able to say whether the effect is a
property of early-exit networks or of our particular way of building them —
which is what R-03's accuracy check and H4b's transformer split exist to
prevent.
