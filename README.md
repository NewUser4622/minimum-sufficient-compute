# MSC — Minimum Sufficient Compute

Research package for: *Is Compute Difficulty Architecture-Agnostic? Measuring and Distilling Per-Sample Minimum Sufficient Computation.*

**Artifacts:** [`huggingface.co/datasets/Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100)

## The question

Adaptive-inference networks decide at runtime how much computation to spend per input. That decision is almost always gated on the deployed model's own confidence — which is exactly the signal a small, poorly-calibrated model is worst at producing. Before building a better gate, this project asks the prior question the field has assumed rather than measured:

> **Is the amount of computation an input requires a property of the input, or of the model?**

If it is a property of the input, a large teacher can supervise a small student's compute-allocation policy. If it is not, a growing line of teacher-guided adaptive-inference work rests on a false premise — and saying so is a contribution.


## ✅ Phase 0 cleared — `FULL-PROGRAM`

Decided 2026-08-02. Full write-up: **[`08_PHASE0_RESULTS.md`](08_PHASE0_RESULTS.md)**

| Gate | Need | Got | |
|---|---|---|---|
| ρ_seed (noise ceiling) | ≥ 0.6 | **0.715** | ✓ |
| T (transfer, disattenuated) | ≥ 0.7 | **0.946** | ✓ |
| ΔR² (irreducibility) | ≥ 0.05 | **0.254** | ✓ 5× |
| Shuffled control | ≈ 0 | **0.007** | ✓ |

Every gate holds across the whole τ grid. All four backbones beat their published
references (resnet32x4 79.59% vs 79.42; wrn-40-2 76.89% vs 75.61).

**One pre-registered hypothesis was refuted:** H2 predicted PC1 ≥ 0.60 across the
compute axes; measured **0.503**. Compute-need is *not* one-dimensional — depth
and resolution correlate at 0.38, precision is nearly independent at ~0.23. That
is a result in its own right, and it says results on depth-based early exit do
not license claims about the other axes.

## Phase 1 atlas — trained 44/45, measured 39/45

Audited against HuggingFace at revision `a4aef3a`, 2026-08-04.

| arch | family | top-1 | published | Δ | ρ_seed @τ=0.1 |
|---|---|---|---|---|---|
| `resnet32x4` | resnet | 79.74 | 79.42 | +0.32 | **0.7256** |
| `wrn_40_2` | wrn | 76.06 | 75.61 | +0.45 | 0.7089 |
| `vgg13` | vgg | 75.70 | 74.64 | +1.06 | 0.6689 |
| `resnet110` | resnet | 74.38 | 74.31 | +0.07 | 0.6339 |
| `wrn_16_2` | wrn | 73.79 | 73.26 | +0.53 | *not measured* |
| `resnet56` | resnet | 73.69 | 72.34 | +1.35 | 0.6217 |
| `resnet8x4` | resnet | 73.26 | 72.50 | +0.76 | 0.6671 |
| `wrn_40_1` | wrn | 72.41 | 71.98 | +0.43 | 0.6559 |
| `shufflenetv2` | mobile | 71.93 | 70.50 | +1.43 | 0.6698 |
| `vgg8` | vgg | 71.73 | 70.36 | +1.37 | **0.7248** |
| `resnet20` | resnet | 70.13 | 69.06 | +1.07 | 0.6425 |
| `mobilenetv2` | mobile | 70.10 | — ⚠ | — | 0.6880 |
| `convnext_femto` | convnext | 62.67 | — | — | 0.7084 |
| `mixer_nano` | **mixer** | 60.23 | — | — | **0.5470** |
| `vit_tiny` | **vit** | 59.33 | — | — | **0.5475** |

ResNet rows are 3-seed means; the rest are a representative seed pending NB15.
⚠ `mobilenetv2`'s published reference is for a half-width model — see D-14.

### New finding: measurement reliability is architecture-dependent

Every convolutional network lands in **[0.622, 0.726]**. Both non-convolutional
models sit at **0.547** — below all twelve CNNs, and below the pre-registered
0.60 reliability threshold. Separation margin 0.074, no overlap (τ ≤ 0.2).

It is not an accuracy artifact: inside the CNN family, ceiling height and top-1
accuracy are **uncorrelated** (Spearman +0.035). `convnext_femto` is the least
accurate CNN — 2.4 points above `mixer_nano` — yet its ceiling is 0.161 higher,
while the 17-point climb from `convnext_femto` to `resnet32x4` buys only +0.017.

This matters beyond this project: **a cross-architecture difficulty study that
does not divide by a per-architecture noise ceiling is comparing quantities
measured with unequal precision.** The example-difficulty literature generally
does not. Phase 0 also happened to pick the two *most* reliable architectures in
the zoo, so its 0.715 was an optimistic sample against an atlas mean of 0.676.

Running record of results, defects and decisions: **[`09_LAB_NOTEBOOK.md`](09_LAB_NOTEBOOK.md)**.

### What is not done yet

| | |
|---|---|
| **Q3 transfer across the atlas** | **1 of 91 pairs computed.** NB11 has not been re-run — the CNN→Transformer question the atlas exists to answer is one CPU-only notebook away |
| Q2 / Q4 across the atlas | still Phase-0-only |
| `wrn_16_2` | trained but unmeasured — the atlas is currently **14** architectures, not 15 (D-15) |
| Q4 battery | still 5 of 7 scores on the `test` split (D-11) — affects the ΔR² = 0.254 headline |

---

## Contents

### Documents

| File | What it is |
|---|---|
| `00_RESEARCH_PROTOCOL.md` | **Master document.** Novelty positioning against SAFE-KD / ERDE / EENet, formal definition of MSC, five pre-registered research questions with both-direction predictions, run matrix, baselines, fair-comparison protocol, paper skeleton, venue plan, risk register. |
| `01_PHASE0_GO_NOGO.md` | The decisive pilot. 4 runs, ~12 GPU-hours, with an explicit decision table. |
| `02_ENGINEERING_SPEC.md` | Original infrastructure contract. **Repo layout superseded by `06_DATA_SCHEMA.md`;** the checkpoint contract and reproducibility requirements still stand. |
| `03_IMPLEMENTATION_PLAN.md` | What the implementation understands from the protocol, every design decision with justification, the compute-configuration grid, and the corrections the preflight surfaced. |
| `04_NOTEBOOK_RUNBOOK.md` | **How to run it.** Kaggle setup, run order, worker splitting, push policy, resumability, troubleshooting. |
| `05_PLAIN_ENGLISH_GUIDE.md` | **Start here if you want to understand the project.** No jargon — what MSC is, why each question matters, what every notebook does. |
| `06_DATA_SCHEMA.md` | The HuggingFace repository structure and the complete data schema — 171 per-epoch columns, 91 final-evaluation columns, mapped to the collection requirements. |
| `08_PHASE0_RESULTS.md` | **The Phase 0 verdict and every number behind it.** Gate table, τ-curves, per-question analysis, and what each result does and does not establish. |
| `09_LAB_NOTEBOOK.md` | **Running record.** Every measured number with its provenance, every defect found with a contamination analysis, every design decision changed, and a crib mapping all of it onto paper sections. Append-only. |
| `07_REPLICATION_PLAYBOOK.md` | **Project-agnostic.** How to rebuild this whole Kaggle + HuggingFace infrastructure in any future project, including the six bugs found along the way. |

### Code

| File | What it is |
|---|---|
| `msc_core.py` | Reference implementation: the MSC oracle plus every analysis statistic. numpy/scipy/pandas/sklearn only — no torch. Self-tested. |
| `msc_torch.py` | Reference model-side components: exit heads, ordinal sufficiency head, three-term loss, Learn-then-Test calibration. |
| `src/msc_lib.py` | The pipeline: HF sync with shared rate limiting, sharded registry, cost-balanced work splitting, 15-architecture zoo, FLOPs budgets, resumable instrumented training, three-axis oracle, MSC-KD, final evaluation, analysis. **157 offline self-checks.** |
| `build_notebooks.py` | Regenerates the 16 Kaggle notebooks, embedding `msc_lib.py` and `msc_core.py` as base64. |
| `notebooks/` | 16 self-contained Kaggle notebooks. |

## The notebooks

| # | Notebook | GPU | Runs | Est. GPU-h | Wall-clock @ 1 worker |
|---|---|---|---|---|---|
| 00 | Setup & Verify | T4 | — | — | ~15 min |
| 01 | Phase 0 — train | T4 | 4 | ~10 | ~10 h (2 sessions) |
| 02 | Phase 0 — measure + final eval | T4 | 4 | ~2 | ~2 h |
| 03 | Phase 0 — decision | **off** | — | — | ~10 min |
| 04 | Atlas — ResNets | T4 | 15 | ~25 | ~25 h (3 sessions) |
| 05 | Atlas — WRN + VGG | T4 | 15 | ~19 | ~19 h (3 sessions) |
| 06 | Atlas — Mobile | T4 | 6 | ~9 | ~9 h (2 sessions) |
| 07 | Atlas — ConvNeXt/ViT/Mixer | T4 | 9 | ~36 | ~36 h (5 sessions) |
| 08 | Atlas — measure + final eval | T4 | 45 | ~27 | ~27 h (4 sessions) |
| 09–12 | Analysis Q1–Q4 | **off** | — | — | 5–20 min each |
| 13 | MSC-KD training | T4 | 9 | ~30 | ~30 h |
| 14 | Method comparison | T4 | — | ~5 | ~5 h |
| 15 | Paper outputs | **off** | — | — | ~10 min |

**Total ≈ 163 GPU-hours.** Every notebook defaults to `NUM_WORKERS = 1`, meaning
one account does all of it. Raise it and run the same notebook on each account
with a different `WORKER_ID` to divide the time — the wall-clock column at
2/4/6 workers is printed inside each notebook.

Sessions assume the 8.5 h pause-and-resume limit. A notebook needing more than
one session resumes automatically: start a fresh session and re-run.

## Why the structure changed from CEB-KD

The previous plan had one structural flaw that mattered more than any technical detail: **its entire value was contingent on one method beating baselines.** If the method lost, there was no paper.

This restructure makes three of the five research questions produce publishable findings *whichever way they resolve*. The method is the last section, not the thesis. Concretely:

- **Seven loss terms → three.** Monotonicity is enforced architecturally rather than by a penalty.
- **"Counterfactual" dropped.** It collides with existing counterfactual-KD work, and what is actually being done is an intervention `do(b)`, not a counterfactual.
- **Energy demoted from novelty to methodology.** Lifecycle accounting of distillation is now published work; report it, cite it, claim nothing.
- **A noise ceiling added.** Every transfer number is divided by seed-to-seed agreement for the same architecture. The example-difficulty literature omits this, which makes its raw cross-architecture correlations hard to interpret.

## The three findings that hold regardless of the method

1. **Is compute need one-dimensional across reduction axes?** Never asked. Every adaptive-inference paper picks one axis (usually depth) and treats it as *the* compute axis. Either answer is a result.
2. **How well does compute need transfer across architectures, corrected for measurement noise?** First noise-ceiling-corrected measurement of this.
3. **Is MSC reducible to classical difficulty scores?** The main threat to the construct, tested head-on with partial correlation and nested ΔR² against a seven-score battery.

## Start here

```bash
python msc_core.py                  # self-test: the oracle and every statistic
python src/msc_lib.py --selftest    # 157 offline checks on the pipeline, no GPU
python build_notebooks.py --check   # confirm the notebooks match the library
```

Then, in order:

1. **Read SAFE-KD (arXiv 2602.03043) as a team.** It is the closest prior art, it appeared in February 2026, and a competing group is active in this space. Write a one-page differentiation memo before writing any code. If it already does cross-architecture transfer, the protocol changes.
2. Verify the 2026 arXiv IDs cited in the protocol — several are recent preprints that may have been revised.
3. Run `notebooks/NB00_Setup_And_Verify` on every account. It ends with a kill-and-resume acceptance test; do not proceed past a failure there.
4. ~~Run Phase 0 (NB01 → NB02 → NB03).~~ **Done — see `08_PHASE0_RESULTS.md`.**
5. ~~Hold the decision meeting.~~ **Verdict: `FULL-PROGRAM`.**
6. ~~NB04 → NB07 (atlas training).~~ **Done — 44/45 trained.**
7. ~~NB08 (measurement).~~ **39/45 measured** — six gaps tracked as D-15.
8. ~~NB09 (Q1).~~ **Done atlas-wide — see the ceiling table above.**
9. **Now: NB11 (Q3).** This is the highest-value remaining action in the whole
   project. All 91 cross-architecture pairs are measured and the ceilings NB11
   needs as denominators now exist, so it runs on CPU in minutes and produces the
   result the atlas was built for. Then NB10 (Q2), NB12 (Q4 on `train_holdout`),
   NB15 (tables and figures).

Phase 0 cost 9.5 GPU-hours and cleared every gate, so the remaining ~150 are
justified. The two open questions it could not answer — does transfer survive the
CNN→Transformer boundary, and does the non-one-dimensionality of compute-need
generalise — are exactly what the atlas is for.

## Running across several accounts

Every account runs the **same notebook**. Two lines differ:

```python
ACCOUNT     = 'acct1'      # label for the run log
NUM_WORKERS = 1            # DEFAULT: this account does everything
WORKER_ID   = 0            # 0..N-1, DIFFERENT on each account
```

Leave `NUM_WORKERS = 1` and one account runs the whole notebook. Raise it only
when you actually have several accounts going at once — then give each a
different `WORKER_ID`.

Work is split by a deterministic cost-balanced scheduler — no communication, no collisions, no gaps. See `04_NOTEBOOK_RUNBOOK.md` §3.
