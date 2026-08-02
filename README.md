# MSC — Minimum Sufficient Compute

Research package for: *Is Compute Difficulty Architecture-Agnostic? Measuring and Distilling Per-Sample Minimum Sufficient Computation.*

**Artifacts:** [`huggingface.co/datasets/Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100)

## The question

Adaptive-inference networks decide at runtime how much computation to spend per input. That decision is almost always gated on the deployed model's own confidence — which is exactly the signal a small, poorly-calibrated model is worst at producing. Before building a better gate, this project asks the prior question the field has assumed rather than measured:

> **Is the amount of computation an input requires a property of the input, or of the model?**

If it is a property of the input, a large teacher can supervise a small student's compute-allocation policy. If it is not, a growing line of teacher-guided adaptive-inference work rests on a false premise — and saying so is a contribution.

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
| `07_REPLICATION_PLAYBOOK.md` | **Project-agnostic.** How to rebuild this whole Kaggle + HuggingFace infrastructure in any future project, including the six bugs found along the way. |

### Code

| File | What it is |
|---|---|
| `msc_core.py` | Reference implementation: the MSC oracle plus every analysis statistic. numpy/scipy/pandas/sklearn only — no torch. Self-tested. |
| `msc_torch.py` | Reference model-side components: exit heads, ordinal sufficiency head, three-term loss, Learn-then-Test calibration. |
| `src/msc_lib.py` | The pipeline: HF sync with shared rate limiting, sharded registry, cost-balanced work splitting, 15-architecture zoo, FLOPs budgets, resumable instrumented training, three-axis oracle, MSC-KD, final evaluation, analysis. **138 offline self-checks.** |
| `build_notebooks.py` | Regenerates the 16 Kaggle notebooks, embedding `msc_lib.py` and `msc_core.py` as base64. |
| `notebooks/` | 16 self-contained Kaggle notebooks. |

## The notebooks

| # | Notebook | GPU | Time | Workers |
|---|---|---|---|---|
| 00 | Setup & Verify | T4 | 15 min | every account |
| 01 | Phase 0 — train | T4 | ~12 GPU-h | 4 |
| 02 | Phase 0 — measure + final eval | T4 | ~2 h | 4 |
| 03 | Phase 0 — decision | off | 10 min | 1 |
| 04–07 | Atlas training (ResNets / WRN+VGG / Mobile / Modern) | T4 | ~110 GPU-h | 6 |
| 08 | Atlas measurement | T4 | ~25 GPU-h | 6 |
| 09–12 | Analysis Q1 / Q2 / Q3 / Q4 | off | 5–20 min | 1 |
| 13–14 | MSC-KD training and comparison | T4 | ~125 GPU-h | 6 |
| 15 | Paper outputs | off | 10 min | 1 |

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
python src/msc_lib.py --selftest    # 138 offline checks on the pipeline, no GPU
python build_notebooks.py --check   # confirm the notebooks match the library
```

Then, in order:

1. **Read SAFE-KD (arXiv 2602.03043) as a team.** It is the closest prior art, it appeared in February 2026, and a competing group is active in this space. Write a one-page differentiation memo before writing any code. If it already does cross-architecture transfer, the protocol changes.
2. Verify the 2026 arXiv IDs cited in the protocol — several are recent preprints that may have been revised.
3. Run `notebooks/NB00_Setup_And_Verify` on every account. It ends with a kill-and-resume acceptance test; do not proceed past a failure there.
4. Run Phase 0 (NB01 → NB02 → NB03).
5. **Hold the decision meeting** against the table in §6 of `01_PHASE0_GO_NOGO.md`, and write the decision into the repo.

The infrastructure for the full program is built and tested, but **Phase 0 still runs first and its gate still governs.** Building the pipeline early costs nothing; skipping the gate costs a month.

## Running across several accounts

Every account runs the **same notebook**. Two lines differ:

```python
ACCOUNT     = 'acct1'      # label for the run log
NUM_WORKERS = 6            # how many accounts you're running
WORKER_ID   = 0            # 0..N-1, DIFFERENT on each account
```

Work is split by a deterministic cost-balanced scheduler — no communication, no collisions, no gaps. See `04_NOTEBOOK_RUNBOOK.md` §3.
