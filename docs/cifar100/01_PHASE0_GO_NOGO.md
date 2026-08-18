# Phase 0 — Decisive Pilot

> ## OUTCOME: `FULL-PROGRAM` (2026-08-02)
> ρ_seed **0.715** · T **0.946** · ΔR² **0.254** · shuffled control **0.007**
>
> Row 1 of the §6 decision table. Every gate cleared at every τ. Full analysis in
> `08_PHASE0_RESULTS.md`; machine-readable record in
> `analysis/phase0_decision.json` on HuggingFace.
>
> One surprise: H2 (PC1 ≥ 0.60) was **refuted** at 0.503 — compute-need is not
> one-dimensional across axes. That question was not part of the gate, so it does
> not change the decision, but it is a finding.

**Purpose:** determine in ~12 GPU-hours whether the MSC construct is real, before committing ~1,200 GPU-hours to the full program.

**Rule:** no other code gets written until this returns numbers. Not the HF push infrastructure, not the method, not the atlas. If Phase 0 fails, all of that work is wasted, and the fastest way to find out is to run the smallest possible experiment that can falsify the premise.

---

## 1. What Phase 0 must answer

| Question | Statistic | Gate |
|---|---|---|
| Is MSC stable across seeds? | $\rho_S^{\text{seed}}$ | ≥ 0.6 pass · 0.4–0.6 marginal · < 0.4 fail |
| Does MSC transfer within a family? | $T(\text{r32x4}, \text{wrn40-2})$ | ≥ 0.7 pass · 0.5–0.7 marginal · < 0.5 informative-fail |
| Is MSC more than difficulty? | $\Delta R^2$ | ≥ 0.05 pass · 0.02–0.05 marginal · < 0.02 reframe |
| Is the signal real at all? | shuffled-target control | shuffled $T$ must be ≈ 0 |

The fourth row is a sanity check on the pipeline, not on the science. If shuffled targets show nonzero transfer, there is a bug — most likely index misalignment between models' per-sample tables. Catch it here.

---

## 2. Runs

Four backbone trainings, CIFAR-100, standard CRD/DKD recipe (240 epochs, SGD, momentum 0.9, wd 5e-4, batch 64, LR 0.05 with ×0.1 decay at 150/180/210, random crop w/ 4px pad + horizontal flip):

| Run ID | Architecture | Seed | Purpose |
|---|---|---|---|
| `p0-r32x4-s1` | resnet32x4 | 1 | reference model A |
| `p0-r32x4-s2` | resnet32x4 | 2 | **noise ceiling for A** |
| `p0-wrn40x2-s1` | wrn-40-2 | 1 | transfer target B |
| `p0-wrn40x2-s2` | wrn-40-2 | 2 | **noise ceiling for B** |

~3 h each on a single T4 → 12 GPU-h, or one afternoon across two accounts.

Expected accuracies to sanity-check against published numbers: resnet32x4 ≈ 79.4%, wrn-40-2 ≈ 75.6%. If you are more than ~1 point below these, fix training before proceeding — a badly-trained teacher produces meaningless MSC.

---

## 3. Budget axes for Phase 0

Keep it minimal. **Depth and resolution only.** Width requires slimmable training (defer to Phase 1c); quantisation requires a PTQ pipeline (defer).

### Depth axis
Attach $K=5$ exit heads at fractional depths $\{0.2, 0.4, 0.6, 0.8, 1.0\}$ of the backbone. Each head: global average pool → BN → linear.

Train heads with the **backbone frozen**, 20 epochs, LR 0.01, cosine decay. ~15 min per model. Freezing is essential — if the backbone adapts, you are measuring a different network at each exit and the "same model under reduced compute" interpretation collapses.

### Resolution axis
Evaluate at $r \in \{16, 20, 24, 28, 32\}$ px via bilinear downsample-then-upsample back to 32 (so the network shape is unchanged and only information content varies). **No retraining.** Cost is measured as the FLOPs of the network run at native $r$ — document this as an idealised cost model, since the actual measured run is at 32px.

> This is a real methodological wrinkle and reviewers will notice. Two honest options: (a) run genuinely at native resolution with adaptive pooling before the classifier and report that, or (b) use the downsample-upsample proxy and label the cost as idealised. Option (a) is cleaner; use it if the architecture tolerates it. Decide in Phase 0 and stay consistent.

---

## 4. Outputs per run

For each run, save a per-sample table over the CIFAR-100 **test set** (10,000 rows), plus the same over a held-out 5,000-sample slice of train (to check whether MSC structure differs on seen data — a free extra finding):

```
sample_idx, label,
  # depth axis, K=5
  pred_d1..pred_d5, top1p_d1..top1p_d5, top2p_d1..top2p_d5,
  # resolution axis, K=5
  pred_r16..pred_r32, top1p_r16..top1p_r32, top2p_r16..top2p_r32,
  # difficulty battery (full compute)
  msp, margin, entropy, ce_loss, el2n, forget_events, pred_depth
```

Store as Parquet. This table is the actual scientific artifact of Phase 0 — the checkpoints matter far less. Push these to HF first.

FLOPs per configuration must be measured once per architecture with `fvcore` or `ptflops` and stored in a `budgets.json` alongside.

---

## 5. Analysis

Run `msc_core.py` (provided). It computes:

1. `compute_msc()` — stable-sufficiency MSC per axis, per $\tau \in \{0, 0.1, 0.2, 0.3, 0.5\}$
2. `seed_ceiling()` — $\rho_S$ between seed 1 and seed 2 of the same architecture
3. `disattenuated_transfer()` — $T(A,B)$ with bootstrap CI
4. `top_decile_jaccard()` — hard-tail agreement
5. `irreducibility()` — partial Spearman + nested $\Delta R^2$ with bootstrap CI
6. `axis_structure()` — PCA over the per-axis MSC matrix (depth vs resolution only in Phase 0; PC1 variance is a preview of Q2)

**Report every statistic as a curve over $\tau$.** A conclusion that survives only one $\tau$ is not a conclusion.

---

## 6. Decision table

| $\rho_S^{\text{seed}}$ | $T$ within-family | $\Delta R^2$ | Decision |
|---|---|---|---|
| ≥ 0.6 | ≥ 0.7 | ≥ 0.05 | **Full program.** Proceed to Phase 1 atlas, build method. Best case. |
| ≥ 0.6 | ≥ 0.7 | < 0.02 | **Reframe.** MSC ≡ difficulty. Paper becomes "cheap difficulty scores are sufficient for compute routing" — a useful engineering result. Skip the multi-axis oracle; keep the routing method with a difficulty-score gate. |
| ≥ 0.6 | < 0.5 | any | **Pivot to the strong negative.** "Per-sample compute requirements are architecture-specific." Expand the atlas across families instead of building the method. This is a *better* paper than the method paper. |
| 0.4–0.6 | any | any | **Marginal.** Coarsen to $K=3$ well-separated budgets and re-run analysis on existing checkpoints (no retraining needed). Re-evaluate. |
| < 0.4 | any | any | **Fail.** MSC is noise-dominated. One retry with $K=3$; if still failing, switch to the fallback direction (§9 of the protocol). |

Note that three of five rows lead to a paper. That is the whole design intent.

---

## 7. Timeline

| Day | Task |
|---|---|
| 1 | Read SAFE-KD (2602.03043) as a team; write the one-page differentiation memo |
| 1–2 | Implement training script + exit heads; verify accuracy against published numbers |
| 2–3 | Launch 4 runs across 2 accounts |
| 4 | Train exit heads; run oracle sweeps; emit per-sample Parquet tables |
| 5 | Run `msc_core.py`; produce the $\tau$-curves |
| 5 | **Team decision meeting against §6.** Write the decision and its justification into the repo. |

One week. Then you know whether you have a project.
