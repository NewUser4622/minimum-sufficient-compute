# Research Protocol v1.0

## Is Compute Difficulty Architecture-Agnostic?
### Measuring and Distilling Per-Sample Minimum Sufficient Computation

**Project code:** MSC
**Status:** protocol frozen for Phase 0 · everything downstream conditional on Phase 0 outcome
**Owner:** Shanmuk4622 + team (6 Kaggle accounts, dual T4)
**Primary artifact repo:** `huggingface.co/Shanmuk4622/msc-kd`

---

## 0. One-paragraph summary

Adaptive-inference networks decide *at runtime* how much computation to spend on each input. Today that decision is almost always made by the deployed model's own confidence — which is exactly the signal a small, poorly-calibrated model is worst at producing. This project asks a prior question that the field has assumed rather than measured: **is the amount of computation an input requires a property of the input, or a property of the model?** We define a per-sample scalar, **Minimum Sufficient Compute (MSC)**, measure it across a matrix of architectures, datasets, and four independent compute-reduction axes, and quantify how well it transfers between models after correcting for seed-level measurement noise. If MSC transfers, a large teacher can supervise a small student's compute-allocation policy, and we build and benchmark that method (MSC-KD). If it does not transfer, that is a directly publishable negative result that invalidates an assumption underlying a growing line of teacher-guided adaptive-inference work.

---

## 1. Why this, and not the previous plan

The earlier CEB-KD/ESB-KD framing had one fatal structural property: **its entire scientific value was contingent on a single method beating baselines.** If MSC-KD had lost to a confidence threshold, there would have been no paper.

This protocol restructures the same underlying intuition so that **three of the four research questions produce publishable findings regardless of which way they resolve.** The method becomes the final section, not the thesis.

Three additional changes, each in response to a specific rejection risk identified in the literature review:

| Rejection risk | Fix in this protocol |
|---|---|
| "This is SAFE-KD (2602.03043) / ERDE (2510.04856) / EENet (2301.07099) with extra steps" | Those are all single-axis (depth), single-model, no transfer study. Our object of study is the **cross-architecture transfer function**, which none of them measure. The method is downstream of a measurement contribution they do not have. |
| "MSC is just sample difficulty renamed" | Q4 is a dedicated irreducibility test with partial correlation and nested-model ΔR² against a seven-score difficulty battery. We treat this as the primary threat, not a footnote. |
| "Seven loss terms and six λ's — you tried everything" | Final objective has **three terms and two weights**. Monotonicity is enforced architecturally, not by a penalty. Feature/attention/Pareto losses are deleted. |
| "Energy claims are unvalidated / CodeCarbon is not a contribution" | Energy is reported as *measurement methodology* with NVML direct sampling, never as novelty. FLOPs is the primary efficiency metric. Batching limitation is stated in the main text, not buried. |

---

## 2. Formal framework

### 2.1 Compute configuration space

Let $f$ be a trained network. A **compute configuration** $c$ is a deterministic modification of $f$'s inference procedure that changes its cost without changing its parameters' semantics. We use four axes:

| Axis | Symbol | Realisation | Requires retraining? |
|---|---|---|---|
| Depth | $A_{\text{d}}$ | exit at intermediate block $k$ via attached head | heads only (backbone frozen) |
| Width | $A_{\text{w}}$ | channel multiplier $w \in \{0.25, 0.5, 0.75, 1.0\}$ | yes (slimmable) — subset of models only |
| Resolution | $A_{\text{r}}$ | input downsampled to $r \in \{16, 20, 24, 28, 32\}$ px | no |
| Precision | $A_{\text{q}}$ | post-training quantisation to INT4/INT6/INT8/FP16/FP32 | no |

Every configuration has a **normalised cost**

$$\rho(c) = \frac{\text{FLOPs}(f, c)}{\text{FLOPs}(f, c_{\text{full}})} \in (0, 1]$$

This normalisation is the load-bearing methodological choice of the project: it puts every architecture and every axis on a **common dimensionless scale**, which is what makes "did MSC transfer from ResNet to ViT?" a well-posed question at all. Absolute FLOPs would not be comparable.

Within an axis, configurations are indexed $c_{a,1}, \dots, c_{a,K}$ ordered by increasing $\rho$, with $c_{a,K} = c_{\text{full}}$ and $\rho(c_{a,K}) = 1$.

### 2.2 Agreement, margin, and stable sufficiency

For input $x$:

- $\hat y_c(x) = \arg\max_j f_c(x)_j$ — decision under configuration $c$
- $\hat y^*(x) = \hat y_{c_{\text{full}}}(x)$ — full-compute reference decision
- $\text{agree}_c(x) = \mathbb{1}[\hat y_c(x) = \hat y^*(x)]$
- $m_c(x) = p_c^{(1)}(x) - p_c^{(2)}(x)$ — top-1 minus top-2 softmax probability

**Definition (stable sufficiency).** Configuration $c_{a,k}$ is *stably sufficient* for $x$ at margin threshold $\tau$ iff

$$\forall j \ge k: \quad \text{agree}_{c_{a,j}}(x) = 1 \;\wedge\; m_{c_{a,j}}(x) \ge \tau$$

The universal quantifier over all *larger* budgets is deliberate and is a departure from prior formulations. Predictions under compute reduction are not monotone — a model can be right at 40% compute, wrong at 60%, right at 100%. Taking the naive minimum over agreeing budgets records the 40% point, which is an accident, not a property of the sample. The stability closure records the point past which the decision has *settled*. It also guarantees that the sufficiency indicator sequence is monotone by construction, which we exploit in §4.2.

**Definition (Minimum Sufficient Compute).**

$$\text{MSC}_a^{f}(x;\tau) = \rho\big(c_{a,k^*}\big), \qquad k^* = \min\{k : c_{a,k} \text{ stably sufficient}\}$$

with $\text{MSC} = 1$ by convention when no $k < K$ qualifies.

### 2.3 The irreducible-uncertainty subpopulation

If $m_{c_{\text{full}}}(x) < \tau$, the full model is itself unconfident and the definition degenerates ($\text{MSC}=1$ trivially). These samples form a distinct population $\mathcal{U}_\tau$. **Do not silently absorb them into the MSC=1 bin.** Report $|\mathcal{U}_\tau|/N$ for every model, exclude $\mathcal{U}_\tau$ from correlation analyses, and report a separate transfer analysis *of membership in $\mathcal{U}_\tau$ itself* — "do different architectures agree on which samples are irreducibly ambiguous?" is a genuinely interesting sub-question and comes free.

### 2.4 Hyperparameter honesty

$\tau$ is a free parameter. **We never select a single value.** Every headline result is reported as a curve over $\tau \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$. If a conclusion holds only at one $\tau$, it is not a conclusion.

---

## 3. Research questions and hypotheses

Each question is stated with a **pre-registered prediction** and an explicit statement of what the opposite outcome would mean. Both directions are written down *before* running anything. This is the discipline that makes the negative results publishable rather than embarrassing.

### Q1 — Is MSC a stable, well-posed quantity?

*Measure:* Spearman $\rho_S$ between $\text{MSC}^{f}$ and $\text{MSC}^{f'}$ where $f, f'$ are the **same architecture, same data, different random seed.**

**H1:** $\rho_S^{\text{seed}} \ge 0.6$ on CIFAR-100 depth axis.

This quantity is the **noise ceiling** for everything else in the project. It is not a side experiment; it is the denominator of the main result. Prior difficulty-transfer work (C-score, prediction depth) reports raw cross-architecture correlations without establishing this ceiling, which makes their numbers uninterpretable — a cross-architecture $\rho$ of 0.6 means something completely different if seed-to-seed is 0.95 than if it is 0.62.

*If H1 fails* ($\rho_S^{\text{seed}} < 0.4$): MSC is dominated by training noise. Pivot immediately — either coarsen the budget grid (fewer, more separated configurations) or abandon and go to the fallback direction in §9.

### Q2 — Is compute need one-dimensional across axes?

*Measure:* For each model, compute the four per-sample vectors $(\text{MSC}_{\text{d}}, \text{MSC}_{\text{w}}, \text{MSC}_{\text{r}}, \text{MSC}_{\text{q}})$. Run PCA and factor analysis on the standardised $N \times 4$ matrix. Report variance explained by PC1 and the full correlation matrix.

**H2:** PC1 explains $\ge 60\%$ of variance — there is a dominant scalar "compute need" factor.

This question has **never been asked**, in this literature or the sample-difficulty literature. Every adaptive-inference paper picks one axis and treats it as *the* compute axis. If H2 holds, that implicit assumption is validated and a single scalar router is justified. If H2 fails, the field has a problem: routing policies are axis-specific and results on depth-based early exit do not license claims about width- or precision-adaptive inference. **Either outcome is a contribution.** This is the highest novelty-per-GPU-hour question in the project — the data comes almost free once the atlas exists.

### Q3 — Does MSC transfer across architectures?

*Measure:* the **disattenuated transfer coefficient**

$$T(A, B) = \frac{\rho_S(\text{MSC}^{A}, \text{MSC}^{B})}{\sqrt{\rho_S^{\text{seed}}(A) \cdot \rho_S^{\text{seed}}(B)}}$$

This is Spearman's classical correction for attenuation, and applying it here is what turns a vague "0.65 seems highish?" into a defensible claim. $T \approx 1$ means transfer is as good as the measurement permits.

Also report **top-decile Jaccard overlap** $J_{10}$ — agreement on *which samples are hardest*. For a routing application this matters more than global rank correlation, because the router's job is identifying the expensive tail.

**H3:** $T$ is ordered: within-family (ResNet→ResNet) > across-CNN-family (ResNet→VGG) > CNN→ViT / CNN→MLP-Mixer, with within-family $T > 0.8$ and CNN→ViT $T < 0.6$.

This ordering is the prediction implied by Kwok et al. (arXiv 2401.01867), who found a dominant shared difficulty component plus a detectable inductive-bias-specific minority. We are testing whether the same structure holds for *compute* rather than *difficulty*.

*If transfer is uniformly high:* the applied method (Q5) becomes easy and the paper leans applied.
*If transfer is uniformly low:* **this is the strongest possible result.** It says teacher-guided adaptive inference rests on a false premise, and it explains why. Write that paper.

### Q4 — Is MSC irreducible to existing difficulty measures?

**This is the question that decides whether the project has a new object or a rebranded one. Treat it as the primary threat to the work.**

Difficulty battery $\mathcal{D}$, all computed at full compute:
1. max softmax probability
2. top-1/top-2 margin
3. predictive entropy
4. cross-entropy loss under ground truth
5. EL2N score (Paul et al., NeurIPS 2021) — use the during-training variant, **not** GraNd-at-init, which failed reproduction (arXiv 2303.14753)
6. forgetting events (Toneva et al., ICLR 2019)
7. prediction depth (Baldock, Maennel & Neyshabur, NeurIPS 2021)

Two tests:

**(a) Partial correlation.** Compute $\rho_S(\text{MSC}^A, \text{MSC}^B \mid \mathcal{D}^A)$. If MSC is a monotone reparameterisation of difficulty, this collapses to ~0.

**(b) Nested predictive model.** Fit gradient-boosted regressors predicting $\text{MSC}^B$ from (i) $\mathcal{D}^A$ alone, (ii) $\mathcal{D}^A \cup \{\text{MSC}^A\}$. Report $\Delta R^2$ with bootstrap CI over 1000 resamples.

**H4:** $\Delta R^2 \ge 0.05$ and partial $\rho_S \ge 0.3$ — MSC carries information beyond the battery.

*If H4 fails:* MSC ≡ difficulty. **This is still publishable and you must not hide it** — "per-sample compute requirements are fully explained by classical difficulty scores" is a clean, useful, citable finding that saves the community effort. But the method section becomes "use a cheap difficulty score instead of a multi-axis oracle," which is a *better* engineering result anyway.

### Q5 — Does distilled MSC beat self-confidence routing?

*Measure:* accuracy at matched average FLOPs; area under the accuracy-vs-FLOPs curve; and the risk-controlled operating point.

**H5:** MSC-KD > confidence-threshold routing by $\ge 1.0$ point top-1 at matched average FLOPs on CIFAR-100, with the gap widening as student capacity decreases.

The mechanism claim is specific and testable: **small students are miscalibrated, so their own confidence is a poor gate; a large teacher's compute assessment is a cleaner signal.** Therefore the gap should be *larger* for smaller students. Test this explicitly by sweeping student capacity — if the gap does not widen, the stated mechanism is wrong even if the method wins, and you must say so.

---

## 4. The method (MSC-KD)

Only built if Q3 clears its gate (§5.2). Deliberately minimal.

### 4.1 Architecture

Student backbone + $K$ exit heads (linear on pooled features) + one **sufficiency head** $g_\phi$ operating on an intermediate feature map (choose the earliest exit's features, so the routing decision is available cheaply and early).

### 4.2 Monotone ordinal sufficiency head

$g_\phi$ outputs a scalar $u(x)$ and $K{-}1$ learned thresholds parameterised as

$$\theta_1 \in \mathbb{R}, \qquad \theta_{k+1} = \theta_k + \text{softplus}(\delta_k)$$

so that $\theta_1 < \theta_2 < \cdots$ by construction. The predicted sufficiency curve is

$$\hat s_k(x) = \sigma\big(\theta_k - u(x)\big)$$

which is **non-decreasing in $k$ automatically**. This is a cumulative-link ordinal regression head. It replaces the auxiliary monotonicity penalty $\mathcal{L}_{\text{mono}}$ from the earlier plan: an architectural constraint is strictly better than a soft penalty because it cannot be violated, adds no hyperparameter, and cannot trade off against other loss terms.

### 4.3 Objective

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \alpha\,\mathcal{L}_{\text{KD}} + \beta\,\mathcal{L}_{\text{MSC}}$$

$$\mathcal{L}_{\text{MSC}} = \frac{1}{K}\sum_{k=1}^{K} \text{BCE}\big(\hat s_k(x),\; s_k^T(x)\big), \qquad s_k^T(x) = \mathbb{1}\big[\rho_k \ge \text{MSC}^T(x)\big]$$

Three terms. Two weights. $\alpha$ and $\beta$ tuned on a validation split with an explicitly reported search budget, identical for every baseline (see §6.3).

### 4.4 Risk-controlled deployment

Route to $\hat k(x) = \min\{k : \hat s_k(x) \ge \gamma\}$.

Calibrate $\gamma$ on a held-out split using **Learn-then-Test** (Angelopoulos et al.) so that

$$\mathbb{P}\big(\text{accuracy drop vs.\ full compute} > \epsilon\big) \le \delta$$

with distribution-free finite-sample validity. Cite Jazbec et al. (NeurIPS 2024) as the precedent for risk control in early exit. **This is a tool we adopt, not a contribution we claim** — SAFE-KD already combines conformal risk control with early-exit distillation. Our differentiation is the supervision signal, not the calibration machinery.

---

## 5. Experimental design

### 5.1 Phase 0 — decisive pilot (specified in `01_PHASE0_GO_NOGO.md`)

~4 runs, ~12 GPU-hours, one week. Answers Q1 and a first cut at Q3 on a single dataset. **Nothing else is built until Phase 0 returns numbers.**

### 5.2 Gate criteria

| Phase 0 outcome | Action |
|---|---|
| $\rho_S^{\text{seed}} \ge 0.6$ **and** within-family $T \ge 0.7$ | Proceed to full atlas + method. Primary path. |
| $\rho_S^{\text{seed}} \ge 0.6$, within-family $T < 0.5$ | **Better outcome than it looks.** Drop the method; build the paper around "compute requirements are architecture-specific." Expand the atlas instead. |
| $\rho_S^{\text{seed}} < 0.4$ | MSC is noise-dominated. Coarsen budget grid and retry once; if still failing, switch to fallback (§9). |
| $\Delta R^2 < 0.02$ in the Q4 pilot | MSC ≡ difficulty. Reframe as "cheap difficulty scores suffice for compute routing" and skip the multi-axis oracle. |

### 5.3 Model zoo

Chosen for **inductive-bias diversity**, which is the independent variable in Q3 — not for accuracy.

| Family | Models | Role |
|---|---|---|
| ResNet (CIFAR) | resnet32x4, resnet56, resnet110, resnet8x4, resnet20 | within-family transfer; matches CRD/DKD benchmark pairs |
| WideResNet | wrn-40-2, wrn-16-2, wrn-40-1 | within-family, different width regime |
| VGG | vgg13, vgg8 | across-CNN-family (no residuals) |
| Mobile | MobileNetV2, ShuffleNetV2 | across-family, depthwise-separable |
| Modern CNN | ConvNeXt-Femto (CIFAR-adapted) | modern CNN inductive bias |
| Transformer | ViT-Tiny / DeiT-Tiny (CIFAR-adapted, DeiT recipe) | **critical** — different inductive bias entirely |
| MLP | MLP-Mixer-Nano | **critical** — weakest spatial prior; the extreme point of H3 |

The ViT and Mixer entries are the ones that make Q3 interesting. Do not drop them for convenience — without them the transfer study only covers CNNs and H3 becomes untestable.

Using the standard CRD/DKD pairs for the CNN core means **published baseline numbers can be cited rather than re-run** (see benchmark table in `02_ENGINEERING_SPEC.md`), and reviewers can place your numbers instantly.

### 5.4 Datasets

| Dataset | Role | Notes |
|---|---|---|
| CIFAR-100 | primary | full atlas, all axes, 3 seeds |
| CIFAR-10 | secondary | tests whether transfer depends on class granularity |
| Tiny ImageNet | scale check | 200 classes @ 64px; full method comparison |
| ImageNet-100 | generalisation | fixed 100-class subset; **document the exact subset**, no canonical split exists |

### 5.5 Run matrix

| Phase | Content | Runs | Est. T4-h |
|---|---|---|---|
| 0 | Pilot: 2 arch × 2 seeds, CIFAR-100, depth axis | 4 | 12 |
| 1 | Atlas: 10 arch × 3 seeds × CIFAR-100 | 30 | 110 |
| 1b | Exit heads + oracle sweeps (inference-heavy) | 30 | 25 |
| 1c | Slimmable variants for width axis (3 arch × 3 seeds) | 9 | 45 |
| 2 | Atlas: CIFAR-10 + Tiny ImageNet, 6 arch × 3 seeds | 36 | 200 |
| 3 | Method: 4 pairs × 9 configs × 3 seeds, CIFAR-100 | 108 | 320 |
| 4 | Method on Tiny ImageNet, 2 pairs × 9 configs × 2 seeds | 36 | 180 |
| 5 | ImageNet-100 generalisation, 2 pairs × 5 configs × 2 seeds | 20 | 200 |
| 6 | Ablations + sensitivity sweeps, 1 seed | 30 | 90 |

**Total ≈ 1,180 T4-hours.** At 6 accounts × ~30 GPU-h/week ≈ 180 h/week, that is **~7 weeks of wall-clock** with perfect utilisation, realistically 9–11 weeks. Comfortably within reach.

---

## 6. Baselines and controls

### 6.1 Method baselines (Q5)

Every one at **matched average FLOPs**, which is the only comparison that means anything.

| ID | Baseline | Why it's here |
|---|---|---|
| B1 | Static student, no adaptivity | floor |
| B2 | Multi-exit + **confidence threshold** | **the true rival** — the thing everyone actually deploys |
| B3 | Multi-exit + gate trained on student's own correctness | isolates "teacher signal" vs "any learned gate" |
| B4 | MSDNet | classic dynamic-depth reference |
| B5 | Zero Time Waste (ZTW) | strong early-exit baseline |
| B6 | L2W-DEN (Han et al., ECCV 2022) | meta-learned sample weighting for exits |
| B7 | EENet (2301.07099) | closest formalisation of per-sample minimum exit |
| B8 | ERDE (2510.04856) | closest KD × early-exit method |
| B9 | **SAFE-KD (2602.03043)** | **closest overall prior art — mandatory** |
| B10 | MSC-KD (ours) | |
| B11 | Oracle: route by student's own true post-hoc MSC | **ceiling** |

B2 vs B10 vs B11 is the paper's central figure. B2 is where the field is, B11 is the ceiling, and the fraction of the B2→B11 gap that B10 closes *is the result*.

### 6.2 Mandatory ablations

| Ablation | Tests |
|---|---|
| **Shuffled MSC targets** (permute within batch) | Is the signal real, or is $\mathcal{L}_{\text{MSC}}$ just a regulariser? **Run this early.** If shuffled ≈ real, the method is a regulariser and you need to know before writing anything. |
| MSC from teacher confidence alone (no multi-axis oracle) | Does the expensive oracle earn its cost over a free proxy? |
| Single-axis vs multi-axis oracle | Justifies the multi-axis claim, which is the differentiator from SAFE-KD/ERDE/EENet |
| Naive-min MSC vs stable-sufficiency MSC | Validates the §2.2 definitional choice |
| $\tau$ sweep | Hyperparameter honesty |
| $K$ sweep (2, 4, 6, 8 budgets) | Granularity sensitivity |

### 6.3 Fair-comparison protocol

The KD literature has a documented reproducibility problem — torchdistill (Matsubara, arXiv 2011.12913) found most re-implemented ImageNet KD methods do not beat vanilla Hinton KD under matched settings, and the semantic-segmentation study (arXiv 2309.03659) showed distillation gains vanish under sufficient hyperparameter tuning. Compliance is non-negotiable:

1. **Identical hyperparameter search budget** for every method — fixed number of trials, same search space shape, logged.
2. **≥3 seeds**, report mean ± std. Single-run numbers are not reportable.
3. **Paired statistical tests** (Wilcoxon signed-rank across seeds/pairs), Holm-Bonferroni correction, effect sizes, win/tie/loss tables.
4. **Equal-total-compute comparison** in addition to epoch-matched — KD requires teacher forward passes; report both accountings.
5. All configs, seeds, and logs pushed to HF. Every number in the paper traceable to a run ID.

---

## 7. Measurement methodology

### 7.1 Efficiency metrics — primary and secondary

**Primary: theoretical FLOPs.** Deterministic, hardware-independent, reproducible.

**Secondary: measured batch-1 latency and energy** via NVML/pynvml direct power sampling at ≥10 Hz, five repeated measurements, median reported, warm-up discarded.

### 7.2 The batching caveat — state it in the main text

Per-sample dynamic routing yields **no wall-clock speedup under batched inference** unless the batch is split by route. A reviewer will raise this. Pre-empt it: report the batched throughput honestly, discuss batch-splitting overhead, and frame the deployment claim for the batch-1 / edge / streaming regime where it actually holds.

### 7.3 Energy is reported, not claimed

The lifecycle question is largely answered — end-to-end energy accounting of distillation pipelines (arXiv 2605.13981), the amortisation-threshold framing (arXiv 2311.10267), and Rafat et al. (PLOS ONE 2023) reporting KD consuming 13.5–17.9× the teacher's carbon due to temperature tuning. **Report carbon, cite these, claim nothing.** Note that FLOP-based proxies can underestimate real energy by 2–6× due to memory and kernel-launch overheads — which is precisely why NVML sampling is used and why FLOPs is labelled *theoretical*.

---

## 8. Paper structure and novelty statements

### 8.1 Contribution list (in claim order)

1. **Minimum Sufficient Compute**, a per-sample, cost-normalised, multi-axis, stability-closed operationalisation of how much computation an input requires — comparable across architectures for the first time.
2. **The first measurement of whether compute requirements are one-dimensional across reduction axes** (Q2). Untouched in the literature.
3. **The first noise-ceiling-corrected cross-architecture transfer study** of compute requirements (Q3), with disattenuated transfer coefficients — a methodological correction the sample-difficulty literature omits.
4. **An irreducibility analysis** separating MSC from seven classical difficulty scores (Q4).
5. **MSC-KD**, a three-term, architecturally-monotone, risk-controlled method distilling compute requirements teacher→student, benchmarked against SAFE-KD, ERDE, EENet, ZTW, L2W-DEN at matched FLOPs (Q5).
6. **A fully reproducible artifact**: every checkpoint, config, per-sample MSC table, and log on Hugging Face.

Contributions 2–4 hold irrespective of whether 5 succeeds. That is the point of the restructure.

### 8.2 Defensible novelty paragraph (draft for the intro)

> Adaptive-inference networks allocate computation per input, almost universally gating on the deployed model's own confidence. This paper asks whether the required computation is a property of the input rather than the model. We introduce Minimum Sufficient Compute (MSC), the smallest cost-normalised configuration at which a network's decision has stably settled to its full-compute decision, defined uniformly over depth, width, resolution, and precision reduction. Using MSC we report three measurements absent from the literature: whether compute requirements along different reduction axes share a common scalar factor; how MSC transfers across architectures once corrected for seed-level measurement noise; and whether MSC is reducible to classical example-difficulty scores. Building on these, we introduce MSC-KD, which distils a teacher's per-sample compute requirement into a student's monotone routing policy with distribution-free risk control. Unlike risk-controlled early-exit distillation (SAFE-KD), entropy-regularised early-exit distillation (ERDE), and learned minimum-exit assignment (EENet) — all of which operate on a single model along the depth axis — our supervision signal is an explicit, multi-axis, cross-architecture compute target.

### 8.3 Related-work positioning table (goes in the paper)

| Prior work | Multi-axis budget? | Explicit per-sample budget target? | Cross-architecture transfer? | Risk control? |
|---|---|---|---|---|
| MSDNet / BranchyNet / SDN | ✗ (depth) | ✗ | ✗ | ✗ |
| Slimmable / US-Net / DS-Net | ✗ (width) | ✗ | ✗ | ✗ |
| Once-for-All | ✓ (NAS-time) | ✗ (not per-sample) | ✗ | ✗ |
| EENet (2301.07099) | ✗ (depth) | ✓ (within one model) | ✗ | ✗ |
| ZTW / L2W-DEN | ✗ (depth) | ✗ | ✗ | ✗ |
| Jazbec et al. (NeurIPS'24) | ✗ (depth) | ✗ | ✗ | ✓ |
| ERDE (2510.04856) | ✗ (depth) | ✗ | ✗ | ✗ |
| **SAFE-KD (2602.03043)** | ✗ (depth) | ✗ (confidence threshold) | ✗ | ✓ |
| **MSC (ours)** | **✓** | **✓** | **✓** | ✓ (adopted) |

### 8.4 Paper skeleton

```
1  Introduction                          — the assumption nobody measured
2  Related Work
   2.1 Knowledge distillation
   2.2 Adaptive / dynamic inference       ← SAFE-KD, ERDE, EENet positioned here
   2.3 Example difficulty & transferability
   2.4 Risk control for early exit
3  Minimum Sufficient Compute
   3.1 Compute configuration space & cost normalisation
   3.2 Stable sufficiency
   3.3 Irreducible-uncertainty subpopulation
4  Measurement Study
   4.1 Protocol, noise ceiling, disattenuation
   4.2 [Q2] Axis structure
   4.3 [Q3] Cross-architecture transfer
   4.4 [Q4] Irreducibility to difficulty scores
5  MSC-KD
   5.1 Ordinal sufficiency head
   5.2 Objective
   5.3 Risk-controlled routing
6  Experiments
   6.1 Setup & fair-comparison protocol
   6.2 Main results at matched FLOPs
   6.3 Ablations
   6.4 Efficiency: FLOPs, batch-1 latency, energy
7  Limitations                            — batching, T4-only, subset scales
8  Conclusion
```

Section 4 is the paper's spine. Section 5–6 are the application. A reader who rejects the method still leaves with Section 4.

---

## 9. Fallback direction

If Phase 0 kills MSC entirely (§5.2 row 3, twice), switch to:

**"What does distillation lose?"** — an equal-compute, multi-seed audit of how logit vs feature vs relation KD affect calibration (ECE), OOD robustness (CIFAR-100-C), and spurious-correlation reliance, reconciling the contradictory 2024–2026 findings (calibration transfer, ACCV 2024, vs "Do Students Debias Like Teachers?", arXiv 2510.26038, which finds KD *amplifies* spurious-feature reliance without diminishing as teachers scale). Same infrastructure, same model zoo, same rigor protocol — **the atlas built in Phase 1 is directly reusable.** Novelty ~7/10, execution-heavy but low scientific risk, and a natural TMLR paper.

Build the infrastructure so this pivot costs days, not weeks.

---

## 10. Venue plan

| Stage | Target | Rationale |
|---|---|---|
| Month 3 | Workshop paper (BMVC / WACV / NeurIPS or ICML efficiency workshop) | **Stake priority fast.** SAFE-KD appeared Feb 2026; this space is moving and a competing group is active. |
| Month 6–8 | **TMLR** (primary) | Rolling submission, values correctness over novelty-for-novelty, ideal home for measurement/transferability work and for negative results. |
| Alternative | BMVC / WACV → IEEE TNNLS or Pattern Recognition | If Q5 produces a clear method win over SAFE-KD at matched FLOPs. |
| Not recommended | IEEE Access | 27% acceptance, $2,160 APC, binary accept/reject, and its review model rewards technical correctness over novelty — the wrong signal for this work. Keep only as a speed fallback. |

---

## 11. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| A competing group publishes MSC-like transfer study | Medium | High | Workshop paper at month 3 |
| MSC ≡ difficulty (Q4 fails) | **Medium-high** | Medium | Pre-planned reframe; result is still publishable |
| Method loses to confidence threshold | Medium | **Low** | By design — §4 is one section, not the thesis |
| ViT/Mixer fail to train adequately on CIFAR-100 from scratch | Medium | Medium | Use DeiT-style recipe + strong augmentation; if still weak, report their lower ceiling explicitly rather than dropping them |
| SAFE-KD reimplementation is not faithful | Medium | High | Contact authors for code; if unavailable, state reimplementation details fully and report as "our reimplementation" |
| Seed noise swamps signal | Low-medium | High | Phase 0 gate catches this in week 1 |
| 6-account coordination causes duplicated/lost runs | Medium | Medium | Run registry + claim protocol (`02_ENGINEERING_SPEC.md`) |

---

## 12. Immediate next actions

1. **Read SAFE-KD (arXiv 2602.03043) in full, as a team.** It is the closest prior art and appeared five months ago. Write a one-page differentiation memo before anything else. If it already does cross-architecture transfer, this protocol changes.
2. Verify the 2026 arXiv IDs cited here — several are recent preprints and some may have been revised or withdrawn.
3. Run Phase 0 per `01_PHASE0_GO_NOGO.md`. Four runs. Report $\rho_S^{\text{seed}}$, $T$, and $\Delta R^2$.
4. **Do not build the full training infrastructure until Phase 0 returns numbers.**
