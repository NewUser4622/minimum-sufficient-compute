# How Much Computation Does an Image Need? A Noise-Ceiling-Corrected Atlas of Per-Sample Compute Requirements Across Fifteen Architectures

**Draft v1 — for submission to TMLR / IEEE TPAMI / IJCV.**
Every number in this manuscript is traceable to
`huggingface.co/datasets/Shanmuk4622/msc-cifar100`.

---

## Abstract

Adaptive-inference networks decide at runtime how much computation to spend on
each input. Almost universally, that decision is gated on the deployed model's
own confidence — precisely the signal a small, poorly calibrated network is
worst at producing. A growing body of work therefore proposes to have a large
teacher supervise a small student's compute-allocation policy. This rests on an
assumption that has been stated but, to our knowledge, never measured: that the
amount of computation an input requires is a property of the *input* rather than
of the *model*.

We define **Minimum Sufficient Compute (MSC)**: the smallest cost-normalised
configuration at which a network's decision on a given input has *stably
settled* — agreeing at that budget and at every larger one. We measure MSC along
three reduction axes (depth, resolution, numerical precision) for **45 trained
networks spanning 15 architectures and 3 seeds each** on CIFAR-100, and correct
every cross-architecture comparison by a per-architecture **noise ceiling**
estimated from independent seeds.

Four results follow. **(i)** Compute requirement transfers strongly across
architectures: 92% of the measurement ceiling within an architecture family, 88%
across convolutional families, and **71% across the convolution/attention
boundary**, with no overlap between the three distributions. We had
pre-registered that transfer would collapse below 60% at that boundary; it did
not. **(ii)** Compute requirement is *not* one-dimensional across reduction
axes: in none of 15 architectures does a single principal component explain more
than 53% of the variance, so results obtained on depth-based early exit do not
license claims about resolution or precision. **(iii)** Measurement reliability
is itself architecture-dependent — all thirteen convolutional networks yield
seed-to-seed reliabilities in [0.62, 0.73] while both non-convolutional models
sit at 0.55 — which makes noise-ceiling correction a demonstrated necessity
rather than a methodological preference. **(iv)** MSC is not reducible to a
seven-score battery of classical example-difficulty measures for convolutional
pairs (median ΔR² = 0.155, 78/78 pairs clearing a pre-registered threshold), but
*is* only marginally irreducible for pairs involving a transformer (median
ΔR² = 0.043, 11/27 clearing) — a limit we report rather than omit.

We release the full atlas: 49 trained networks, per-sample measurements on three
compute axes at five confidence thresholds, and 171 per-epoch telemetry columns
per run.

---

## 1. Introduction

Not every input to a neural network is equally hard. This observation motivates
adaptive inference: early-exit networks, cascades, token pruning, dynamic
resolution and mixed-precision routing all spend less computation on inputs the
model finds easy. The engineering question — *how do we decide, at runtime, when
to stop?* — has received substantial attention. The scientific question
underneath it has received much less.

That question is: **is "how much computation this input needs" a property of the
input, or of the model?**

The distinction matters, and it is not academic. If compute requirement is
largely a property of the input, then a large, well-calibrated teacher can
estimate it and supervise a small student's routing policy — the student need
not rely on its own confidence, which is exactly what it is worst at. Several
recent lines of work take this route. If compute requirement is instead largely
model-specific, that entire strategy is built on sand, and the field should know.

Answering the question requires a measurement, and the measurement is harder
than it first appears for three reasons.

**First, "how much compute" is not one number.** A network can be made cheaper
by exiting early, by lowering input resolution, or by reducing numerical
precision. Almost all published work picks one axis — usually depth — and
treats it as *the* compute axis. Whether these axes agree on which inputs are
expensive is an empirical question that, to our knowledge, has not been asked.

**Second, the measurement is noisy, and the noise is not uniform.** Two networks
of the same architecture trained with different random seeds do not assign
identical compute requirements to the same images. Any cross-architecture
correlation is therefore attenuated by measurement error. Reporting raw
correlations without correcting for this — as the example-difficulty literature
generally does — makes results incomparable across studies and, as we show,
across architectures within a single study.

**Third, a new construct must be shown not to be an old one.** Per-sample
compute requirement might simply be a re-parameterisation of example difficulty,
for which many cheap scores already exist. If so, the useful engineering
recommendation is to use a cheap score, and the honest scientific outcome is to
say so.

This paper addresses all three. We define MSC as a *stability* criterion rather
than a first-success criterion, measure it on three axes, correct every
cross-architecture comparison by a per-architecture noise ceiling, and test
irreducibility against a seven-score difficulty battery with nested ΔR².

### Contributions

1. **A definition and a measurement protocol.** MSC as the smallest
   cost-normalised budget at which a decision has stably settled, measured
   across depth, resolution and precision, with an explicit treatment of
   samples on which the full model is itself unconfident.
2. **A noise-ceiling correction for cross-architecture difficulty comparison**,
   and evidence that it is necessary: measurement reliability varies enough
   across architectures to change conclusions.
3. **An atlas**: 15 architectures × 3 seeds, fully measured, released.
4. **Four empirical results**, two of which refute hypotheses we registered in
   advance — one unfavourably to a common assumption, one favourably.

We also report, without hedging, the places where our own construct is weakest.

---

## 2. Related work

**Adaptive inference and early exit.** Multi-exit architectures attach
classifiers to intermediate layers and halt when a confidence criterion is met.
Downstream variants add learned gates, cascades of separate models, or
input-dependent resolution and precision. Our work is not a competitor to these;
it is a measurement of a quantity they all implicitly assume exists.

**Teacher-guided routing.** Several recent approaches distil a teacher's
assessment of input difficulty into a student's exit policy. The premise — that
the teacher's assessment is informative about the *student's* requirement — is
what Section 6.2 measures directly.

**Example difficulty.** A substantial literature scores training examples by
prediction depth, margin, loss, gradient norm (EL2N), forgetting events, and
related quantities, typically to prune data or schedule curricula. These scores
are the natural null hypothesis for MSC, and we test against seven of them
jointly in Section 6.4. Two of the seven are *training-set* quantities, a detail
that materially affected our own results (Section 5.4).

**Reliability correction.** Disattenuation — dividing an observed correlation by
the geometric mean of the two measures' reliabilities — is standard in
psychometrics and, as far as we can tell, essentially absent from the
example-difficulty literature. Section 6.1 shows why its absence matters here.

> *Bibliographic note.* Reference details are omitted from this draft
> deliberately. Every citation must be verified against the primary source
> before submission; several of the closest prior works are recent preprints
> whose venue and version may have changed. We would rather ship a draft with
> an explicit gap than one with plausible-looking but unchecked citations.

---

## 3. Minimum Sufficient Compute

### 3.1 Definition

Let *f* be a trained network and *c* a compute configuration drawn from an
ordered set *C* = {c₁ ≺ … ≺ c_K}, where ≺ orders configurations by cost. Define
the cost ratio

> **ρ(c) = FLOPs(f, c) / FLOPs(f, c_K)** ∈ (0, 1].

For an input *x*, let ŷ(x, c) be the network's prediction under configuration
*c*, and let m(x, c) be its margin — the gap between the top two class
probabilities.

**Definition (MSC).** The minimum sufficient compute of *x* at tolerance τ is

> **MSC_τ(x) = min { ρ(c_k) : ŷ(x, c_j) = ŷ(x, c_K) and m(x, c_j) ≥ τ for all j ≥ k }.**

Two aspects are deliberate.

**Stability, not first success.** MSC is the smallest budget from which the
decision agrees with the full-budget decision *and continues to agree at every
larger budget*. A configuration that happens to be right and is then overturned
by more computation was not sufficient. First-correct-exit criteria, which are
common, admit exactly that failure and consequently produce optimistically low
estimates.

**An explicit tolerance.** A decision that is technically correct at a margin of
10⁻⁴ has not settled in any useful sense. τ makes the confidence requirement
explicit and, more importantly, *tunable* — we report every result as a curve
over τ ∈ {0, 0.1, 0.2, 0.3, 0.5} rather than at a single value. No conclusion in
this paper depends on the choice of τ.

### 3.2 The irreducible subpopulation

For some inputs the *full* model is itself unconfident: m(x, c_K) < τ. For these,
MSC is degenerate — it evaluates to 1.0 by construction, not because the input
is expensive but because the network has no usable opinion about it. We denote
this set U_τ, exclude it from all correlations, and report |U_τ|/N separately.
At τ = 0.1 it is 8–35% of the test set depending on architecture. Including
these samples would inflate every agreement statistic, since two models would
"agree" merely by both being confused.

### 3.3 Three axes

- **Depth.** *K* exits attached to a frozen backbone at increasing prefixes,
  each with a linear head trained post hoc. The number of exits is *adaptive*:
  a backbone with fewer blocks than requested carries fewer distinct budgets.
  (`resnet8x4` yields 3 rather than 5; forcing 5 produces duplicate ρ values
  and a silently degenerate measurement.)
- **Resolution.** Native evaluation where the architecture tolerates it, plus a
  downsample-then-upsample proxy applicable to all architectures. The proxy is
  used as the primary resolution axis so that all 15 architectures are
  comparable; it is labelled idealised throughout, since it does not realise the
  FLOPs saving it models. One architecture (MLP-Mixer) cannot run at any
  non-native resolution at all, its token-mixing layer being tied to a fixed
  token count — an architectural fact, not an implementation limit.
- **Precision.** Simulated fake quantisation at 4, 6, 8, 16 and 32 bits, costed
  analytically as ρ = bits/32. No T4 kernel exists to time INT4/INT6, so these
  are never reported as measured latency.

### 3.4 Cost accounting, and a caveat we take seriously

ρ is defined in FLOPs. Across our zoo, the ratio of wall-clock time to FLOPs
varies by a factor of **17.9**: `shufflenetv2` performs 7.1× fewer FLOPs than
`wrn_40_2` yet takes 14.5% *more* wall-clock time and 89% of the energy, because
depthwise and grouped convolutions are memory-bandwidth-bound rather than
arithmetic-bound.

Because ρ is a ratio taken *within* a single architecture, this does not
threaten MSC. But it constrains the language: an MSC of 0.65 means 35% fewer
FLOPs. It does not mean 35% less time or 35% less energy, and we never state it
that way.

---

## 4. Correcting for measurement noise

### 4.1 The noise ceiling

Train two networks of the same architecture with different seeds. They will not
assign identical MSC to the same images. Define the **noise ceiling** of
architecture *A* as

> **ceiling_A = ρ_S( MSC^{A,seed 1}, MSC^{A,seed 2} )**,

the Spearman correlation between per-sample MSC of two seeds. This is the
maximum agreement any *other* model could plausibly reach with *A*, because it
is the agreement *A* reaches with itself.

### 4.2 Disattenuated transfer

For architectures *A* and *B*, define

> **T(A, B) = ρ_S(A, B) / √( ceiling_A · ceiling_B )**.

T ≈ 1 means transfer is as complete as the measurement permits. T well below 1
means genuine architecture-specific structure rather than noise. This is
Spearman's classical correction for attenuation; the contribution here is
recognising that it is required, and demonstrating what happens without it.

### 4.3 A control that must pass

Every cross-architecture correlation is computed on per-sample tables that must
be row-aligned. We verify alignment by hash and, additionally, by permuting one
side and recomputing: shuffled transfer must be indistinguishable from zero.

Calibrating this control is less trivial than it appears. Under a random
permutation the rank correlation has mean 0 and standard deviation exactly
1/√(n−1) ≈ 0.013 at our sample sizes. A fixed cutoff such as |T| < 0.05 is 2.6
standard deviations at n ≈ 6,000 but 5 at n ≈ 25,000, and — because T divides by
√(ceiling_A · ceiling_B) — it is systematically stricter for *low-ceiling* pairs,
which are exactly the pairs carrying our most interesting result. We therefore
test the *raw* correlation against its exact permutation null and require both
statistical and practical significance (|z| > 5 **and** |ρ| > 0.10). All 78
tested pairs pass, with a maximum |z| of 3.30.

---

## 5. Experimental setup

### 5.1 Architectures

Fifteen architectures spanning six families, 3 seeds each (45 networks), plus a
4-network pilot: ResNet-20/56/110/8×4/32×4; WRN-16-2/40-1/40-2; VGG-8/13;
MobileNetV2; ShuffleNetV2; ConvNeXt-Femto; ViT-Tiny; MLP-Mixer-Nano.

CIFAR-100. Convolutional networks use the standard CRD/DKD recipe (240 epochs,
SGD 0.05, ×0.1 at 150/180/210, batch 64, weight decay 5×10⁻⁴, random crop and
horizontal flip). The three modern architectures use 300 epochs.

Ten of the twelve architectures with an established published reference exceed
it (Table 1). We treat this as an acceptance test for the whole pipeline: MSC
computed from an under-trained network is meaningless, and an under-trained
network is otherwise easy to miss.

### 5.2 Table 1 — the atlas

| architecture | family | top-1 | published | Δ | ρ_seed (τ=0.1) |
|---|---|---|---|---|---|
| ResNet-32×4 | resnet | 79.74 | 79.42 | +0.32 | **0.726** |
| WRN-40-2 | wrn | 76.06 | 75.61 | +0.45 | 0.709 |
| VGG-13 | vgg | 75.70 | 74.64 | +1.06 | 0.669 |
| ResNet-110 | resnet | 74.38 | 74.31 | +0.07 | 0.634 |
| WRN-16-2 | wrn | 73.79 | 73.26 | +0.53 | 0.633 |
| ResNet-56 | resnet | 73.69 | 72.34 | +1.35 | 0.622 |
| ResNet-8×4 | resnet | 73.26 | 72.50 | +0.76 | 0.667 |
| WRN-40-1 | wrn | 72.41 | 71.98 | +0.43 | 0.656 |
| ShuffleNetV2 | mobile | 71.93 | 70.50 | +1.43 | 0.670 |
| VGG-8 | vgg | 71.73 | 70.36 | +1.37 | 0.722 |
| ResNet-20 | resnet | 70.13 | 69.06 | +1.07 | 0.643 |
| MobileNetV2 | mobile | 70.10 | — † | — | 0.688 |
| ConvNeXt-Femto | convnext | 62.67 | — | — | 0.708 |
| MLP-Mixer-Nano | **mixer** | 60.23 | — | — | **0.547** |
| ViT-Tiny | **vit** | 59.33 | — | — | **0.548** |

† The commonly cited MobileNetV2 reference for CIFAR-100 is for a *half-width*
variant (≈0.81 M parameters); our model has 2.35 M. We therefore claim no
comparison rather than a flattering one. The three modern architectures have no
established from-scratch CIFAR-100 reference; their 59–63% top-1 is the expected
range for attention and MLP models trained on 50 k images without heavy
augmentation, and is not evidence of under-training.

### 5.3 Pre-registered hypotheses

- **H2**: a single principal component explains ≥ 60% of the variance in
  per-sample MSC across the three axes.
- **H3**: transfer is ordered within-family > across-CNN-family >
  CNN→transformer, with within-family > 0.8 and CNN→transformer < 0.6.
- **H4**: MSC is irreducible to the difficulty battery — ΔR² ≥ 0.05 and partial
  correlation ≥ 0.30.

### 5.4 A methodological error worth reporting

Our pilot computed the irreducibility test on the *test* split with five of the
seven difficulty scores, because EL2N and forgetting events are training-set
quantities that cannot be attached to test images. Handicapping the battery does
not add noise — it *biases in one direction*, leaving more variance for MSC to
explain. Re-run correctly on a held-out slice of training data with all seven
scores, our headline ΔR² fell from 0.254 to 0.121, and the partial correlation
from 0.489 to a value that no longer clearly clears its threshold.

We report this because the incorrect number was internally consistent,
survived review, and flattered our own construct. Anomalously *favourable*
results receive less scrutiny than unfavourable ones, and that asymmetry is a
methodological hazard in this subfield specifically, where the null hypothesis
is "your new score is an old score."

---

## 6. Results

### 6.1 Measurement reliability is architecture-dependent

At τ = 0.1, all thirteen convolutional networks yield noise ceilings in
**[0.622, 0.726]**. Both non-convolutional models sit at **0.547** — below every
CNN, below our pre-registered 0.60 threshold, with a separation margin of 0.074
and no overlap. (This holds for τ ≤ 0.2; at τ = 0.3, ViT-Tiny — the only
architecture whose reliability *increases* with τ — overtakes ResNet-56.)

**This is not an accuracy artifact.** The obvious objection is that ViT-Tiny
(59.3%) and Mixer-Nano (60.2%) are also our two least accurate models. Three
observations rule that reading out:

1. Within the thirteen CNNs, noise ceiling and top-1 accuracy are
   **uncorrelated** (Spearman +0.035, Pearson −0.007).
2. ConvNeXt-Femto breaks the confound directly. At 62.67% it is the least
   accurate CNN — only 2.4 points above Mixer-Nano — yet its ceiling is
   **0.161 higher**, fourth-highest overall. Moving *17 points* of accuracy from
   ConvNeXt-Femto up to ResNet-32×4 buys only **+0.017** of ceiling. The
   2.4-point step across the architectural boundary is roughly an order of
   magnitude larger than the 17-point step within it.
3. ConvNeXt-Femto and the two non-CNNs share the same 300-epoch schedule, so
   schedule length is not the difference either.

**Honest caveat.** CNN accuracies (62.7–79.7) and non-CNN accuracies (59.3–60.2)
do not overlap, so family and accuracy remain partly confounded at the level of
the atlas. The ConvNeXt comparison is the strongest available evidence and it
rests on one architecture. A decisive test would train a non-convolutional model
to CNN-level accuracy, or a CNN down to ≈60%.

**Consequence.** A cross-architecture difficulty study that does not divide by a
per-architecture reliability is comparing quantities measured with unequal
precision. Our pilot happened to select the first- and third-most reliable
architectures in the zoo, so its ceiling of 0.715 was an optimistic sample
against an atlas mean of 0.676.

### 6.2 Compute requirement transfers across architectures

**105 architecture pairs**, disattenuated, τ = 0.1, depth axis, 1000 bootstrap
resamples.

| pair type | n | **mean T** | sd | range |
|---|---|---|---|---|
| within-family | 12 | **0.920** | 0.041 | 0.877 – 1.005 |
| across-CNN-family | 43 | **0.878** | 0.070 | 0.732 – 0.966 |
| **CNN → transformer** | 22 | **0.710** | 0.034 | 0.657 – 0.777 |
| transformer → transformer | 1 | 0.886 | — | — |

H3's **ordering is confirmed**, with *complete separation*: the weakest
within-family pair (0.877) exceeds the strongest CNN→transformer pair (0.777).

H3's **magnitude prediction is refuted, favourably**. We predicted transfer would
fall below 0.6 across the convolution/attention boundary — that compute
requirement would become architecture-specific there. It held at **0.710**, far
above the 0.5 line we had set for "the field assumption is wrong."

Two observations sharpen this.

**ResNet-110 × ResNet-56 reaches T = 1.005, CI [0.979, 1.029].** The interval
includes 1.0: cross-architecture agreement is statistically indistinguishable
from same-architecture, different-seed agreement. For that pair, *which* of the
two networks you measure makes no detectable difference to per-sample MSC. (T
slightly above 1 is expected occasionally, since disattenuation divides by an
*estimated* ceiling; it should always be read with its interval.)

**ConvNeXt-Femto transfers like a transformer, not like a CNN.** Across-CNN
pairs average **0.766** with it and **0.912** without — a gap of 0.146. Ranked by
mean T over all its pairs it falls below every other CNN, just above ViT-Tiny
and Mixer-Nano. ConvNeXt is a deliberately transformer-ised convolutional
network — large depthwise kernels, LayerNorm, inverted bottlenecks, GELU, few
activations — so compute requirement may track those design choices rather than
the convolution/attention label as such. We state this as a hypothesis: n = 1,
and confounded with the 300-epoch schedule.

Notably this cuts *against* the grouping in Section 6.1: ConvNeXt-Femto has a
CNN-like *ceiling* but transformer-like *transfer*. Reliability and
transferability are separable properties of an architecture, and one model in
our zoo separates them.

### 6.3 Compute requirement is not one-dimensional

PCA over per-sample MSC on {depth, resolution-proxy, precision}, τ = 0.1:

| | value |
|---|---|
| runs reaching PC1 ≥ 0.60 | **0 of 15** |
| highest PC1 in the atlas | **0.532** (ShuffleNetV2) |
| lowest | 0.441 (Mixer-Nano) |
| mean ± sd | 0.500 ± 0.025 |

**H2 is refuted, in every architecture.** The highest value anywhere is 0.068
below the threshold, and the spread across fifteen architectures is only 0.09
wide — compute requirement is *reliably* three-dimensional, not marginally so.

The axes decouple further in non-convolutional models: depth↔precision
correlation is 0.143 for ViT and Mixer against 0.260 for CNNs, and for
Mixer-Nano it falls to **0.096** — precision requirement is very nearly
independent of depth and resolution requirement.

The practical implication is blunt. Results obtained on depth-based early exit
do not license claims about resolution or precision routing, in any architecture
we tried, and the assumption is *worst* in the architectures the field is
currently moving toward.

### 6.4 Irreducibility, and where it fails

Nested ΔR²: predict one model's per-sample MSC from a seven-score difficulty
battery (MSP, margin, entropy, cross-entropy loss, EL2N, forgetting events,
prediction depth), then again with the *other* model's MSC added. Held-out
training slice, 105 pairs.

| | median ΔR² | range | clearing ΔR² ≥ 0.05 |
|---|---|---|---|
| **CNN-only** (78 pairs) | **0.155** | 0.078 – 0.245 | **78 / 78 = 100%** |
| **transformer-involving** (27) | **0.043** | 0.017 – 0.090 | 11 / 27 = **41%** |
| all 105 | 0.121 | 0.017 – 0.245 | 89 / 105 = 85% |

**For convolutional networks MSC is decisively irreducible.** Every one of 78
pairs clears the threshold, at a median three times it. Knowing another CNN's
per-sample compute requirement raises explained variance by 0.155 *over and
above* the full classical battery.

**For pairs involving a transformer it is not.** The median falls below the
threshold and fewer than half the pairs clear it.

We stress that this is **not an independent finding**. Section 6.1 established
that MSC is measured less reliably in ViT and Mixer; a noisier measurement
necessarily explains less variance. The two results share a cause and must be
reported together, or a reader will double-count them. What we can say is that
*as currently measured*, MSC adds little beyond classical difficulty scores for
transformer pairs — whether that reflects the construct or our measurement of it
is not resolved by this design.

---

## 7. Discussion

### What this licenses

The central result is permissive. Per-sample compute requirement transfers at
71% of the measurement ceiling across the largest architectural gap in our zoo,
and at 88–92% within the convolutional world. **A large teacher's assessment of
how much computation an input needs is substantially informative about a small
student's requirement.** The premise beneath teacher-guided adaptive inference
is sound, at least at this scale and on this dataset.

### What this forbids

Two things.

**Do not generalise across axes.** Compute requirement is three-dimensional in
every architecture we measured. A method validated on depth-based early exit has
not been validated for resolution or precision routing, and the assumption is
weakest in transformers.

**Do not compare difficulty scores across architectures without correcting for
reliability.** We can now put a number on the cost of not doing so: our own
pilot, using two architectures that happened to be the most reliably measured in
the zoo, would have reported a ceiling 6% higher than the atlas mean and
correspondingly deflated transfer estimates elsewhere.

### The transformer story is coherent across three measurements

ViT and Mixer are measured less reliably (6.1), transfer less well (6.2), and
carry less unique information beyond classical scores (6.4). These are not three
findings; they are most likely one, seen three ways. We flag the common cause
rather than presenting them as mutually reinforcing evidence.

---

## 8. Limitations

- **One dataset.** CIFAR-100 only. Nothing here is demonstrated at ImageNet
  scale, and the ViT/Mixer results in particular may be specific to the
  small-data regime where those architectures are known to underperform.
- **Family and accuracy are partly confounded** in Section 6.1. The ConvNeXt
  comparison is the strongest available evidence and rests on one architecture.
- **The ConvNeXt transfer result is n = 1** and confounded with schedule length.
- **Section 6.4's transformer result cannot be separated** from Section 6.1's
  reliability result by this design.
- **The precision axis is simulated.** INT4/INT6 are fake-quantised and costed
  analytically; no kernel exists on our hardware to time them. They are never
  reported as measured latency.
- **The resolution axis is primarily a proxy.** Downsample-then-upsample does
  not realise the FLOPs saving it models. It is used because one architecture
  cannot run at any other native resolution, and comparability across all
  fifteen was judged more valuable than realism on fourteen.
- **FLOPs are not time.** The wall-clock-per-FLOP ratio varies 17.9× across our
  zoo. MSC is a within-architecture ratio so this does not threaten it, but
  FLOPs-denominated savings must not be restated as time or energy savings.
- **The method is not evaluated here.** A distillation method that trains a
  student router from teacher MSC is implemented and trained, but its oracle
  ceiling requires per-sample measurement of each *student*, which we have not
  yet performed. We therefore make no claim about it, and this paper is a
  measurement paper.

---

## 9. Conclusion

We asked whether the amount of computation an input requires is a property of
the input or of the model, and measured the answer across fifteen architectures
with an explicit correction for measurement noise.

It is substantially a property of the input. Compute requirement transfers at
0.71 of the measurement ceiling even across the convolution/attention boundary,
refuting our own pre-registered prediction that it would collapse there. It is
also, in every architecture we tried, irreducibly three-dimensional — so
single-axis results do not generalise. And measurement reliability is itself
architecture-dependent to a degree that changes conclusions, which makes the
noise-ceiling correction we advocate a necessity rather than a refinement.

We release the full atlas so that these numbers can be checked, extended, and
argued with.

---

## Reproducibility statement

All 49 trained networks, their per-sample measurements on three compute axes at
five tolerance values, 171 per-epoch telemetry columns per run, and every
analysis output are public at
`huggingface.co/datasets/Shanmuk4622/msc-cifar100`. The measurement pipeline
carries 232 offline self-checks. All 15 architectures share a single data
ordering hash, so per-sample tables are index-aligned by construction and
verified by a permutation control before any correlation is computed. The
complete engineering record — including all 36 defects found during development,
each with an analysis of whether it contaminated a reported number — is released
alongside the code.

Two reported numbers were corrected during development and the originals are
withdrawn: the irreducibility figure (Section 5.4) and the MobileNetV2
comparison (Table 1, footnote).

---

## Appendix A — reviewer questions we expect, and our answers

**"Isn't MSC just prediction depth?"** Prediction depth is one of the seven
scores in the battery of Section 6.4, and MSC adds 0.155 median ΔR² over the
battery for CNN pairs. For transformer pairs the answer is less comfortable and
is reported as such.

**"Why is transfer measured on the depth axis only?"** Because depth is the axis
on which all fifteen architectures support genuine, non-proxied budget
reduction. Section 6.3 shows the axes are not interchangeable, which is
precisely why we do not pool them.

**"τ = 0.1 seems arbitrary."** Every result is reported over τ ∈ {0, 0.1, 0.2,
0.3, 0.5}. No conclusion depends on the choice; where a conclusion is
τ-sensitive — the CNN/non-CNN separation in Section 6.1, which holds for τ ≤ 0.2
— we say so explicitly.

**"Your two non-convolutional models are also your least accurate."** Yes.
Section 6.1 addresses this at length and states the residual confound rather
than dismissing it.

**"Only CIFAR-100?"** Yes, and we say so first in Section 8.
