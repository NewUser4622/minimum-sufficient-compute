# ImageNet-100 — FINDINGS

**What was predicted, what was measured, and what is actually new.**

Companion to `docs/cifar100/10_FINAL_RESULTS.md`. Operational state is in
`24_IN100_STATUS.md`; what went wrong on the way is in `22_IN100_LAB_NOTEBOOK.md`.

---

## 1. The pre-registered scorecard

The five hypotheses were registered in `docs/cifar100/00_RESEARCH_PROTOCOL.md`
before either study ran. Applying CIFAR's thresholds verbatim to ImageNet-100
is itself a choice — it is the strict reading, and it is the one taken here.

| | prediction | CIFAR-100 | ImageNet-100 | |
|---|---|---|---|---|
| **H1** | ρ_seed ≥ 0.60 | CNN 0.62–0.72<br>non-CNN 0.547 | `resnet50` **0.822**<br>`vit_small_p16` **0.649** | IN-100 **supported**; CIFAR partial |
| **H2** | PC1 ≥ 0.60 — one dominant factor | max **0.532**, 0 of 15 | **0.547** / **0.522** | **refuted in both** |
| **H3** | CNN→transformer T **< 0.6** | T = **0.710** | T = **0.640** | **refuted in both, favourably** |
| **H4** | ΔR² ≥ 0.05 **and** partial ρ ≥ 0.30 | median **0.0425** (transformer pairs) | ΔR² **0.0411**, partial ρ **0.282** | **missed in both** |
| **H5** | MSC-KD ≥ +1.0 pt vs confidence | **untestable** — B11 never computed (O-21) | **−0.88 pt**, B11 flat | **falsified on IN-100** |

**One of five is cleanly supported.** That is the honest headline, and it is not
a bad outcome — four of the five failures are *informative*, and two of them
were already known from CIFAR and now replicate at 40× the data.

---

## 2. What is actually novel

### 2.1 The oracle ceiling — CIFAR's own "one substantive gap remaining"

`10_FINAL_RESULTS.md` §6 states it plainly:

> **Incomplete: B11, the oracle ceiling, is unavailable.** … Without it, the
> headline *"fraction of the B2→B11 gap closed"* cannot be computed. … tracked
> as **O-21** — the one substantive gap remaining.

**This study closes it.** B11 was computed for all 18 students, and the answer
is decisive:

| router (matched FLOPs, ρ = 0.806) | vs B2 confidence | runs |
|---|---|---|
| **B11** — route by the student's own *true* post-hoc MSC | **+0.00007** (sd 0.00036) | headroom in **0/18** |
| **B10** — MSC-KD | **−0.0088** (sd 0.0068) | worse in **18/18** |

This is the contribution that changes what the negative result *means*. Without
B11 you can only report "our distillation underperformed" — a statement about
an implementation. With B11 you can report **"MSC-based routing has no headroom
over confidence thresholding at this operating point"** — a statement about the
premise. CIFAR could not make that distinction. ImageNet can.

It also retroactively explains CIFAR's Q5: the method was never going to win,
and the missing baseline was the reason nobody could tell.

### 2.2 Three-dimensionality replicates at scale

H2 predicted a dominant scalar "compute need". CIFAR refuted it (max PC1
0.532 across 15 runs). ImageNet-100 refutes it again at 0.547 and 0.522.

**Single-axis adaptive-inference results do not generalise across axes, at
either scale.** Two datasets, 49× apart in pixels, non-overlapping
architectures, same conclusion. This is the most robust finding in the project.

### 2.3 Cross-architecture transfer exceeds the pre-registered bound, twice

H3 predicted CNN→transformer T **< 0.6** — i.e. that compute-need would largely
*not* transfer across the convolution/attention boundary. Measured:

- CIFAR-100: **T = 0.710**
- ImageNet-100: **T = 0.640**, CI [0.614, 0.664], shuffled control **0.037**

Both above the bound, so the hypothesis is refuted in the *favourable*
direction: compute-need is substantially architecture-transferable. The
17× separation from the shuffled control is what makes the ImageNet number a
measurement rather than an alignment artifact.

### 2.4 Noise-ceiling correction, demonstrated rather than argued

ρ_seed differs by architecture in both studies, in the same direction:

| | CNN | non-CNN / ViT |
|---|---|---|
| CIFAR-100 | 0.668 (12 archs) | 0.547 |
| ImageNet-100 | 0.822 (`resnet50`) | 0.649 (`vit_small_p16`) |

Any transfer number not divided by a per-architecture ceiling is comparing
quantities measured with different amounts of noise. The ImageNet replication
uses **non-overlapping architectures**, which makes it independent evidence
that the effect is not a property of one model family's implementation.

### 2.5 A construct limit, stated twice

H4 misses in both studies — CIFAR's transformer pairs at ΔR² 0.0425, ImageNet
at 0.0411 with partial ρ 0.282. MSC adds *some* information beyond a seven-score
difficulty battery (the CI excludes zero) but **not enough to clear the
pre-registered bar, at either scale**. That is a limit on the construct, and
reporting it twice is worth more than reporting it once.

---

## 3. What is NOT novel, and what is not supported

Read this before writing an abstract.

1. **No architecture appears in both studies.** CIFAR ran `vit_tiny` and twelve
   small CNNs; ImageNet ran `vit_small_p16` and `resnet50`. Every cross-study
   *magnitude* confounds architecture, resolution and dataset. The **ordering**
   replicates; the numbers do not transfer. See `24_IN100_STATUS.md`
   §FIRST RESULT.
2. **n = 1 architecture per family on ImageNet-100**, 2 seeds each. ρ_seed has
   no error bar and no family-level claim is possible.
3. **`vit_small_p16` at 0.649 sits inside the CIFAR CNN range (0.622–0.722).**
   "ViT is unreliable" is not scale-invariant; "ViT is less reliable *than a CNN
   trained alongside it*" is what holds.
4. **The ViT arm overfits**: train 98.7% vs val 60.6%, a 38-point gap against
   16 for ResNet-50. It is the no-strong-augmentation arm behaving as designed,
   but a badly-overfit ViT may not be the right input to a reliability
   comparison. Decide deliberately and say which way.
5. **Equal epochs for all architectures** is a chosen confound. State it.
6. **MSC-KD's negative is at one operating point** (ρ = 0.806). The B11 result
   makes it unlikely that another point rescues the method, but it does not
   prove it.

---

## 4. The strongest honest framing

> Minimum Sufficient Compute is a reliable, three-dimensional, largely
> architecture-transferable per-sample quantity — and it does **not** yield
> better inference routing than confidence thresholding, because the oracle
> ceiling for MSC-based routing is itself flat.

Every clause is measured, in two studies where the design allows. The last
clause is the one no previous work could state, because the oracle baseline had
never been computed.

This is a **limits paper with a decisive mechanism**, not a failed method paper.
The difference is entirely due to B11.

---

## 5. What would strengthen it, in cost order

| | runs | GPU-h | buys |
|---|---|---|---|
| **A** | 2 | ~12 | a 3rd seed on both architectures → an error bar on ρ_seed |
| **B** | 8 | ~64 | A + `shufflenetv2` and `convnext_tiny` at 3 seeds. **Both exist in the CIFAR zoo**, so `table6_cifar_vs_imagenet.csv` stops being empty and the cross-study comparison becomes same-architecture |
| **C** | 24 | ~334 | the full 8-architecture atlas |

**B is the one that changes what can be claimed.** It converts §2.4 from "the
ordering replicates on different architectures" to "the same architecture, at
both scales, moves this much" — the only version that supports a quantitative
statement about scale.

Estimates use pre-`channels_last`-fix throughput for the convnets and are
therefore **over**-estimates; `python tools/conv_sweep.py --arch <name>` gives
the real figure.

---

## 6. Provenance

Every number here is in `C:\msc_results\analysis\` and reproducible by NB4/NB5:

| claim | file |
|---|---|
| ρ_seed, Jaccard@10, τ-curves | `q1_seed_ceilings_all.csv` |
| PC1 | `q2_axis_structure_all.csv` |
| T, CI, shuffled control | `q3_transfer_matrix.csv`, `q3_shuffled_control.csv` |
| ΔR², partial ρ | `q4_irreducibility_all.csv` |
| B1/B2/B10/B11 per student | `q5_method_comparison.csv` |

`per_sample/test.parquet` per run is the underlying artifact; everything above
is computed from it, and `sample_idx` is a global pack index so tables from
different runs join directly.
