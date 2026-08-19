# Why Study 1 fell short

Study 1 is not worthless — it produced two results that Study 2 is built on,
and 45 measured runs that Study 2 reuses. But as a *paper* it does not stand up,
and the reasons are structural rather than bad luck. Five diagnoses, each with
the evidence.

This is about the science. The engineering failures are in
`docs/imagenet100/22_IN100_LAB_NOTEBOOK.md` (D-37…D-86) and are a separate
matter.

---

## 1. The ceiling was measured after the method, not before

**The single most expensive mistake in the project.**

MSC-KD was the method: distil the teacher's per-sample compute requirement into
a student's routing policy. Eighteen students were trained — 3 architectures ×
3 seeds × 2 arms — for roughly **79 GPU-hours**.

The B11 baseline asks a prior question: *if the student routed by its own
**true** post-hoc MSC — the best any MSC-based router could do — how much better
than confidence thresholding would it be?*

Answer, measured afterwards in about two hours:

```
B11 (oracle MSC)  −  B2 (confidence)  =  +0.00007   (sd 0.00036)
```

**There was no gap.** The method could not have won, and one cheap measurement
before training would have said so.

The CIFAR study had the same blind spot and recorded it honestly — B11 is
listed in `docs/cifar100/10_FINAL_RESULTS.md` §6 as *"the one substantive gap
remaining"* (O-21), never computed. So both studies built or planned a method on
top of an unverified premise.

> **Rule for Study 2.** Every "signal X could improve Y" claim is preceded by
> "what is the best X could possibly do", and that number is a **gate**.

## 2. Every claim was conditional on MSC being the right construct

The five hypotheses all presuppose MSC. H1 is *MSC's* reliability. H2 is *MSC's*
dimensionality. H3 is *MSC's* transfer. H4 is *MSC's* irreducibility. H5 is a
method built from MSC.

So when H5 failed and H4 missed, there was no claim left that did not sound like
a defence of the metric. A paper whose every result is "our quantity is good"
has nowhere to stand when one result says otherwise.

**Contrast.** "Difficulty scores have unreported noise ceilings" is true or
false regardless of whether MSC is any good — MSC becomes one column in the
table. Study 2 makes no claim that depends on a metric we invented.

## 3. The two strongest results were framed as scaffolding

| what it was called | what it actually is |
|---|---|
| Q1, "the denominator of every transfer claim" | **ρ_seed varies 0.547–0.822 by architecture** — a methodological critique of a literature that does not measure it |
| Q5, "a negative result" | **the oracle ceiling for per-sample routing is flat** — a bound on a family of methods |

Both were treated as supporting apparatus for the MSC story. Both are more
interesting than the MSC story. Study 1 buried its findings under its framing.

## 4. Cross-scale claims were designed out before the first run

The stated purpose of the ImageNet-100 port was: *does the CIFAR seed-reliability
gap survive at scale, or was it a small-data artifact?*

That question requires an architecture measurable at both scales. The zoos share
**none**:

| | CIFAR-100 | ImageNet-100 |
|---|---|---|
| non-CNN | `vit_tiny`, `mixer_nano` | `vit_small_p16` |
| CNN | 13 small CNNs | `resnet50` |

`table6_cifar_vs_imagenet.csv` states it in the data: `same_architecture=False`
on every row, CIFAR column empty. Every cross-study number confounds
architecture, resolution and dataset simultaneously. The headline question was
**unanswerable by construction**, and nobody noticed until the analysis ran.

`shufflenetv2` and `convnext` exist in both zoos. Choosing them would have cost
the same and answered the question.

## 5. The pre-registration was inherited rather than re-derived

H1–H5 were written for CIFAR-100 and applied verbatim to ImageNet-100, where
several were not testable:

- **H3** predicts an *ordering* — within-family > across-CNN-family > CNN→ViT.
  ImageNet-100 has two architectures, hence **one pair**. An ordering over one
  element is not a test.
- **H1's** 0.60 threshold was chosen for 32px CIFAR. Nothing argued it should
  transfer to 224px.

A pre-registration is a commitment about *this* experiment. Copying one across
a design change keeps the ritual and loses the function.

---

## What Study 1 got right, and Study 2 keeps

1. **Noise-ceiling correction as standard practice.** Disattenuating by
   √(ρ_a·ρ_b) is correct and the results show it matters.
2. **Shuffled-target controls.** T = 0.640 versus 0.037 shuffled is what makes
   the transfer number a measurement rather than an artifact of alignment.
3. **Per-sample tables keyed by a global index.** `sample_idx` being a pack
   index rather than a split position is why 49 runs can be joined at all —
   and it is what makes Study 2 cheap.
4. **Reporting misses as misses.** H2, H3 and H4 are recorded as refuted or
   missed, with numbers. That honesty is what makes the corpus reusable.

---

## The through-line

Every one of these five is the same shape: **the expensive thing was done
before the cheap thing that would have told you whether to do it.**

- 79 GPU-h of students before a 2-hour ceiling measurement.
- 45 backbone runs before checking the two zoos shared an architecture.
- Five hypotheses about a metric before asking whether the metric needed to be
  special for the paper to work.

Study 2 inverts the order deliberately: **P0 costs 2 CPU-hours and P1 costs 6
GPU-hours, and between them they decide whether P2 is worth running at all.**
