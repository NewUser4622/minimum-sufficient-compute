# Decisions — settled

**All five are decided.** You asked me to make the call, so I have. Each is
recorded with the reasoning so it can be overturned by evidence rather than
preference — and D1 and D3 moved after the literature search
(`08_RELATED_WORK.md`).

If you disagree with any of them, say so and I will change it; nothing here is
irreversible and none of it blocks P0.

---

## D1 — Is the framing right?

Study 2's thesis:

> Per-sample difficulty scores are measured with architecture-dependent noise
> the literature does not report, and correcting for it removes the headroom
> that motivates per-sample adaptive inference.

This is a **limits/methods paper**. It bounds a family of methods and criticises
a measurement practice. It does not propose a new technique.

Some venues take that readily (TMLR, the reproducibility and
critique tracks); some do not. If the goal is a method paper, Study 2 as
designed is the wrong shape and we should talk about that before P0.

**DECIDED: yes — but reframed after the literature search.**

The original framing put *"difficulty scores are noisy"* first. That is already
known qualitatively (`08_RELATED_WORK.md` §4), so it cannot lead. The
**optimism bias** leads, and reliability becomes the *instrument* that measures
it rather than a finding in its own right:

> Oracle upper bounds for early-exit routing are computed from the model being
> routed and are therefore optimistically biased. Using a second training seed
> as an unbiased instrument, we measure that bias across 8 difficulty scores and
> 15 architectures, relate it to each score's measured reliability, and report
> the corrected headroom.

Still a limits/methods paper. TMLR remains the natural venue.

## D2 — Does MSC stay in the paper?

Three options:

| | MSC's role | consequence |
|---|---|---|
| **A** | one of eight scores, not singled out | cleanest; the paper cannot be sunk by MSC being unremarkable |
| **B** | one of eight, plus a short section on what makes it different | keeps Study 1's contribution visible; costs a section and invites "why is your metric special?" |
| **C** | the subject, with the other seven as context | reverts to Study 1's structural flaw (`01_POSTMORTEM.md` §2) |

**Recommendation: A**, with Study 1 cited for the construct. B is defensible.
C is the mistake we just diagnosed.

**DECIDED: A** — MSC is one of eight, not singled out.

Study 1 is cited for the construct and the 45 measured runs. The paper cannot be
sunk by MSC turning out unremarkable, which is the specific failure mode of
`01_POSTMORTEM.md` §2.

## D3 — Does ImageNet-100 appear at all?

CIFAR-100 alone supports the whole thesis: 15 architectures, 3 seeds, 8 scores.
ImageNet-100 adds 2 architectures × 2 seeds, no shared architecture, no error
bars.

| | |
|---|---|
| **CIFAR only** | clean, defensible, no confounds. Reviewers ask "does it hold at scale?" |
| **Both, ImageNet as a bounded appendix** | pre-empts the question; must be explicit that no cross-scale *magnitude* claim is made |
| **Both, with P2b** (a shared architecture, ~18 GPU-h) | the only way to make a real scale claim |

**Recommendation: appendix**, unless a reviewer-proof scale claim is worth the
extra runs — in which case P2b, and `shufflenetv2` is the natural choice since
it exists in both zoos.

**DECIDED: CIFAR-100 only for the main result; ImageNet-100 mentioned as a
consistency note, not an appendix section.**

Moved from "appendix" after the search. With 2 architectures × **2 seeds**,
ImageNet cannot support a cross-seed bias estimate with any confidence — the
bias needs seed pairs, and 2 seeds gives exactly one pair per architecture with
no spread. Presenting it as an appendix implies more than it carries.

CIFAR-100 alone gives 15 architectures × 3 seeds = **3 seed pairs each**, which
is what the design actually needs. ImageNet appears in one paragraph: *the same
direction is observed on 2 architectures at 224px, n too small for an interval.*

P2b (`shufflenetv2` on ImageNet, ~18 GPU-h) stays available if a reviewer wants
scale, but is not planned.

## D4 — Is +1.0 accuracy point the right gate for R3?

It is inherited from Study 1's H5, which keeps the two commensurable. But it is
inherited, and `01_POSTMORTEM.md` §5 is about exactly that failure mode.

Arguments for a **lower** bar (say +0.3): at 10,000 samples, 2 SE ≈ 0.011, so
+0.3 points is still ~2.7σ and detectable. A method that reliably bought 0.3
points at 20% less compute would be interesting.

Arguments for keeping +1.0: it is what the field's papers typically claim, and
what Study 1 pre-registered.

**Recommendation: keep +1.0 as the pre-registered gate, and report the measured
headroom with its noise floor regardless**, so a reader can apply their own bar.
That is honest either way and costs nothing.

**DECIDED: keep +1.0 as the pre-registered gate; always report the measured
headroom with its noise floor.**

Unchanged from the recommendation. The bar is inherited but *deliberately* so —
it keeps Study 1 and Study 2 commensurable, and it is the effect size the field
typically claims. Reporting the raw number alongside lets any reader apply their
own bar, which costs nothing and pre-empts the objection.

## D5 — What happens to Study 1's write-up?

| | |
|---|---|
| **Keep as-is** | complete, honest, on HuggingFace, cited by Study 2 for the construct |
| **Cut down** | reduce to a technical report on the measurement machinery |
| **Merge** | fold the useful parts into Study 2 as background |

**Recommendation: keep as-is and cite it.** It is internally honest — H2, H3 and
H4 are recorded as refuted or missed, with numbers — and the 49 measured runs
are Study 2's input. A study that reports its own misses is worth citing.

`PAPER.md` stays the CIFAR-100 manuscript with its scope note.

**DECIDED: keep as-is and cite it.**

It is internally honest — H2, H3 and H4 are recorded as refuted or missed, with
numbers — and its 49 measured runs are Study 2's entire input. `PAPER.md` stays
the CIFAR-100 manuscript with its scope note.

---

## Summary

| | decision |
|---|---|
| **D1** framing | yes — **optimism bias leads**, reliability is the instrument |
| **D2** MSC | **A** — one of eight |
| **D3** ImageNet | **CIFAR-only** main result; ImageNet is one paragraph |
| **D4** gate | keep **+1.0**, always report the noise floor |
| **D5** Study 1 | keep as-is, cite it |

**Next action is P0a**, the collinearity matrix — and the literature search
promoted it from a sanity check to a **decision point**. One paper reports that
difficulty scoring functions agree with each other **>70% in almost every case**
(`08_RELATED_WORK.md` §6). If our eight collapse into three or four families,
the grid has far fewer independent cells than it looks and the analysis must be
recomputed on families, not scores.

Run `S2_NB1` as far as the P0a output and stop there.

---

## And one thing I should say plainly

Study 1's failures were substantially mine — the ceiling measured after the
method, the reader wired without a writer, eight defects where the verification
and the artifact were not the same object. The design above is shaped by those
specific mistakes, which is the only useful thing to do with them.

But a design document is not evidence. **The first thing Study 2 does is check
that its own inventory file is true** (`03_INVENTORY.md`, last section) rather
than trusting it — because assuming the data was as described is how several of
those defects happened.
