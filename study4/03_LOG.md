# Study 4 — live log

**Newest first.** One entry per session: what changed, what it cost, what it
**settled**, what is next.

A phase is `done` only when its artifact exists on disk **and has been opened
and checked**.

---

## Machine facts

| | value | wired in as |
|---|---|---|
| CIFAR-100 | `C:\Users\Administrator\Desktop\New folder\cifar-100-python` | `CIFAR_DIR` → `MSC_CIFAR_DIR` → `locate_cifar100()` |
| ImageNet-100 pack | `C:\msc_data\in100` | `MSC_IN100_DIR` |
| results root | `C:\msc_results` | `MSC_ROOT`, auto-resolved |
| GPU | RTX 4000 Ada, 20 GB | — |
| HF repo | `Shanmuk4622/msc-cifar100` | `MSC_HF_REPO` (library default is the **ImageNet** repo) |
| network | **intermittent** | notebooks run OFFLINE; a publish notebook uploads once at the end |

**Measured timings — use these, not guesses.**

| run | measured | use for planning? |
|---|---|---|
| CIFAR-100, 240–300 ep | 1.3–3.7 GPU-h | yes |
| `vit_small_p16` IN-100, 100 ep | 5.7 GPU-h | yes |
| `resnet50` IN-100, 100 ep | **41.5 GPU-h** | **NO — that is the D-59 layout run.** Documented correct figure ~6 h. P2 gates on epoch-1 throughput |

---

## Status board

| | what | cost | state | artifact |
|---|---|---|---|---|
| **P0** | Figure 1 (ρ-sweep) + bootstrap CIs | free | **DONE — 3/3 CIs exclude zero** | `analysis/s4_bootstrap.csv`, `paper/figures/fig1_headroom.png` |
| **P1** | margin + patience baselines | free | **RUN, RESULTS VOID (D-89) — re-run** | `analysis/s4_baselines.csv` |
| **P2** | ImageNet-100 + transformer | ~20 GPU-h | **DONE — H4 and H4b SUPPORTED** | `runs/p6-*-jointexit-s1` |
| **P3** | MSDNet, a designed early-exit net | ~15 GPU-h | **not built yet** — needs new zoo code, own pass | `runs/p7-msdnet-*` |

**Next action: re-run `S4_NB1_Baselines`.** Its router was broken (D-89);
the fix is in and verified, but the results on disk are void.

---

## Pre-registered predictions — fill in as results arrive

Do not edit the prediction column.

| | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| **H6** | conclusion is baseline-independent | negative at all 7 budgets | **VOID — D-89** | re-run required |
| **H4** | excess holds at ImageNet-100 scale | ≥ 2.0 pt, 2 of 2 | **7.39 / 6.91 pt** | **SUPPORTED** |
| **H4b** | it holds on the **transformer** specifically | ≥ 2.0 pt | **6.91 pt** | **SUPPORTED** |
| **H5** | excess holds on MSDNet | ≥ 2.0 pt, 2 of 2 seeds | _pending_ | _pending_ |

---

## 2026-08-20 (RESULTS) · P0 and P2 land; P1 is VOID — D-89

### P0 — the intervals are tight and all exclude zero

| arch | excess | 95 % CI over 10,000 test samples |
|---|---|---|
| `resnet20` | **10.64** | [10.00, 11.28] |
| `resnet32x4` | **8.55** | [7.98, 9.10] |
| `vgg8` | **9.15** | [8.61, 9.71] |

**3 of 3 exclude zero**, and the widths are ~1.2 pt against effects of 8–11 pt.
The refusal condition did not trigger. Figure 1 is written to
`paper/figures/fig1_headroom.png`.

*(Sample interval, not a seed interval — the label says so, and the n is read
from the data rather than assumed.)*

### P2 — the scale objection is closed, transformer included

| arch | excess | full compute |
|---|---|---|
| `resnet50` | **7.39 pt** | 81.58 % |
| **`vit_small_p16`** | **6.91 pt** | 63.23 % |

**H4 SUPPORTED (2 of 2). H4b SUPPORTED — the transformer shows 6.91 pt.**

This is the result Study 4 existed for. The excess is no longer a CIFAR-100
finding on small convolutional nets: it holds at **224 px on ImageNet-100** and
on a **vision transformer**, at magnitudes (6.9–7.4 pt) close to CIFAR's 6.86.
The claim in `../PAPER_CLAIM.md` can drop "on CIFAR-100" from its scope.

### D-89 — P1's router had an inverted bisection, and a canary that could not fail

`S4_NB1` produced this, which is impossible:

| ρ | 0.40 | 0.60 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|
| confidence accuracy | 71.57 | 71.57 | 69.52 | 27.97 | **19.21** |

**Accuracy fell as the budget rose.** More compute cannot hurt.

**Cause.** Cost is non-decreasing in the threshold — a higher bar means fewer
early exits and more compute. So when the achieved cost is *below* target the
bar must go **up**. `route_threshold` lowered it. The search ran the budget
backwards, which is exactly the inverted curve above.

**Why it survived.** Two compounding mistakes of mine:

1. **`achieved_cost` echoed the target.** For the threshold rules I recorded
   `tr` — the requested budget — instead of the measured cost. So the overspend
   check I had *just built* could never fire for them, and the inversion was
   invisible in the table.
2. **The canary asserted `cost <= target`.** A constant satisfies that at every
   loose budget. It passed while the function ignored the budget entirely.
   **A canary a broken function passes is not a canary** — the lesson this
   project keeps re-learning, in a new place.

**Fixed:** correct bisection direction, `lo` side chosen so it never overspends,
measured cost recorded, and the canary strengthened to require that **cost
tracks the budget** (spread > 0.15, not constant) and that **accuracy never
falls as the budget rises**. Verified: cost spread 0.394, monotone accuracy.

### A near-miss worth recording

While checking whether Studies 2–3 were contaminated, my first diagnostic used
**binary** synthetic confidence. Only three costs are reachable there, so *both*
the old and new functions sat on the same plateau and the test reported
`route_confidence` as **ALSO BROKEN**. It is not. With continuous scores both
track the budget exactly and agree to three decimals.

**I nearly reported Studies 2–3 as invalid on the strength of a bad test.**
Study 2/3's `route_confidence` raises the bar with `lo = th` — the correct
direction — and was right all along. D-89 is confined to Study 4's P1.

**Consequence:** `s4_baselines.csv` is void and withheld from `RESULTS.md`
rather than published beside correct numbers. H6 is unanswered until
`S4_NB1` is re-run.

---

## 2026-08-20 (built) · four notebooks, and three defects the harness caught

`build_notebooks_study4.py` → `notebooks_study4/`. All gates pass, including a
**new** one: every `M.*` and `sess.*` the notebooks reference is checked against
the library AST at build time. `check_names` cannot see attributes, which is how
`sess.hub.resolve_meta` reached a GPU in Study 3.

| notebook | phase | cost |
|---|---|---|
| `S4_NB0_Figures` | P0 | free, CPU minutes |
| `S4_NB1_Baselines` | P1 | free, CPU minutes |
| `S4_NB2_ImageNet` | P2 | ~20 GPU-h |
| `S4_NB3_Publish` | — | minutes, the only online notebook |

**MSDNet (P3) is deliberately not built yet.** It needs a new architecture in
the zoo rather than a config flag, and this authoring environment has no torch —
so it would ship entirely unexecuted, which is the pattern that has cost the most
in this project. It gets its own pass, after P0–P2 have produced something.

### Three defects caught by executing the notebooks, not reading them

`tools/s4_harness.py` runs NB0 and NB1's **real cells** against synthetic data
with a known answer. It found:

1. **A canary that could not fail.** The bootstrap "CI narrows with n" check
   sliced a *sorted* array, so the small sample had zero excess and zero width —
   the canary reported FAIL while the code was correct. Now samples randomly.
2. **A hardcoded `n=10,000`** in the manuscript label, printed regardless of the
   actual split size. Now read from the data: quoting 10,000 when the split is
   smaller would be a fabricated detail in the paper itself.
3. **An unfair baseline ranking — the serious one.** `winner = idxmax()` over
   raw accuracy let **patience win by overspending**: it reached 0.747 at a 0.70
   budget, bought accuracy it was not entitled to, and would have been declared
   "the strongest baseline", triggering a recomputation of every headroom figure
   in Studies 2–3 against a baseline that had simply cheated on cost.

The fix took three attempts, each caught by the harness:

* symmetric ±0.05 tolerance — **still let patience win**, because it overshot
  *within* tolerance;
* asymmetric (may underspend, never overspend) — **still unfair**, because
  patience then qualified at only some budgets and its median was computed over
  a different set;
* **common budgets only** — rank on the budgets where every baseline has a
  qualifying row. Correct.

**And one of my own harness assumptions was wrong.** I expected confidence to
win, but the synthetic world gave patience perfect agreement signal too, so
patience winning was right. The discriminating test is that the **noise**
baseline must never win — which is what the harness now asserts.

PABEE has only K discrete operating points and frequently cannot hit a
requested budget. That is a property of the method, and the notebook now reports
achieved cost beside accuracy rather than tabulating them as if the budgets
matched.

---

## 2026-08-20 · Study 4 planned

Written after [`../PAPER_CLAIM.md`](../PAPER_CLAIM.md) identified the three
objections standing between Paper A and an archival venue. Each maps to one
phase.

**Two corrections to the estimates in that document**, both from checking rather
than assuming — which is the habit Study 3 was supposed to teach:

1. **"Extra baselines are free" was only two-thirds true.** Margin
   (`top1p − top2p`) and patience (`pred_dk` agreement) are computable from the
   existing parquets. **Entropy is not** — only the top-2 probabilities were
   stored, never full logits. Approximating it would be a fabricated baseline.
   The protocol omits entropy and says why, in the manuscript as well as here.

2. **"ImageNet ≈ 25 GPU-h" quoted a broken run.**
   `p0-resnet50-imagenet100-base-s1` took **41.5 GPU-h** — the D-59
   `channels_last` disaster, four times the documented ~6 h. `vit_small_p16`
   took 5.7 h. Revised estimate ~20 GPU-h, **with a throughput gate in epoch 1**
   that aborts if the regression has returned. That gate is R-02 and it is the
   difference between losing one epoch and losing 35 hours.

**Ordering.** P0 and P1 are free and go first — not only because they are cheap,
but because **P1 can invalidate the headline**. If margin or patience beats
confidence thresholding, every headroom number in Studies 2–3 was computed
against a weak comparator and must be recomputed. Finding that out costs CPU
minutes; finding it out after P2 and P3 costs 35 GPU-hours.

**P3 is last and is the riskiest.** It needs a genuinely new architecture in the
zoo rather than a config flag, and this authoring environment has no torch, so
it ships unexecuted until it reaches the GPU. R-03 requires checking MSDNet's
accuracy against the published number before H5 is treated as tested at all.

**The falsifier worth stating up front:** if MSDNet shows no excess, that is not
a failure. It converts the claim into something sharper and more actionable —
*oracle bounds are inflated for attached exits and sound for designed ones* —
and it must be reported with equal prominence.

### What is deliberately not planned

No ImageNet-1k. No further pruning work (that is Paper B, and it needs its own
rerun with the selection rule inverted). No new difficulty scores. Every one of
those is downstream of P2/P3, and Study 1 spent 79 GPU-hours on work that a
later cheap check showed was pointless.
