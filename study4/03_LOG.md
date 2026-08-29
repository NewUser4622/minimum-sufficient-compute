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
| **P0** | Figure 1 (ρ-sweep) + bootstrap CIs | **free**, CPU min | planned | `analysis/s4_bootstrap.csv`, `paper/figures/fig1_headroom.png` |
| **P1** | margin + patience baselines | **free**, CPU min | planned | `analysis/s4_baselines.csv` |
| **P2** | ImageNet-100 + transformer | ~20 GPU-h | planned — gated on P0/P1 | `runs/p6-*-jointexit-s1` |
| **P3** | MSDNet, a designed early-exit net | ~15 GPU-h | planned — highest risk | `runs/p7-msdnet-*` |

**Next action: build and run P0 + P1.** They cost nothing, they produce the
paper's main figure and its intervals, and P1 can overturn the headline numbers
by finding a stronger baseline — which is exactly why it runs before any GPU
time is spent.

---

## Pre-registered predictions — fill in as results arrive

Do not edit the prediction column.

| | prediction | threshold | measured | verdict |
|---|---|---|---|---|
| **H6** | conclusion is baseline-independent | honest headroom negative for margin + patience at all 7 budgets | _pending_ | _pending_ |
| **H4** | excess holds at ImageNet-100 scale | ≥ 2.0 pt, 2 of 2 | _pending_ | _pending_ |
| **H4b** | it holds on the **transformer** specifically | ≥ 2.0 pt, `vit_small_p16` | _pending_ | _pending_ |
| **H5** | excess holds on MSDNet | ≥ 2.0 pt, 2 of 2 seeds | _pending_ | _pending_ |

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
