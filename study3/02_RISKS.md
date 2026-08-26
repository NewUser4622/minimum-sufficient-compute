# Study 3 — risk register

Each risk has a **detector** (how we find out), a **trigger** (when), and a
**response** (what we do), all fixed before any result exists.

Study 2's register worked: R-02, R-03, R-04 and R-06 all fired, and because the
responses were written in advance none of them turned into an argument after the
fact. R-06 ("I break something in the re-analysis") fired *repeatedly* and is
the reason this file leads with it.

---

## R-01 · The measurement is wrong again · **HIGH**

Study 2's routing code was implemented incorrectly **three times** (+5.165,
−10, −8 accuracy points), each producing a plausible, publishable-looking
number. Study 3 reuses that code and adds two new measurements.

**Detector.** `tools/s2_routing_canaries.py` (18) and `s2_canaries.py` (3) must
pass unchanged before any Study 3 number is believed. Each new statistic gets
its own canary, including the load-bearing kind: *it must detect an effect in a
world where the effect certainly exists*.

**Trigger.** Before the first result is written down.

**Response.** A number without a passing canary is not reported. If a canary
fails, fix the measurement before looking at the data again — Study 2's canary
10 found a bug inside the fix for canary 9.

**Specific new canaries required:**

| measurement | canary |
|---|---|
| joint-exit training (Q1) | a network with *identical* exits must give pool = 0; a network whose early exits are random must give a large pool |
| learned router (Q2) | a router given the label must capture ~100 % of the gap; a router given noise must capture ~0 % |
| pruning (Q3) | pruning that keeps *everything* must reproduce the full-data baseline exactly |

## R-02 · Joint training changes accuracy, confounding the pairing · **HIGH**

Q1 compares frozen against joint exits. Joint training also changes the
backbone, so a raw pool comparison confounds "better exits" with "different
network".

**Detector.** Compare `acc_full` between the frozen and joint runs of the same
architecture.
**Trigger.** P1, immediately.
**Response.** Report the pool raw *and* conditioned on `acc_full`. If final
accuracy moves by more than 1 pt, the conditioned number is primary. If it moves
by more than 3 pt, add a matched-accuracy checkpoint (early-stopped joint run) so
the comparison is like-for-like.

## R-03 · "Jointly trained" is underspecified · **MEDIUM**

Deep supervision has many variants — uniform loss weights, linearly decaying
weights, gradient rescaling, gradient equilibrium. Results may depend on which.

**Detector.** Literature check before implementing; record the recipe chosen and
its source.
**Trigger.** Before P1 is written.
**Response.** Use one standard recipe, name it, and state that sensitivity to the
variant is untested. If P1 lands near a threshold, run one alternative weighting
as a robustness check — **not** as a second chance at a favourable result.

## R-04 · The learned router memorises seed noise · **MEDIUM — and it is the point**

A router trained on seed *i*'s per-exit correctness can fit that seed's noise and
report a large capture fraction that means nothing.

**Detector.** Train on seed *i*, evaluate on seed *j*. Also a held-out split
within seed *i*.
**Trigger.** P2, built in from the start — not added after a good number appears.
**Response.** Report both the in-seed and cross-seed capture. The in-seed number
alone is uninterpretable and must never be quoted on its own. This is the same
error Study 2 spent three rounds correcting, in a new place.

## R-05 · The pruning experiment cannot show a difference · **MEDIUM**

If the saturated and unsaturated sources select nearly the same samples, no
downstream difference is possible and 18 GPU-h buys nothing.

**Detector.** Kept-set overlap at each retention rate. Costs minutes.
**Trigger.** **Before** P3 runs.
**Response.** If overlap ≥ 90 %, cancel P3 and report the overlap itself — "the
scores disagree on reliability but agree on which samples to keep" is a finding,
and a cheap one.

## R-06 · Q1 falsifies Study 2 · **MEDIUM — accepted, and survivable**

If jointly trained exits remove the pool, Study 2's headline was an artifact of
post-hoc exits.

**Response.** Report it plainly and with the same prominence. It converts to a
real, useful claim: *oracle bounds are inflated when exits are trained post-hoc,
and sound when trained jointly* — which is directly actionable for anyone
building an early-exit system. Study 2's other findings (collinearity, the
reliability atlas, the memorisation collapse) are untouched by this, because none
of them depend on the routing machinery.

**This is why Q3 is scheduled independently.** If Q1 goes badly there is still a
paper, and it does not need Q1 to be true.

## R-07 · Someone publishes this first · **LOW**

**Response.** Q1 is ~10 GPU-h and settles the blocker. Do it first. The
literature check for R-03 doubles as a freshness check on the whole idea.

## R-08 · Hardware · **LOW, but it has bitten before**

Multi-exit training holds K heads and K losses; memory is higher than Study 1's
single-head runs. A benchmark once crashed this machine and cost two hours.

**Detector.** Dry run at batch size 1, then the target batch size, with a memory
probe, before any full run.
**Trigger.** Before P1's first epoch.
**Response.** Reduce batch size rather than risk the machine. Study 1's timings
(1.3–3.7 GPU-h per run) assume a 20 GB card is not being pushed to its limit.

## R-09 · The network drops mid-run · **OCCURRED — now designed out**

**It happened on the first joint run.** A background HuggingFace uploader
retried, hit a 403 (read-only token, and the wrong repo), and the run stopped.

**Response, already applied.** Training and analysis notebooks run with
`enable_hf=False`. Nothing is uploaded while work is in progress; the local
tree is complete and authoritative. `S3_NB5_Publish` uploads once, at the end,
with a network you know is up, and it:

* checks the token BEFORE uploading (`hf_token_check` — valid / **write** /
  right namespace), so a 403 names its own cause (D-84);
* sizes the upload before moving a byte;
* uploads folder-at-a-time via `hf_upload_resilient`, which survives a DNS drop
  (D-86);
* verifies with `resolve_meta` — a drained queue is not confirmation
  (rules 9, 10).

**Detector:** the canary suite asserts HF is off in every training notebook and
on only in the publisher. **Trigger:** build time, every time.

---

## What success means

Not "the hypotheses are confirmed". Success is that **at the end we know whether
Study 2's number is a property of oracle bounds or a property of how we trained
our exits** — and that we can say which, with a number and an interval.

By that standard every branch succeeds. The branch that would make Study 3 a
waste is spending the GPU-hours and still not being able to tell the two apart,
which is what R-02's matched-accuracy response exists to prevent.
