# Minimum Sufficient Compute (MSC)

**How much compute does *this* sample actually need, and is that a property of
the sample or of the seed?**

MSC is a per-sample, cost-normalised, multi-axis, stability-closed measure of
the compute a trained network needs to reach its final decision. Two studies
live in this repository:

| study | dataset | scale | architectures | status |
|---|---|---|---|---|
| **Study 1 · CIFAR-100** | 50k @ 32px | small | 15 | complete — [`docs/cifar100/10_FINAL_RESULTS.md`](docs/cifar100/10_FINAL_RESULTS.md) |
| **Study 1 · ImageNet-100** | 130k @ 224px | 40× data, 49× pixels | 2 (pilot) + 3 students | pilot complete — [`docs/imagenet100/26_IN100_FINDINGS.md`](docs/imagenet100/26_IN100_FINDINGS.md) |
| **Study 2** | CIFAR-100, re-analysed | — | 15 | **complete** — [`study2/PAPER.md`](study2/PAPER.md) |
| **Study 3** | CIFAR-100 | — | 3 + pruning | **complete** — [`study3/04_FINDINGS.md`](study3/04_FINDINGS.md) |
| **Study 4** | CIFAR-100 + ImageNet-100 | 32px + 224px | 2 + MSDNet | **P0–P2 complete, P3 built** — [`study4/README.md`](study4/README.md) |

**Study 3 runs offline.** Training and analysis notebooks never touch the network; `S3_NB5_Publish` uploads the finished tree in one pass at the end.

### Study 2, in one line

> **Oracle upper bounds for early-exit routing are inflated by +22.41 accuracy
> points — more than the entire headroom they appear to show — and the excess is
> per-exit noise that does not survive a change of training seed.**

At ρ = 0.80, across 15 architectures × 3 seeds (90 ordered seed pairs): an
oracle scored from the **same** seed it routes reaches **78.30 %**; scored from a
**different** seed it reaches **54.50 %**; a deployable confidence baseline
reaches **62.39 %**. **0 of 15** architectures keep positive honest headroom, at
any budget from ρ = 0.40 to 0.95.

The load-bearing part is an **exact identity**: the in-seed oracle beats the
network's *own full-compute accuracy* (71.21 %) in 100 % of runs by a median of
**+6.86 pt**, which equals the fraction of samples that are right at some early
exit and **wrong at the final layer**. A bound above full compute cannot be
reached by any router. That component needs no second seed; the remaining
+15.33 pt of the bias measures cross-seed non-transfer and is reported
separately as weaker evidence.

**This confirms Study 1 rather than overturning it.** B11's +0.00007 was read as
a possible MSC artifact; it was not.

Study 2 cost **no training and no new runs** — Study 1's per-sample parquets
already held per-exit predictions, so it is CPU re-analysis. Start at
[`study2/PAPER.md`](study2/PAPER.md).

**That blocker is cleared.** Study 3 Q1 retrained three architectures with exits
trained **jointly** — the way MSDNet and BranchyNet do it — and the excess came
out **larger**, not smaller: **8.55 / 9.15 / 10.64 pt** against 6.42 / 7.95 /
6.69 frozen, surviving conditioning on backbone accuracy. Weak exits are right
on almost nothing, so they cannot rescue a sample the final layer gets wrong;
rescues need competent exits, and the pool grows with exit quality.

Study 3 Q2 then asked whether a *learned* gate can reach that gap **at
ρ = 0.80**. It captures **1.7 %**, and in-seed capture is no higher than
cross-seed, so it is not even memorising noise.

**Study 4 closed the scale objection and qualified the claim.** The excess holds
at 224 px on ImageNet-100 — `resnet50` **+7.39 pt**, `vit_small_p16`
**+6.91 pt**, a vision transformer. But sweeping the compute budget, the honest
cross-seed ceiling **changes sign at ρ ≈ 0.65**: negative above it, **positive
below** (+7.74 pt at ρ = 0.40), with three independent exit rules agreeing
within 1.8 pt. *"No headroom"* is true only at generous budgets.
Q3 (does the memorisation collapse damage pruning?) came back inconclusive with
a confounded design, recorded rather than re-cut. See
[`study3/04_FINDINGS.md`](study3/04_FINDINGS.md).

---

## The results, and what to do with them

| | |
|---|---|
| [`RESULTS.md`](RESULTS.md) / [`RESULTS.csv`](RESULTS.csv) | **every headline number**, generated from the artifacts by `tools/build_results.py` — never hand-edited |
| [`PAPER_CLAIM.md`](PAPER_CLAIM.md) | **what is publishable, where, and what is missing** — two papers, with a costed gap analysis |

All source CSVs are on
[`Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100)
under `analysis/`, verified byte-identical to the local copies (12/12).

---

## Start here

**If you want the science**, read [`study2/PAPER.md`](study2/PAPER.md) — the
newest and strongest result, and the one that does not depend on MSC being a
good metric. For Study 1, read
[`docs/imagenet100/26_IN100_FINDINGS.md`](docs/imagenet100/26_IN100_FINDINGS.md),
which scores all five pre-registered hypotheses against **both** datasets.

**If you want to know how the work goes wrong**, read
[`study2/README.md`](study2/README.md) §"What actually went wrong" — the same
measurement was implemented incorrectly three times, each producing a plausible
number, before a canary caught it.

**If you want the current operational state**, read
[`docs/imagenet100/24_IN100_STATUS.md`](docs/imagenet100/24_IN100_STATUS.md).

**If something is broken**, read
[`docs/imagenet100/23_IN100_RUNBOOK.md`](docs/imagenet100/23_IN100_RUNBOOK.md)
— a symptom → cause → fix table.

**If you are picking this project up**, read in this order:

1. [`docs/cifar100/09_LAB_NOTEBOOK.md`](docs/cifar100/09_LAB_NOTEBOOK.md) —
   36 defects, each with a contamination analysis. The most important file.
2. [`docs/cifar100/10_FINAL_RESULTS.md`](docs/cifar100/10_FINAL_RESULTS.md)
3. [`docs/cifar100/07_REPLICATION_PLAYBOOK.md`](docs/cifar100/07_REPLICATION_PLAYBOOK.md)
4. [`docs/cifar100/00_RESEARCH_PROTOCOL.md`](docs/cifar100/00_RESEARCH_PROTOCOL.md)
5. [`docs/imagenet100/22_IN100_LAB_NOTEBOOK.md`](docs/imagenet100/22_IN100_LAB_NOTEBOOK.md)
   — defects D-37…D-85 from the ImageNet port.

---

## Repository layout

```
README.md              you are here
RESULTS.md / .csv      every headline number, GENERATED by tools/build_results.py
PAPER_CLAIM.md         what is publishable, where, and what is missing
PAPER.md               Study 1 manuscript (CIFAR-100)
requirements.txt

src/msc_lib.py         the library -- data, zoo, training, measurement, HF sync
msc_core.py            reference maths: the MSC definition and every statistic
msc_torch.py           torch helpers

build_notebooks.py           generates notebooks/        (Study 1, CIFAR-100)
build_notebooks_in100.py     generates notebooks_in100/  (Study 1, ImageNet-100)
build_notebooks_study2.py    generates notebooks_study2/
build_notebooks_study3.py    generates notebooks_study3/
build_notebooks_study4.py    generates notebooks_study4/

tools/                 validators, canaries, harnesses, results generator
docs/cifar100/         Study 1 protocol, lab notebook, results
docs/imagenet100/      Study 1 ImageNet-100 port
study2/ study3/ study4/  per-study README, protocol, risk register, log, findings
benchmark/             throughput measurements (D-59 lives here)
```

**Notebooks are generated, never hand-edited.** Each embeds `msc_lib` as base64
and writes it out at run time, so a notebook is self-contained and its library
version is checkable. Editing a notebook is undone by the next build; edit the
generator.

**Data is not in this repo.** Runs, checkpoints and per-sample tables live on
HuggingFace at
[`Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100)
(107 run directories) and
[`Shanmuk4622/msc-imagenet100`](https://huggingface.co/datasets/Shanmuk4622/msc-imagenet100).
`msc_data/` and `msc_results/` are gitignored local storage roots.

### Verification

```
python src/msc_lib.py --selftest        483 checks
python tools/s2_canaries.py               3
python tools/s2_routing_canaries.py      18
python tools/s3_canaries.py              75
python tools/s4_msdnet_canaries.py       47  (each proven able to fail)
python tools/s4_harness.py               executes S4_NB0/NB1/NB4 on known data
python tools/s3_nb0_harness.py           executes S3_NB0 on known data
python tools/s3_nb3_harness.py           executes S3_NB3 on known data
python tools/check_links.py              every doc cross-reference resolves
python tools/check_names.py <nb>...      no undefined names across cells
python tools/build_results.py            regenerates RESULTS.md / RESULTS.csv
```

**Every canary must be able to fail.** Several were found that could not — the
most recent (D-89) asserted `cost <= target`, which a constant satisfies, and
passed while the router ignored the budget entirely.

---

## Layout

```
├── src/msc_lib.py              the pipeline: data, zoo, training, measurement
│                               ONE library, parameterised by dataset
├── msc_core.py                 the reference maths -- the MSC definition and
│                               every statistic in the paper. Imported, never
│                               reimplemented.
├── build_notebooks.py          generates notebooks/       (CIFAR-100, 14)
├── build_notebooks_in100.py    generates notebooks_in100/ (ImageNet-100, 6)
├── build_notebooks_study2.py   generates notebooks_study2/ (Study 2, 3)
├── notebooks/                  GENERATED -- do not edit
├── notebooks_in100/            GENERATED -- do not edit
├── tools/
│   ├── validate_notebooks.py   column names, repo paths, call arity, stages
│   ├── check_names.py          undefined names (NameError before it happens)
│   ├── check_links.py          every document cross-reference resolves
│   ├── conv_sweep.py           per-architecture throughput, isolated + capped
│   ├── bisect_speed.py         where the time goes: compute/augment/data
│   ├── diagnose_epochs.py      read the timing split out of a finished run
│   ├── verify_loader.py        loader throughput, no model attached
│   ├── verify_d55.py           memory-format A/B
│   └── pack_imagenet100.py     JPEG tree -> uint8 memmap
├── benchmark/                  throughput measurements + their JSON
├── docs/cifar100/              00-10  protocol, spec, schema, results
├── docs/imagenet100/           20-26  port plan, delta, lab notebook, status,
│                                      data card, FINDINGS
├── notebooks_study2/           GENERATED -- CPU re-analysis, no GPU
├── study2/                     PROPOSAL -- postmortem, protocol, inventory,
│                               design, decisions, RISK REGISTER, progress
└── PAPER.md
```

**The notebooks are generated.** Editing one does nothing that survives a
rebuild — edit `src/msc_lib.py` or the generator, then:

```bash
python build_notebooks_in100.py
```

The build refuses to emit if any of these fail: undefined names, cells that do
not parse as Python 3.10, unknown column names, literal repo paths, wrong call
arity, or a stage predicate that would skip the work it was asked to do.

---

## Verifying a checkout

```bash
python src/msc_lib.py --selftest      # 454 checks, each with a canary
python tools/check_names.py           # no NameErrors waiting in a branch
python tools/check_links.py           # every doc reference resolves
python build_notebooks_in100.py       # regenerate + all six validation layers
```

The self-test is designed to be able to fail: it carries a deliberate canary
that must report `FAIL`, and a floor on the number of checks. A suite that
cannot fail is [D-37](docs/imagenet100/22_IN100_LAB_NOTEBOOK.md).

---

## Running the ImageNet-100 pipeline

```
NB1_Setup  →  NB2_Train  →  NB3_Measure  →  NB4_Analysis  →  NB5_Method  →  NB6_Publish
```

Full instructions, including the close-without-saving step that Jupyter makes
necessary, are in
[`docs/imagenet100/24_IN100_STATUS.md`](docs/imagenet100/24_IN100_STATUS.md) §4.

Requirements: `requirements.txt`. Runs fully offline; nothing is downloaded at
run time. Artifacts stay on local disk and are pushed to HuggingFace only by
`NB6_Publish`.

---

## Results in one paragraph

> MSC is a reliable, three-dimensional, largely architecture-transferable
> per-sample quantity — and it does **not** yield better inference routing than
> confidence thresholding, because the oracle ceiling for MSC-based routing is
> itself flat.

Compute-need is three-dimensional in both studies (PC1 max 0.532 on CIFAR,
0.547 on ImageNet, against a pre-registered 0.60 — refuted twice). It transfers
across the convolution/attention boundary above the pre-registered bound in
both (T = 0.710 and 0.640 against a predicted < 0.6, with a shuffled control at
0.037). Seed-reliability is architecture-dependent in both, same ordering, on
non-overlapping architectures — which is what makes noise-ceiling correction a
demonstrated necessity rather than an argued one.

**The new result is the oracle ceiling.** The CIFAR study lists B11 as its
*"one substantive gap remaining"* (O-21) — without it, "did the method fail or
is the premise empty?" is unanswerable. It is now computed for 18 students:
B11 is **+0.00007** over confidence routing, so there was never headroom, and
MSC-KD's −0.0088 is a fact about the premise rather than the distillation.

Scorecard, caveats and cost of closing the gaps:
[`docs/imagenet100/26_IN100_FINDINGS.md`](docs/imagenet100/26_IN100_FINDINGS.md).
