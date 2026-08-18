# Minimum Sufficient Compute (MSC)

**How much compute does *this* sample actually need, and is that a property of
the sample or of the seed?**

MSC is a per-sample, cost-normalised, multi-axis, stability-closed measure of
the compute a trained network needs to reach its final decision. Two studies
live in this repository:

| study | dataset | scale | architectures | status |
|---|---|---|---|---|
| **CIFAR-100** | 50k @ 32px | small | 14 | complete — [`docs/cifar100/10_FINAL_RESULTS.md`](docs/cifar100/10_FINAL_RESULTS.md) |
| **ImageNet-100** | 130k @ 224px | 40× data, 49× pixels | 2 (pilot) + 3 students | pilot complete — [`docs/imagenet100/24_IN100_STATUS.md`](docs/imagenet100/24_IN100_STATUS.md) |

---

## Start here

**If you want the current state of the ImageNet-100 work**, read
[`docs/imagenet100/24_IN100_STATUS.md`](docs/imagenet100/24_IN100_STATUS.md).
It opens with what is proven, what is not, and why.

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
   — defects D-37…D-81 from the ImageNet port.

---

## Layout

```
├── src/msc_lib.py              the pipeline: data, zoo, training, measurement
│                               ONE library, parameterised by dataset
├── msc_core.py                 the reference maths -- the MSC definition and
│                               every statistic in the paper. Imported, never
│                               reimplemented.
├── build_notebooks.py          generates notebooks/       (CIFAR-100, 14)
├── build_notebooks_in100.py    generates notebooks_in100/ (ImageNet-100, 5)
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
├── docs/imagenet100/           20-25  port plan, delta, lab notebook, status
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
python src/msc_lib.py --selftest      # 445 checks, each with a canary
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
NB1_Setup  →  NB2_Train  →  NB3_Measure  →  NB4_Analysis  →  NB5_Method
```

Full instructions, including the close-without-saving step that Jupyter makes
necessary, are in
[`docs/imagenet100/24_IN100_STATUS.md`](docs/imagenet100/24_IN100_STATUS.md) §4.

Requirements: `requirements.txt`. Runs fully offline; nothing is downloaded at
run time. Artifacts stay on local disk and are pushed to HuggingFace only by
`NB6_Publish`.

---

## Results in one paragraph

On CIFAR-100, non-convolutional models had markedly lower seed-reliability
(ρ_seed ≈ 0.547) than CNNs (0.62–0.72). On ImageNet-100 the **ordering
replicates** on non-overlapping architectures — `resnet50` 0.822 versus
`vit_small_p16` 0.649 — but no architecture appears in both studies, so
cross-study *magnitudes* are confounded by architecture, resolution and dataset
together. Disattenuated transfer between the CNN and the ViT is
T = 0.640 [0.614, 0.664] against a shuffled control of 0.037. MSC-KD is a clean
negative: at matched FLOPs (ρ = 0.806) it sits 0.9 points *below* confidence
thresholding in 18/18 runs, and the oracle ceiling — routing by the student's
own true post-hoc MSC — offers only +0.00007 over confidence, so the limitation
is the premise rather than the distillation.

Numbers, caveats and what would close the gaps:
[`docs/imagenet100/24_IN100_STATUS.md`](docs/imagenet100/24_IN100_STATUS.md).
