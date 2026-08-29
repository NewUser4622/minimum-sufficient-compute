# Results — every headline number, generated from the artifacts

**Do not hand-edit.** Regenerate with `python tools/build_results.py`.
Machine-readable copy: [`RESULTS.csv`](RESULTS.csv).

Source CSVs are on HuggingFace at
[`Shanmuk4622/msc-cifar100`](https://huggingface.co/datasets/Shanmuk4622/msc-cifar100)
under `analysis/`, byte-identical to the local copies.

## Study 2 — oracle ceiling

| metric | scope | value | source |
|---|---|---|---|
| confidence baseline | CIFAR-100, 90 seed pairs, rho=0.80 | **62.39 %** | `s2_true_oracle.csv` |
| full compute | CIFAR-100, 90 seed pairs, rho=0.80 | **71.21 %** | `s2_true_oracle.csv` |
| oracle, in-seed | CIFAR-100, 90 seed pairs, rho=0.80 | **78.3 %** | `s2_true_oracle.csv` |
| oracle, cross-seed | CIFAR-100, 90 seed pairs, rho=0.80 | **54.5 %** | `s2_true_oracle.csv` |
| in-seed - baseline | CIFAR-100, 90 seed pairs, rho=0.80 | **12.2 pt** | `s2_true_oracle.csv` |
| cross-seed - baseline | CIFAR-100, 90 seed pairs, rho=0.80 | **-7.9 pt** | `s2_true_oracle.csv` |
| optimism bias | CIFAR-100, 90 seed pairs, rho=0.80 | **22.41 pt** | `s2_true_oracle.csv` |
| early-right/final-wrong pool | CIFAR-100, 90 seed pairs, rho=0.80 | **6.86 pt** | `s2_true_oracle.csv` |
| oracle ABOVE own full-compute accuracy | 90/90 runs | **6.86 pt** | `s2_true_oracle.csv` |

## Study 2 — reliability atlas

| metric | scope | value | source |
|---|---|---|---|
| rho_seed range across the grid | 15 archs x 5 scores | **0.667 rho** | `s2_reliability_grid.csv` |
| rho_seed minimum | mixer_nano / entropy | **0.207 rho** | `s2_reliability_grid.csv` |
| rho_seed maximum | mobilenetv2 / ce_loss | **0.874 rho** | `s2_reliability_grid.csv` |

## Study 2 — memorisation collapse

| metric | scope | value | source |
|---|---|---|---|
| Spearman(softmax saturation, rho_seed drop) | 15 architectures | **0.832 rho** | `s2_memorisation.csv` |
| Spearman(train accuracy, rho_seed drop) | 15 architectures | **0.746 rho** | `s2_memorisation.csv` |
| Spearman(TEST accuracy (control), rho_seed drop) | 15 architectures | **-0.114 rho** | `s2_memorisation.csv` |
| largest rho_seed drop, test -> train_holdout | convnext_femto | **0.558 rho** | `s2_memorisation.csv` |

## Study 3 — Q1 joint exits

| metric | scope | value | source |
|---|---|---|---|
| excess, resnet20 | frozen | **6.69 pt** | `s3_q1_comparison.csv` |
| excess, resnet20 | JOINT | **10.64 pt** | `s3_q1_comparison.csv` |
| excess, resnet32x4 | frozen | **6.42 pt** | `s3_q1_comparison.csv` |
| excess, resnet32x4 | JOINT | **8.55 pt** | `s3_q1_comparison.csv` |
| excess, vgg8 | frozen | **7.95 pt** | `s3_q1_comparison.csv` |
| excess, vgg8 | JOINT | **9.15 pt** | `s3_q1_comparison.csv` |
| H1: excess >= 2.0 pt under joint training | 3/3 architectures | **9.15 pt** | `s3_q1_comparison.csv` |
| change frozen -> joint (raw median) | 3 architectures | **2.13 pt** | `s3_q1_comparison.csv` |

## Study 3 — Q2 learned router

| metric | scope | value | source |
|---|---|---|---|
| capture fraction, in-seed | 3 architectures, rho=0.80 | **2.36 %** | `s3_router_capture.csv` |
| capture fraction, cross-seed | 3 architectures, rho=0.80 | **1.73 %** | `s3_router_capture.csv` |
| oracle gap available to be captured | median over architectures | **11.62 pt** | `s3_router_capture.csv` |

## Study 3 — Q3 pruning

| metric | scope | value | source |
|---|---|---|---|
| target accuracy, rand | keep 30% | **19.16 %** | `s3_pruning.csv` |
| target accuracy, sat | keep 30% | **16.46 %** | `s3_pruning.csv` |
| target accuracy, uns | keep 30% | **7.92 %** | `s3_pruning.csv` |
| target accuracy, rand | keep 50% | **25.58 %** | `s3_pruning.csv` |
| target accuracy, sat | keep 50% | **25.05 %** | `s3_pruning.csv` |
| target accuracy, uns | keep 50% | **14.76 %** | `s3_pruning.csv` |
| target accuracy, full | keep 100% | **70.24 %** | `s3_pruning.csv` |
