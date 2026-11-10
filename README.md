# NP-TRBR on the public ULB benchmark

[简体中文](README_zh.md)

This repository reproduces the matched **TabM + NP-TRBR** experiments on the public ULB/Worldline credit-card fraud benchmark, [OpenML data ID 1597](https://www.openml.org/d/1597). It includes the official TabM backbone, the four gate/loss combinations, and the fixed positive-weighted BCE control. This is the version used for the manuscript's matched ULB results, distinct from the earlier local STaR/TabM-style experiments.

Only source code, configuration, tests and public aggregate reference metrics are included. The archive contains no transaction rows, fitted weights, row-level predictions or non-public datasets.

## 1. Install and check

Run the commands from the repository root. They work as single lines in PowerShell, Bash and similar terminals. Python 3.11 or 3.12 is recommended; the CPU checks used Python 3.12. Install a PyTorch build appropriate for your machine using the [PyTorch installation page](https://pytorch.org/get-started/locally/). A CUDA-enabled build is required for `--device cuda:0`; keep your existing compatible CUDA build if you already have one.

```text
python -m pip install -r requirements.txt
python reproduce_ulb.py smoke
python -m unittest discover -s tests -v
```

The smoke test runs a reduced, offline CPU experiment on synthetic data. It does not train on or download ULB.

## 2. Prepare the public split

```text
python reproduce_ulb.py prepare
python reproduce_ulb.py audit
```

`prepare` downloads OpenML 1597 without credentials. If you have a copy of the original public CSV in the same row order, use `python reproduce_ulb.py prepare --input creditcard.csv`. If you already have the evaluated prepared CSV, place it at `data/ulb_prepared.csv` and use `python reproduce_ulb.py split`. The commands do not overwrite a prepared CSV. Pass `--data` and `--split-dir` consistently if you use different locations.

| Partition | Records | Positives |
|---|---:|---:|
| Training | 182,277 | 365 |
| Early validation | 30,839 | 32 |
| Calibration | 14,730 | 20 |
| Test | 56,961 | 75 |

The only variable inputs are `V1`–`V28` and `Amount`. The label is `Class`. The evaluated export omitted `Time`. We preserve source order, do not verify transaction chronology, and never use row ID or time as a model feature. A constant date is retained as preprocessing metadata and its constant derived columns are dropped in the matched official model. Split metadata records the prepared CSV and manifest SHA-256 hashes. A partition mismatch fails instead of silently reshuffling the data.

## 3. Train the five matched arms

```text
python -u reproduce_ulb.py train --device cuda:0 --seeds 42 7 2027
python reproduce_ulb.py development --device cuda:0 --seeds 42 7 2027
```

Replace `cuda:0` with `cpu` to run on CPU; use the same device for the corresponding training and evaluation commands. The default suite fits **15 models**. To run one fit first, use:

```text
python -u reproduce_ulb.py train --device cuda:0 --arms nptrbr --seeds 42
```

| Arm | MECG gate | NP-TRBR | BCE positive weight |
|---|---|---|---:|
| `bce` | Off | Off | 1 |
| `mecg_bce` | On | Off | 1 |
| `nptrbr` | Off | On | 1 |
| `mecg_nptrbr` | On | On | 1 |
| `wbce_w2` | Off | Off | 2 |

The frozen recipe uses `tabm==0.0.3`, 32 members, three blocks of width 128, 16 piecewise-linear bins, 8-dimensional numerical embeddings, FP32, a 1:10 target positive-to-negative ratio per member, and up to 100 epochs. The early-validation AP selects up to five checkpoints per seed. The fixed weight 2 replays the manuscript's chosen control; the release does not repeat the cross-dataset weight search.

Training and development do not score the test set. The same command skips verified completed runs and restores interrupted runs from the last saved epoch boundary, subject to matching code, data, split, configuration, dependency versions and device. Use a new `--output` for a changed setup. Checkpoints and pickles must be trusted local outputs; never load artifacts from an untrusted source.

## 4. Evaluate and build tables

After the fits are complete, explicitly unlock the retrospective test evaluation:

```text
python reproduce_ulb.py evaluate --device cuda:0 --seeds 42 7 2027 --unlock-test
python summarize_ulb.py
```

The evaluator averages checkpoint probabilities within each seed, then averages seed probabilities. It fits Platt calibration and thresholds **only on the calibration split**. The four reported rules are calibration FPR caps of 0.1% and 0.5%, calibrated probability 0.5, and maximum F1 on calibration. Observed test FPR can differ from the calibration cap. AP and AUROC use uncalibrated scores and are not tied to a threshold.

- `runs/ulb_paper/development_metrics.csv`: out-of-fold calibration diagnostics.
- `runs/ulb_paper/evaluation/test_metrics.csv`: per-seed and ensemble AP, AUROC, TP/FP/TN/FN, precision, recall, F1 and FPR.
- `runs/ulb_paper/evaluation/test_ensemble_metrics.csv`: ensemble-only results.
- `runs/ulb_paper/evaluation/paired_intervals.csv`: 2,000 paired, class-stratified row-bootstrap draws for the specified loss contrasts, conditional on the fitted models and frozen thresholds. These intervals do not establish equivalence or future deployment performance.
- `tables/ulb/`: ranking, decision and matched factorial CSVs generated by `summarize_ulb.py`.

For a single-arm check, pass the same `--arms` and `--seeds` to evaluation and add `--bootstrap-repeats 0 --report-dir runs/ulb_one_arm`. Use a distinct `--report-dir` for any changed evaluation selection or bootstrap configuration; the fitted runs stay in `--output`.

[Archived ULB aggregates](reference_results/) are reference values, not predictions bundled with the code. The paper's NP-TRBR ensemble reached AP **0.792128** and AUROC **0.987409**; matched BCE reached **0.798344** and **0.988100**. The paper's exact numbers are not asserted for every library version or GPU. To regenerate summary tables directly from the archived aggregates:

```text
python summarize_ulb.py --metrics reference_results/test_ensemble_metrics.csv --output tables/ulb_reference
```

## 5. Optional separately fitted baselines

These runners use the same public split but their own model, weighting and checkpoint recipes. Do not substitute their scores for the matched loss/gate contrasts. To keep this distinction visible, they write to separate directories.

```text
python -m pip install -r requirements-baselines.txt
python run_extra_baselines.py --models random_forest xgboost lightgbm ft_transformer official_tabm --device cuda:0 --tree-device cuda --unlock-test
python -m mecg_star_tabm suite --models logistic,catboost,vanilla_tabm --seeds 42,7,2027 --run-mode final --device cuda:0 --output-root runs/ulb_legacy_baselines --resume
```

For the extra baselines on CPU, pass both `--device cpu --tree-device cpu`. Omit `--unlock-test` for the extra baselines to train and score development only; for the legacy CLI use `--run-mode tune` with a separate output root. The earlier local 32-member STaR recipe is also available as `--models full --config mecg_star_tabm/configs/legacy_k32.json` with the legacy CLI. Its command reruns the frozen configuration, not the preceding tuning/confirmation history. The standalone official TabM and local `vanilla_tabm` runners are not the matched `bce` control.

## Attribution and release

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset and method citations. Upstream packages and public data are obtained separately. Before making the repository public, confirm the project copyright notice and release approval with the rights holders.
