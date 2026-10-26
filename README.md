# Sparse private regression experiments

Implementation of U2R, 2-SLR, LDP-IHT, SampleSDP, and non-private Lasso, with entry points for the frozen synthetic and real-data experiments.

## Installation

Use Python 3.13 and the pinned dependencies. Run commands from this directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -B -m scripts.check
```

## Synthetic experiments

```bash
python -B -m scripts.run_synthetic --study all --out outputs/synthetic --plan-only
python -B -m scripts.run_synthetic --study all --out outputs/synthetic --resume --workers 2
```

Individual studies are `privacy`, `users`, `dimension`, `support_m`, and `support_n`. The complete grid and ordered seeds are in `configs/experiments.json`. The privacy-budget and user-count error studies use the same 30 seeds; the dimension and support studies use ten seeds. The user-count error grid is 8000, 10000, 12000, 14000, 16000, 18000, and 20000. The separate support-recovery user-count grid retains 15000.

Default settings are n=15000, m=50, d=256, k=8, residual standard deviation 0.25, epsilon=4, and delta=1e-5. Nonzero coefficients equal 1/k. Training and test samples use separate random streams. Each job runs all five estimators. All requested seeds and finite outcomes are retained.

Use `--seeds 123` for a separate trial. An existing output directory requires `--resume` and an identical manifest. The full grid is computationally demanding; set `--workers` to fit the available CPU and RAM. Outputs include raw and aggregate CSV files.

## Real-data experiment

The task predicts nacelle temperature from contemporaneous Kelmarsh sensor measurements. It is not forecasting or evaluation on unseen turbines. Data are obtained from the pinned [Kelmarsh SCADA publisher record](https://zenodo.org/records/16807551), subject to its license.

```bash
python -B -m scripts.download_kelmarsh --raw-dir data/kelmarsh/raw
python -B -m scripts.prepare_kelmarsh --work-dir data/kelmarsh
python -B -m scripts.run_real --data-dir data/kelmarsh/nacelle_temperature_v2 --out outputs/kelmarsh --plan-only
python -B -m scripts.run_real --data-dir data/kelmarsh/nacelle_temperature_v2 --out outputs/kelmarsh --resume --workers 2
```

The downloader checks publisher file sizes (when supplied) and CSV headers. Preparation assigns whole UTC dates to disjoint public preprocessing, validation, training, and test subsets. Imputation, scaling, and feature-domain choices use public preprocessing data. Public validation supplies parameters; Lasso cross-validation groups training dates. The target and duplicate target summaries are excluded from predictors.

The grid uses n=10000 simulated users, m in {10,50,100}, epsilon in {1,4}, and the same ten seeds for all methods. Training draws and the Lasso denominator are paired across methods. Real-data estimator settings are in `FROZEN_PARAMETERS.json`. SampleSDP uses Ridge stabilization with multiplier 4. All requested seeds are retained without score-based exclusions.

The `--data-dir` input is checked for the expected arrays, dimensions, row identities, and date-disjoint splits. Input-file sizes and CRC32 values are computed when a run starts and saved in its output manifest; resumption requires matching values. The test split is used for metric evaluation after fitting, and all requested seeds are retained.

## Privacy and outputs

U2R and 2-SLR use user-level units; LDP-IHT and SampleSDP retain their record-level units. Sample-level epsilon is not divided by m. Approximate-DP mechanisms use delta=1e-5. Equal epsilon across different privacy units and trust models is not an equal-guarantee comparison.

These are research simulations. Saved seeds, coefficients, caches, and diagnostic outputs are not an end-to-end private release. The real-data U2R support cap, SampleSDP Ridge stabilization, and LDP-IHT gradient clipping are explicit adaptations. Real observations are not asserted to satisfy every utility-theorem assumption.

`scripts.run_real` writes per-cell JSON, `results.csv`, `summary.csv`, and `completion.json`. Scores include MSE, RMSE, MAE, R-squared, and MSE relative to the paired non-private Lasso. The unit suite also checks fitting/evaluation boundaries and result retention. Numerical regression tests generate seeded synthetic matrices. Tests covering the Clarabel fallback require the pinned `clarabel` dependency.

## Layout and attribution

- `paper_suite/`, `lasso_threshold_v1/`, `dp_safety/`: frozen estimators and privacy-accounting primitives.
- `exp5/`: real-data adapters and solvers.
- `scripts/`: supported download, preparation, checking, and experiment commands.
- `configs/`: fixed grids, seeds, and input-file requirements.
- `tests/`, `paper_suite/tests/`: regression and boundary checks.
- `validation/`: optional synthetic diagnostics and parameter checks.
- `tests/real/synthetic_cases.py`: deterministic synthetic numerical regression inputs.

`VERSION_LOCK.json` checks packaged file sizes and CRC32 values. CRC32 detects many accidental file changes; it does not authenticate files or protect against deliberate modification. Random-stream names are encoded directly as numeric seed components, so repeated runs of this version remain deterministic. The supported workflows are the `scripts` commands above.

The archive contains source code and configurations; it does not contain raw or prepared datasets, credentials, execution logs, caches, or Git metadata. Numerical regression tests use deterministic synthetic inputs. Dataset attribution and third-party licenses are provided in `THIRD_PARTY_NOTICES.md`.

## Optional external input registration

The generic `paper_suite.run_real` interface accepts explicit `--csv` and `--target` arguments. Optional registered inputs for the general dispatcher use an external JSON registry specified by `SDP_DATASET_REGISTRY`, with entries of the form `{"example": {"format": "csv", "path": "/path/to/input.csv", "target": "response"}}`. The pinned Kelmarsh protocol is run through `scripts.run_real`.

## Reproducibility boundaries

Prepared Kelmarsh arrays are generated by the download and preparation scripts. Per-run file-size/CRC32 checks and split identities provide basic consistency checks. Estimator parameters and experiment seeds are specified in `FROZEN_PARAMETERS.json` and `configs/experiments.json`.
