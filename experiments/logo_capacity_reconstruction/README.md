# Training-seed audit: held-out-cell capacity reconstruction

This directory reports a bounded negative result from the repository at
commit `bf53f86805ed1637b5ad1dc3d014976be71415b8`.

The experiment used three training seeds (42, 123, 456) and six leave-one-cell-out
(LOGO) folds over repository snapshots named CS2_33 through CS2_38. In each fold,
PINN and LSTM models were trained on five clean capacity trajectories and evaluated
on the sixth trajectory after fixed Gaussian corruption (severity 0.5; corruption
seed 42). Both outputs received the same EMA (`alpha=0.15`) followed by a running
minimum.

## What the experiment measures

The held-out trajectory's corrupted capacity is an input feature. This is therefore
capacity reconstruction/denoising on six fixed snapshots. It is **not** future
capacity forecasting, RUL prediction, a complete reproduction of a CALCE study,
or evidence of deployment safety. The snapshots were inherited from this repository;
this audit did not independently rebuild them from the original CALCE archive.

## Negative result

| Model | RMSE over 18 seed-folds, mean ± SD (Ah) | Mean monotonicity-violation rate |
|---|---:|---:|
| LSTM | 0.2221 ± 0.0352 | 0.00% |
| PINN | 0.9814 ± 1.6640 | 0.00% |

PINN mean RMSE varied sharply by training seed: 0.3124 (seed 42), 2.3953
(seed 123), and 0.2365 (seed 456). LSTM means were 0.2216, 0.2224, and 0.2222.
The worst PINN fold was seed 123 / CS2_37 (RMSE 4.9901 Ah), versus 0.2198 Ah
for LSTM in the same fold. These observations do not support PINN superiority
under this protocol.

The 0.00% violation rate is also not a PINN-specific safety result. The shared
running-minimum transform deterministically prevents upward steps for both models.

## Published artifacts

- `protocol.json`: fixed protocol and interpretation boundary.
- `per_fold_metrics.csv`: 36 aggregate fold/model rows; no point-level records.
- `summary_by_seed.csv` and `summary_by_model.csv`: deterministic aggregates.
- `verification.json`: checks performed against the retained local detailed run.
- `verify_published_summary.py`: validates row structure and recomputes both summaries.

This audit directory adds no copies of raw CSVs, point-level predictions, training
losses, model weights, or machine-local paths. The pre-existing repository still
contains `data/calce/*.csv`; the SHA-256 digests in `protocol.json` identify which
six of those snapshots were used.

## Reproduce with separately obtained data

Use Python 3.10+ and the repository dependencies. The repository's existing
`scripts/validate_real_data_logo.py` contains the underlying single-seed training
path. Validate the published aggregate-only package with:

```bash
python experiments/logo_capacity_reconstruction/verify_published_summary.py \
  --results experiments/logo_capacity_reconstruction
```

The detailed private audit retained point-level predictions and independently
recalculated all 36 fold/model metrics. Those point records are intentionally not
published. Cross-host numerical identity has not been established.
