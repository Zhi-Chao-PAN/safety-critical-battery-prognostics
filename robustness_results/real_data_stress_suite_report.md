# Real-Data Multi-Seed Corruption Stress Suite

## Experimental Setup

- **Protocols**: same-cell noise robustness and LOGO cross-cell validation
- **Cells**: CALCE CS2_33-CS2_38
- **Training seed**: 42 for each protocol fold
- **Seeds**: 42, 123, 456, 789, 1024
- **Corruptions**: Gaussian noise, bias drift, impulse spikes, missing segments
- **Severity**: 50% scale relative to per-cell capacity variability
- **Reporting**: seed-level cell averages summarized as mean ± std across seeds

## Same-Cell Noise Robustness

| Corruption | PINN RMSE (mean ± std) | PINN VR (mean ± std) | LSTM RMSE (mean ± std) | LSTM VR (mean ± std) | Hardest Fold |
|------------|------------------------|----------------------|------------------------|----------------------|--------------|
| Gaussian Noise | 0.4012 ± 0.0081 | 0.00% ± 0.00% | 0.2160 ± 0.0002 | 0.00% ± 0.00% | CS2_36 (seed 456, RMSE 1.2406) |
| Bias Drift | 0.3979 ± 0.0028 | 0.00% ± 0.00% | 0.2158 ± 0.0003 | 0.00% ± 0.00% | CS2_36 (seed 123, RMSE 1.2171) |
| Impulse Spikes | 0.3941 ± 0.0036 | 0.00% ± 0.00% | 0.2158 ± 0.0002 | 0.00% ± 0.00% | CS2_36 (seed 456, RMSE 1.2129) |
| Missing Segments | 0.3980 ± 0.0050 | 0.00% ± 0.00% | 0.2158 ± 0.0003 | 0.00% ± 0.00% | CS2_36 (seed 1024, RMSE 1.2288) |

## LOGO Cross-Cell Validation

| Corruption | PINN RMSE (mean ± std) | PINN VR (mean ± std) | LSTM RMSE (mean ± std) | LSTM VR (mean ± std) | Hardest Fold |
|------------|------------------------|----------------------|------------------------|----------------------|--------------|
| Gaussian Noise | 0.2537 ± 0.0038 | 0.00% ± 0.00% | 0.2225 ± 0.0003 | 0.00% ± 0.00% | CS2_33 (seed 1024, RMSE 0.4221) |
| Bias Drift | 0.2572 ± 0.0075 | 0.00% ± 0.00% | 0.2224 ± 0.0002 | 0.00% ± 0.00% | CS2_33 (seed 42, RMSE 0.4253) |
| Impulse Spikes | 0.2499 ± 0.0045 | 0.00% ± 0.00% | 0.2225 ± 0.0001 | 0.00% ± 0.00% | CS2_33 (seed 123, RMSE 0.3779) |
| Missing Segments | 0.2520 ± 0.0043 | 0.00% ± 0.00% | 0.2226 ± 0.0002 | 0.00% ± 0.00% | CS2_33 (seed 456, RMSE 0.3751) |

## Interpretation

- Same-cell and LOGO are reported separately to prevent protocol leakage in the repository narrative.
- Multi-seed statistics show whether the real-data conclusions are stable or just artifacts of one random corruption draw.
- Additional corruption families are reported as stress tests, not as replacements for the baseline Gaussian protocol.