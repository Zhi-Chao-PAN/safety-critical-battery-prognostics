# Multi-Baseline Robustness Benchmark Report

## Experimental Setup
- **Noise Level**: 50% Gaussian (σ_noise = 0.5 × σ_feature)
- **Data**: 200 synthetic battery degradation cycles
- **Seed**: 42 (fixed for reproducibility)
- **Models**: 6 (1 physics-constrained PINN + 5 data-driven baselines)

## Results

| Model | Type | RMSE (Ah) | Violation Rate | Violations | Latency (ms) | Train (s) |
|-------|------|-----------|---------------|------------|-------------|-----------|
| PINN (Ours) | physics | 0.5603 | ✅ 0.00% | 0 | 13 | 5.2 |
| LSTM | data-driven | 0.0571 | ❌ 45.23% | 90 | 970 | 4.3 |
| GRU | data-driven | 0.0712 | ❌ 40.70% | 81 | 967 | 4.5 |
| Transformer | data-driven | 0.3800 | ❌ 53.77% | 107 | 952 | 1.2 |
| TCN | data-driven | 0.9375 | ❌ 60.30% | 120 | 1061 | 3.8 |
| CNN1D | data-driven | 0.0701 | ❌ 49.25% | 98 | 301 | 1.7 |

## Key Findings

1. **Only PINN achieves 0% violation rate** — All 5 data-driven baselines produce physical violations (TCN worst at 60.3%).

2. **Best RMSE**: LSTM (0.0571 Ah). However, this comes at the cost of physical violations.

3. **Fastest inference**: PINN (Ours) (13 ms).

## Conclusion

The PINN's three-layer physics defense is the **only architecture** that guarantees zero physical violations under 50% sensor noise. All data-driven baselines — regardless of architecture (recurrent, attention, convolutional) — produce non-physical capacity rebounds that are unacceptable in safety-critical BMS deployments.