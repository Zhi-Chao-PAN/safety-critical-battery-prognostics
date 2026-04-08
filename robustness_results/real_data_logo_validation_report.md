# Real-World CALCE LOGO Cross-Cell Robustness Report

## Experimental Setup
- **Dataset**: CALCE CS2 series lithium-ion batteries
- **Protocol**: leave-one-cell-out; train on all other clean cells and evaluate on the held-out cell
- **Conditions**: clean held-out trajectory and 50% Gaussian noisy held-out trajectory
- **Defense**: Full three-layer physics shield with identical post-processing for all models
- **Seed**: 42

## Clean Results

| Cell | Cycles | PINN RMSE | PINN VR | LSTM RMSE | LSTM VR |
|------|--------|-----------|---------|-----------|---------|
| CS2_33 | 864 | 0.3459 | 0.00% | 0.2799 | 0.00% |
| CS2_34 | 774 | 0.1892 | 0.00% | 0.1797 | 0.00% |
| CS2_35 | 932 | 0.2437 | 0.00% | 0.1988 | 0.00% |
| CS2_36 | 970 | 0.2634 | 0.00% | 0.2498 | 0.00% |
| CS2_37 | 1037 | 0.2357 | 0.00% | 0.2194 | 0.00% |
| CS2_38 | 1076 | 0.2207 | 0.00% | 0.2063 | 0.00% |
| **Average** | - | **0.2497** | **0.00%** | **0.2223** | **0.00%** |

## Noisy Results

| Cell | Cycles | PINN RMSE | PINN VR | LSTM RMSE | LSTM VR |
|------|--------|-----------|---------|-----------|---------|
| CS2_33 | 864 | 0.4263 | 0.00% | 0.2796 | 0.00% |
| CS2_34 | 774 | 0.1869 | 0.00% | 0.1806 | 0.00% |
| CS2_35 | 932 | 0.2372 | 0.00% | 0.2007 | 0.00% |
| CS2_36 | 970 | 0.2589 | 0.00% | 0.2508 | 0.00% |
| CS2_37 | 1037 | 0.2250 | 0.00% | 0.2201 | 0.00% |
| CS2_38 | 1076 | 0.2349 | 0.00% | 0.2073 | 0.00% |
| **Average** | - | **0.2615** | **0.00%** | **0.2232** | **0.00%** |

## Key Findings

- **Clean**: PINN average VR = 0.00% (RMSE 0.2497) vs LSTM average VR = 0.00% (RMSE 0.2223).
- **Noisy**: PINN average VR = 0.00% (RMSE 0.2615) vs LSTM average VR = 0.00% (RMSE 0.2232).