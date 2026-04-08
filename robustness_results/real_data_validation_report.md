# Real-World CALCE Same-Cell Noise Robustness Report

## Experimental Setup
- **Dataset**: CALCE CS2 series lithium-ion batteries
- **Protocol**: train on each cell's clean trajectory, evaluate on a noisy version of the same trajectory
- **Noise Level**: 50% Gaussian (sigma_noise = 0.5 x sigma_capacity)
- **Defense**: Full three-layer physics shield with identical post-processing for all models
- **Seed**: 42

## Noisy Results

| Cell | Cycles | PINN RMSE | PINN VR | LSTM RMSE | LSTM VR |
|------|--------|-----------|---------|-----------|---------|
| CS2_33 | 864 | 0.2893 | 0.00% | 0.2771 | 0.00% |
| CS2_34 | 774 | 0.1567 | 0.00% | 0.1417 | 0.00% |
| CS2_35 | 932 | 0.2085 | 0.00% | 0.2000 | 0.00% |
| CS2_36 | 970 | 1.1494 | 0.00% | 0.2491 | 0.00% |
| CS2_37 | 1037 | 0.2847 | 0.00% | 0.2216 | 0.00% |
| CS2_38 | 1076 | 0.2205 | 0.00% | 0.2063 | 0.00% |
| **Average** | - | **0.3848** | **0.00%** | **0.2160** | **0.00%** |

## Key Findings

- **Noisy**: PINN average VR = 0.00% (RMSE 0.3848) vs LSTM average VR = 0.00% (RMSE 0.2160).