# 📂 Project Directory Structure & Module Guide

This document provides a high-level overview of the repository structure to assist with navigation and research integration.

```text
.
├── 📂 configs/             # YAML configurations for experiments & model hyperparameters
├── 📂 data/                # Dataset storage (NASA B-series, CALCE)
│   └── 📂 battery_data/    # Raw .mat and .csv telemetry
├── 📂 deployment/          # Edge deployment scripts (ONNX, TensorRT)
├── 📂 docs/                # Technical documentation, API specs, and FMEA reports
├── 📂 experiments/         # Jupyter notebooks and standalone research scripts
├── 📂 scripts/             # Utility scripts (ETL pipelines, Benchmarking, Plotting)
├── 📂 src/                 # Main source code
│   ├── 📂 models/          # SOTA Prognostics Models (PINN, TCN, Chronos, etc.)
│   ├── 📂 physics/         # Electrochemical Models (SPM, FDM solvers)
│   ├── 📂 safety/          # ISO 26262 FMEA Agents and Decision Logic
│   ├── 📂 training/        # Training pipelines and loss functions
│   └── 📂 uncertainty/     # Bayesian NNs and OOD detection logic
├── 📂 tests/               # Unit and integration tests (PyTest)
├── 📄 main.py              # Central entry point for full-pipeline execution
├── 📄 requirements.txt     # Consolidated & version-locked dependencies
└── 📄 .env.example         # Template for API keys and environment variables
```

## 🧠 Key Modules

### 1. `src/physics/electrochemistry`
Contains the **PyTorch-based SPM (Single Particle Model)**. 
- **Innovation**: Semi-Implicit solver for numerical stability and fully differentiable parameters for online calibration.

### 2. `src/models/pinn_model.py`
Integrates the physics model with Deep Learning.
- **Innovation**: Adaptive loss weighting that dynamically trusts physics over data during the "knee" region of battery aging.

### 3. `src/safety/fmea`
LLM-based diagnostic agent.
- **Function**: Translates raw physics tensors (mechanical stress, concentration gradients) into human-readable ISO 26262 FMEA reports.

### 4. `src/uncertainty`
Handles epistemic and aleatoric uncertainty.
- **Function**: Implements OOD (Out-of-Distribution) detection to prevent over-confident predictions in safety-critical scenarios.
