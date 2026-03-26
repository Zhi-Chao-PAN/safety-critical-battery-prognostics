# 📂 Project Directory Structure

This document provides a high-level overview of the repository hierarchy to assist with navigation, architectural compliance, and research integration.

```text
safety-critical-battery-prognostics/
├── 📂 configs/             # YAML configurations for experiments & model hyperparameters
├── 📂 data/                # Dataset storage (NASA B-series, CALCE)
│   └── 📂 battery_data/    # Raw .mat and .csv telemetry
├── 📂 deployment/          # Edge deployment scripts (ONNX, TensorRT)
├── 📂 docs/                # Technical documentation, API specs, and FMEA reports
├── 📂 experiments/         # Standalone research scripts and ablation studies
├── 📂 scripts/             # Utility scripts (ETL pipelines, Benchmarking, Plotting)
├── 📂 src/                 # Main source codebase
│   ├── 📂 models/          # SOTA Prognostics Models (PINN, Chronos, etc.)
│   ├── 📂 physics/         # Electrochemical Models (SPM, FDM solvers)
│   ├── 📂 safety/          # ISO 26262 FMEA Agents and Decision Logic
│   ├── 📂 training/        # Training pipelines and unified loss scheduling
│   ├── 📂 evaluation/      # Generalization and strict performance metrics
│   └── 📂 uncertainty/     # Bayesian NNs and OOD detection bounds
├── 📂 tests/               # Unit, integration, and safety-critical tests (PyTest)
├── 📂 notebooks/           # Interactive Jupyter analytical playgrounds
├── 📂 results/             # Checkpoints, cached artifacts, and exported metrics
├── 📄 main_simple.py       # Minimal reproducible sandbox entry point
├── 📄 main.py              # Central entry point for full-pipeline execution (Legacy)
├── 📄 pyproject.toml       # Strict PEP compliant build configurations and dependency locking
├── 📄 requirements.txt     # Consolidated runtime dependencies
└── 📄 .env.example         # Template for contextual environment variables
```

## 🧠 Core Modules Breakdown

### 1. `src/physics/electrochemistry`
Houses the **PyTorch-based Single Particle Model (SPM)**.
- **Innovation**: Employs an unconditionally stable **Semi-Implicit Backward Euler** solver, fused with fully differentiable tensors (`nn.Parameter`) for dynamic online calibration of solid-phase diffusion coefficients ($D_s$).

### 2. `src/models/pinn_model.py`
Fuses the rigorous physics priors with highly parameterized Deep Learning networks.
- **Innovation**: Features an algorithmic **Adaptive Loss Weighting** mechanism via Sigmoid scheduling. It dynamically enforces physics dominance during early cycles, transferring trust to data-driven residuals during the highly non-linear "knee" degradation phase.

### 3. `src/safety/fmea`
The LLM-based intelligent safety diagnostic agent.
- **Function**: Translates raw numerical abnormalities (e.g., predicted mechanical stress fractures, lithium plating concentration gradients) into formal, human-readable **ISO 26262 FMEA** hazard reports.

### 4. `src/uncertainty`
Engineers the bounds for Epistemic and Aleatoric uncertainty mapping.
- **Function**: Operates deep **Out-of-Distribution (OOD)** boundary detection utilizing Mahalanobis distance embeddings to intercept extrapolation failures in sparse operational profiles.
