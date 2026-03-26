<div align="center">

# 🔋 Micro-Macro Time-Scale Decoupling for Battery RUL Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python)]()
[![GitHub stars](https://img.shields.io/github/stars/Zhi-Chao-PAN/safety-critical-battery-prognostics?style=social)]()

*A State-of-the-Art Production-Ready Prognostics Engine for Academic Research and Industrial BMS Edge Deployment.*

[🇨🇳 简体中文 (Simplified Chinese)](README_zh.md)

</div>

---

## 📚 Project Overview

This repository explores a novel paradigm for predicting the Remaining Useful Life (RUL) of lithium-ion batteries. The core methodology bridges physics-informed priors and data-driven dynamics via **Micro-Macro Time-Scale Decoupling**, ensuring both numerical stability and industrial-grade safety compliance.

### ✨ Key Innovations
- **Physics-Informed Neural Networks (PINNs)**: A hybrid architecture incorporating Semi-Implicit Backward Euler Solvers tailored for robust electrochemical dynamics.
- **Adaptive Loss Weighting**: Dynamic sigmoid-scheduled balancing of physics priors and neural residuals.
- **Safety-Critical Compliance**: ISO 26262 ASIL-C level hazard mitigation with an LLM-assisted Failure Mode diagnostic agent.
- **Uncertainty Quantification**: Conformal Prediction bounds for Aleatoric and Epistemic risk evaluation.
- **Edge Native**: Native ONNX export pipeline optimized for Cortex-M and NVIDIA Jetson inference requirements (<0.1ms latency).

---

## 🚀 Quick Start

### 5-Minute Sandbox (Demonstration)

A sterile, pre-configured sandbox environment that **guarantees execution**:

```bash
# 1. Clone the repository
git clone https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics

# 2. Install zero-conflict dependencies
pip install -r requirements.txt

# 3. Launch the sandbox demonstration
python main_simple.py
```

*This demonstration synthesizes dynamic battery cycling data, initializes a baseline physics model, and renders point-prediction results directly.*

### Full Integration Suite
For rigorous empirical analysis:
1. Download the targeted testbed datasets (NASA or CALCE).
2. Configure data ingestion paths inside `configs/`.
3. Reference the comprehensive guides under the `docs/` directory.

---

## 📚 Official Documentation

Detailed architectural and deployment guides are available in the `docs/` tree:
- [**API & Module Reference**](docs/technical/API_REFERENCE.md) - Core classes and differentiable solvers.
- [**Directory Architecture**](docs/DIRECTORY_STRUCTURE.md) - System layout and functional definitions.
- [**Edge Deployment SOP**](docs/deployment/DEPLOYMENT_GUIDE.md) - TensorRT optimization and hardware constraints.
- [**ISO 26262 Safety Case**](docs/industrial/ISO26262_Safety_Case.md) - Functional safety analysis and FMEA mitigations.

---

## 📁 Repository Structure

```text
safety-critical-battery-prognostics/
├── src/                    # Core source codebase
│   ├── data/               # Ingestion and normalization
│   ├── models/             # Neural network definitions (PINN, Chronos)
│   ├── features/           # Temporal feature engineering
│   ├── training/           # Unification pipelines (LOGO-CV)
│   ├── evaluation/         # Strict performance metrics
│   ├── uncertainty/        # OOD and Epistemic bounds
│   ├── physics/            # Differentiable SPM & degradation modules
│   ├── safety/             # LLM-FMEA & diagnostic engines
│   ├── deployment/         # ONNX compilation scripts
│   └── utils/              # Auxiliary utilities
├── data/                   # Target datasets directory
├── docs/                   # Documentation and reports
├── notebooks/              # Interactive Jupyter playgrounds
├── tests/                  # Unit & Integration test suites
├── figures/                # Auto-generated visualization assets
├── main_simple.py          # Minimal reproducible entry point
├── main.py                 # Full experimental pipeline (Legacy)
├── pyproject.toml          # Strict Python project configurations
└── README.md               # You are here
```

---

## 📄 Academic Citation

If you utilize this repository for academic research or industrial baselines, please cite this work:

```bibtex
@software{pan2026battery,
  author = {Pan, Zhichao},
  title = {Micro-Macro Time-Scale Decoupling for Battery RUL Prediction},
  year = {2026},
  url = {https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics}
}
```

---

## 🤝 Contribution Guidelines

We adhere to a strict zero-regression policy. Open-source contributions are broadly welcomed!
- [CONTRIBUTING.md](CONTRIBUTING.md) - Developer setup and PR guidelines.
- [FAQ.md](docs/FAQ.md) - Frequently Discussed Questions.
- [ROADMAP.md](docs/ROADMAP.md) - Project Evolution Milestones.

---

## 📬 Academic & Professional Inquiries

- **Email**: [18652585856@163.com](mailto:18652585856@163.com)
- **Issues**: [GitHub Issue Tracker](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/discussions)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<i>A production-grade, SOTA formulation architecture. If you find this engine useful, please consider giving it a ⭐ Star!</i>
</div>
