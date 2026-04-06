<div align="center">

# 🔋 Physics-Shielded Battery Prognostics

### Micro-Macro Time-Scale Decoupled PINN for Safety-Critical RUL Prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-67%2F67%20Passing-brightgreen.svg)](tests/)
[![Violation Rate](https://img.shields.io/badge/Physics%20Violations-0.00%25%20(3--Layer%20Defense)-success.svg)]()
[![GitHub stars](https://img.shields.io/github/stars/Zhi-Chao-PAN/safety-critical-battery-prognostics?style=social)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics)

*A three-layer physics defense achieving **0.00% physical violation rate**¹ on 6 real CALCE battery cells under extreme noise — validated without hyperparameter retuning.*

<sub>¹ 0.00% VR is achieved with the complete three-layer defense architecture (Constraint Training + Residual Clamping + Monotonic Projection). See [Section V.E](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) for ablation results and [Section V.K](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) for fair comparison with identical post-processing applied to all baselines.</sub>

[📄 Paper Draft](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) · [📊 Full Results](docs/comprehensive_experimental_results.md) · [🇨🇳 简体中文](README_zh.md)

</div>

---

## 🎯 The Problem

In safety-critical Battery Management Systems (BMS), a single **non-physical prediction** — such as forecasting capacity *increase* during battery aging — can trigger catastrophic failures: false range estimates in EVs, premature grid storage shutdowns, or missed thermal runaway warnings.

Pure data-driven models (LSTM, Transformer) routinely produce such violations under sensor noise. **This project eliminates them entirely.**

---

## 🛡️ Three-Layer Physics Defense Architecture

Our core contribution is a **cascading physics shield** that guarantees monotonic capacity degradation predictions, even under 50% Gaussian noise:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Constraint Training                           │
│  → Embeds physics prior via differentiable penalty      │
│  → Soft regularization during optimization              │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Residual Clamping                             │
│  → Bounds NN residual to training-time observed range   │
│  → Prevents OOD explosion (RMSE ↓77%)                  │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Monotonic Projection                          │
│  → EMA smoothing (α=0.15) + running-minimum             │
│  → Hard guarantee: 0.00% physical violations            │
└─────────────────────────────────────────────────────────┘
```

### Why three layers?

Each layer addresses a **distinct failure mode**. Our ablation study proves they are complementary and non-redundant:

| Defense Configuration | RMSE (Ah) | Violation Rate | Role |
|----------------------|-----------|---------------|------|
| No Defense | 1.748 | 50.75% | Baseline (catastrophic) |
| + Constraint Training | 3.348 | 48.24% | Weak regularization |
| + Residual Clamping | **0.759** | 48.74% | **Accuracy** (RMSE ↓77%) |
| + Monotonic Projection | 2.589 | **0.00%** | **Safety** guarantee |
| **Full Defense (Ours)** | **0.323** | **0.00%** | **Best of both** |

> **Key Insight**: Removing clamping degrades accuracy (RMSE 2.589). Removing projection breaks safety (VR ~48%). You need all three.

---

## 📊 Experimental Results

### Robustness: PINN vs LSTM under 50% Gaussian Noise

| Metric | PINN (Ours) | LSTM Baseline |
|--------|:-----------:|:-------------:|
| Physical Violation Rate | **0.00%** ✅ | 18.55% ❌ |
| Inference Latency | **11 ms** ⚡ | 2,230 ms |
| Speed Advantage | **203× faster** | — |

### Cross-Cell Generalization (6 Real CALCE Batteries)

Validated on real-world data **without any hyperparameter retuning**:

| Cell | Cycles | PINN VR | LSTM VR |
|------|--------|---------|---------|
| CS2_33 | 864 | **0.00%** ✅ | 47.97% |
| CS2_34 | 774 | **0.00%** ✅ | 49.29% |
| CS2_35 | 932 | **0.00%** ✅ | 48.87% |
| CS2_36 | 970 | **0.00%** ✅ | 48.30% |
| CS2_37 | 1,037 | **0.00%** ✅ | 49.52% |
| CS2_38 | 1,076 | **0.00%** ✅ | 49.95% |

> **6/6 cells at 0.00% violation rate with three-layer defense.** The physics shield is not an artifact of synthetic data — see [fairness validation (Section V.K)](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) for identical post-processing applied to all baselines.

### Computational Efficiency

| Metric | Value |
|--------|-------|
| Peak VRAM | 8.14 MB |
| ONNX INT8 Inference | < 0.1 ms |
| AMP Training Speedup | 2× (Tensor Core) |
| MC Dropout Speedup | 100× (Batched) |

---

## 🏗️ Architecture

```
                    ┌──────────────────────┐
                    │   Raw Battery Data    │
                    │  (V, I, T, cycles)    │
                    └─────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
   ┌──────────────────┐            ┌──────────────────┐
   │  Micro-Scale SPM  │           │  Macro-Scale NN   │
   │  (Intra-cycle)    │           │  (Inter-cycle)    │
   │                    │           │                    │
   │  Fick's Diffusion  │──feat──▶ │  TCN + Attention  │
   │  FDM Sandbox       │           │  Adaptive λ(t)    │
   └──────────────────┘            └────────┬─────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              ▼              ▼              ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────┐
                        │ Layer 1  │  │ Layer 2  │  │ Layer 3  │
                        │Constraint│→ │ Clamp    │→ │ Project  │
                        │Training  │  │ Residual │  │ Monotone │
                        └──────────┘  └──────────┘  └──────────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │  Safe RUL Pred  │
                                    │  0.00% VR ✅    │
                                    └────────────────┘
```

### Key Innovations

1. **Micro-Macro Time-Scale Decoupling** — Resolves the "time-scale black hole" by isolating fast SPM dynamics (seconds) from slow degradation prediction (months)
2. **Adaptive Physics Loss Weighting** — Sigmoid-scheduled λ(t) trusts data early, physics late
3. **Three-Layer Physics Shield** — Cascading defense guaranteeing 0% physical violations
4. **Batched MC Dropout** — 100× faster uncertainty quantification via tensor expansion
5. **AMP Training** — 2× speedup with 41% VRAM reduction on RTX 4060

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics

# Install dependencies
pip install -r requirements.txt

# Run basic demonstration
python main.py

# Run robustness test (PINN vs LSTM under 50% noise)
python robustness_test.py

# Run defense layer ablation study
python scripts/ablation_defense_layers.py

# Run real-world CALCE validation
python scripts/validate_real_data.py

# Run unit tests
python -m pytest tests/ -v
```

---

## 📁 Repository Structure

```text
safety-critical-battery-prognostics/
├── src/                        # Core source code
│   ├── models/                 #   PINN, LSTM, Chronos, Online Adapter
│   ├── physics/                #   Differentiable SPM, constraint system
│   ├── training/               #   Mixed precision, LOGO-CV
│   ├── data/                   #   Data ingestion & normalization
│   ├── evaluation/             #   Metrics & performance profiling
│   ├── uncertainty/            #   Conformal prediction, MC Dropout
│   ├── safety/                 #   LLM-FMEA diagnostic engine
│   ├── deployment/             #   ONNX export & quantization pipeline
│   └── infrastructure/         #   Config schema, dataset management
├── scripts/                    # Experiment scripts
│   ├── ablation_defense_layers.py  # Defense layer ablation (5 variants)
│   ├── validate_real_data.py       # Cross-cell CALCE validation
│   ├── run_ablation_study.py       # Architecture ablation
│   └── ...
├── robustness_results/         # All robustness experiment outputs
│   ├── ablation_defense_layers.png # IEEE-grade ablation figure
│   ├── real_data_validation.png    # 12-panel cross-cell figure
│   └── *.md, *.csv                 # Reports and raw data
├── data/                       # NASA + CALCE datasets
├── tests/                      # 67 unit tests (100% passing)
├── docs/                       # Documentation & paper draft
├── configs/                    # YAML configurations (schema + experiments)
└── robustness_test.py          # Main robustness pipeline
```

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Comprehensive Results](docs/comprehensive_experimental_results.md) | Full 10-section experimental report |
| [Project Progress](docs/project_progress.md) | 16 milestones with metrics |
| [IEEE Paper Draft](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) | Full IEEE Transactions-style paper |
| [Architecture Guide](docs/PROJECT_ARCHITECTURE.md) | System design documentation |
| [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md) | Edge BMS deployment SOP |

---

## 🧪 Reproducibility

All experiments are fully reproducible with fixed random seeds:

```bash
# Reproduce defense ablation (Table III in paper)
python scripts/ablation_defense_layers.py
# → robustness_results/ablation_defense_layers.png
# → robustness_results/ablation_defense_report.md

# Reproduce cross-cell validation (Table IV in paper)
python scripts/validate_real_data.py
# → robustness_results/real_data_validation.png
# → robustness_results/real_data_validation_report.md
```

---

## 📄 Citation

```bibtex
@software{pan2026pinn_battery,
  author = {Pan, Zhichao},
  title = {Physics-Shielded Battery Prognostics: Micro-Macro Time-Scale
           Decoupled PINN with Three-Layer Defense},
  year = {2026},
  url = {https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics},
  note = {0.00\% physical violation rate on 6 real CALCE cells under 50\% noise}
}
```

---

## 📬 Contact

- **Author**: Zhichao Pan
- **Email**: [18652585856@163.com](mailto:18652585856@163.com)
- **GitHub**: [@Zhi-Chao-PAN](https://github.com/Zhi-Chao-PAN)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*If this project advances your research, please consider giving it a ⭐*

**0.00% physical violations (three-layer defense) · 203× faster inference · 6/6 real cells validated**

</div>
