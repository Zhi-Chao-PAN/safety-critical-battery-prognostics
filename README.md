<div align="center">

# 🔋 Physics-Shielded Battery Prognostics

### Micro-Macro Time-Scale Decoupled PINN for Safety-Critical Battery Prognostics

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml)
[![Synthetic VR](https://img.shields.io/badge/Synthetic%20VR-0.00%25%20(3--Layer%20Defense)-success.svg)](docs/comprehensive_experimental_results.md)
[![GitHub stars](https://img.shields.io/github/stars/Zhi-Chao-PAN/safety-critical-battery-prognostics?style=social)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics)

*A three-layer physics defense with **0.00% physical violation rate**¹ on the synthetic robustness benchmark and bounded, fairness-matched real-data evidence on 6 CALCE cells.*

<sub>¹ The model-specific 0.00% VR headline comes from the synthetic `robustness_test.py` benchmark. The real-data same-cell, LOGO, and multi-seed corruption reports all apply identical EMA smoothing + monotonic projection to PINN and baselines; in those fairness-matched protocols, both PINN and LSTM reach 0.00% VR while PINN trails LSTM on RMSE. See [Comprehensive Results](docs/comprehensive_experimental_results.md) and the [Claim-Evidence Matrix](docs/claim_evidence_matrix.md) for protocol boundaries.</sub>

[📄 Paper Draft](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) · [📊 Full Results](docs/comprehensive_experimental_results.md) · [🇨🇳 简体中文](README_zh.md)

> **Repository note**: the authoritative, up-to-date claim boundaries for this repository live in this `README`, [Comprehensive Results](docs/comprehensive_experimental_results.md), and the [Claim-Evidence Matrix](docs/claim_evidence_matrix.md). Files under `docs/archive/` are preserved as historical deliverables.

</div>

---

## 🎯 The Problem

In safety-critical Battery Management Systems (BMS), a single **non-physical prediction** — such as forecasting capacity *increase* during battery aging — can trigger catastrophic failures: false range estimates in EVs, premature grid storage shutdowns, or missed thermal runaway warnings.

Pure data-driven models (LSTM, Transformer) routinely produce such violations under sensor noise. **This repository shows how a three-layer physics defense removes them on the synthetic robustness benchmark while keeping the real-data claims explicitly bounded and protocol-specific.**

---

## 🛡️ Three-Layer Physics Defense Architecture

Our core contribution is a **cascading physics shield** that guarantees monotonic capacity degradation predictions in the synthetic robustness benchmark under 50% Gaussian noise:

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

### Same-Cell Noise Robustness (Fairness-Matched)

The seeded rerun of `scripts/validate_real_data.py` keeps the protocol scoped to same-cell noise rejection and applies the same post-processing chain to both models:

| Condition | PINN Avg RMSE | PINN Avg VR | LSTM Avg RMSE | LSTM Avg VR |
|-----------|---------------|-------------|---------------|-------------|
| 50% noisy same-cell trajectory | 0.3848 | 0.00% | 0.2160 | 0.00% |

> **Bounded interpretation**: with identical EMA smoothing + running-minimum projection, both PINN and LSTM are monotone on all 6 real cells in this same-cell protocol. This is no longer evidence of a PINN-specific real-data safety advantage, and CS2_36 is the hardest PINN fold (RMSE 1.1494).

### LOGO Cross-Cell Validation (Held-Out Cells, Bounded Conclusion)

The repository now includes an executed leave-one-cell-out validation on the same 6 CALCE cells:

```bash
python scripts/validate_real_data_logo.py
```

This protocol trains on all non-held-out clean cells and evaluates on the held-out cell under both clean and noisy conditions. It is the correct route for real cross-cell evidence and is intentionally kept separate from the same-cell noise robustness table above.

| Condition | PINN Avg RMSE | PINN Avg VR | LSTM Avg RMSE | LSTM Avg VR |
|-----------|---------------|-------------|---------------|-------------|
| Clean held-out cell | 0.2497 | 0.00% | 0.2223 | 0.00% |
| 50% noisy held-out cell | 0.2615 | 0.00% | 0.2232 | 0.00% |

> **Bounded interpretation**: the seeded LOGO rerun again shows that both PINN and LSTM remain at 0.00% violation rate under the shared post-processing stack on held-out cells, while PINN still lags LSTM on RMSE. The hardest PINN fold is now CS2_33 rather than a uniform failure mode across cells.

### Multi-Seed Corruption Stress Suite

The repository now also includes a seeded stress-suite report across 5 corruption seeds and 4 corruption families for both same-cell and LOGO protocols:

| Protocol | PINN RMSE Range Across Corruptions | LSTM RMSE Range Across Corruptions | Shared VR |
|----------|------------------------------------|------------------------------------|-----------|
| Same-cell | 0.3941-0.4012 | 0.2158-0.2160 | 0.00% for both |
| LOGO | 0.2499-0.2572 | 0.2224-0.2226 | 0.00% for both |

See [real_data_stress_suite_report.md](robustness_results/real_data_stress_suite_report.md) for the per-corruption `mean ± std` tables and hardest-fold breakdowns.

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
3. **Three-Layer Physics Shield** — Cascading defense with 0% synthetic-benchmark physical violations
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

# Run real-world LOGO cross-cell validation
python scripts/validate_real_data_logo.py

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
│   ├── validate_real_data.py       # Same-cell robust validation (noise)
│   ├── validate_real_data_logo.py  # LOGO cross-cell validation
│   ├── validate_real_data_stress_suite.py # Multi-seed corruption suite
│   ├── run_ablation_study.py       # Architecture ablation
│   └── ...
├── robustness_results/         # All robustness experiment outputs
│   ├── ablation_defense_layers.png # IEEE-grade ablation figure
│   ├── real_data_validation.png    # 12-panel same-cell noise figure
│   ├── real_data_logo_validation.png # LOGO clean/noisy figure
│   ├── real_data_logo_validation_report.md # LOGO markdown summary
│   ├── real_data_stress_suite_report.md # Multi-seed corruption report
│   └── *.md, *.csv                 # Reports and raw data
├── data/                       # NASA + CALCE datasets
├── tests/                      # 91 automated tests (100% passing)
├── docs/                       # Documentation & paper draft
├── configs/                    # YAML configurations (schema + experiments)
└── robustness_test.py          # Main robustness pipeline
```

---

## 📄 Documentation

| Document | Description |
|----------|-------------|
| [Comprehensive Results](docs/comprehensive_experimental_results.md) | Full experimental report |
| [Claim-Evidence Matrix](docs/claim_evidence_matrix.md) | Verified claims, bounded claims, and future work |
| [Contributing Guide](CONTRIBUTING.md) | How to set up a dev environment, run checks, and submit PRs |
| [Code of Conduct](CODE_OF_CONDUCT.md) | Community expectations for respectful collaboration |
| [Security Policy](SECURITY.md) | How to report security, safety, or documentation misuse concerns |
| [Project Progress](docs/project_progress.md) | 16 milestones with metrics |
| [IEEE Paper Draft](docs/archive/IEEE_Whitepaper_PINN_Battery_RUL_Complete.md) | Full IEEE Transactions-style paper |
| [Architecture Guide](docs/PROJECT_ARCHITECTURE.md) | System design documentation |
| [Deployment Guide](docs/deployment/DEPLOYMENT_GUIDE.md) | Edge BMS deployment SOP |

Active sources of truth are the first four entries above. Archive materials remain available for provenance, but they should not be used as the primary source for current benchmark claims.

---

## 🧪 Reproducibility

All experiments are fully reproducible with fixed random seeds:

```bash
# Reproduce defense ablation (Table III in paper)
python scripts/ablation_defense_layers.py
# → robustness_results/ablation_defense_layers.png
# → robustness_results/ablation_defense_report.md

# Reproduce same-cell noise validation (Table IV in paper)
python scripts/validate_real_data.py
# → robustness_results/real_data_validation.png
# → robustness_results/real_data_validation_report.md

# Reproduce LOGO cross-cell validation
python scripts/validate_real_data_logo.py
# → robustness_results/real_data_logo_validation.png
# → robustness_results/real_data_logo_validation_report.md

# Reproduce the multi-seed corruption stress suite
python scripts/validate_real_data_stress_suite.py
# → robustness_results/real_data_stress_suite_report.md
# → robustness_results/real_data_stress_suite_summary.csv
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
  note = {0.00\% physical violation rate on the synthetic robustness benchmark; fairness-matched same-cell and LOGO real-data reports are included separately}
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

**0.00% synthetic-benchmark violations · 203× faster inference · same-cell, LOGO, and stress-suite reports included**

</div>
