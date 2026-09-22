<div align="center">

# 🔋 Battery Capacity Modeling Research Audit

### Bounded synthetic benchmarks and held-out-cell reconstruction experiments

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/Zhi-Chao-PAN/safety-critical-battery-prognostics?style=social)](https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics)

*Research code for capacity-space models, monotonic post-processing, and explicitly bounded robustness audits.*

The strongest current real-data result is a negative one: across 3 training seeds × 6 held-out-cell folds, PINN error is unstable and does not beat LSTM. Both models reach zero upward-step violations only after the same deterministic running-minimum transform. See the [training-seed audit](experiments/logo_capacity_reconstruction/README.md).

[📊 Training-seed audit](experiments/logo_capacity_reconstruction/README.md) · [🧾 Claim-evidence matrix](docs/claim_evidence_matrix.md) · [🇨🇳 简体中文](README_zh.md)

> **Repository note**: current claim boundaries live in this README, the [training-seed audit](experiments/logo_capacity_reconstruction/README.md), and the [Claim-Evidence Matrix](docs/claim_evidence_matrix.md). Older reports are retained for provenance and may contain stronger historical wording.

</div>

---

## 🎯 Research question

Monotonic capacity outputs can be desirable in some degradation models, but enforcing monotonicity is not equivalent to accurate forecasting or operational safety.

This repository studies training constraints, residual clamping, and post-processing. Its real-cell experiments consume the observed capacity trajectory as an input, so they measure reconstruction/denoising rather than future capacity or RUL forecasting.

---

## Three-stage experimental pipeline

The inherited pipeline combines a training penalty, residual clamping, and a deterministic running-minimum projection. The last step enforces non-increasing output by construction within the evaluated sequence; it is not a system-safety guarantee.

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
│  → Deterministic non-increasing output on each sequence │
└─────────────────────────────────────────────────────────┘
```

### Synthetic ablation retained from the original benchmark

The following single-seed synthetic ablation is kept as a benchmark result, with no claim of real-world generalization:

| Defense Configuration | RMSE (Ah) | Violation Rate | Role |
|----------------------|-----------|---------------|------|
| No Defense | 1.748 | 50.75% | Baseline |
| + Constraint Training | 3.348 | 48.24% | Weak regularization |
| + Residual Clamping | **0.759** | 48.74% | **Accuracy** (RMSE ↓77%) |
| + Monotonic Projection | 2.589 | **0.00%** | Deterministic output constraint |
| **Full pipeline** | **0.323** | **0.00%** | Reported synthetic result |

> This table alone does not establish necessity across seeds, cells, datasets, or deployment conditions.

---

## 📊 Experimental Results

### Robustness: PINN vs LSTM under 50% Gaussian Noise

These are retained single-environment synthetic benchmark measurements and were not revalidated by the training-seed audit:

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

### LOGO held-out-cell reconstruction (single training seed)

The repository now includes an executed leave-one-cell-out validation on the same 6 CALCE cells:

```bash
python scripts/validate_real_data_logo.py
```

This protocol trains on all non-held-out clean cells and reconstructs the held-out cell while its observed capacity remains an input feature. It does not test future forecasting.

| Condition | PINN Avg RMSE | PINN Avg VR | LSTM Avg RMSE | LSTM Avg VR |
|-----------|---------------|-------------|---------------|-------------|
| Clean held-out cell | 0.2497 | 0.00% | 0.2223 | 0.00% |
| 50% noisy held-out cell | 0.2615 | 0.00% | 0.2232 | 0.00% |

> **Bounded interpretation**: both models are monotone after shared deterministic post-processing, while PINN trails LSTM on RMSE.

### Training-seed audit (3 seeds × 6 held-out cells)

| Model | RMSE over 18 seed-folds, mean ± SD (Ah) | Mean violation rate after shared projection |
|---|---:|---:|
| LSTM | 0.2221 ± 0.0352 | 0.00% |
| PINN | 0.9814 ± 1.6640 | 0.00% |

PINN mean RMSE rises to 2.3953 at seed 123, versus 0.2224 for LSTM. This does not support PINN superiority or stable initialization behavior under this protocol. Aggregate-only artifacts and verification code are in [`experiments/logo_capacity_reconstruction`](experiments/logo_capacity_reconstruction/README.md).

### Multi-Seed Corruption Stress Suite

The repository now also includes a seeded stress-suite report across 5 corruption seeds and 4 corruption families for both same-cell and LOGO protocols:

| Protocol | PINN RMSE Range Across Corruptions | LSTM RMSE Range Across Corruptions | Shared VR |
|----------|------------------------------------|------------------------------------|-----------|
| Same-cell | 0.3941-0.4012 | 0.2158-0.2160 | 0.00% for both |
| LOGO | 0.2499-0.2572 | 0.2224-0.2226 | 0.00% for both |

See [real_data_stress_suite_report.md](robustness_results/real_data_stress_suite_report.md) for the per-corruption `mean ± std` tables and hardest-fold breakdowns.

### Earlier computational measurements

These values come from earlier repository reports and are environment-specific; they are not deployment qualification evidence:

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
                                    │ Postprocessed  │
                                    │ capacity output│
                                    └────────────────┘
```

### Implemented components

1. **Micro-Macro Time-Scale Decoupling** — Separates fast SPM features from slower capacity modeling
2. **Adaptive Physics Loss Weighting** — Implements a sigmoid-scheduled loss weight
3. **Three-stage pipeline** — Training constraint, residual clamp, and monotonic projection
4. **Batched MC Dropout** — Implements tensor-expanded uncertainty sampling
5. **AMP Training** — Provides a mixed-precision training path

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

# Run repository-snapshot reconstruction validation (data acquired separately)
python scripts/validate_real_data.py

# Run held-out-cell reconstruction validation
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
├── data/                       # Existing repository snapshots and data instructions
├── tests/                      # Automated tests
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

The repository provides fixed-seed scripts. Exact cross-host numerical identity has not been established:

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

**Research code · bounded claims · negative results reported**

</div>
