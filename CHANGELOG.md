# 📜 CHANGELOG

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-03-26

**The "SOTA Hardening" Release**

### 🚀 Added
- **Differentiable Physics**: Re-implemented SPM with `nn.Parameter` support for online calibration.
- **Adaptive Loss Weighting**: PINN model now dynamically trusts physics over data during "knee" degradation regions.
- **OOD Detection**: Integrated Mahalanobis distance and Epistemic surge monitoring for safety-critical uncertainty quantification.
- **ISO 26262 Proof-of-Concept**: Added FMEA LLM Agent for real-time failure mode analysis.
- **Chronos-PINN Integration**: Support for Amazon Chronos foundation models in the physics-informed pipeline.

### 🛡️ Changed
- **Numerical Solver**: Upgraded SPM from Explicit Euler to **Semi-Implicit (Backward Euler Diagonal)** for unconditional stability.
- **Flux Discretization**: Improved surface flux accuracy using **2nd-order Taylor expansion**.
- **Repository Standard**: Migrated all dependencies to a unified `requirements.txt` and `pyproject.toml`.
- **Project Structure**: Cleaned up experimental artifacts and synchronized `autodl-tmp` logic into core `src`.

### 🧹 Removed
- Legacy redundant files in `autodl-tmp/`.
- Hardcoded API keys in `llm_agent.py` (migrated to `.env`).

### 🔧 Fixed
- Silent convergence failures in low-state-of-charge (SoC) regions.
- Memory leaks in long-running Streamlit sessions.
- Inconsistent naming conventions in the training pipeline.
