# Project Roadmap

The **Safety-Critical Battery Prognostics** engine is under active development. Below is our strategic outline for future iterations.

---

## ✅ Completed Milestones

### v1.0 & v2.0 - Core Engine Implementation
- [x] Micro-Macro Time-Scale Decoupling architecture.
- [x] SPM-FDM Microscopic Electrochemical Sandboxing.
- [x] Physical Feature Extractors & Empirical Degradation Priors.
- [x] Macroscopic RUL Prediction Networks (PINNs).
- [x] High-Order Differentiable Physical Constraints.
- [x] Sub-millisecond ONNX Exportation & Quantization.
- [x] Conformal Quantile Regression (CQR) Uncertainty Framework.
- [x] LLM-Assisted FMEA Diagnostic Agent (ISO 26262 Aligned).
- [x] SOTA Refactoring (Semi-Implicit Backward Euler Solvers).

---

## 🚧 In Progress

### v2.1 - DX (Developer Experience) & Hardening
- [ ] Improved Command Line Interface (CLI) configuration overrides.
- [ ] Stricter schema validation for `configs/` JSON/YAML.
- [ ] Live telemetry and hardware-in-the-loop progression bars.

### v2.2 - Universal Dataset Integration
- [ ] Abstraction layers for diverse battery chemistries (LFP, NCM).
- [ ] LG Chem Dataset Integration.
- [ ] Samsung Dataset Integration.
- [ ] Panasonic Dataset Integration.
- [ ] Unified multi-dataset `DataLoader` abstractions.

---

## 🔮 Future Architecture Vision

### v3.0 - Algorithmic Expansions
- [ ] Multi-Task Learning frameworks (simultaneous SOC & SOH tracking).
- [ ] Domain Adaptation and Transfer Learning bounds.
- [ ] Active Learning strategies for data-sparse edge scenarios.
- [ ] Knowledge Distillation from foundation models (e.g., Chronos) to ultra-light edge networks.

### v3.1 - Advanced Electrochemical Twinning
- [ ] Pseudo-Two-Dimensional (P2D) partial differential equation coupling.
- [ ] Deep Electrochemical-Thermal aging models.
- [ ] Dynamic parameter identifiability routines.

### v3.2 - Edge & Cloud Ecosystem
- [ ] Full TensorRT dynamic batching support.
- [ ] TFLite integration for Cortex-M bare-metal deployment.
- [ ] Real-time MQTT telemetry dashboards.
- [ ] Edge-to-Cloud Federated Learning hooks.

---

## 💡 Open Issues (Contributions Welcomed!)

### Algorithmic Research
- [ ] Advancing Epistemic vs. Aleatoric uncertainty separation.
- [ ] Extreme Out-of-Distribution (OOD) anomaly clustering.
- [ ] Few-shot meta-learning approaches for novel battery modules.

### Engineering & MLOps
- [ ] Distributed multi-GPU data parallelism via `DistributedDataParallel` (DDP).
- [ ] Automatic hyperparameter tuning via Optuna.
- [ ] Advanced CI/CD hardware testing runners on ARM instances.

---

## 🎯 Long-Term Vision

### Academic Leadership
- Establish this engine as the de facto SOTA baseline for all PINN-based battery life prediction literature.
- Curate open, standardized benchmarks and leaderboards for prognostics research.

### Industrial Viability
- Achieve fully certifiable ASIL-D functional safety modules.
- Provide plug-and-play BMS integrations for Tier-1 automotive software stacks.
- Maintain stability and zero-allocation efficiency for >1,000,000 edge assets.

---

## 📬 Feedback & Prioritization

We heavily weigh community input when selecting the next milestone:
1. **Community Demand**: (Monitored via GitHub Discussions and Issue upvotes).
2. **Technological Value**: Direct impact on industrial robustness or academic rigor.
3. **Vision Alignment**: Adherence to our safety-critical and decoupled-physics ethos.

If you have proposals, please open a PR or a Discussion thread!
