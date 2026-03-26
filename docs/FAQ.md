# Frequently Asked Questions (FAQ)

## General Information

### Q: What is the purpose of this project?
A: This is an open-source prognostics engine designed for Lithium-ion Battery Remaining Useful Life (RUL) prediction. It bridges physics-informed neural networks (PINNs) and data-driven methods through a novel **Micro-Macro Time-Scale Decoupling** architecture to achieve highly efficient and reliable battery life forecasting.

### Q: Who is the target audience?
A:
- 🎓 **Academic Researchers**: Scholars and students investigating battery health management and prognostic algorithms.
- 🏭 **Industrial Engineers**: BMS (Battery Management System) engineers aiming for edge deployment.
- 💻 **AI Enthusiasts**: Developers interested in Physics-Informed ML and time-series uncertainty quantification.

### Q: What are the core academic innovations?
A:
1. **Micro-Macro Time-Scale Decoupling**: Solves the OOM (Out of Memory) problem typical of PINNs simulating over thousands of cycles, reducing VRAM usage to ~8.14MB.
2. **Hard Physical Constraints**: 100% guarantee that prognostic predictions lie within thermodynamically feasible boundaries.
3. **Edge-Readiness**: ONNX INT8 inference latency is measured at just 0.078ms natively.
4. **Distribution-Free Uncertainty**: Provides mathematically guaranteed 95% coverage intervals using Conformal Quantile Regression (CQR).
5. **Safety Diagnostic Framework**: Integrates an ISO 26262-aligned Failure Mode and Effects Analysis (FMEA) system via an LLM agent.

---

## Installation & Environment

### Q: What are the hardware requirements?
A:
- **Minimum**: 8GB RAM, modern multi-core CPU.
- **Recommended for Training**: 16GB RAM, NVIDIA GPU (>= 4GB VRAM).
- **Edge Deployment Target**: Compatible with low-power architectures (Raspberry Pi, NVIDIA Jetson, STM32).

### Q: Which Python versions are supported?
A: Python 3.8+ (3.10 or 3.11 is highly recommended).

### Q: I'm encountering installation dependency errors. How do I fix this?
A:
1. Ensure you are operating within an isolated virtual environment (`venv` or `conda`).
2. Upgrade your package manager: `pip install --upgrade pip`.
3. If encountering CUDA-related conflicts, you may fall back to the CPU version of PyTorch.
4. For persistent errors, please open a GitHub Issue attaching the complete traceback and your `pip freeze` output.

---

## Usage & Execution

### Q: Where can I obtain the datasets?
A:
- **NASA Dataset**: [NASA Prognostics Center of Excellence](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)
- **CALCE Dataset**: [UMD CALCE Battery Group](https://web.calce.umd.edu/batteries/data.htm)

### Q: How do I train the model on a custom battery dataset?
A:
1. Reformat your proprietary data to match the tensor shapes expected by the dataloader (reference the templates in `data/`).
2. Utilize `src/data/validator.py` to assert data integrity and check for NaN or infinite values.
3. Update the data paths within the global configurations.
4. Follow the specific steps illustrated in the codebase documentation.

### Q: How long does the training pipeline take?
A:
- On an NVIDIA RTX 4060: **~5 minutes**.
- On a CPU (e.g., Intel Core i7): **~30-60 minutes**.
- *Note:* Thanks to the decoupled time-scale architecture, our training convergence is exponentially faster than traditional sequential PINNs.

### Q: I am facing Out-Of-Memory (OOM) errors during training. What should I adjust?
A:
1. Decrease the `sequence_length` context window.
2. Reduce the `batch_size`.
3. Enable mixed-precision training (FP16/BF16).
4. Our baseline VRAM blueprint is exceptionally low (8.14MB); if you encounter OOM on a standard GPU, please file a bug report.

---

## Technical Deep-Dive

### Q: What exactly is the "Time-Scale Black Hole" problem?
A:
- Microscopic electrochemical processes (Li+ diffusion) occur on the order of **seconds**.
- Macroscopic capacity degradation occurs over **months or years**.
- Directly bridging these scales using traditional recurrent solvers triggers massive Backpropagation Through Time (BPTT) computational graphs, leading to immediate memory explosions. We bypass this via hierarchical decoupling.

### Q: How are the hard physical constraints enforced?
A: We overlay differentiable clamping layers and thermodynamic boundary conditions acting as inductive priors:
$$ 0 < C_{pred} \le C_{nominal} $$

### Q: Why utilize Conformal Prediction over Bayesian Neural Networks (BNNs)?
A:
- ✅ Mathematically guaranteed finite-sample coverage without distributional assumptions.
- ✅ Orders of magnitude faster to compute than MCMC or variational ensembles.
- ✅ Conceptually simpler to validate for industrial functional safety audits.

---

## Contribution & Community

### Q: How can I contribute to the codebase?
A: Please review our [CONTRIBUTING.md](../CONTRIBUTING.md) file which outlines the PR workflow and coding standards in detail.

### Q: How should I report a Bug?
A:
1. Search the existing GitHub Issues to prevent duplicates.
2. Use the standard Bug Report template.
3. Supply exact reproduction steps, environment context, and full tracebacks.

---

## Academic & Publication

### Q: How do I cite this project in my research?
A: Please reference the "Academic Citation" section located in the main `README.md`, or utilize the structured `CITATION.cff` metadata file.

### Q: Can I leverage this architecture for my own academic publications?
A: Absolutely! This repository is fully open-sourced under the MIT License for academic and industrial utilization. We kindly request that you cite our baseline repository in your manuscript.

---

## Other Inquiries

### Q: Is this repository actively maintained?
A: Yes. Please view the [ROADMAP.md](ROADMAP.md) to track our upcoming release milestones.

### Q: How can I communicate directly with the author(s)?
A:
- Open a **GitHub Issue** for bugs.
- Start a **GitHub Discussion** for architectural questions.
- Email the primary maintainer directly (found in the README).
