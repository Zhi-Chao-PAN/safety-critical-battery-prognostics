# Technical Report: Uncertainty-Aware Battery Health Management
**A Comparative Study of Deep Learning and Hierarchical Bayesian Inference for Safety-Critical Systems**

<p style="text-align: center; font-size: 14pt; margin-top: -20px;">
    <strong>Zhichao Pan</strong>
</p>

## Abstract
Prognostics and Health Management (PHM) for Lithium-ion batteries is critical for the safety of electric vehicles and aerospace systems. While deep learning models like LSTMs provide state-of-the-art accuracy, they typically lack intrinsic uncertainty quantification. This report evaluates a **Hierarchical Bayesian Degradation Model** against a deterministic **LSTM baseline**. Our results, based on the NASA PCoE dataset, demonstrate that the Bayesian approach captures *epistemic* uncertainty more effectively, providing an average "Safety Buffer" width of **32.2 cycles** (vs. 9.1 cycles for LSTM), significantly reducing the risk of silent catastrophic failure in out-of-distribution scenarios.

---

## 1. Introduction
Modern energy storage systems require high-fidelity Remaining Useful Life (RUL) predictions. Traditional deterministic models provide point estimates ($\hat{y}$), which can be dangerously overconfident. This research implements a probabilistic framework compliant with **ISO 26262** safety standards, focusing on the trade-off between *precision* and *risk awareness*.

## 2. Methodology

### 2.1 Deterministic Baseline: LSTM + MC Dropout
We implemented a Long Short-Term Memory (LSTM) network with Monte Carlo (MC) Dropout to approximate Bayesian inference.
*   **Architecture**: 2-layer LSTM (64 units) -> Dropout (0.2) -> Linear Head.
*   **Inference**: 50 forward passes with dropout enabled to estimate predictive variance.
*   **Limitation**: Captures *aleatoric* uncertainty but struggles with *epistemic* uncertainty (novel degradation patterns).

### 2.2 Proposed: Hierarchical Bayesian Degradation Model
We model the battery degradation path as a generative hierarchical process, implemented in **PyMC**.
*   **Likelihood**: $y_{ij} \sim \mathcal{N}(\alpha_j + \mathbf{x}_{ij}^\top \boldsymbol{\beta}, \sigma)$
*   **Partial Pooling**: Battery-specific intercepts $\alpha_j$ are drawn from a population distribution $\alpha_j \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha)$.
*   **Advantage**: "Borrows strength" from the population to regularize predictions for individual batteries, preventing overfitting to noise.
*   **Inference**: No-U-Turn Sampler (NUTS) with 2,000 samples.

## 3. Empirical Results

### 3.1 Quantitative Metrics (Leave-One-Group-Out CV)
The table below summarizes the performance across 4 test batteries.

| Metric | Hierarchical Bayesian (Proposed) | LSTM (MC Dropout) | Implication |
| :--- | :---: | :---: | :--- |
| **B0006 RMSE** | **8.95** | 28.01 | Bayesian adapts faster to healthy battery curves. |
| **B0018 RMSE** | **18.85** | 19.56 | Comparable accuracy on complex aging profiles. |
| **Start-of-Life RMSE (B0005)** | 25.47 | 33.52 | Bayesian prior stabilizes early predictions. |
| **Avg. Uncertainty Width** | **32.23 cycles** | 9.05 cycles | **Safety Critical Finding**: Bayesian model provides a 3.5x wider safety buffer. |

### 3.2 Key Findings: The "Safety Gap"
*   **B0006 Success Case**: The Hierarchical Bayesian model achieved near-perfect calibration (95.4% HDI Coverage), while the LSTM failed to capture the degradation trend (11.4% Coverage).
*   **B0007 Failure Mode**: Both models struggled with B0007 (an anomaly), but the Bayesian model correctly signaled complete ignorance (Width ~30) whereas the LSTM remained confidently wrong (Width ~8).

**Figure 1** below visualizes the "Safety Buffer Zone" on Test Battery `B0018`. The Green Zone (Bayesian HDI) correctly expands during key regeneration spikes (Cycle 60-80), protecting the system from false positives.

![Figure 1: Comparison of Bayesian vs LSTM Model](results/rigor/comparison_B0018_generated.png)

## 4. Discussion & Conclusion
The LSTM model is highly effective for capturing non-linear temporal dynamics when training data is abundant. However, for **Safety-Critical Industrial AI** with limited samples ($N=4$), the Hierarchical Bayesian model is superior because:
1.  **Risk Aversion**: It prefers to be "vaguely right" (wide intervals) rather than "precisely wrong."
2.  **Data Efficiency**: Hierarchical priors stabilize learning on small datasets.
3.  **Explainability**: Variance components ($\sigma_\alpha$, $\sigma$) directly map to manufacturing variability and sensor noise.

## 5. Future Work
*   **Hybrid Models**: Combining LSTM feature extraction with Bayesian output layers (Bayesian LSTM).
*   **Online Learning**: Implementing Variational Inference (VI) for real-time parameter updating on embedded BMS chips.
