# Technical Report: Uncertainty-Aware Battery Health Management
**A Comparative Study of Deep Learning and Hierarchical Bayesian Inference**

<div class="author-block">
    <strong>Author: Zhichao Pan</strong>
</div>

## Abstract
Prognostics and Health Management (PHM) for Lithium-ion batteries is critical for the safety of electric vehicles and aerospace systems. While deep learning models like LSTMs provide state-of-the-art accuracy, they typically lack intrinsic uncertainty quantification. This report evaluates a Hierarchical Bayesian approach against a deterministic LSTM baseline. Our results demonstrate that while the LSTM achieves low RMSE, the Bayesian model provides a 100% coverage rate and a quantifiable "Safety Buffer Zone," enabling risk-aware decision-making in safety-critical applications.

---

## 1. Introduction
Modern energy storage systems require high-fidelity Remaining Useful Life (RUL) predictions. Traditional deterministic models provide point estimates ($y_{\text{pred}}$), which can be dangerously overconfident during the non-linear degradation phases of a battery. This research implements a probabilistic framework to capture both aleatoric (noise) and epistemic (model) uncertainty.

## 2. Methodology

### 2.1 Deterministic Baseline: LSTM
We implemented a Long Short-Term Memory (LSTM) network with a sliding window architecture (window size = 10). The model architecture includes:
- **Input**: Multimodal sequences (Discharge Time, Max Temperature).
- **Hidden Layers**: 2-layer LSTM (64 units each) with Dropout (0.2).
- **Output**: Linear projection to RUL.

### 2.2 Probabilistic Proposal: Hierarchical Probabilistic Degradation Model
The proposed model utilizes partial pooling to learn degradation trends across a battery population while adapting to individual battery quirks (intercepts $\alpha_j$ and slopes $\beta_j$).
- **Likelihood**: $y_{ij} \sim \mathcal{N}(\alpha_j + \mathbf{x}_{ij}^\top \boldsymbol{\beta}, \sigma)$
- **Hyperpriors**: Informative priors centered on population means to stabilize MCMC sampling.
- **Inference**: No-U-Turn Sampler (NUTS) with 2,000 samples.

## 3. Empirical Results

### 3.1 Quantitative Metrics
Evaluated on Test Battery **B0018** (Zero-shot prognosis):

| Model | Metric | Value | Implications |
| :--- | :--- | :--- | :--- |
| **LSTM** | RMSE | **36.53 Cycles** | High precision but lacks risk bounds. |
| **Bayesian** | HDI Coverage | **100.00%** | All true values captured within 95% credible interval. |

### 3.2 Qualitative Analysis: Figure 1
![Figure 1: Comparison of Bayesian vs LSTM Model](results/final_comparison_B0018.png)

The comparison plot above shows that the Bayesian High Density Interval (HDI) widens as the battery approaches its End-of-Life (EOL). This widening signals increased model ignorance, effectively creating a **"Safety Buffer Zone"** for control systems.

## 4. Discussion & Conclusion
The LSTM model is highly effective for capturing temporal trends when data is abundant. However, the Hierarchical Bayesian model is superior for **Safety-Critical Industrial AI** because:
1. It avoids "catastrophic overconfidence" at failure points.
2. It handles inter-battery variability via partial pooling.
3. It provides a mathematically rigorous basis for failsafe protocols.

In conclusion, for applications where the cost of failure is high, the Bayesian framework's ability to be "vaguely right" (accurate uncertainty) outweighs the deterministic model's "precise wrongness."
