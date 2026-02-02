# Technical Report: Uncertainty Quantification in Battery Health Management

### 1. Abstract
This study investigates the limitations of deterministic Deep Learning in prognostic applications. While LSTMs achieve low RMSE on training data, they exhibit **overconfidence** on out-of-distribution (OOD) test data (Battery B0018). We propose a Bayesian framework to quantify this epistemic uncertainty.

### 2. Methodology
We utilized the NASA PCoE dataset (Run-to-Failure).
* **Data Processing**: Raw voltage/current curves were processed into cycle-level health indicators.
* **Model A (Baseline)**: A 2-layer LSTM with Dropout ($p=0.2$). Loss function: MSE.
* **Model B (Proposed)**: A Hierarchical Bayesian Model.
    $$RUL_{ij} \sim \mathcal{N}(\mu_{ij}, \sigma)$$
    $$\mu_{ij} = \alpha_{j} + \beta \cdot Features_{ij}$$
    Where $\alpha_j$ allows each battery to have its own baseline health (Partial Pooling).

### 3. Critical Analysis of Results
Testing on Battery B0018 revealed:
1.  **Non-Linearity Handling**: The LSTM adapts well to the general trend but struggles with capacity regeneration (local peaks).
2.  **Safety Margins**: The Bayesian model's **95% Credible Interval** consistently bracketed the true RUL, even when the mean prediction deviated.
3.  **Conclusion**: For safety-critical applications, optimizing for **Coverage Probability** (via Bayes) is superior to optimizing for pure Accuracy (RMSE).

### 4. Future Directions
* Integration of Bayesian Neural Networks (BNN) using Variational Inference.
* Deployment on Edge AI using probabilistic layers.
