# 📖 API Documentation & Module Reference

This document provides a detailed reference for the core classes and functions in the `safety-critical-battery-prognostics` library.

---

## 1. Physics Engine (`src.physics`)

### `PyTorchSPM` (in `src.physics.electrochemistry.spm`)
*SOTA Differentiable Single Particle Model.*

- **`__init__(n_shells=10, trainable=False, device="cpu")`**
    - `n_shells`: Number of radial discretization shells.
    - `trainable`: If `True`, diffusion coefficients ($D_s$) are optimized during training.
- **`forward(c_anode, c_cathode, j_n_anode, j_n_cathode, dt)`**
    - Performs one time step using a **Semi-Implicit Backward Euler** solver.
    - Returns: `(updated_c_anode, updated_c_cathode)`.

### `PhysicsModel` (in `src.physics.aging.degradation`)
*Empirical Capacity Fade Prior.*

- **`fit(cycles, capacities, battery_id="global")`**
    - Fits the empirical model $Q(n) = Q_0 - a\sqrt{n} - bn$.
- **`predict(cycles, battery_id="global")`**
    - Returns the predicted capacity baseline.

---

## 2. Deep Learning Models (`src.models`)

### `PINNModel` (in `src.models.pinn_model`)
*Physics-Informed Neural Network with Adaptive Loss Weighting.*

- **`fit(X, y, **kwargs)`**
    - Trains the model to predict residuals $y - Physics(n)$.
    - Features: Adaptive loss weighting (Sigmoid schedule) and physics calibration.
- **`predict(X)`**
    - Performs inference with **MC Dropout**.
    - Returns: `(mean, lower_bound, upper_bound)`.

### `ChronosZeroShotModel` (in `src.models.chronos_model`)
*Foundation Model for Battery Forecasting.*

- **`load_pretrained(model_name="amazon/chronos-t5-small")`**
    - Loads the pre-trained forecasting weights.
- **`forecast(history, horizon)`**
    - Returns probabilistic forecasts for future capacity.

---

## 3. Uncertainty & Safety (`src.uncertainty`, `src.safety`)

### `OODDetector` (in `src.uncertainty.bayesian.ood_detector`)
*Out-of-Distribution Detection.*

- **`fit(X_train, train_stds)`**
    - Learns the reference distribution using Mahalanobis distance and Epistemic surge baselines.
- **`detect(X_test, test_stds)`**
    - Categorizes samples as `IN_DISTRIBUTION`, `BORDERLINE`, or `OUT_OF_DISTRIBUTION`.

### `FMEAAgent` (in `src.safety.fmea.llm_agent`)
*LLM-based Safety Diagnostics.*

- **`detect_anomalies(...)`**
    - Monitors physics tensors for mechanical stress or concentration gradient breaches.
- **`generate_fmea_report(trigger)`**
    - Queries the LLM to generate an ISO 26262 compliant Failure Mode report.

---

## 4. Training Utilities (`src.training`)

### `TrainingPipeline` (in `src.training.pipeline`)
*Unified Experiment Logic.*

- **`train_and_evaluate(df, model, seed=42)`**
    - Executes full **Leave-One-Battery-Out (LOGO) cross-validation**.
    - Logic: Auto-checkpointing, error logging, and metric aggregation.
