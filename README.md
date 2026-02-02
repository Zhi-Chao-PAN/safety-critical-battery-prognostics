# Safety-Critical Battery Prognostics: Bayesian vs. Deterministic
**A Comparative Study on Uncertainty Quantification in Industrial AI**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-red.svg)](https://pytorch.org/)
[![PyMC](https://img.shields.io/badge/PyMC-Bayesian-green.svg)](https://www.pymc.io/)
[![Dataset](https://img.shields.io/badge/Dataset-NASA_PCoE-orange.svg)](https://www.nasa.gov/)
[![CI](https://github.com/Zhi-Chao-PAN/spatial-bayes-vs-deep-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhi-Chao-PAN/spatial-bayes-vs-deep-learning/actions)
[![Docker](https://img.shields.io/badge/docker-build-blue)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## 📌 Project Overview
> "In safety-critical systems (e.g., EVs, Aerospace), a confident wrong prediction is fatal. Knowing *what* you don't know is crucial."

This project addresses the challenge of **Remaining Useful Life (RUL)** prediction for Lithium-ion batteries. Using the **NASA PCoE Dataset**, we compare:
1.  **Deterministic Deep Learning (LSTM)**: High accuracy but lacks risk awareness.
2.  **Hierarchical Bayesian Modeling (PyMC)**: Provides **Uncertainty Quantification (UQ)** via posterior distributions.

The goal is to demonstrate that Bayesian methods offer a necessary "safety buffer" for industrial decision-making.

## 🚀 Key Features
* **Physics-Informed Feature Engineering**: Extraction of `Discharge_Time`, `Max_Temp`, and `Voltage_Drop` based on electrochemical principles.
* **Hybrid Architecture**:
    * **LSTM**: Sliding-window temporal feature extraction (PyTorch).
    * **Bayesian**: Hierarchical linear regression with partial pooling to handle inter-battery variability (PyMC).
* **Real-World Validation**: Trained on Batteries B0005, B0006, B0007; Tested on **B0018**.

## 📊 The "Killer Result"
*(Please ensure the image `results/comparison_B0018.png` is displayed below)*

![B0018 Prediction Analysis](results/comparison_B0018.png)

*Figure 1: RUL Prediction on Test Battery B0018. Note the critical difference:*
* 🔴 **Red Line (LSTM)**: Provides a point estimate. It fails to capture the risk when the battery exhibits regeneration phenomena.
* 🟢 **Green Zone (Bayesian)**: The 95% High Density Interval (HDI). It successfully widens as the battery approaches failure, signaling the system to "trust the prediction less." **This is the safety mechanism required for ISO 26262 compliance.**

## 💻 Tech Stack & Environment
* **Hardware**: Intel Core Ultra 9 (High-performance MCMC sampling), RTX 4060 (CUDA acceleration).
* **Software**: PyTorch 2.0, PyMC 5.0, ArviZ, Pandas.

## 🛠️ Quick Start
1.  **Clone the Repo**
    ```bash
    git clone https://github.com/YourUsername/safety-critical-battery-prognostics.git
    cd safety-critical-battery-prognostics
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Pipeline**
    ```bash
    # 1. Train LSTM Baseline
    python src/train_nn_baseline.py
    
    # 2. Run Bayesian Inference (MCMC)
    python src/train_bayes_hierarchical.py
    
    # 3. Generate Comparison Plot
    python src/compare_models.py
    ```
