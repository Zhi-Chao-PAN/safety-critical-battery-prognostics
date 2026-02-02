# Safety-Critical Battery Prognostics 🔋
**A Comparative Study: Hierarchical Bayesian Methods vs. Deep Learning (LSTM) for RUL Prediction**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![PyMC](https://img.shields.io/badge/PyMC-Bayesian-green.svg)](https://www.pymc.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Core Research Question**: In safety-critical industrial systems, is it better to be *precisely wrong* (Deterministic DL) or *vaguely right* (Bayesian Inference)?

## 📖 Project Overview
This research project addresses the "Black Box" problem in predictive maintenance. While Long Short-Term Memory (LSTM) networks offer high accuracy for Remaining Useful Life (RUL) prediction, they often fail to capture **epistemic uncertainty**—a critical flaw when making decisions for high-stakes assets like EV batteries or aerospace components.

Using the **NASA PCoE Battery Dataset**, this repository implements and compares:
1.  **Baseline**: A deterministic LSTM (Sliding Window approach).
2.  **Proposed**: A Hierarchical Probabilistic Degradation Model (Probabilistic approach).

## 🛡️ Impact & Safety Significance
This project aligns with **ISO 26262** functional safety requirements by providing probabilistic bounds rather than blind point estimates. Such systems are critical for **Level 4 autonomous systems**, where failure to quantify uncertainty could lead to catastrophic battery thermal runaway.

## 📊 Key Findings (The "Safety Gap")
The visualization below demonstrates the critical advantage of the Bayesian approach on Test Battery `B0018`:

![Comparison Plot](results/final_comparison_B0018.png)

* **Red Line (LSTM)**: Provides a single point estimate. It fails to account for the capacity regeneration phenomena (spikes) and offers no warning of confidence loss.
* **Green Zone (Bayesian)**: The 95% High Density Interval (HDI) successfully widens as data becomes scarce or noisy, providing a necessary **"Safety Buffer"** for human operators.

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Zhi-Chao-PAN/safety-critical-battery-prognostics.git
cd safety-critical-battery-prognostics
pip install -r requirements.txt
```
