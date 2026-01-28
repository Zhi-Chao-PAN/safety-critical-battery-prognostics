# Reproducing Results

This document provides instructions for reproducing all experimental results from scratch.

## Quick Reproduce (All Results)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run rigorous cross-validation (15 runs)
python src/evaluate_rigor.py
# Output: results/rigor/summary_metrics.csv

# 3. Train Bayesian hierarchical model (~15 min)
python src/train_bayes_hierarchical.py
# Output: results/bayes_hierarchical/trace_hierarchical.nc

# 4. Generate figures
python src/visualize_results.py
# Output: results/figures/*.png

# 5. Run tests
pytest tests/ -v
```

## Reproducibility Guarantees

| Component | Mechanism |
|:----------|:----------|
| Data splits | `StratifiedKFold` with fixed seeds (42, 101, 2024) |
| NN weights | `torch.manual_seed(seed)` before each fold |
| Bayesian MCMC | PyMC default seed + 4 chains for convergence diagnostics |

## Hardware Requirements

- **Minimum**: 8GB RAM, any modern CPU
- **Recommended**: 16GB RAM for Bayesian sampling
- **Expected runtime**: ~30 minutes total on CPU
