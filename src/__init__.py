# src/__init__.py
"""
Spatial Bayesian vs Deep Learning - Source Package.

This package contains all training, evaluation, and analysis scripts
for comparing Bayesian hierarchical models with deep neural networks
on spatial housing price prediction.

Modules:
    - train_bayes_hierarchical: Hierarchical Bayesian model with PyMC
    - train_bayes_pooled: Pooled (non-hierarchical) Bayesian model
    - train_nn_baseline: PyTorch MLP baseline
    - train_nn_spatial: MLP with spatial cluster embeddings
    - evaluate_rigor: Rigorous cross-validation evaluation
    - visualize_results: Publication-quality figure generation
"""
