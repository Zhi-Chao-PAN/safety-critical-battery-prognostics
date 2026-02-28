"""
Uncertainty Decomposition - Separate aleatoric and epistemic uncertainty.

Total Variance = E[Var(y|w)] + Var[E(y|w)]
               = Aleatoric    + Epistemic
"""

import numpy as np


def decompose_ensemble(
    predictions: np.ndarray,
    aleatoric_stds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Decompose uncertainty from ensemble/MC predictions.

    Args:
        predictions: (N_samples, N_points) - multiple predictions per input
        aleatoric_stds: (N_samples, N_points) - per-sample aleatoric std
                        If None, assumes homoscedastic (estimated from residuals)

    Returns:
        dict with keys: total_std, aleatoric_std, epistemic_std, mean
    """
    mean = predictions.mean(axis=0)  # (N_points,)

    # Epistemic: Variance of means across samples
    epistemic_var = predictions.var(axis=0)  # (N_points,)

    # Aleatoric: Mean of per-sample variances
    if aleatoric_stds is not None:
        aleatoric_var = np.mean(aleatoric_stds ** 2, axis=0)
    else:
        # Estimate from within-sample variance (rough approximation)
        aleatoric_var = np.zeros_like(epistemic_var)

    total_var = aleatoric_var + epistemic_var

    return {
        "mean": mean,
        "total_std": np.sqrt(np.maximum(total_var, 1e-12)),
        "aleatoric_std": np.sqrt(np.maximum(aleatoric_var, 1e-12)),
        "epistemic_std": np.sqrt(np.maximum(epistemic_var, 1e-12)),
    }


def decompose_from_model(
    model, X: np.ndarray, n_forward: int = 100
) -> dict[str, np.ndarray]:
    """
    Decompose uncertainty by running multiple forward passes.
    Works with any model that has predict_distribution().
    """
    samples = model.predict_distribution(X, n_samples=n_forward)
    return decompose_ensemble(samples)
