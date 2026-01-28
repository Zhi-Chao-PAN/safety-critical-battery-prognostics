# src/train_bayes_hierarchical.py
"""
Hierarchical Bayesian Model for Spatial Housing Price Prediction.

This module implements a Multi-Slope Hierarchical Bayesian model using PyMC,
with varying intercepts and slopes by spatial cluster (partial pooling).

Key Features:
    - Non-centered parameterization for robust NUTS sampling
    - Explicit quantification of spatial parameter variance
    - Full posterior sampling for uncertainty quantification
"""

from typing import Union
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path

from utils.schema import load_schema


def standardize_series(series: pd.Series) -> np.ndarray:
    """
    Standardize a pandas Series to zero mean and unit variance.
    
    Args:
        series: Input pandas Series to standardize.
        
    Returns:
        Standardized numpy array with mean ≈ 0 and std ≈ 1.
        
    Example:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> z = standardize_series(s)
        >>> np.abs(z.mean()) < 1e-10
        True
    """
    return (series - series.mean()) / series.std()


def standardize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each column of a DataFrame to zero mean and unit variance.
    
    Args:
        df: Input DataFrame with numeric columns.
        
    Returns:
        DataFrame with each column standardized independently.
    """
    return (df - df.mean()) / df.std()


def build_model(
    y: np.ndarray,
    x: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    n_features: int
) -> pm.Model:
    """
    Build a Multi-Slope Hierarchical Bayesian Model.
    
    This model implements partial pooling where both intercepts (alpha) and
    slopes (beta) vary by spatial cluster. Uses non-centered parameterization
    for improved NUTS sampler convergence.
    
    Model Structure:
        y ~ Normal(mu, sigma)
        mu = alpha[group] + sum(beta[group, feature] * x[feature])
        
        # Hyperpriors
        mu_alpha ~ Normal(0, 1)
        sigma_alpha ~ HalfNormal(1)
        mu_beta[feature] ~ Normal(0, 1)
        sigma_beta[feature] ~ HalfNormal(1)
        
        # Group-level (non-centered)
        alpha[group] = mu_alpha + alpha_raw[group] * sigma_alpha
        beta[group, feature] = mu_beta[feature] + beta_raw[group, feature] * sigma_beta[feature]
    
    Args:
        y: Target variable array of shape (n_samples,).
        x: Feature matrix of shape (n_samples, n_features).
        group_idx: Group indices for each sample, shape (n_samples,).
        n_groups: Total number of unique groups/clusters.
        n_features: Number of features with varying slopes.
        
    Returns:
        PyMC Model object ready for sampling.
        
    References:
        Gelman & Hill (2006). Data Analysis Using Regression and 
        Multilevel/Hierarchical Models.
    """
    with pm.Model() as model:
        # Hyperpriors
        # mu_beta: shape (n_features,)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, shape=n_features)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, shape=n_features)

        # mu_alpha: scalar (intercept)
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)

        # Group-level parameters (Non-centered parameterization)
        # This improves sampling geometry for hierarchical models
        # alpha: (n_groups,)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_groups)
        alpha_group = pm.Deterministic(
            "alpha_group", mu_alpha + alpha_raw * sigma_alpha
        )

        # beta: (n_groups, n_features)
        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=(n_groups, n_features))
        
        # Broadcasting: mu_beta (n_features) + beta_raw (n_groups, n_features) * sigma_beta (n_features)
        beta_group = pm.Deterministic(
            "beta_group", mu_beta + beta_raw * sigma_beta
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear combination
        slope_effect = (beta_group[group_idx] * x).sum(axis=1)
        
        mu = alpha_group[group_idx] + slope_effect

        # Likelihood
        pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y
        )
    
    return model


def main() -> None:
    """
    Main entry point for training the Hierarchical Bayesian model.
    
    Loads data according to schema, builds the model, runs NUTS sampling,
    and saves the trace to disk.
    """
    schema = load_schema()

    data_path: str = schema["dataset"]["path"]
    target: str = schema["target"]["name"]
    group_var: str = schema["bayesian"]["hierarchical"]["group"]
    
    # Handle scalar vs list slope definition
    slope_vars: Union[str, list] = schema["bayesian"]["hierarchical"]["slope"]
    if isinstance(slope_vars, str):
        slope_vars = [slope_vars]

    print("Loading data...")
    print(f"Using data file: {data_path}")

    df = pd.read_csv(data_path)

    # ===== Extract =====
    y_raw = df[target].values
    
    # X matrix (n_samples, n_features)
    x_raw_df = df[slope_vars]
    n_features = len(slope_vars)

    # group indexing
    group_codes, group_idx = np.unique(df[group_var], return_inverse=True)
    n_groups = len(group_codes)

    # ===== Standardization =====
    if target in schema["modeling"].get("standardize", []):
        y = standardize_series(y_raw)
    else:
        y = y_raw

    x_df = standardize_matrix(x_raw_df)
    x = x_df.values

    print(f"Hierarchical groups: {n_groups}")
    print(f"Slope features: {slope_vars}")

    print("Building model...")
    model = build_model(y, x, group_idx, n_groups, n_features)

    print("Sampling hierarchical Bayesian model (Multi-Slope)...")
    with model:
        idata = pm.sample(
            draws=3000,
            tune=3000,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )

    results_dir = Path("results/bayes_hierarchical")
    results_dir.mkdir(parents=True, exist_ok=True)

    trace_path = results_dir / "trace_hierarchical.nc"
    az.to_netcdf(idata, trace_path)

    print("\n===== Hierarchical Bayesian Results =====")
    print(az.summary(idata, var_names=["mu_beta"], kind="stats"))
    print(f"Saved to: {results_dir}")


if __name__ == "__main__":
    main()
