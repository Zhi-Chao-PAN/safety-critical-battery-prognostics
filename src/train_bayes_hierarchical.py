# src/train_bayes_hierarchical.py
"""
Hierarchical Bayesian Model for Battery RUL Prediction.

This module implements a Multi-Slope Hierarchical Bayesian model using PyMC,
with varying intercepts and slopes by Battery ID (partial pooling).

Key Changes for RUL:
    - Grouping by 'battery_id' instead of spatial cluster.
    - Predicting RUL based on degradation features.
"""

from typing import Union
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path

from utils.schema import load_schema
from data_loader import load_battery_data


def standardize_series(series: pd.Series) -> np.ndarray:
    """Standardize a pandas Series to zero mean and unit variance."""
    if series.std() == 0:
        return np.zeros_like(series)
    return (series - series.mean()) / series.std()


def standardize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize each column of a DataFrame."""
    # Avoid division by zero for constant columns
    return (df - df.mean()) / df.std().replace(0, 1)


def build_model(
    y: np.ndarray,
    x: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    n_features: int
) -> pm.Model:
    """
    Build a Multi-Slope Hierarchical Bayesian Model.
    
    y ~ Normal(mu, sigma)
    mu = alpha[group] + sum(beta[group, feature] * x[feature])
    """
    with pm.Model() as model:
        # Hyperpriors
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, shape=n_features)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, shape=n_features)

        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)

        # Group-level parameters (Non-centered)
        alpha_raw = pm.Normal("alpha_raw", mu=0, sigma=1, shape=n_groups)
        alpha_group = pm.Deterministic(
            "alpha_group", mu_alpha + alpha_raw * sigma_alpha
        )

        beta_raw = pm.Normal("beta_raw", mu=0, sigma=1, shape=(n_groups, n_features))
        
        # Broadcasting: (n_features) + (n_groups, n_features) * (n_features)
        beta_group = pm.Deterministic(
            "beta_group", mu_beta + beta_raw * sigma_beta
        )

        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear combination
        # x is (n_samples, n_features)
        # beta_group[group_idx] is (n_samples, n_features)
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
    schema = load_schema()

    data_path: str = schema["dataset"]["path"]
    target: str = schema["target"]["name"]
    group_var: str = schema["bayesian"]["hierarchical"]["group"]
    
    slope_vars: Union[str, list] = schema["bayesian"]["hierarchical"]["slope"]
    if isinstance(slope_vars, str):
        slope_vars = [slope_vars]

    print("Loading battery data...")
    print(f"Data directory: {data_path}")
    
    # NEW: Load via data_loader
    df = load_battery_data(data_path)

    # ===== Extract =====
    y_raw = df[target].values
    
    # X matrix (n_samples, n_features)
    x_raw_df = df[slope_vars]
    n_features = len(slope_vars)

    # group indexing
    # Ensure group_var is treated as string/categorical for consistent encoding
    df[group_var] = df[group_var].astype(str)
    group_codes, group_idx = np.unique(df[group_var], return_inverse=True)
    n_groups = len(group_codes)

    # ===== Standardization =====
    if target in schema["modeling"].get("standardize", []):
        print(f"Standardizing target '{target}'...")
        y = standardize_series(df[target])
    else:
        y = y_raw

    print(f"Standardizing features: {slope_vars}")
    x_df = standardize_matrix(x_raw_df)
    x = x_df.values

    print(f"Hierarchical groups ({group_var}): {n_groups} {group_codes}")
    print(f"Slope features: {slope_vars}")

    print("Building model...")
    model = build_model(y, x, group_idx, n_groups, n_features)

    print("Sampling hierarchical Bayesian model...")
    with model:
        # Reduced chains/draws for faster feedback in this refactor, 
        # normally you'd want more.
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
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
