# src/train_bayes_pooled.py

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path

from utils.schema import load_schema


def standardize(series):
    return (series - series.mean()) / series.std()


def main():
    schema = load_schema()

    data_path = schema["dataset"]["path"]
    target = schema["target"]["name"]
    slope_var = schema["bayesian"]["pooled"]["slope"]

    print("Loading data...")
    print(f"Using data file: {data_path}")

    df = pd.read_csv(data_path)

    # ===== Extract variables from schema =====
    y_raw = df[target].values
    x_raw = df[slope_var].values

    # ===== Standardization (schema-driven) =====
    if target in schema["modeling"].get("standardize", []):
        y = standardize(y_raw)
    else:
        y = y_raw

    if slope_var in schema["modeling"].get("standardize", []):
        x = standardize(x_raw)
    else:
        x = x_raw

    print("Sampling pooled Bayesian model...")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = alpha + beta * x

        y_obs = pm.Normal(
            "y_obs",
            mu=mu,
            sigma=sigma,
            observed=y
        )

        idata = pm.sample(
            draws=3000,
            tune=3000,
            chains=4,
            target_accept=0.9,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )

    results_dir = Path("results/bayes_pooled")
    results_dir.mkdir(parents=True, exist_ok=True)

    trace_path = results_dir / "trace_pooled.nc"
    az.to_netcdf(idata, trace_path)

    beta_mean = idata.posterior["beta"].mean().item()

    print("\n===== Pooled Bayesian Results =====")
    print(f"beta_{slope_var} (std units if standardized): {beta_mean}")
    print(f"Saved to: {results_dir}")


if __name__ == "__main__":
    main()
