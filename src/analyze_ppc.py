# src/analyze_ppc.py

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils.schema import load_schema
from train_bayes_hierarchical import build_model, standardize_matrix, standardize_series

def load_data_and_model():
    schema = load_schema()
    data_path = schema["dataset"]["path"]
    target = schema["target"]["name"]
    group_var = schema["bayesian"]["hierarchical"]["group"]
    slope_vars = schema["bayesian"]["hierarchical"]["slope"]
    if isinstance(slope_vars, str):
        slope_vars = [slope_vars]

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    # Prepare data exactly as in training
    y_raw = df[target].values
    x_raw_df = df[slope_vars]
    
    group_codes, group_idx = np.unique(df[group_var], return_inverse=True)
    n_groups = len(group_codes)
    n_features = len(slope_vars)

    if target in schema["modeling"].get("standardize", []):
        y = standardize_series(y_raw)
    else:
        y = y_raw

    x_df = standardize_matrix(x_raw_df)
    x = x_df.values

    print("Building model context...")
    model = build_model(y, x, group_idx, n_groups, n_features)
    
    return df, model, y

def main():
    trace_path = Path("results/bayes_hierarchical/trace_hierarchical.nc")
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}. Run training first.")

    print("Loading original data and model definition...")
    df, model, y_std = load_data_and_model()

    print(f"Loading trace from {trace_path}...")
    idata = az.from_netcdf(trace_path)

    print("Sampling Posterior Predictive...")
    with model:
        # Extend idata with posterior predictive samples
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, progressbar=True)

    # 1. PPC Plot (Distribution)
    print("Generating PPC density plot...")
    az.plot_ppc(idata, num_pp_samples=100)
    plt.savefig("results/bayes_hierarchical/ppc_density.png")
    plt.close()

    # 2. LOO-PIT (Calibration)
    print("Generating LOO-PIT plot...")
    az.plot_loo_pit(idata, y="y_obs")
    plt.savefig("results/bayes_hierarchical/loo_pit.png")
    plt.close()

    # 3. Spatial Residual Map
    print("Calculating residuals...")
    # Posterior mean prediction
    y_pred_mean = idata.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values
    
    # Residuals (in standardized units if y was standardized)
    residuals = y_std - y_pred_mean

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df["longitude"], 
        df["latitude"], 
        c=residuals, 
        cmap="coolwarm", 
        s=10, 
        alpha=0.7,
        vmax=np.percentile(residuals, 98),
        vmin=np.percentile(residuals, 2)
    )
    plt.colorbar(sc, label="Residual (std units)")
    plt.title("Spatial Residual Map (Hierarchical Model)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig("results/bayes_hierarchical/spatial_residuals.png")
    plt.close()
    
    print("\n===== Diagnostics Completed =====")
    print("Saved plots to results/bayes_hierarchical/")
    print("- ppc_density.png: Check if model captures data distribution")
    print("- loo_pit.png: Check if uncertainty is calibrated (uniform is good)")
    print("- spatial_residuals.png: Check for unmodeled spatial patterns")

if __name__ == "__main__":
    main()
