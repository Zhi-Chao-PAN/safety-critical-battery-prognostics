import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import BatteryDataLoader
import os

def train_bayesian_model(output_dir="results/bayes_hierarchical"):
    print("Loading Battery Data...")
    loader = BatteryDataLoader()
    df = loader.load_data()
    
    # Encode Battery IDs to integers for PyMC indexing
    battery_ids = df['battery_id'].unique()
    battery_idx = pd.Categorical(df['battery_id'], categories=battery_ids).codes
    coords = {'battery_id': battery_ids}
    
    # Standardize features
    X = df[['discharge_time', 'max_temp', 'voltage_drop']].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    y = df['rul'].values

    print("Building Hierarchical Model...")
    n_samples, n_features = X_scaled.shape
    n_batteries = len(battery_ids)
    
    with pm.Model() as hierarchical_model:
        # Hyperpriors
        mu_alpha = pm.Normal("mu_alpha", mu=100, sigma=50)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=20)
        
        # Priors
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_batteries)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=n_features)
        
        # Error
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # Linear Model
        # alpha[battery_idx] -> (N,)
        # dot(X, beta) -> (N,)
        mu = alpha[battery_idx] + pm.math.dot(X_scaled, beta)
        
        # Likelihood
        rul_obs = pm.Normal("rul_obs", mu=mu, sigma=sigma, observed=y)
        
        # Inference
        print("Sampling (MCMC)...")
        # Ensure we use 4 chains for academic rigor check
        trace = pm.sample(1000, tune=1000, chains=4, cores=4, target_accept=0.9)
        
        # Saving
        os.makedirs(output_dir, exist_ok=True)
        az.to_netcdf(trace, os.path.join(output_dir, "trace.nc"))
        print(f"Trace saved to {output_dir}/trace.nc")
        
        # Summary
        summary = az.summary(trace)
        summary.to_csv(os.path.join(output_dir, "summary.csv"))

if __name__ == "__main__":
    train_bayesian_model()
