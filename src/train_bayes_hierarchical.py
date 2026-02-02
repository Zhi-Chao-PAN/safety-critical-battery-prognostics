import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import os
from src.data_loader import BatteryDataLoader

def train():
    print("Loading Data...")
    print("Loading Data...")
    loader = BatteryDataLoader()
    # Explicitly exclude test battery B0018 to prevent data leakage
    train_batteries = ['B0005', 'B0006', 'B0007']
    print(f"Training on: {train_batteries}")
    df = loader.load_data(battery_ids=train_batteries)
    
    # 1. Feature Engineering (Standardization)
    X_cols = ['discharge_time', 'max_temp']
    X = df[X_cols].values
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    y = df['rul'].values

    # 2. Hierarchical Grouping by Battery ID
    battery_ids = df['battery_id'].unique()
    battery_idx = pd.Categorical(df['battery_id'], categories=battery_ids).codes
    coords = {'battery_id': battery_ids, 'features': X_cols}

    print(f"Training Hierarchical Bayesian Model on {len(df)} cycles...")
    
    with pm.Model(coords=coords) as model:
        # Mutable Data Containers
        X_data = pm.Data("X_data", X_scaled)
        battery_idx_data = pm.Data("battery_idx", battery_idx)
        
        # Hyperpriors (Partial Pooling)
        mu_alpha = pm.Normal("mu_alpha", mu=100, sigma=50)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=20)
        
        # Priors for each battery intercept
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, dims="battery_id")
        
        # Slopes for features
        beta = pm.Normal("beta", mu=0, sigma=10, dims="features")
        
        # Model Error
        sigma = pm.HalfNormal("sigma", sigma=10)
        
        # Linear Function
        mu = alpha[battery_idx_data] + pm.math.dot(X_data, beta)
        
        # Likelihood
        rul_obs = pm.Normal("rul_obs", mu=mu, sigma=sigma, observed=y)
        
        # Sampling (MCMC)
        cores = os.cpu_count() or 1
        trace = pm.sample(500, tune=500, chains=2, cores=cores, target_accept=0.9)
        
        save_path = "results/bayes_hierarchical"
        os.makedirs(save_path, exist_ok=True)
        az.to_netcdf(trace, os.path.join(save_path, "trace.nc"))
        print(f"✅ Bayesian Model Trained & Saved to {save_path}")

if __name__ == "__main__":
    train()