import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, Dict, List, Tuple, Any
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

class HierarchicalBayesianModel(BaseEstimator, RegressorMixin):
    """
    Hierarchical Bayesian Regression Model for Battery RUL.
    
    Implements a partial pooling model:
        y ~ Normal(alpha[j] + beta * X, sigma)
        alpha[j] ~ Normal(mu_alpha, sigma_alpha)
        
    Adheres to scikit-learn interface guidelines.
    """
    
    def __init__(self, 
                 samples: int = 1000, 
                 tune: int = 1000, 
                 target_accept: float = 0.95,
                 chains: int = 2,
                 random_seed: int = 42):
        self.samples = samples
        self.tune = tune
        self.target_accept = target_accept
        self.chains = chains
        self.random_seed = random_seed
        self.trace = None
        self.model = None
        self.feature_names_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, feature_names: Optional[List[str]] = None) -> 'HierarchicalBayesianModel':
        """
        Fit the Hierarchical Bayesian model using NUTS sampler.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            groups: Group labels (battery IDs) for partial pooling
            feature_names: Optional list of feature names
        """
        # Standardization (handling internally for safety)
        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0) + 1e-6
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        # Encoding groups
        self.groups_ = np.unique(groups)
        group_map = {g: i for i, g in enumerate(self.groups_)}
        group_idx = np.array([group_map[g] for g in groups])
        n_groups = len(self.groups_)
        
        n_features = X.shape[1]
        self.feature_names_ = feature_names if feature_names else [f"f{i}" for i in range(n_features)]
        
        coords = {
            "battery_idx": np.arange(n_groups),
            "features": self.feature_names_,
            "obs_id": np.arange(len(y))
        }
        
        logger.info(f"Building PyMC model for {n_groups} groups and {n_features} features...")
        
        with pm.Model(coords=coords) as self.model:
            # Data containers (Mutable for predictions, though PyMC 5+ prefers explicit replacement)
            # For simplicity in this sklearn wrapper, we train largely on static data here, 
            # prediction will require rebuilding/substituting. 
            # Better pattern: Use pm.Data to allow swapping X for approx inference or posterior predictive.
            
            X_data = pm.Data("X_data", X_scaled)
            group_idx_data = pm.Data("group_idx", group_idx)
            y_data = pm.Data("y_data", y)
            
            # Hyperpriors
            mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=10.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=5.0)
            
            # Group-level intercepts (Partial Pooling)
            alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, dims="battery_idx")
            
            # Global Slopes (could also be hierarchical, but keeping fixed for stability on small data)
            beta = pm.Normal("beta", mu=0.0, sigma=5.0, dims="features")
            
            # Model Error
            sigma = pm.HalfNormal("sigma", sigma=5.0)
            
            # Linear combination
            mu = alpha[group_idx_data] + pm.math.dot(X_data, beta)
            
            # Likelihood
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma, observed=y_data)
            
            # Sampling
            logger.info("Starting NUTS sampling...")
            self.trace = pm.sample(
                draws=self.samples,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                progressbar=False # Reduce log noise
            )
            
        return self
    
    def predict(self, X: np.ndarray, group_id: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate probabilistic predictions.
        
        Args:
            X: Input features
            group_id: The battery ID to predict for. 
                      If seen during training, uses that alpha.
                      If new (OOD), uses population mu_alpha.
        
        Returns:
            mean_pred: Average prediction
            hdi_low: 2.5% percentile
            hdi_high: 97.5% percentile
        """
        if self.trace is None:
            raise RuntimeError("Model not fitted yet.")
            
        X_scaled = (X - self.X_mean_) / self.X_std_
        
        posterior = self.trace.posterior
        
        # Manual flattening to avoid xarray version issues with .stack()
        # Shape: (chains, draws, n_groups) or (chains, draws, n_features)
        
        # Alpha: (chains, draws, n_groups) -> (n_groups, total_samples)
        alpha_vals = posterior["alpha"].values
        n_chains, n_draws, n_groups = alpha_vals.shape
        alpha_samples = alpha_vals.reshape(n_chains * n_draws, n_groups).T
        
        # Beta: (chains, draws, n_features) -> (n_features, total_samples)
        beta_vals = posterior["beta"].values
        n_chains_b, n_draws_b, n_features = beta_vals.shape
        beta_samples = beta_vals.reshape(n_chains_b * n_draws_b, n_features).T
        
        # Mu_alpha: (chains, draws) -> (total_samples,)
        mu_vals = posterior["mu_alpha"].values
        mu_alpha_samples = mu_vals.flatten()
        
        # Decide which alpha to use
        if group_id in self.groups_:
            idx = np.where(self.groups_ == group_id)[0][0]
            chosen_alpha = alpha_samples[idx, :] # (total_samples,)
        else:
            chosen_alpha = mu_alpha_samples
            
        # Compute predictions: y = alpha + X * beta
        # X: (n_obs, n_features)
        # beta: (n_features, total_samples)
        # alpha: (total_samples,)
        
        term1 = np.dot(X_scaled, beta_samples) # (n_obs, total_samples)
        preds = term1 + chosen_alpha[None, :]  # (n_obs, total_samples)
        
        mean_pred = np.mean(preds, axis=1)
        hdi_low = np.percentile(preds, 2.5, axis=1)
        hdi_high = np.percentile(preds, 97.5, axis=1)
        
        return mean_pred, hdi_low, hdi_high
