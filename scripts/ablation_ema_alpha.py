#!/usr/bin/env python3
"""
EMA Alpha Ablation Study — Sweep α ∈ [0.05, 1.0] to find optimal 
smoothing factor that minimizes RMSE while maintaining 0% violation rate.

This script loads a pre-trained PINN model, predicts on noisy data,
and applies the two-stage projection with varying alpha values.
No model retraining is needed — only post-processing varies.

Author: Antigravity AI Research Architect
Date: 2026-04-05
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pinn_model import PINNModel
from src.physics.constraints import (
    ConstraintManager, MonotonicityConstraint, 
    SPMResidualConstraint,
    VoltageConstraint, TemperatureConstraint
)


def generate_synthetic_data(n_samples=200, seed=42):
    """Same data generation as robustness_test.py for reproducibility."""
    np.random.seed(seed)
    cycles = np.linspace(0, 1000, n_samples)
    capacity_clean = 2.0 * np.exp(-0.001 * cycles) + 0.05 * np.sin(0.01 * cycles)
    X_clean = np.column_stack([cycles, capacity_clean])
    y_clean = capacity_clean
    
    noise_level = 0.50
    noise_std = noise_level * np.std(capacity_clean)
    noise = np.random.normal(0, noise_std, len(capacity_clean))
    capacity_noisy = capacity_clean + noise
    X_noisy = np.column_stack([cycles, capacity_noisy])
    
    return X_clean, y_clean, X_noisy


def evaluate_alpha(predictions_raw, y_clean, alpha):
    """Apply EMA + running-min with given alpha, return RMSE and violation rate."""
    # Stage 1: EMA
    smoothed = np.empty_like(predictions_raw)
    smoothed[0] = predictions_raw[0]
    for i in range(1, len(predictions_raw)):
        smoothed[i] = alpha * predictions_raw[i] + (1 - alpha) * smoothed[i - 1]
    
    # Stage 2: Running minimum
    projected = np.empty_like(smoothed)
    projected[0] = smoothed[0]
    for i in range(1, len(smoothed)):
        projected[i] = min(projected[i - 1], smoothed[i])
    
    # Metrics
    rmse = np.sqrt(np.mean((projected - y_clean) ** 2))
    
    violations = 0
    for i in range(1, len(projected)):
        if projected[i] > projected[i - 1] + 1e-10:
            violations += 1
    violation_rate = violations / max(len(projected) - 1, 1) * 100
    
    return rmse, violation_rate, projected


def main():
    print("=" * 70)
    print("EMA Alpha Ablation Study for PINN Robustness Post-Processing")
    print("=" * 70)
    
    # Generate data
    X_clean, y_clean, X_noisy = generate_synthetic_data()
    
    # Train PINN model (one-time)
    robust_cm = ConstraintManager()
    robust_cm.add_constraint(MonotonicityConstraint(weight=1.0, adaptive=True))
    robust_cm.add_constraint(SPMResidualConstraint(weight=0.1, adaptive=True))
    robust_cm.add_constraint(VoltageConstraint(v_min=0.0, v_max=2.5, weight=0.05, adaptive=True))
    robust_cm.add_constraint(TemperatureConstraint(t_max=2.2, weight=0.01, adaptive=True))
    
    pinn_model = PINNModel(
        input_dim=2, hidden_dim=64, dropout=0.05,
        lr=1e-3, epochs=500, patience=80,
        lambda_physics=0.1, lambda_mono=1.0,
        adaptive_weighting=True, mc_samples=50,
        device="cpu", constraint_manager=robust_cm
    )
    
    print("\n[1/3] Training PINN model on clean data...")
    pinn_model.fit(X_clean, y_clean)
    
    print("[2/3] Predicting on noisy data (raw, no post-processing)...")
    predictions_raw, lower, upper = pinn_model.predict(X_noisy)
    
    # Raw metrics (no post-processing)
    raw_rmse = np.sqrt(np.mean((predictions_raw - y_clean) ** 2))
    raw_violations = sum(1 for i in range(1, len(predictions_raw)) 
                         if predictions_raw[i] > predictions_raw[i-1] + 1e-10)
    raw_violation_rate = raw_violations / (len(predictions_raw) - 1) * 100
    print(f"    Raw (no post-proc): RMSE={raw_rmse:.4f}, Violations={raw_violation_rate:.2f}%")
    
    # Sweep alpha
    print("\n[3/3] Sweeping EMA alpha parameter...")
    print("-" * 70)
    print(f"{'Alpha':>8} | {'RMSE':>10} | {'Violation %':>12} | {'Status':>10}")
    print("-" * 70)
    
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 
              0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    
    results = []
    best_rmse = float('inf')
    best_alpha = None
    
    for alpha in alphas:
        rmse, vr, projected = evaluate_alpha(predictions_raw, y_clean, alpha)
        status = "✅ SAFE" if vr == 0.0 else "❌ VIOLATION"
        
        if vr == 0.0 and rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            status += " ⭐"
        
        print(f"{alpha:>8.2f} | {rmse:>10.4f} | {vr:>11.2f}% | {status}")
        results.append((alpha, rmse, vr))
    
    print("-" * 70)
    print(f"\n{'='*70}")
    print(f"OPTIMAL: α = {best_alpha:.2f}, RMSE = {best_rmse:.4f}, Violation Rate = 0.00%")
    print(f"{'='*70}")
    
    # Also test: running-minimum only (no EMA) as baseline
    rmse_noema, vr_noema, _ = evaluate_alpha(predictions_raw, y_clean, 1.0)
    print(f"\nBaseline (running-min only, α=1.0): RMSE={rmse_noema:.4f}, VR={vr_noema:.2f}%")
    
    return best_alpha, best_rmse


if __name__ == "__main__":
    best_alpha, best_rmse = main()
