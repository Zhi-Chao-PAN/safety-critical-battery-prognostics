import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

def plot_safety_comparison(
    cycles: np.ndarray,
    gt_rul: np.ndarray,
    lstm_preds: np.ndarray,
    bayes_metrics: dict,
    battery_id: str,
    save_path: str = "results/final_comparison.png"
):
    """
    Generate the 'Safety Buffer' comparison plot (Figure 1 in Report).
    
    Args:
        cycles: Array of cycle numbers
        gt_rul: Ground Truth RUL
        lstm_preds: LSTM Point Estimates (aligned to cycles, with NaNs for window)
        bayes_metrics: Dictionary containing 'hdi_low', 'hdi_high' (arrays)
        battery_id: ID of the test battery
        save_path: Output path
    """
    plt.style.use('default') 
    # Attempt to use Serif font for academic look
    try:
        plt.rcParams["font.family"] = "serif"
    except:
        pass
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ground Truth
    ax.plot(cycles, gt_rul, 'k-', linewidth=2, label='True RUL (Ground Truth)')
    
    # Bayesian (Safety Buffer)
    if 'hdi_low' in bayes_metrics and 'hdi_high' in bayes_metrics:
        ax.fill_between(
            cycles, 
            bayes_metrics['hdi_low'], 
            bayes_metrics['hdi_high'], 
            color='green', 
            alpha=0.3, 
            label='Bayesian 95% HDI'
        )
    
    # LSTM
    ax.plot(cycles, lstm_preds, 'r--', linewidth=2, label='LSTM (Point Estimate)')
    
    # Annotation
    eol_idx = np.argmin(gt_rul)
    if eol_idx < len(cycles) and 'hdi_high' in bayes_metrics:
        ax.annotate(
            'Safety Buffer Zone\n(Uncertainty Widens)', 
            xy=(cycles[eol_idx], bayes_metrics['hdi_high'][eol_idx]), 
            xytext=(cycles[eol_idx]-50, bayes_metrics['hdi_high'][eol_idx]+50),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10, 
            fontweight='bold', 
            color='darkgreen'
        )

    ax.set_title(f"Figure 1: RUL Comparison on Test Battery {battery_id}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Cycle Number", fontsize=12)
    ax.set_ylabel("Remaining Useful Life (RUL)", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to {out_path}")
