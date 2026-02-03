import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_safety_comparison(
    cycles: np.ndarray,
    gt_rul: np.ndarray,
    lstm_preds: np.ndarray,
    bayes_metrics: dict,
    battery_id: str,
    save_path: str = "results/final_comparison.png"
):
    """
    Generate IEEE-style 'Safety Buffer' comparison plot.
    """
    # IEEE Column Width ~3.5 inches, but for Readme we go wider.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (8, 5)
    })
    
    fig, ax = plt.subplots()
    
    # 1. Ground Truth
    ax.plot(cycles, gt_rul, color='black', linestyle='-', linewidth=2.5, label='True RUL')
    
    # 2. Bayesian HDI (Safety Buffer)
    if 'hdi_low' in bayes_metrics and 'hdi_high' in bayes_metrics:
        # Check alignment
        if len(bayes_metrics['hdi_low']) == len(cycles):
            ax.fill_between(
                cycles, 
                bayes_metrics['hdi_low'], 
                bayes_metrics['hdi_high'], 
                color='#2ca02c', # Medical Green
                alpha=0.25, 
                label='Bayesian 95% HDI (Safety Buffer)'
            )
            # Plot mean for reference?
            # ax.plot(cycles, (bayes_metrics['hdi_low'] + bayes_metrics['hdi_high'])/2, color='#2ca02c', linewidth=1, alpha=0.5)

    # 3. LSTM Point Estimate
    ax.plot(cycles, lstm_preds, color='#d62728', linestyle='--', linewidth=2, label='LSTM (Point Estimate)')
    
    # Annotations
    ax.set_xlabel("Charge/Discharge Cycle")
    ax.set_ylabel("Remaining Useful Life (Cycles)")
    
    # Grid
    ax.grid(True, which='major', linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
    
    plt.tight_layout()
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved IEEE plot to {save_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    plt.close()
