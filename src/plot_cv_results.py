# src/plot_cv_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    # Load summary metrics
    results_path = Path("results/rigor/summary_metrics.csv")
    if not results_path.exists():
        print("Error: Rigor results not found. Run evaluate_rigor.py first.")
        return

    df = pd.read_csv(results_path)
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create colors
    colors = ['#4c72b0' if "Linear" in name else '#dd8452' for name in df["Model"]]
    
    # Plot Error Bars
    # x = Model, y = mean, yerr = std
    bars = plt.bar(
        df["Model"], 
        df["mean"], 
        yerr=df["std"], 
        capsize=10, 
        color=colors, 
        alpha=0.8,
        edgecolor='black'
    )
    
    # Add values on top
    for bar, mean, std in zip(bars, df["mean"], df["std"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + std + 0.01,
            f'{mean:.3f}\n±{std:.3f}',
            ha='center', 
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    plt.title("Model Performance: 5-Fold Cross-Validation (n=15 runs)", fontsize=14, pad=20)
    plt.ylabel("RMSE (Lower is Better)", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylim(0, df["mean"].max() * 1.3) # Add headroom for text
    
    # Add context text box
    plt.text(
        0.95, 0.95, 
        "Dataset Regime: Small Tabular\nFinding: Occam's Razor wins\n(Linear > Neural)", 
        transform=plt.gca().transAxes, 
        fontsize=10, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )

    # Save
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cv_error_bars.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
