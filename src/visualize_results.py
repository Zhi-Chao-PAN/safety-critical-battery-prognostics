# src/visualize_results.py
"""
Visualization module for generating publication-quality figures.

This module provides functions to create visualizations for:
    - Model performance comparison (error bars)
    - Bayesian posterior distributions
    - Spatial cluster analysis
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10
})


def plot_cv_error_bars(
    results_path: str = "results/rigor/raw_cv_results.csv",
    output_path: str = "results/figures/cv_error_bars.png"
) -> None:
    """
    Create error bar plot comparing model RMSE with standard deviations.
    
    Args:
        results_path: Path to CSV with cross-validation results.
        output_path: Path to save the output figure.
    """
    df = pd.read_csv(results_path)
    
    # Compute statistics
    summary = df.groupby("Model")["RMSE"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    x_pos = np.arange(len(summary))
    
    bars = ax.bar(x_pos, summary["mean"], yerr=summary["std"], 
                  capsize=8, color=colors[:len(summary)], 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary["Model"], fontweight='bold')
    ax.set_ylabel("RMSE (Mean ± Std)", fontweight='bold')
    ax.set_title("Cross-Validated Model Performance\n(5-Fold × 3 Seeds = 15 runs)", 
                 fontweight='bold', fontsize=14)
    
    # Add value labels
    for bar, mean, std in zip(bars, summary["mean"], summary["std"]):
        ax.annotate(f'{mean:.3f}±{std:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01),
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim(0, max(summary["mean"] + summary["std"]) * 1.15)
    ax.axhline(y=summary["mean"].min(), color='green', linestyle='--', 
               alpha=0.5, label='Best (Linear Reg.)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_posterior_distributions(
    trace_path: str = "results/bayes_hierarchical/trace_hierarchical.nc",
    output_dir: str = "results/figures"
) -> None:
    """
    Create posterior distribution plots for Bayesian model parameters.
    
    Generates ridge plots showing the posterior distributions of
    global slope parameters (mu_beta) and their spatial variance (sigma_beta).
    
    Args:
        trace_path: Path to ArviZ InferenceData NetCDF file.
        output_dir: Directory to save output figures.
    """
    try:
        import arviz as az
    except ImportError:
        print("ArviZ not installed. Skipping posterior plots.")
        return
    
    trace_file = Path(trace_path)
    if not trace_file.exists():
        print(f"Trace file not found: {trace_path}")
        print("Run train_bayes_hierarchical.py first to generate the trace.")
        return
    
    idata = az.from_netcdf(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Global means (mu_beta)
    fig, ax = plt.subplots(figsize=(10, 4))
    az.plot_posterior(idata, var_names=["mu_beta"], ax=ax, kind="hist")
    ax.set_title("Global Slope Parameters (μ_β)\nAcross All Spatial Clusters", 
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_mu_beta.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'posterior_mu_beta.png'}")
    
    # 2. Between-cluster variance (sigma_beta)
    fig, ax = plt.subplots(figsize=(10, 4))
    az.plot_posterior(idata, var_names=["sigma_beta"], ax=ax, kind="hist")
    ax.set_title("Spatial Heterogeneity (σ_β)\nHow Slopes Vary by Cluster", 
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "posterior_sigma_beta.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'posterior_sigma_beta.png'}")
    
    # 3. Forest plot for group-level effects
    fig, ax = plt.subplots(figsize=(12, 8))
    az.plot_forest(idata, var_names=["beta_group"], combined=True, ax=ax)
    ax.set_title("Group-Level Slope Estimates with 94% HDI\n(Partial Pooling Effect)", 
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "forest_beta_group.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'forest_beta_group.png'}")


def plot_spatial_clusters(
    data_path: str = "data/processed/housing_with_spatial_clusters.csv",
    output_path: str = "results/figures/spatial_clusters_map.png"
) -> None:
    """
    Create a geographic visualization of spatial clusters and prices.
    
    Args:
        data_path: Path to dataset with spatial coordinates and cluster labels.
        output_path: Path to save the output figure.
    """
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Price heatmap
    scatter1 = axes[0].scatter(
        df['longitude'], df['latitude'],
        c=df['price'], cmap='RdYlGn_r',
        alpha=0.6, s=3, edgecolors='none'
    )
    axes[0].set_xlabel('Longitude', fontweight='bold')
    axes[0].set_ylabel('Latitude', fontweight='bold')
    axes[0].set_title('Housing Prices by Location', fontweight='bold', fontsize=13)
    cbar1 = plt.colorbar(scatter1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Price (scaled)', fontweight='bold')
    
    # 2. Cluster assignments
    if 'spatial_cluster' in df.columns:
        n_clusters = df['spatial_cluster'].nunique()
        scatter2 = axes[1].scatter(
            df['longitude'], df['latitude'],
            c=df['spatial_cluster'], cmap='tab10',
            alpha=0.6, s=3, edgecolors='none'
        )
        axes[1].set_xlabel('Longitude', fontweight='bold')
        axes[1].set_ylabel('Latitude', fontweight='bold')
        axes[1].set_title(f'Spatial Clusters (n={n_clusters})\nHierarchical Model Groups', 
                         fontweight='bold', fontsize=13)
        cbar2 = plt.colorbar(scatter2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Cluster ID', fontweight='bold')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_income_slope_heterogeneity(
    data_path: str = "data/processed/housing_with_spatial_clusters.csv",
    output_path: str = "results/figures/income_slope_by_cluster.png"
) -> None:
    """
    Visualize how income-price relationship varies by spatial cluster.
    
    This demonstrates the key insight captured by Bayesian hierarchical models:
    the slope (elasticity) of income on price differs across neighborhoods.
    
    Args:
        data_path: Path to dataset with spatial coordinates and cluster labels.
        output_path: Path to save the output figure.
    """
    df = pd.read_csv(data_path)
    
    if 'spatial_cluster' not in df.columns:
        print("Dataset missing 'spatial_cluster' column. Skipping.")
        return
    
    n_clusters = df['spatial_cluster'].nunique()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot regression line for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i, cluster in enumerate(sorted(df['spatial_cluster'].unique())):
        cluster_df = df[df['spatial_cluster'] == cluster]
        
        # Compute cluster-specific regression
        x = cluster_df['median_income']
        y = cluster_df['price']
        
        # Simple linear fit
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = slope * x_line + intercept
        
        ax.scatter(x, y, alpha=0.3, s=5, color=colors[i], label=f'Cluster {cluster}')
        ax.plot(x_line, y_line, color=colors[i], linewidth=2, 
                label=f'Slope={slope:.3f}' if i == 0 else None)
    
    ax.set_xlabel('Median Income', fontweight='bold', fontsize=12)
    ax.set_ylabel('Price', fontweight='bold', fontsize=12)
    ax.set_title('Income-Price Relationship by Spatial Cluster\n'
                 '(Demonstrates Heterogeneous Slopes Captured by Bayesian Model)',
                 fontweight='bold', fontsize=13)
    
    # Legend with cluster info
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
              title='Cluster', title_fontsize=10)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_figures() -> None:
    """Generate all visualization figures for the project."""
    print("=" * 50)
    print("GENERATING PROJECT VISUALIZATIONS")
    print("=" * 50)
    
    # Check for raw CV results
    cv_path = Path("results/rigor/raw_cv_results.csv")
    if cv_path.exists():
        print("\n[1/4] Cross-validation error bars...")
        plot_cv_error_bars()
    else:
        print(f"\n[1/4] Skipping CV plot (run evaluate_rigor.py first)")
    
    print("\n[2/4] Posterior distributions...")
    plot_posterior_distributions()
    
    print("\n[3/4] Spatial cluster map...")
    plot_spatial_clusters()
    
    print("\n[4/4] Income slope heterogeneity...")
    plot_income_slope_heterogeneity()
    
    print("\n" + "=" * 50)
    print("VISUALIZATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_figures()
