"""Generate OOD detection and dynamic safety boundary plots."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from src.uncertainty.ood_detector import OODDetector, OODLevel

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.features.extractor import FeatureExtractor
from src.models import BTCNModel

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
FIG = ROOT / "figures"

def main():
    print("Loading data...")
    loader = UnifiedDataLoader()
    df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
    validator = DataValidator()
    df, _ = validator.validate(df)
    extractor = FeatureExtractor()
    df = extractor.extract_all(df)

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("cycle", "rul")]
    df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)

    # Train heavily on B0005
    train_df = df[df["battery_id"] == "B0005"].copy()

    # Use B0006 as the testing base, and artificially inject OOD condition
    test_df = df[df["battery_id"] == "B0006"].copy().reset_index(drop=True)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["rul"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["rul"].values.astype(np.float32)
    cycles = test_df["cycle"].values

    # Simulate a high temperature shock / capacity dive starting at cycle #80
    shock_idx = 80
    if len(X_test) > shock_idx:
        for c_idx, c_name in enumerate(feature_cols):
            if "temp" in c_name.lower():
                X_test[shock_idx:, c_idx] *= 1.8  # 80% increase in temperature features
            if "capacity" in c_name.lower():
                X_test[shock_idx:, c_idx] *= 0.7  # Accelerated capacity fade simulation

    print("Training BTCN Model...")
    btcn = BTCNModel(input_dim=len(feature_cols), num_channels=[16, 16], seq_length=15, epochs=30, n_samples=30)
    btcn.fit(X_train, y_train)

    print("Extracting train standards for OOD Detection...")
    mean_train, lo_train, hi_train = btcn.predict(X_train)
    epistemic_stds_train = btcn.get_epistemic_uncertainty(X_train)

    detector = OODDetector(safety_margin=3.0)
    detector.fit(X_train[-len(epistemic_stds_train):], epistemic_stds_train)

    print("Evaluating Test data (with synthetic shock)...")
    mean_test, lo_test, hi_test = btcn.predict(X_test)
    epistemic_stds_test = btcn.get_epistemic_uncertainty(X_test)

    results = detector.detect(X_test[-len(epistemic_stds_test):], epistemic_stds_test)
    mean_adj, lo_adj, hi_adj = detector.adjust_predictions(mean_test, lo_test, hi_test, results)

    c_t = cycles[-len(mean_test):]
    y_test_t = y_test[-len(mean_test):]

    OOD_flags = np.array([1 if r.level == OODLevel.OUT_OF_DISTRIBUTION else 0 for r in results])

    print("Plotting Figure 11...")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(c_t, y_test_t, 'k-', lw=2, label='Ground Truth RUL')
    ax.plot(c_t, mean_test, 'b-', lw=1.5, label='BTCN Mean Prediction')

    # Standard bounds
    ax.fill_between(c_t, lo_test, hi_test, alpha=0.4, color='royalblue', label='Standard CI (95%)')
    # Adjusted OOD bounds
    ax.fill_between(c_t, lo_adj, hi_adj, alpha=0.3, color='crimson', label='Dynamic Safety Margin (OOD Adjusted)')

    # Highlight OOD region
    ood_indices = np.where(OOD_flags == 1)[0]
    if len(ood_indices) > 0:
        ax.axvspan(c_t[ood_indices[0]], c_t[-1], color='crimson', alpha=0.1, label='OOD Danger Zone')
        ax.text(c_t[ood_indices[0]] + 2, max(y_test_t)*0.85, '⚠️ LOW CONFIDENCE\nUnknown Operating Params\nSafety Protocol Triggered', color='darkred', weight='bold')

    ax.set_title('OOD Double Defense: Dynamic Safety Boundaries During Thermal/Capacity Shock')
    ax.set_xlabel('Cycles')
    ax.set_ylabel('RUL (cycles)')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')

    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    filepath = FIG / "fig11_ood_dynamic_boundary.png"
    fig.savefig(filepath, dpi=300)
    plt.close()
    print(f"Fig 11 saved to {filepath}")

if __name__ == '__main__':
    main()
