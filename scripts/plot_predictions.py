"""Generate prediction vs ground truth plots for best models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.data.splitter import DataSplitter
from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.features.extractor import FeatureExtractor
from src.models import BayesianNNModel, TCNModel

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"

loader = UnifiedDataLoader()
df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
validator = DataValidator()
df, _ = validator.validate(df)
extractor = FeatureExtractor()
df = extractor.extract_all(df)

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("cycle", "rul")]
df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)

splitter = DataSplitter()
folds = splitter.logo_cv(df)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for i, (train_df, test_df, test_id) in enumerate(folds):
    bat_id = test_df["battery_id"].iloc[0]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["rul"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["rul"].values.astype(np.float32)
    cycles = test_df["cycle"].values

    # TCN
    tcn = TCNModel(input_dim=len(feature_cols), num_channels=[32,32,64,64], seq_length=30, epochs=100, mc_samples=50)
    tcn.fit(X_train, y_train)
    mean_t, lo_t, hi_t = tcn.predict(X_test)

    c_t = cycles[-len(mean_t):]
    y_test_t = y_test[-len(mean_t):]

    ax = axes[0, i]
    ax.plot(c_t, y_test_t, 'k-', lw=1.5, label='Ground Truth')
    ax.plot(c_t, mean_t, 'b-', lw=1.2, label='TCN Pred')
    ax.fill_between(c_t, lo_t, hi_t, alpha=0.2, color='blue', label='95% CI')
    ax.set_title(f'TCN — {bat_id}')
    ax.set_xlabel('Cycle'); ax.set_ylabel('RUL')
    if i == 0: ax.legend(fontsize=8)

    # BNN
    bnn = BayesianNNModel(input_dim=len(feature_cols), hidden_dim=64, epochs=100, n_samples=100)
    bnn.fit(X_train, y_train)
    mean_b, lo_b, hi_b = bnn.predict(X_test)

    c_b = cycles[-len(mean_b):]
    y_test_b = y_test[-len(mean_b):]

    ax = axes[1, i]
    ax.plot(c_b, y_test_b, 'k-', lw=1.5, label='Ground Truth')
    ax.plot(c_b, mean_b, 'r-', lw=1.2, label='BNN Pred')
    ax.fill_between(c_b, lo_b, hi_b, alpha=0.2, color='red', label='95% CI')
    ax.set_title(f'BayesianNN — {bat_id}')
    ax.set_xlabel('Cycle'); ax.set_ylabel('RUL')
    if i == 0: ax.legend(fontsize=8)

plt.suptitle('Prediction vs Ground Truth with Uncertainty Bounds', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIG / "fig10_prediction_comparison.png")
plt.close()
print("Fig 10 saved.")
