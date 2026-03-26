"""Run feature importance analysis on best model (TCN)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.data.unified_loader import UnifiedDataLoader
from src.data.validator import DataValidator
from src.evaluation.feature_importance import permutation_importance
from src.features.extractor import FeatureExtractor
from src.models import TCNModel

ROOT = Path(__file__).parent.parent

loader = UnifiedDataLoader()
df = loader.load_all(nasa_dir=str(ROOT / "data" / "battery_data"))
validator = DataValidator()
df, _ = validator.validate(df)
extractor = FeatureExtractor()
df = extractor.extract_all(df)

feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("cycle", "rul")]
df = df.dropna(subset=feature_cols + ["rul"]).reset_index(drop=True)

X = df[feature_cols].values.astype(np.float32)
y = df["rul"].values.astype(np.float32)

model = TCNModel(input_dim=len(feature_cols), num_channels=[32,32,64,64], seq_length=30, epochs=100, mc_samples=50)
model.fit(X, y)

importances = permutation_importance(model, X, y, feature_cols, n_repeats=10)
print("\nFeature Importance (top 10):")
for name, imp in sorted(importances.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:30s} {imp:.4f}")

# Save
import json

with open(ROOT / "results" / "feature_importance.json", "w") as f:
    json.dump(dict(sorted(importances.items(), key=lambda x: -x[1])), f, indent=2)
print("\nSaved to results/feature_importance.json")
