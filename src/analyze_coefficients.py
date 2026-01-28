import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================
# 修改为你当前最好的 ridge 模型路径（最新时间戳）
MODEL_PATH = "results/models/ridge_20260125_123821.joblib"
DATA_PATH = "data/processed/housing.csv"
TARGET_COL = "price"
OUTPUT_PATH = "results/feature_coefficients_ridge.csv"

# =========================
# Load model and data
# =========================
print(f"Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

print(f"Loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])

# =========================
# Extract coefficients
# =========================
# Pipeline: [scaler, model]
scaler = model.named_steps["scaler"]
ridge = model.named_steps["model"]

feature_names = X.columns

# Ridge coefficients are in standardized space
coefs = ridge.coef_

coef_df = pd.DataFrame({
    "feature": feature_names,
    "ridge_coef_standardized": coefs,
    "abs_coef": np.abs(coefs),
})

coef_df = coef_df.sort_values("abs_coef", ascending=False)

print("\n=== Ridge Coefficients (Standardized) ===")
print(coef_df)

# =========================
# Save
# =========================
Path("results").mkdir(exist_ok=True)

coef_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved coefficients to: {OUTPUT_PATH}")
