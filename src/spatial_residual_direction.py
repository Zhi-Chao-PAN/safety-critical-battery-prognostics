import pandas as pd
import joblib

DATA_PATH = "data/processed/housing_with_spatial_clusters.csv"
MODEL_PATH = "results/models/ridge_spatial_20260125_142901.joblib"
TARGET_COL = "price"

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

df["pred"] = model.predict(X)
df["residual"] = y - df["pred"]

summary = (
    df.groupby("spatial_cluster")["residual"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .sort_values("mean")
)

print("\n=== Spatial Residual Direction ===")
print(summary)

summary.to_csv("results/spatial_residual_direction.csv", index=False)
