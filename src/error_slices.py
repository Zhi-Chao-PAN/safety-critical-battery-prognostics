import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

# =========================
# CONFIG
# =========================
DATA_PATH = "data/processed/housing_with_spatial_clusters.csv"
MODEL_PATH = "results/models/ridge_spatial_20260125_142901.joblib"
TARGET_COL = "price"

# =========================
# MAIN
# =========================
def main():
    print(f"Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # =========================
    # Predict + residuals
    # =========================
    y_pred = model.predict(X)
    df["pred"] = y_pred
    df["residual"] = y - y_pred
    df["abs_error"] = np.abs(df["residual"])

    # =========================
    # Create slices
    # =========================
    # Income buckets
    df["income_bucket"] = pd.qcut(df["median_income"], 4, labels=[
        "low", "mid-low", "mid-high", "high"
    ])

    # Age buckets
    df["age_bucket"] = pd.qcut(df["house_age"], 4, labels=[
        "new", "mid-new", "mid-old", "old"
    ])

    # Latitude buckets (north/south)
    df["lat_bucket"] = pd.qcut(df["latitude"], 4, labels=[
        "south", "mid-south", "mid-north", "north"
    ])

    slice_summaries = []

    for col in ["income_bucket", "age_bucket", "lat_bucket", "spatial_cluster"]:
        grp = df.groupby(col)["abs_error"].agg(["mean", "count"]).reset_index()
        grp["slice_type"] = col
        grp.rename(columns={col: "slice"}, inplace=True)
        slice_summaries.append(grp)

    summary = pd.concat(slice_summaries, ignore_index=True)
    summary_sorted = summary.sort_values("mean", ascending=False)

    print("\n=== Worst Error Slices (Top 15) ===")
    print(summary_sorted.head(15))

    # =========================
    # Save
    # =========================
    out_path = "results/error_slices_summary_spatial.csv"
    summary_sorted.to_csv(out_path, index=False)
    print(f"\nSaved slice analysis to: {out_path}")


if __name__ == "__main__":
    main()
