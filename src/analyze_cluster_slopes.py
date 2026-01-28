import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = "results/models/ridge_cluster_slopes_20260126_133807.joblib"
DATA_PATH = "data/processed/housing_with_spatial_clusters.csv"
OUTPUT_PATH = "results/cluster_income_slopes.csv"


def main():
    print(f"Loading model: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    print(f"Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Get feature names after preprocessing
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        print("ERROR: Could not extract feature names from ColumnTransformer.")
        return

    coefs = model.coef_

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    })

    # Filter income_cluster interaction terms
    income_cluster_mask = coef_df["feature"].str.contains("income_cluster_")
    income_cluster_df = coef_df[income_cluster_mask].copy()

    # Parse cluster id
    income_cluster_df["cluster"] = (
        income_cluster_df["feature"]
        .str.extract(r"income_cluster_(\d+)")
        .astype(int)
    )

    income_cluster_df = income_cluster_df.sort_values("cluster")

    print("\n=== Cluster-Specific Income Slopes ===")
    print(income_cluster_df[["cluster", "coef"]])

    # Save results
    Path("results").mkdir(exist_ok=True)
    income_cluster_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved cluster-specific slopes to: {OUTPUT_PATH}")

    # Also show summary stats
    print("\n=== Summary ===")
    print(income_cluster_df["coef"].describe())


if __name__ == "__main__":
    main()
