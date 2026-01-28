import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


DATA_PATH = "data/processed/housing_with_spatial_clusters.csv"
TARGET = "price"
CLUSTER_COL = "spatial_cluster"

NUM_FEATURES = [
    "median_income",
    "house_age",
    "avg_rooms",
    "avg_bedrooms",
    "population",
    "latitude",
    "longitude",
]


def build_cluster_slope_design_matrix(df, num_features, cluster_col, all_clusters):
    """
    Build FIXED design matrix with cluster-specific income slopes.
    Ensures all interaction columns exist for all folds.
    """
    X = df[num_features].copy()

    # Always create ALL cluster interaction columns
    for c in all_clusters:
        mask = (df[cluster_col] == c).astype(int)
        X[f"median_income_cluster_{c}"] = df["median_income"] * mask

    # Drop base median_income to avoid collinearity
    X = X.drop(columns=["median_income"])

    return X


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Get full cluster set ONCE (fixed design)
    all_clusters = sorted(df[CLUSTER_COL].unique())
    print(f"Detected clusters: {all_clusters}")

    results = []

    print("\nRunning Leave-One-Cluster-Out CV...")

    for held_out in all_clusters:
        print(f"\n=== Holding out cluster {held_out} ===")

        train_df = df[df[CLUSTER_COL] != held_out].reset_index(drop=True)
        test_df = df[df[CLUSTER_COL] == held_out].reset_index(drop=True)

        # Build fixed design matrices
        X_train = build_cluster_slope_design_matrix(
            train_df, NUM_FEATURES, CLUSTER_COL, all_clusters
        )
        X_test = build_cluster_slope_design_matrix(
            test_df, NUM_FEATURES, CLUSTER_COL, all_clusters
        )

        y_train = train_df[TARGET]
        y_test = test_df[TARGET]

        # Define preprocessing (scale everything)
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), X_train.columns.tolist())
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Cluster {held_out} RMSE: {rmse:.4f}")

        results.append(
            {
                "held_out_cluster": held_out,
                "rmse": rmse,
                "n_test": len(test_df),
            }
        )

    results_df = pd.DataFrame(results)
    print("\n================ LOCO Summary ================")
    print(results_df)
    print("\nOverall RMSE (mean over clusters):", results_df["rmse"].mean())

    out_path = "results/loco_cluster_rmse.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved LOCO results to: {out_path}")


if __name__ == "__main__":
    main()
