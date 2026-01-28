# src/train_gam.py

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from pygam import LinearGAM, s, te


def main():
    data_path = "data/processed/housing_with_spatial_clusters.csv"
    target_col = "price"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # =========================
    # Feature setup
    # =========================
    num_features = [
        "median_income",
        "house_age",
        "avg_rooms",
        "avg_bedrooms",
        "population",
    ]

    lat_col = "latitude"
    lon_col = "longitude"

    all_features = num_features + [lat_col, lon_col]

    X = df[all_features].values
    y = df[target_col].values

    # =========================
    # Train / Val / Test split
    # =========================
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42
    )

    print(f"Train size: {X_train.shape}")
    print(f"Val size:   {X_val.shape}")
    print(f"Test size:  {X_test.shape}")

    # =========================
    # Build GAM
    # =========================
    # Feature index map
    idx_income = all_features.index("median_income")
    idx_lat = all_features.index("latitude")
    idx_lon = all_features.index("longitude")

    print("\nBuilding GAM with:")
    print("- s(median_income)  [nonlinear income effect]")
    print("- te(latitude, longitude) [smooth spatial surface]")
    print("- linear terms for other variables")

    gam = LinearGAM(
        # Nonlinear income
        s(idx_income, n_splines=15)
        +
        # Spatial smooth (2D tensor product)
        te(idx_lat, idx_lon, n_splines=15)
        +
        # Linear terms for remaining numeric variables
        s(all_features.index("house_age"), n_splines=5, lam=0.1)
        +
        s(all_features.index("avg_rooms"), n_splines=5, lam=0.1)
        +
        s(all_features.index("avg_bedrooms"), n_splines=5, lam=0.1)
        +
        s(all_features.index("population"), n_splines=5, lam=0.1)
    )

    print("\nFitting GAM...")
    gam.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    val_pred = gam.predict(X_val)
    test_pred = gam.predict(X_test)

    val_rmse = rmse(y_val, val_pred)
    test_rmse = rmse(y_test, test_pred)

    print("\n================ GAM Results ================")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print("============================================")

    # =========================
    # Save model
    # =========================
    os.makedirs("results/models", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"results/models/gam_spatial_{timestamp}.joblib"

    joblib.dump(gam, model_path)
    print(f"Saved GAM model to: {model_path}")

    # =========================
    # Save partial dependence for income
    # =========================
    os.makedirs("results", exist_ok=True)

    XX = gam.generate_X_grid(term=0)  # s(median_income)
    pd_income = gam.partial_dependence(term=0, X=XX)

    income_curve = pd.DataFrame({
        "median_income": XX[:, idx_income],
        "partial_effect": pd_income,
    })

    curve_path = "results/gam_income_curve.csv"
    income_curve.to_csv(curve_path, index=False)
    print(f"Saved GAM income curve to: {curve_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
