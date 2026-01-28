import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def main():
    # =========================
    # Paths
    # =========================
    data_path = Path("data/processed/housing.csv")
    out_dir = Path("results/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(data_path)

    target = "price"
    income_feature = ["median_income"]
    structural_features = [
        "house_age",
        "avg_rooms",
        "avg_bedrooms",
        "population",
        "latitude",
        "longitude",
    ]

    X = df[income_feature + structural_features]
    y = df[target]

    # =========================
    # Train / Test Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # =========================
    # Stage 1: NONLINEAR Income Model
    # =========================
    X1_train = X_train[income_feature]
    X1_test  = X_test[income_feature]

    stage1 = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )

    stage1.fit(X1_train, y_train)

    y1_train_pred = stage1.predict(X1_train)
    y1_test_pred  = stage1.predict(X1_test)

    stage1_rmse = rmse(y_test, y1_test_pred)

    # =========================
    # Stage 2: Residual Model (Linear on Structural Features)
    # =========================
    # Residuals
    r_train = y_train - y1_train_pred
    r_test  = y_test  - y1_test_pred

    X2_train = X_train[structural_features]
    X2_test  = X_test[structural_features]

    scaler = StandardScaler()
    X2_train_scaled = scaler.fit_transform(X2_train)
    X2_test_scaled  = scaler.transform(X2_test)

    stage2 = Ridge(alpha=1.0, random_state=42)
    stage2.fit(X2_train_scaled, r_train)

    r_test_pred = stage2.predict(X2_test_scaled)

    # =========================
    # Two-stage Combined Prediction
    # =========================
    y_two_stage_pred = y1_test_pred + r_test_pred
    two_stage_rmse = rmse(y_test, y_two_stage_pred)

    # =========================
    # Print Results
    # =========================
    print("\n=== Two-Stage Results (NONLINEAR Stage 1) ===")
    print(f"Stage 1 (income only, nonlinear) RMSE: {stage1_rmse:.4f}")
    print(f"Two-stage combined RMSE:           {two_stage_rmse:.4f}")

    # =========================
    # Residual Model Coefficients
    # =========================
    coef_df = pd.DataFrame({
        "feature": structural_features,
        "residual_coef": stage2.coef_,
        "abs_coef": np.abs(stage2.coef_),
    }).sort_values("abs_coef", ascending=False)

    print("\n=== Residual Model Coefficients ===")
    print(coef_df)

    # =========================
    # Save models
    # =========================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib.dump(stage1, out_dir / f"two_stage_stage1_nonlinear_{ts}.joblib")
    joblib.dump(stage2, out_dir / f"two_stage_stage2_residual_{ts}.joblib")
    joblib.dump(scaler, out_dir / f"two_stage_scaler_{ts}.joblib")

    coef_df.to_csv(f"results/feature_coefficients_two_stage_{ts}.csv", index=False)


if __name__ == "__main__":
    main()
