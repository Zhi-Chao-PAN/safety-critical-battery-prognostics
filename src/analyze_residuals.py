# src/analyze_residuals.py

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/housing.csv"
MODEL_PATH = "results/models/ridge_20260125_123820.joblib"
TARGET = "price"
RANDOM_STATE = 42

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    results = X_test.copy()
    results["y_true"] = y_test.values
    results["y_pred"] = y_pred
    results["residual"] = residuals
    results["abs_error"] = np.abs(residuals)

    # 1. Error by price buckets
    results["price_bucket"] = pd.qcut(results["y_true"], q=5)

    bucket_rmse = results.groupby("price_bucket").apply(
        lambda g: np.sqrt(np.mean((g["residual"]) ** 2))
    )

    print("\n=== RMSE by True Price Bucket ===")
    print(bucket_rmse)

    # 2. Correlation between residual and features
    corr = results.drop(columns=["y_true", "y_pred", "price_bucket"]).corr()["residual"].sort_values()

    print("\n=== Correlation with Residual ===")
    print(corr)

    # Save for further analysis
    results.to_csv("results/residual_analysis_test.csv", index=False)
    print("\nSaved: results/residual_analysis_test.csv")

if __name__ == "__main__":
    main()
