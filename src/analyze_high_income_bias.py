import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def main():
    data_path = "data/processed/housing.csv"
    feature = "median_income"
    target = "price"

    df = pd.read_csv(data_path)

    X = df[feature].values.reshape(-1, 1)
    y = df[target].values

    # =========================
    # Train models
    # =========================
    lin = LinearRegression()
    lin.fit(X, y)

    nl = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.1,
        max_iter=300,
        random_state=42
    )
    nl.fit(X, y)

    y_hat_lin = lin.predict(X)
    y_hat_nl = nl.predict(X)

    # =========================
    # Residuals
    # =========================
    resid_lin = y - y_hat_lin
    resid_nl = y - y_hat_nl

    df["resid_linear"] = resid_lin
    df["resid_nonlinear"] = resid_nl

    # =========================
    # Define high-income regime (top 20%)
    # =========================
    q80 = df[feature].quantile(0.8)
    high_df = df[df[feature] >= q80]

    print("\n=== High-Income Regime (Top 20%) ===")
    print(f"Income threshold (80th pct): {q80:.3f}")
    print(f"Number of samples: {len(high_df)}")

    print("\nResidual summary (HIGH income):")
    print(high_df[["resid_linear", "resid_nonlinear"]].describe())

    # =========================
    # RMSE by regime
    # =========================
    rmse_lin_high = np.sqrt(mean_squared_error(
        high_df[target],
        high_df[target] - high_df["resid_linear"]
    ))
    rmse_nl_high = np.sqrt(mean_squared_error(
        high_df[target],
        high_df[target] - high_df["resid_nonlinear"]
    ))

    print("\nRMSE (HIGH income only):")
    print(f"Linear RMSE:    {rmse_lin_high:.4f}")
    print(f"Nonlinear RMSE: {rmse_nl_high:.4f}")

    # =========================
    # Plot residuals vs income
    # =========================
    plt.figure(figsize=(8, 6))

    plt.scatter(df[feature], df["resid_linear"], alpha=0.4, label="Linear residuals")
    plt.scatter(df[feature], df["resid_nonlinear"], alpha=0.4, label="Nonlinear residuals")

    plt.axhline(0, linestyle="--")
    plt.axvline(q80, linestyle=":", label="High-income threshold (80%)")

    plt.xlabel("Median Income")
    plt.ylabel("Residual (y - y_hat)")
    plt.title("Residuals vs Income: Linear vs Nonlinear")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
