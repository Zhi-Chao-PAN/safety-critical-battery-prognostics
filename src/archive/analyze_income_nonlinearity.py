import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


def main():
    data_path = "data/processed/housing.csv"
    target = "price"
    feature = "median_income"

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Using target column: {target}")

    # =========================
    # Prepare data
    # =========================
    X = df[feature].values
    y = df[target].values

    # Grid for smooth curve
    X_plot = np.linspace(X.min(), X.max(), 200)

    # =========================
    # Nonlinear model (HGB)
    # =========================
    print("Training nonlinear income-only model (HistGradientBoosting)...")
    nonlinear_model = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.1,
        max_iter=300,
        random_state=42
    )
    nonlinear_model.fit(X.reshape(-1, 1), y)

    y_pred_nonlinear = nonlinear_model.predict(X_plot.reshape(-1, 1))

    # Save nonlinear curve
    curve_df = pd.DataFrame({
        "median_income": X_plot,
        "predicted_price_nonlinear": y_pred_nonlinear
    })
    curve_path = "results/income_nonlinear_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"Saved nonlinear income curve to: {curve_path}")

    # Save nonlinear model
    nonlinear_model_path = "results/models/income_only_hgb_nonlinear.joblib"
    joblib.dump(nonlinear_model, nonlinear_model_path)
    print(f"Saved income-only nonlinear model to: {nonlinear_model_path}")

    # =========================
    # Linear baseline model
    # =========================
    print("Training linear income-only baseline (LinearRegression)...")
    linear_model = LinearRegression()
    linear_model.fit(X.reshape(-1, 1), y)

    y_pred_linear = linear_model.predict(X_plot.reshape(-1, 1))

    # =========================
    # Diagnostics: RMSE on training (for shape comparison only)
    # =========================
    y_hat_nl_train = nonlinear_model.predict(X.reshape(-1, 1))
    y_hat_lin_train = linear_model.predict(X.reshape(-1, 1))

    rmse_nl = np.sqrt(mean_squared_error(y, y_hat_nl_train))
    rmse_lin = np.sqrt(mean_squared_error(y, y_hat_lin_train))

    print("\n=== Income Nonlinearity Diagnostics ===")
    print(f"Linear RMSE (train):    {rmse_lin:.4f}")
    print(f"Nonlinear RMSE (train): {rmse_nl:.4f}")

    # =========================
    # Plot: Linear vs Nonlinear
    # =========================
    plt.figure(figsize=(8, 6))

    plt.plot(X_plot, y_pred_nonlinear, label="Nonlinear model")
    plt.plot(X_plot, y_pred_linear, linestyle="--", label="Linear model")

    plt.xlabel("Median Income")
    plt.ylabel("Predicted Price")
    plt.title("Linear vs Nonlinear: Income → Housing Price")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # =========================
    # Extra: where linear deviates most
    # =========================
    diff = y_pred_nonlinear - y_pred_linear
    print("\n=== Linear vs Nonlinear Deviation (on curve grid) ===")
    print(pd.Series(diff).describe())


if __name__ == "__main__":
    main()
