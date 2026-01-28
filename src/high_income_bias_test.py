import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor


def main():
    df = pd.read_csv("data/processed/housing.csv")

    X = df[["median_income"]].values
    y = df["price"].values

    # Train models
    lin = LinearRegression().fit(X, y)
    nl = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.1, max_iter=300, random_state=42
    ).fit(X, y)

    df["resid_linear"] = y - lin.predict(X)
    df["resid_nonlinear"] = y - nl.predict(X)

    # High-income regime
    q80 = df["median_income"].quantile(0.8)
    high = df[df["median_income"] >= q80]

    print("\n=== High-Income Residual Bias Test ===")
    print(f"Income threshold (80%): {q80:.3f}")
    print(f"Samples in high-income: {len(high)}")

    # Mean residuals
    mean_lin = high["resid_linear"].mean()
    mean_nl = high["resid_nonlinear"].mean()

    print("\nMean residual (high-income):")
    print(f"Linear:    {mean_lin:.4f}")
    print(f"Nonlinear: {mean_nl:.4f}")

    # One-sample t-test vs 0
    t_lin, p_lin = stats.ttest_1samp(high["resid_linear"], 0.0)
    t_nl, p_nl = stats.ttest_1samp(high["resid_nonlinear"], 0.0)

    print("\nOne-sample t-test (H0: mean residual = 0):")
    print(f"Linear:    t = {t_lin:.3f}, p = {p_lin:.4g}")
    print(f"Nonlinear: t = {t_nl:.3f}, p = {p_nl:.4g}")

    # Save summary
    summary = pd.DataFrame({
        "model": ["linear", "nonlinear"],
        "mean_residual_high_income": [mean_lin, mean_nl],
        "t_stat": [t_lin, t_nl],
        "p_value": [p_lin, p_nl],
        "n_high_income": [len(high), len(high)]
    })

    out_path = "results/high_income_bias_test.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSaved statistical test results to: {out_path}")


if __name__ == "__main__":
    main()
