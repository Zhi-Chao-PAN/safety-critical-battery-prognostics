import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

DATA_PATH = "data/processed/housing_with_spatial_clusters.csv"
TARGET = "price"

def main():
    df = pd.read_csv(DATA_PATH)

    results = []

    for c in sorted(df["spatial_cluster"].unique()):
        sub = df[df["spatial_cluster"] == c]

        X = sub[["median_income"]].values
        y = sub[TARGET].values

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        resid = y - y_pred

        corr = np.corrcoef(sub["median_income"], y)[0,1]

        results.append({
            "cluster": int(c),
            "n": len(sub),
            "income_slope": float(model.coef_[0]),
            "income_intercept": float(model.intercept_),
            "income_price_corr": float(corr),
            "residual_std": float(np.std(resid))
        })

    res_df = pd.DataFrame(results)

    print("\n=== Cluster Distribution Diagnostics ===")
    print(res_df)

    print("\n=== Summary ===")
    print(res_df.describe())

    res_df.to_csv("results/cluster_distribution_diagnostics.csv", index=False)
    print("\nSaved to: results/cluster_distribution_diagnostics.csv")

if __name__ == "__main__":
    main()
