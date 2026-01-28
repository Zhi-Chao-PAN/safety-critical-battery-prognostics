import pandas as pd

def main():
    ridge_path = "results/cluster_income_slopes.csv"
    bayes_path = "results/bayes_cluster_income_slopes.csv"

    ridge = pd.read_csv(ridge_path)
    bayes = pd.read_csv(bayes_path)

    ridge = ridge.rename(columns={"coef": "ridge_slope"})
    bayes = bayes.rename(columns={"posterior_mean_income_slope": "bayes_slope"})

    df = pd.merge(ridge, bayes, on="cluster")

    global_mean = df["ridge_slope"].mean()
    df["shrinkage"] = df["bayes_slope"] - df["ridge_slope"]
    df["toward_global"] = df["bayes_slope"] - global_mean

    print("\n=== Shrinkage Comparison ===")
    print(df)

    print("\n=== Summary ===")
    print(df[["ridge_slope", "bayes_slope"]].describe())

    df.to_csv("results/shrinkage_comparison.csv", index=False)
    print("\nSaved to: results/shrinkage_comparison.csv")

if __name__ == "__main__":
    main()
