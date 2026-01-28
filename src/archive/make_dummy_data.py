import os
import numpy as np
import pandas as pd


def main():
    os.makedirs("data/processed", exist_ok=True)

    np.random.seed(42)
    n_samples = 2000

    df = pd.DataFrame({
        "median_income": np.random.normal(5, 2, n_samples),
        "house_age": np.random.randint(1, 50, n_samples),
        "avg_rooms": np.random.normal(5, 1.5, n_samples),
        "avg_bedrooms": np.random.normal(2, 0.5, n_samples),
        "population": np.random.randint(100, 5000, n_samples),
        "latitude": np.random.uniform(32, 42, n_samples),
        "longitude": np.random.uniform(-124, -114, n_samples),
    })

    # Synthetic price with noise (regression target)
    df["price"] = (
        0.5 * df["median_income"]
        - 0.02 * df["house_age"]
        + 0.1 * df["avg_rooms"]
        - 0.05 * df["population"] / 1000
        + np.random.normal(0, 0.5, n_samples)
    )

    output_path = "data/processed/housing.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved synthetic dataset to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
