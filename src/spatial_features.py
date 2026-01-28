import pandas as pd
from sklearn.cluster import KMeans


def main():
    data_path = "data/processed/housing.csv"
    out_path = "data/processed/housing_with_spatial_clusters.csv"

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    coords = df[["latitude", "longitude"]]

    print("Fitting KMeans on spatial coordinates...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df["spatial_cluster"] = kmeans.fit_predict(coords)

    print("Cluster counts:")
    print(df["spatial_cluster"].value_counts().sort_index())

    df.to_csv(out_path, index=False)
    print(f"Saved augmented dataset to: {out_path}")


if __name__ == "__main__":
    main()
