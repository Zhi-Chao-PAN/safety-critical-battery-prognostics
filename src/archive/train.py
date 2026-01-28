import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def resolve_data_path(cfg):
    data_cfg = cfg.get("data", {})
    for key in ["path", "file", "input_path", "csv", "processed_path"]:
        if key in data_cfg:
            return data_cfg[key]
    raise KeyError(
        f"Could not find data path in config. "
        f"Expected one of: data.path / data.file / data.input_path / data.csv / data.processed_path. "
        f"Found keys: {list(data_cfg.keys())}"
    )


def add_cluster_interactions(df, cluster_values):
    """
    Add income * cluster_k interaction columns to df.
    """
    df = df.copy()
    for k in cluster_values:
        col = f"income_cluster_{k}"
        df[col] = df["median_income"] * (df["spatial_cluster"] == k).astype(int)
    return df


def build_pipeline(interaction_cols):
    num_base = [
        "median_income",
        "house_age",
        "avg_rooms",
        "avg_bedrooms",
        "population",
        "latitude",
        "longitude",
    ]

    numeric_features = num_base + interaction_cols

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("scaler", StandardScaler())
                ]),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                ["spatial_cluster"],
            ),
        ]
    )

    model = Ridge(alpha=1.0, random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    return pipeline, numeric_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_path = resolve_data_path(cfg)
    target = cfg["data"]["target"]

    print(f"Loaded data from: {data_path}")
    df = pd.read_csv(data_path)

    print(f"Data shape: {df.shape}")

    # =========================
    # Split
    # =========================
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]

    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=42)
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(df_trainval, test_size=val_ratio, random_state=42)

    # =========================
    # Add cluster-specific interaction features
    # =========================
    cluster_values = sorted(df["spatial_cluster"].unique().tolist())
    interaction_cols = [f"income_cluster_{k}" for k in cluster_values]

    df_train = add_cluster_interactions(df_train, cluster_values)
    df_val   = add_cluster_interactions(df_val, cluster_values)
    df_test  = add_cluster_interactions(df_test, cluster_values)

    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    X_val = df_val.drop(columns=[target])
    y_val = df_val[target]

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    print(f"Train size: {X_train.shape}")
    print(f"Val size:   {X_val.shape}")
    print(f"Test size:  {X_test.shape}")

    # =========================
    # Build pipeline
    # =========================
    pipeline, numeric_features = build_pipeline(interaction_cols)

    print("=" * 60)
    print("Model object (Cluster-specific income slopes):")
    print(pipeline)
    print("Interaction columns:", interaction_cols)
    print("=" * 60)

    # =========================
    # Train
    # =========================
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    val_rmse = mean_squared_error(y_val, val_pred, squared=False)
    test_rmse = mean_squared_error(y_test, test_pred, squared=False)

    print("=" * 60)
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print("=" * 60)

    # =========================
    # Save
    # =========================
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ridge_cluster_slopes_{ts}"

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/config_snapshots", exist_ok=True)

    model_path = f"results/models/{model_name}.joblib"
    joblib.dump(pipeline, model_path)

    cfg_snapshot_path = f"results/config_snapshots/{model_name}.json"
    with open(cfg_snapshot_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Config snapshot saved to: {cfg_snapshot_path}")


if __name__ == "__main__":
    main()
