import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# 明确目标数据路径和特征
DATA_PATH = "data/processed/housing_with_spatial_clusters.csv"
TARGET = "price"
GROUP_COL = "spatial_cluster"
NUM_FEATURES = [
    "median_income",
    "house_age",
    "avg_rooms",
    "avg_bedrooms",
    "population",
    "latitude",
    "longitude",
]
CLUSTER_COL = "spatial_cluster"


def build_cluster_slope_design_matrix(df, num_features, cluster_col):
    """
    Cluster-specific slopes for median_income
    """
    X = df[num_features].copy()
    for c in sorted(df[cluster_col].unique()):
        mask = (df[cluster_col] == c).astype(int)
        X[f"median_income_cluster_{c}"] = df["median_income"] * mask
    X = X.drop(columns=["median_income"])
    return X


def evaluate_cv(X, y, groups=None, spatial=False, n_splits=5):
    rmses = []

    if spatial:
        # 空间分层交叉验证：保证每个空间块不会重复使用
        cv = GroupKFold(n_splits=n_splits)
        splits = cv.split(X, y, groups=groups)
        tag = "Spatial CV"
    else:
        # 常规随机交叉验证
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = cv.split(X, y)
        tag = "Random CV"

    for fold, split in enumerate(splits, 1):
        train_idx, val_idx = split
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 构建和训练模型
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmses.append(rmse)

        print(f"{tag} | Fold {fold} RMSE: {rmse:.4f}")

    return np.mean(rmses), np.std(rmses)


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("\nUsing target:", TARGET)
    y = df[TARGET]
    groups = df[GROUP_COL]

    print("\nBuilding cluster-specific slope design matrix...")
    X = build_cluster_slope_design_matrix(
        df,
        NUM_FEATURES,
        CLUSTER_COL
    )

    print("\n==============================")
    print("Random Cross-Validation")
    print("==============================")
    rand_mean, rand_std = evaluate_cv(X, y, spatial=False)

    print("\n==============================")
    print("Spatial (Group) Cross-Validation")
    print("==============================")
    spat_mean, spat_std = evaluate_cv(X, y, groups=groups, spatial=True)

    results = pd.DataFrame({
        "cv_type": ["random", "spatial_group"],
        "rmse_mean": [rand_mean, spat_mean],
        "rmse_std": [rand_std, spat_std]
    })

    out_path = Path("results/spatial_cv_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print("\n==============================")
    print("Summary")
    print("==============================")
    print(results)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
