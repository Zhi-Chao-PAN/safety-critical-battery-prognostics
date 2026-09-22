"""Validate the public aggregate-only LOGO seed-sweep artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"verification failed: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    protocol = json.loads((args.results / "protocol.json").read_text())
    verification = json.loads((args.results / "verification.json").read_text())
    metrics = pd.read_csv(args.results / "per_fold_metrics.csv")
    by_seed = pd.read_csv(args.results / "summary_by_seed.csv")
    by_model = pd.read_csv(args.results / "summary_by_model.csv")
    keys = ["training_seed", "held_out_cell", "model"]
    expected = len(protocol["training_seeds"]) * len(protocol["cells"]) * 2
    require(len(metrics) == expected, "metric row count")
    require(not metrics.duplicated(keys).any(), "duplicate seed/cell/model row")
    require(set(metrics.training_seed) == set(protocol["training_seeds"]), "training seeds")
    require(set(metrics.held_out_cell) == set(protocol["cells"]), "cells")
    require(set(metrics.model) == {"pinn", "lstm"}, "models")
    require((metrics.violation_count == 0).all(), "published violation counts")
    for filename, key in (
        ("per_fold_metrics.csv", "per_fold_metrics_sha256"),
        ("summary_by_seed.csv", "summary_by_seed_sha256"),
        ("summary_by_model.csv", "summary_by_model_sha256"),
    ):
        require(digest(args.results / filename) == verification[key], f"{filename} hash")

    expected_seed = (metrics.groupby(["training_seed", "model"], as_index=False)
                     .agg(rmse_mean=("rmse", "mean"), rmse_std_over_cells=("rmse", "std"),
                          violation_rate_mean=("violation_rate", "mean"),
                          violation_rate_std_over_cells=("violation_rate", "std"),
                          folds=("rmse", "size")))
    expected_model = (metrics.groupby("model", as_index=False)
                      .agg(rmse_mean=("rmse", "mean"), rmse_std_over_seed_cell=("rmse", "std"),
                           violation_rate_mean=("violation_rate", "mean"),
                           violation_rate_std_over_seed_cell=("violation_rate", "std"),
                           folds=("rmse", "size")))
    for expected_frame, actual_frame, label in (
        (expected_seed, by_seed, "seed summary"),
        (expected_model, by_model, "model summary"),
    ):
        require(list(expected_frame.columns) == list(actual_frame.columns), f"{label} columns")
        require(np.allclose(expected_frame.select_dtypes("number"),
                            actual_frame.select_dtypes("number"), rtol=0, atol=1e-12), label)
        require((expected_frame.select_dtypes(exclude="number").to_numpy() ==
                 actual_frame.select_dtypes(exclude="number").to_numpy()).all(), label)

    print(json.dumps({"status": "VERIFIED", "metric_rows": len(metrics),
                      "summaries_recomputed": True}, indent=2))


if __name__ == "__main__":
    main()
