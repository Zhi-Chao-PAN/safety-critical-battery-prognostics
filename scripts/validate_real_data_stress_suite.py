#!/usr/bin/env python3
"""
Phase 7: Real-Data Multi-Seed Corruption Stress Suite

Runs both same-cell and LOGO validation protocols across multiple random seeds
and multiple corruption families without retraining the same fold repeatedly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.real_data_validation import (
    CORRUPTION_DISPLAY_NAMES,
    evaluate_models_on_signals,
    build_condition_signals,
    load_calce_cells,
    summarize_condition,
    train_reference_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RealDataStressSuite")

TARGET_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]
SEEDS = [42, 123, 456, 789, 1024]
CORRUPTIONS = ("gaussian", "bias_drift", "impulse_spikes", "missing_segments")
MODEL_ORDER = ("pinn", "lstm")
TRAINING_SEED = 42
PROTOCOL_LABELS = {
    "same_cell": "Same-Cell Noise Robustness",
    "logo": "LOGO Cross-Cell Validation",
}


def load_target_cells() -> dict[str, tuple]:
    """Load the active CALCE target cells for real-data validation."""
    cell_paths = [Path("data/calce") / f"{cell_name}.csv" for cell_name in TARGET_CELLS]
    available_paths = [path for path in cell_paths if path.exists()]
    missing = [path for path in cell_paths if not path.exists()]
    for path in missing:
        logger.warning("Cell file not found: %s", path)
    if len(available_paths) < 2:
        raise RuntimeError("At least two CALCE cells are required for the stress suite.")
    return load_calce_cells(available_paths)


def prepare_same_cell_protocol(cells: dict[str, tuple]) -> dict[str, dict[str, object]]:
    """Train one model pair per cell for same-cell evaluation."""
    protocol_bundle: dict[str, dict[str, object]] = {}
    for cell_id, (cycles, capacity) in cells.items():
        logger.info("Training same-cell models for %s", cell_id)
        protocol_bundle[cell_id] = {
            "models": train_reference_models({cell_id: (cycles, capacity)}, training_seed=TRAINING_SEED),
            "cycles": cycles,
            "capacity": capacity,
        }
    return protocol_bundle


def prepare_logo_protocol(cells: dict[str, tuple]) -> dict[str, dict[str, object]]:
    """Train one model pair per held-out LOGO fold."""
    protocol_bundle: dict[str, dict[str, object]] = {}
    for held_out_cell, (cycles, capacity) in cells.items():
        train_cells = {cell_id: payload for cell_id, payload in cells.items() if cell_id != held_out_cell}
        logger.info("Training LOGO fold with held-out cell %s", held_out_cell)
        protocol_bundle[held_out_cell] = {
            "models": train_reference_models(train_cells, training_seed=TRAINING_SEED),
            "cycles": cycles,
            "capacity": capacity,
        }
    return protocol_bundle


def run_protocol_sweep(
    protocol_name: str,
    protocol_bundle: dict[str, dict[str, object]],
    severity: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one protocol across all configured seeds and corruption families."""
    detailed_rows: list[dict[str, object]] = []
    seed_summary_rows: list[dict[str, object]] = []

    for corruption in CORRUPTIONS:
        for seed in SEEDS:
            seed_results = []
            for cell_id, payload in protocol_bundle.items():
                cycles = payload["cycles"]
                capacity = payload["capacity"]
                models = payload["models"]
                signals = build_condition_signals(
                    capacity=capacity,
                    corruption=corruption,
                    severity=severity,
                    seed=seed,
                    include_clean=False,
                    noisy_label=corruption,
                )
                result = evaluate_models_on_signals(
                    models=models,
                    cell_id=cell_id,
                    cycles=cycles,
                    capacity=capacity,
                    signals=signals,
                    scatter_condition=corruption,
                )
                seed_results.append(result)

                for model_name in MODEL_ORDER:
                    metrics = result.evaluations[model_name][corruption]
                    detailed_rows.append(
                        {
                            "protocol": protocol_name,
                            "cell_id": cell_id,
                            "seed": seed,
                            "corruption": corruption,
                            "model": model_name,
                            "rmse": metrics.rmse,
                            "violation_rate": metrics.violation_rate,
                            "violation_count": metrics.violation_count,
                        }
                    )

            for model_name in MODEL_ORDER:
                summary = summarize_condition(seed_results, model_name, corruption)
                seed_summary_rows.append(
                    {
                        "protocol": protocol_name,
                        "seed": seed,
                        "corruption": corruption,
                        "model": model_name,
                        "rmse": summary["rmse"],
                        "violation_rate": summary["violation_rate"],
                    }
                )

    detailed_df = pd.DataFrame(detailed_rows)
    seed_summary_df = pd.DataFrame(seed_summary_rows)
    return detailed_df, seed_summary_df


def aggregate_seed_summaries(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate seed-level protocol summaries into mean/std tables."""
    summary_df = (
        seed_summary_df.groupby(["protocol", "corruption", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            violation_rate_mean=("violation_rate", "mean"),
            violation_rate_std=("violation_rate", "std"),
        )
        .sort_values(["protocol", "corruption", "model"])
    )
    return summary_df.fillna(0.0)


def build_markdown_report(
    summary_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write the multi-seed, multi-corruption summary report."""
    lines = [
        "# Real-Data Multi-Seed Corruption Stress Suite",
        "",
        "## Experimental Setup",
        "",
        "- **Protocols**: same-cell noise robustness and LOGO cross-cell validation",
        "- **Cells**: CALCE CS2_33-CS2_38",
        "- **Training seed**: 42 for each protocol fold",
        "- **Seeds**: 42, 123, 456, 789, 1024",
        "- **Corruptions**: Gaussian noise, bias drift, impulse spikes, missing segments",
        "- **Severity**: 50% scale relative to per-cell capacity variability",
        "- **Reporting**: seed-level cell averages summarized as mean ± std across seeds",
        "",
    ]

    for protocol_name, protocol_label in PROTOCOL_LABELS.items():
        lines.append(f"## {protocol_label}")
        lines.append("")
        lines.append("| Corruption | PINN RMSE (mean ± std) | PINN VR (mean ± std) | LSTM RMSE (mean ± std) | LSTM VR (mean ± std) | Hardest Fold |")
        lines.append("|------------|------------------------|----------------------|------------------------|----------------------|--------------|")

        for corruption in CORRUPTIONS:
            protocol_slice = summary_df[
                (summary_df["protocol"] == protocol_name) & (summary_df["corruption"] == corruption)
            ]
            pinn = protocol_slice[protocol_slice["model"] == "pinn"].iloc[0]
            lstm = protocol_slice[protocol_slice["model"] == "lstm"].iloc[0]

            hardest = (
                detailed_df[
                    (detailed_df["protocol"] == protocol_name)
                    & (detailed_df["corruption"] == corruption)
                    & (detailed_df["model"] == "pinn")
                ]
                .sort_values("rmse", ascending=False)
                .iloc[0]
            )
            hardest_fold = f"{hardest['cell_id']} (seed {int(hardest['seed'])}, RMSE {hardest['rmse']:.4f})"

            lines.append(
                "| "
                f"{CORRUPTION_DISPLAY_NAMES[corruption]} | "
                f"{pinn['rmse_mean']:.4f} ± {pinn['rmse_std']:.4f} | "
                f"{pinn['violation_rate_mean']:.2f}% ± {pinn['violation_rate_std']:.2f}% | "
                f"{lstm['rmse_mean']:.4f} ± {lstm['rmse_std']:.4f} | "
                f"{lstm['violation_rate_mean']:.2f}% ± {lstm['violation_rate_std']:.2f}% | "
                f"{hardest_fold} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Same-cell and LOGO are reported separately to prevent protocol leakage in the repository narrative.",
            "- Multi-seed statistics show whether the real-data conclusions are stable or just artifacts of one random corruption draw.",
            "- Additional corruption families are reported as stress tests, not as replacements for the baseline Gaussian protocol.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Stress-suite report saved: %s", output_path)


def main() -> None:
    output_dir = Path("robustness_results")
    output_dir.mkdir(exist_ok=True)

    cells = load_target_cells()
    same_cell_bundle = prepare_same_cell_protocol(cells)
    logo_bundle = prepare_logo_protocol(cells)

    same_cell_detail, same_cell_seed = run_protocol_sweep("same_cell", same_cell_bundle)
    logo_detail, logo_seed = run_protocol_sweep("logo", logo_bundle)

    detailed_df = pd.concat([same_cell_detail, logo_detail], ignore_index=True)
    seed_summary_df = pd.concat([same_cell_seed, logo_seed], ignore_index=True)
    summary_df = aggregate_seed_summaries(seed_summary_df)

    detailed_path = output_dir / "real_data_stress_suite_details.csv"
    seed_summary_path = output_dir / "real_data_stress_suite_seed_summary.csv"
    summary_path = output_dir / "real_data_stress_suite_summary.csv"
    report_path = output_dir / "real_data_stress_suite_report.md"

    detailed_df.to_csv(detailed_path, index=False)
    seed_summary_df.to_csv(seed_summary_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    build_markdown_report(summary_df, detailed_df, report_path)

    logger.info("Detailed rows saved: %s", detailed_path)
    logger.info("Seed summary saved: %s", seed_summary_path)
    logger.info("Aggregated summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
