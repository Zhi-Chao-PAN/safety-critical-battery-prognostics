#!/usr/bin/env python3
"""
Phase 6: Real-World Leave-One-Cell-Out (LOGO) Validation

Protocol:
  For each held-out CALCE cell, train on all other clean cells and evaluate on
  the held-out cell under both clean and noisy conditions.

This script provides the real cross-cell robustness evidence that is distinct
from the same-cell noise protocol in `scripts/validate_real_data.py`.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.real_data_validation import (
    CellResult,
    evaluate_models_on_cell,
    generate_figure,
    generate_report,
    load_calce_cells,
    train_reference_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LOGOValidation")

TARGET_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]


def run_logo_fold(
    held_out_cell: str,
    cells: dict[str, tuple],
    noise_level: float = 0.5,
    seed: int = 42,
) -> CellResult:
    """Train on all non-held-out cells and evaluate on the held-out cell."""
    logger.info("\n%s", "=" * 60)
    logger.info("LOGO fold: held-out cell = %s", held_out_cell)
    logger.info("%s", "=" * 60)

    train_cells = {cell_id: payload for cell_id, payload in cells.items() if cell_id != held_out_cell}
    eval_cycles, eval_capacity = cells[held_out_cell]

    if not train_cells:
        raise ValueError("LOGO fold requires at least one training cell")

    logger.info("  Training on %d cells, evaluating on %s", len(train_cells), held_out_cell)
    models = train_reference_models(train_cells)
    result = evaluate_models_on_cell(
        models=models,
        cell_id=held_out_cell,
        cycles=eval_cycles,
        capacity=eval_capacity,
        noise_level=noise_level,
        seed=seed,
        conditions=("clean", "noisy"),
    )

    for condition in ("clean", "noisy"):
        pinn = result.evaluations["pinn"][condition]
        lstm = result.evaluations["lstm"][condition]
        logger.info(
            "  %s -> PINN: RMSE=%.4f, VR=%.2f%% | LSTM: RMSE=%.4f, VR=%.2f%%",
            condition,
            pinn.rmse,
            pinn.violation_rate,
            lstm.rmse,
            lstm.violation_rate,
        )

    return result


def main():
    output_dir = Path("robustness_results")
    output_dir.mkdir(exist_ok=True)

    cell_paths = [Path("data/calce") / f"{cell_name}.csv" for cell_name in TARGET_CELLS]
    missing = [str(path) for path in cell_paths if not path.exists()]
    for path in missing:
        logger.warning("Cell file not found: %s", path)

    cell_paths = [path for path in cell_paths if path.exists()]
    if len(cell_paths) < 2:
        logger.error("LOGO validation requires at least two available CALCE cells.")
        return

    cells = load_calce_cells(cell_paths)
    results: list[CellResult] = []
    for held_out_cell in TARGET_CELLS:
        if held_out_cell not in cells:
            continue
        try:
            results.append(run_logo_fold(held_out_cell, cells))
        except Exception as exc:
            logger.error("Failed on held-out cell %s: %s", held_out_cell, exc)

    if not results:
        logger.error("No LOGO folds completed successfully.")
        return

    generate_figure(
        results,
        output_dir / "real_data_logo_validation.png",
        title="Real-World CALCE LOGO Cross-Cell Robustness: PINN vs LSTM",
        condition_order=("clean", "noisy"),
    )
    generate_report(
        results,
        output_dir / "real_data_logo_validation_report.md",
        protocol_title="Real-World CALCE LOGO Cross-Cell Robustness Report",
        setup_lines=[
            "- **Dataset**: CALCE CS2 series lithium-ion batteries",
            "- **Protocol**: leave-one-cell-out; train on all other clean cells and evaluate on the held-out cell",
            "- **Conditions**: clean held-out trajectory and 50% Gaussian noisy held-out trajectory",
            "- **Defense**: Full three-layer physics shield with identical post-processing for all models",
            "- **Seed**: 42",
        ],
        condition_order=("clean", "noisy"),
    )

    logger.info("\n%s", "=" * 60)
    logger.info("LOGO CROSS-CELL SUMMARY")
    logger.info(
        "%-10s | %10s | %8s | %10s | %8s | %10s | %8s | %10s | %8s",
        "Cell",
        "PINN C",
        "VR C",
        "LSTM C",
        "VR C",
        "PINN N",
        "VR N",
        "LSTM N",
        "VR N",
    )
    logger.info("%s", "-" * 110)
    for result in results:
        pinn_clean = result.evaluations["pinn"]["clean"]
        lstm_clean = result.evaluations["lstm"]["clean"]
        pinn_noisy = result.evaluations["pinn"]["noisy"]
        lstm_noisy = result.evaluations["lstm"]["noisy"]
        logger.info(
            "%-10s | %10.4f | %7.2f%% | %10.4f | %7.2f%% | %10.4f | %7.2f%% | %10.4f | %7.2f%%",
            result.cell_id,
            pinn_clean.rmse,
            pinn_clean.violation_rate,
            lstm_clean.rmse,
            lstm_clean.violation_rate,
            pinn_noisy.rmse,
            pinn_noisy.violation_rate,
            lstm_noisy.rmse,
            lstm_noisy.violation_rate,
        )


if __name__ == "__main__":
    main()
