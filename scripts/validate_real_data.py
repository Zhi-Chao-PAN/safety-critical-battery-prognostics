#!/usr/bin/env python3
"""
Phase 6: Real-World Same-Cell Noise Robustness Validation

Protocol:
  Each CALCE cell is trained on its own clean trajectory, then evaluated on a
  noisy version of that same trajectory. This validates same-cell sensor-noise
  rejection only. It does NOT measure cross-cell generalization.

For leave-one-cell-out cross-cell validation, use:
  python scripts/validate_real_data_logo.py
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
    load_calce_cell,
    train_reference_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SameCellValidation")

TARGET_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]


def run_cell_experiment(cell_path: Path, noise_level: float = 0.5, seed: int = 42) -> CellResult:
    """Train and evaluate the reference models on one clean/noisy cell pair."""
    cell_id = cell_path.stem
    logger.info("\n%s", "=" * 60)
    logger.info("Processing same-cell robustness for %s", cell_id)
    logger.info("%s", "=" * 60)

    cycles, capacity = load_calce_cell(cell_path)
    if len(cycles) < 20:
        raise ValueError(f"Too few cycles for {cell_id}: {len(cycles)}")

    logger.info(
        "  Loaded %d cycles, capacity range=[%.4f, %.4f]",
        len(cycles),
        float(capacity.min()),
        float(capacity.max()),
    )

    models = train_reference_models({cell_id: (cycles, capacity)})
    result = evaluate_models_on_cell(
        models=models,
        cell_id=cell_id,
        cycles=cycles,
        capacity=capacity,
        noise_level=noise_level,
        seed=seed,
        conditions=("noisy",),
    )

    pinn = result.evaluations["pinn"]["noisy"]
    lstm = result.evaluations["lstm"]["noisy"]
    logger.info(
        "  PINN noisy: RMSE=%.4f, VR=%.2f%% | LSTM noisy: RMSE=%.4f, VR=%.2f%%",
        pinn.rmse,
        pinn.violation_rate,
        lstm.rmse,
        lstm.violation_rate,
    )
    return result


def main():
    output_dir = Path("robustness_results")
    output_dir.mkdir(exist_ok=True)

    calce_dir = Path("data/calce")
    results: list[CellResult] = []

    for cell_name in TARGET_CELLS:
        cell_path = calce_dir / f"{cell_name}.csv"
        if not cell_path.exists():
            logger.warning("Cell file not found: %s", cell_path)
            continue
        try:
            results.append(run_cell_experiment(cell_path))
        except Exception as exc:
            logger.error("Failed on %s: %s", cell_name, exc)

    if not results:
        logger.error("No cells processed successfully.")
        return

    logger.info("\n%s", "=" * 60)
    logger.info("Generating same-cell noise robustness summary (%d cells)...", len(results))
    logger.info("%s", "=" * 60)

    generate_figure(
        results,
        output_dir / "real_data_validation.png",
        title="Real-World CALCE Same-Cell Noise Robustness: PINN vs LSTM",
        condition_order=("noisy",),
    )
    generate_report(
        results,
        output_dir / "real_data_validation_report.md",
        protocol_title="Real-World CALCE Same-Cell Noise Robustness Report",
        setup_lines=[
            "- **Dataset**: CALCE CS2 series lithium-ion batteries",
            "- **Protocol**: train on each cell's clean trajectory, evaluate on a noisy version of the same trajectory",
            "- **Noise Level**: 50% Gaussian (sigma_noise = 0.5 x sigma_capacity)",
            "- **Defense**: Full three-layer physics shield with identical post-processing for all models",
            "- **Seed**: 42",
        ],
        condition_order=("noisy",),
    )

    logger.info("\n%s", "=" * 60)
    logger.info("SAME-CELL NOISE SUMMARY")
    logger.info("%-10s | %10s | %8s | %10s | %8s", "Cell", "PINN RMSE", "PINN VR", "LSTM RMSE", "LSTM VR")
    logger.info("%s", "-" * 60)
    for result in results:
        pinn = result.evaluations["pinn"]["noisy"]
        lstm = result.evaluations["lstm"]["noisy"]
        logger.info(
            "%-10s | %10.4f | %7.2f%% | %10.4f | %7.2f%%",
            result.cell_id,
            pinn.rmse,
            pinn.violation_rate,
            lstm.rmse,
            lstm.violation_rate,
        )
    logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()
