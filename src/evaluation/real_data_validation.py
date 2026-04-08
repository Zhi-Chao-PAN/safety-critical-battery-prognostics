"""
Shared utilities for CALCE real-data robustness validation protocols.

The same helpers are used by:
- same-cell noise robustness
- leave-one-cell-out (LOGO) cross-cell robustness
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.lstm_model import LSTMModel
from src.models.pinn_model import PINNModel
from src.physics.constraints import (
    ConstraintManager,
    MonotonicityConstraint,
    SPMResidualConstraint,
    TemperatureConstraint,
    VoltageConstraint,
)
from src.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {
    "pinn": "PINN",
    "lstm": "LSTM",
}

CORRUPTION_DISPLAY_NAMES = {
    "gaussian": "Gaussian Noise",
    "bias_drift": "Bias Drift",
    "impulse_spikes": "Impulse Spikes",
    "missing_segments": "Missing Segments",
}

MODEL_COLORS = {
    "pinn": ("#2ecc71", "#27ae60"),
    "lstm": ("#e74c3c", "#c0392b"),
}


@dataclass
class ModelMetrics:
    """Metrics and predictions for a single model/condition pair."""

    rmse: float
    violation_rate: float
    violation_count: int
    predictions: np.ndarray


@dataclass
class CellResult:
    """Full evaluation payload for one CALCE cell."""

    cell_id: str
    n_cycles: int
    cycles: np.ndarray
    ground_truth: np.ndarray
    noisy_capacity: np.ndarray | None
    evaluations: dict[str, dict[str, ModelMetrics]]


def load_calce_cell(cell_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a CALCE cell CSV and return (cycles, capacities)."""
    df = pd.read_csv(cell_path)
    cycles = df["cycle"].values.astype(float)
    capacity = df["capacity"].values.astype(float)
    return cycles, capacity


def load_calce_cells(cell_paths: list[Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load multiple CALCE cells keyed by cell id."""
    cells = {}
    for cell_path in cell_paths:
        cycles, capacity = load_calce_cell(cell_path)
        cells[cell_path.stem] = (cycles, capacity)
    return cells


def inject_noise(capacity: np.ndarray, noise_level: float = 0.5, seed: int = 42) -> np.ndarray:
    """Backward-compatible alias for Gaussian noise injection."""
    return apply_capacity_corruption(capacity, corruption="gaussian", severity=noise_level, seed=seed)


def inject_bias_drift(capacity: np.ndarray, severity: float = 0.5, seed: int = 42) -> np.ndarray:
    """Apply a gradual upward sensor drift plus light stochastic jitter."""
    rng = np.random.RandomState(seed)
    amplitude = severity * max(float(np.std(capacity)), 1e-8)
    drift = np.linspace(0.0, amplitude, len(capacity))
    jitter = rng.normal(0.0, amplitude * 0.05, len(capacity))
    return capacity + drift + jitter


def inject_impulse_spikes(capacity: np.ndarray, severity: float = 0.5, seed: int = 42) -> np.ndarray:
    """Inject sparse positive spikes that mimic transient sensor glitches."""
    rng = np.random.RandomState(seed)
    signal = capacity.copy()
    n_spikes = max(1, int(round(len(capacity) * 0.03)))
    amplitude = severity * max(float(np.std(capacity)), 1e-8) * 2.0
    spike_indices = rng.choice(len(capacity), size=min(n_spikes, len(capacity)), replace=False)
    spike_values = rng.uniform(amplitude * 0.5, amplitude * 1.5, size=len(spike_indices))
    signal[spike_indices] += spike_values
    return signal


def inject_missing_segments(capacity: np.ndarray, severity: float = 0.5, seed: int = 42) -> np.ndarray:
    """Drop contiguous sensor segments and impute them with forward/back fill."""
    rng = np.random.RandomState(seed)
    signal = capacity.astype(float).copy()
    n_segments = max(1, int(round(1 + severity * 3)))
    segment_length = max(2, int(round(len(capacity) * 0.05)))

    for _ in range(n_segments):
        max_start = max(len(signal) - segment_length, 0)
        start = rng.randint(0, max_start + 1) if max_start > 0 else 0
        signal[start : start + segment_length] = np.nan

    return pd.Series(signal).ffill().bfill().to_numpy(dtype=float)


def apply_capacity_corruption(
    capacity: np.ndarray,
    corruption: str = "gaussian",
    severity: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Apply one supported corruption family to a capacity trajectory."""
    if corruption == "gaussian":
        rng = np.random.RandomState(seed)
        noise_std = severity * np.std(capacity)
        noise = rng.normal(0, noise_std, len(capacity))
        return capacity + noise
    if corruption == "bias_drift":
        return inject_bias_drift(capacity, severity=severity, seed=seed)
    if corruption == "impulse_spikes":
        return inject_impulse_spikes(capacity, severity=severity, seed=seed)
    if corruption == "missing_segments":
        return inject_missing_segments(capacity, severity=severity, seed=seed)
    raise ValueError(f"Unsupported corruption: {corruption}")


def build_condition_signals(
    capacity: np.ndarray,
    corruption: str = "gaussian",
    severity: float = 0.5,
    seed: int = 42,
    include_clean: bool = False,
    noisy_label: str | None = None,
) -> dict[str, np.ndarray]:
    """Create an ordered mapping of evaluation conditions to sensor signals."""
    label = noisy_label or corruption
    signals: dict[str, np.ndarray] = {}
    if include_clean:
        signals["clean"] = capacity
    signals[label] = apply_capacity_corruption(capacity, corruption=corruption, severity=severity, seed=seed)
    return signals


def create_feature_matrix(cycles: np.ndarray, capacity_signal: np.ndarray) -> np.ndarray:
    """Build the 2-feature matrix used by the reference models."""
    return np.column_stack([cycles, capacity_signal])


def compute_violations(predictions: np.ndarray) -> tuple[int, float]:
    """Count monotonicity violations in a capacity trajectory."""
    violations = 0
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i - 1] + 1e-10:
            violations += 1
    rate = violations / max(len(predictions) - 1, 1) * 100.0
    return violations, rate


def apply_ema_smoothing(predictions: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = np.empty_like(predictions)
    smoothed[0] = predictions[0]
    for i in range(1, len(predictions)):
        smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def apply_running_minimum(predictions: np.ndarray) -> np.ndarray:
    """Project predictions to a monotonically non-increasing trajectory."""
    projected = np.empty_like(predictions)
    projected[0] = predictions[0]
    for i in range(1, len(predictions)):
        projected[i] = min(projected[i - 1], predictions[i])
    return projected


def postprocess_capacity_predictions(predictions: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Apply the fairness-matched post-processing chain."""
    predictions = apply_ema_smoothing(predictions, alpha=alpha)
    return apply_running_minimum(predictions)


def build_constraint_manager(capacity: np.ndarray) -> ConstraintManager:
    """Build the shared capacity-space constraint stack for CALCE validation."""
    cm = ConstraintManager()
    cm.add_constraint(MonotonicityConstraint(weight=1.0, adaptive=True))
    cm.add_constraint(SPMResidualConstraint(weight=0.1, adaptive=True))

    capacity_ceiling = max(float(np.max(capacity)) * 1.1, 2.2)
    cm.add_constraint(
        VoltageConstraint(
            v_min=0.0,
            v_max=capacity_ceiling,
            weight=0.05,
            adaptive=True,
        )
    )
    cm.add_constraint(TemperatureConstraint(t_max=2.2, weight=0.01, adaptive=True))
    return cm


def _build_training_payload(
    train_cells: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate clean training cells into shared arrays."""
    features = []
    targets = []
    group_ids = []

    for cell_id, (cycles, capacity) in train_cells.items():
        features.append(create_feature_matrix(cycles, capacity))
        targets.append(capacity)
        group_ids.append(np.repeat(cell_id, len(cycles)))

    return (
        np.concatenate(features, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(group_ids, axis=0),
    )


def train_reference_models(
    train_cells: dict[str, tuple[np.ndarray, np.ndarray]],
    training_seed: int = 42,
) -> dict[str, object]:
    """Train the PINN and LSTM reference models on clean cells."""
    set_global_seed(training_seed)
    X_train, y_train, group_ids = _build_training_payload(train_cells)
    constraint_manager = build_constraint_manager(y_train)

    pinn = PINNModel(
        input_dim=2,
        hidden_dim=64,
        dropout=0.05,
        lr=1e-3,
        epochs=500,
        patience=80,
        lambda_physics=0.1,
        lambda_mono=1.0,
        adaptive_weighting=True,
        mc_samples=50,
        device="cpu",
        constraint_manager=constraint_manager,
    )
    pinn.fit(X_train, y_train)

    lstm = LSTMModel(
        input_dim=2,
        hidden_dim=64,
        dropout=0.2,
        seq_length=5,
        epochs=100,
        lr=1e-3,
        device="cpu",
    )
    lstm.fit(X_train, y_train, group_ids=group_ids)

    return {"pinn": pinn, "lstm": lstm}


def _extract_mean_prediction(raw_prediction: tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray) -> np.ndarray:
    """Extract the mean trajectory from a model prediction return value."""
    if isinstance(raw_prediction, tuple):
        return raw_prediction[0]
    return raw_prediction


def _predict_capacity(
    model: object,
    cell_id: str,
    cycles: np.ndarray,
    capacity_signal: np.ndarray,
) -> np.ndarray:
    """Predict a capacity trajectory and align it back to full cell length."""
    X_eval = create_feature_matrix(cycles, capacity_signal)
    raw_prediction = model.predict(X_eval, group_ids=np.repeat(cell_id, len(X_eval)))
    mean_prediction = _extract_mean_prediction(raw_prediction)

    if len(mean_prediction) == 0:
        raise ValueError(f"Model produced no predictions for cell {cell_id}")

    if len(mean_prediction) < len(capacity_signal):
        pad_width = len(capacity_signal) - len(mean_prediction)
        mean_prediction = np.pad(mean_prediction, (pad_width, 0), mode="edge")

    return postprocess_capacity_predictions(mean_prediction)


def evaluate_models_on_signals(
    models: dict[str, object],
    cell_id: str,
    cycles: np.ndarray,
    capacity: np.ndarray,
    signals: dict[str, np.ndarray],
    scatter_condition: str | None = None,
) -> CellResult:
    """Evaluate trained models against an explicit mapping of condition -> signal."""
    if not signals:
        raise ValueError("At least one evaluation signal is required")

    highlighted_condition = scatter_condition
    if highlighted_condition is None:
        highlighted_condition = next((name for name in signals if name != "clean"), None)
    noisy_capacity = None if highlighted_condition is None else np.asarray(signals[highlighted_condition], dtype=np.float64)

    evaluations: dict[str, dict[str, ModelMetrics]] = {}
    for model_name, model in models.items():
        condition_results: dict[str, ModelMetrics] = {}
        for condition, signal in signals.items():
            predicted_capacity = _predict_capacity(model, cell_id, cycles, signal)
            rmse = float(np.sqrt(np.mean((predicted_capacity - capacity) ** 2)))
            violations, violation_rate = compute_violations(predicted_capacity)
            condition_results[condition] = ModelMetrics(
                rmse=rmse,
                violation_rate=violation_rate,
                violation_count=violations,
                predictions=predicted_capacity,
            )
        evaluations[model_name] = condition_results

    return CellResult(
        cell_id=cell_id,
        n_cycles=len(cycles),
        cycles=cycles,
        ground_truth=capacity,
        noisy_capacity=noisy_capacity,
        evaluations=evaluations,
    )


def evaluate_models_on_cell(
    models: dict[str, object],
    cell_id: str,
    cycles: np.ndarray,
    capacity: np.ndarray,
    noise_level: float = 0.5,
    seed: int = 42,
    conditions: tuple[str, ...] = ("noisy",),
) -> CellResult:
    """Evaluate trained models on one cell under the requested conditions."""
    available_signals = {
        "clean": capacity,
        "noisy": inject_noise(capacity, noise_level=noise_level, seed=seed),
    }
    selected_signals = {condition: available_signals[condition] for condition in conditions}
    scatter_condition = "noisy" if "noisy" in selected_signals else None
    return evaluate_models_on_signals(
        models=models,
        cell_id=cell_id,
        cycles=cycles,
        capacity=capacity,
        signals=selected_signals,
        scatter_condition=scatter_condition,
    )


def summarize_condition(results: list[CellResult], model_name: str, condition: str) -> dict[str, float]:
    """Aggregate RMSE and violation metrics across cells for one condition."""
    metrics = [r.evaluations[model_name][condition] for r in results]
    return {
        "rmse": float(np.mean([m.rmse for m in metrics])),
        "violation_rate": float(np.mean([m.violation_rate for m in metrics])),
    }


def generate_figure(
    results: list[CellResult],
    output_path: Path,
    title: str,
    condition_order: tuple[str, ...],
):
    """Generate a multi-cell validation figure for the requested conditions."""
    model_order = ("pinn", "lstm")
    n_rows = len(model_order)
    n_cols = len(results)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.8 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for col, result in enumerate(results):
        for row, model_name in enumerate(model_order):
            ax = axes[row][col]
            ax.plot(
                result.cycles,
                result.ground_truth,
                "k-",
                linewidth=1.5,
                alpha=0.7,
                label="Ground Truth",
            )

            if "noisy" in condition_order and result.noisy_capacity is not None:
                ax.scatter(
                    result.cycles,
                    result.noisy_capacity,
                    c="salmon",
                    alpha=0.10,
                    s=6,
                    zorder=1,
                    label="Noisy Input" if row == 0 else None,
                )

            base_color, alt_color = MODEL_COLORS[model_name]
            metric_lines = []
            for idx, condition in enumerate(condition_order):
                metrics = result.evaluations[model_name][condition]
                color = base_color if idx == 0 else alt_color
                style = "-" if condition == "noisy" else "--"
                label = f"{MODEL_DISPLAY_NAMES[model_name]} ({condition})"
                ax.plot(
                    result.cycles,
                    metrics.predictions,
                    color=color,
                    linewidth=2.0,
                    linestyle=style,
                    zorder=3 + idx,
                    label=label,
                )
                metric_lines.append(
                    f"{condition}: VR {metrics.violation_rate:.1f}% | RMSE {metrics.rmse:.3f}"
                )

            any_unsafe = any(
                result.evaluations[model_name][condition].violation_rate > 0.0
                for condition in condition_order
            )
            box_color = "#f8d7da" if any_unsafe else "#d4edda"
            ax.text(
                0.03,
                0.03,
                "\n".join(metric_lines),
                transform=ax.transAxes,
                fontsize=8.5,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=box_color, edgecolor="gray", alpha=0.9),
            )
            ax.set_title(f"{result.cell_id} - {MODEL_DISPLAY_NAMES[model_name]}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Capacity (Ah)", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("Cycle", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Validation figure saved: %s", output_path)


def generate_report(
    results: list[CellResult],
    output_path: Path,
    protocol_title: str,
    setup_lines: list[str],
    condition_order: tuple[str, ...],
):
    """Generate a markdown report for a validation protocol."""
    lines = [f"# {protocol_title}", "", "## Experimental Setup"]
    lines.extend(setup_lines)
    lines.append("")

    for condition in condition_order:
        lines.append(f"## {condition.title()} Results")
        lines.append("")
        lines.append("| Cell | Cycles | PINN RMSE | PINN VR | LSTM RMSE | LSTM VR |")
        lines.append("|------|--------|-----------|---------|-----------|---------|")

        for result in results:
            pinn = result.evaluations["pinn"][condition]
            lstm = result.evaluations["lstm"][condition]
            lines.append(
                f"| {result.cell_id} | {result.n_cycles} | {pinn.rmse:.4f} | {pinn.violation_rate:.2f}% "
                f"| {lstm.rmse:.4f} | {lstm.violation_rate:.2f}% |"
            )

        pinn_summary = summarize_condition(results, "pinn", condition)
        lstm_summary = summarize_condition(results, "lstm", condition)
        lines.append(
            f"| **Average** | - | **{pinn_summary['rmse']:.4f}** | **{pinn_summary['violation_rate']:.2f}%** "
            f"| **{lstm_summary['rmse']:.4f}** | **{lstm_summary['violation_rate']:.2f}%** |"
        )
        lines.append("")

    lines.extend(
        [
            "## Key Findings",
            "",
        ]
    )

    for condition in condition_order:
        pinn_summary = summarize_condition(results, "pinn", condition)
        lstm_summary = summarize_condition(results, "lstm", condition)
        lines.append(
            f"- **{condition.title()}**: PINN average VR = {pinn_summary['violation_rate']:.2f}% "
            f"(RMSE {pinn_summary['rmse']:.4f}) vs LSTM average VR = "
            f"{lstm_summary['violation_rate']:.2f}% (RMSE {lstm_summary['rmse']:.4f})."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Validation report saved: %s", output_path)
