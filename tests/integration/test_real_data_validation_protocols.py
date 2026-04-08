import importlib.util
from pathlib import Path

import numpy as np

from src.evaluation.real_data_validation import CellResult, ModelMetrics, evaluate_models_on_cell


ROOT = Path(__file__).parent.parent.parent


def load_script_module(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_dummy_result(cell_id: str, cycles: np.ndarray, capacity: np.ndarray, noisy_capacity: np.ndarray) -> CellResult:
    metrics = ModelMetrics(
        rmse=0.0,
        violation_rate=0.0,
        violation_count=0,
        predictions=capacity.copy(),
    )
    return CellResult(
        cell_id=cell_id,
        n_cycles=len(cycles),
        cycles=cycles,
        ground_truth=capacity,
        noisy_capacity=noisy_capacity,
        evaluations={
            "pinn": {"clean": metrics, "noisy": metrics},
            "lstm": {"clean": metrics, "noisy": metrics},
        },
    )


def test_logo_fold_keeps_train_and_test_cells_disjoint(monkeypatch):
    module = load_script_module("validate_real_data_logo.py")
    captured = {}

    def fake_train_reference_models(train_cells):
        captured["train_cells"] = set(train_cells)
        return {"pinn": object(), "lstm": object()}

    def fake_evaluate_models_on_cell(models, cell_id, cycles, capacity, noise_level, seed, conditions):
        captured["eval_cell"] = cell_id
        return build_dummy_result(cell_id, cycles, capacity, capacity)

    monkeypatch.setattr(module, "train_reference_models", fake_train_reference_models)
    monkeypatch.setattr(module, "evaluate_models_on_cell", fake_evaluate_models_on_cell)

    cells = {
        "CELL_A": (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.8, 1.6])),
        "CELL_B": (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.7, 1.5])),
        "CELL_C": (np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.9, 1.7])),
    }

    result = module.run_logo_fold("CELL_B", cells)

    assert result.cell_id == "CELL_B"
    assert captured["eval_cell"] == "CELL_B"
    assert captured["train_cells"] == {"CELL_A", "CELL_C"}
    assert "CELL_B" not in captured["train_cells"]


def test_real_data_evaluation_uses_shared_postprocessing_for_all_models():
    class FakeModel:
        def __init__(self, raw_prediction):
            self.raw_prediction = np.asarray(raw_prediction, dtype=np.float64)

        def predict(self, X, **kwargs):
            return self.raw_prediction.copy(), self.raw_prediction.copy(), self.raw_prediction.copy()

    cycles = np.arange(1.0, 7.0)
    capacity = np.array([2.0, 1.92, 1.85, 1.78, 1.70, 1.62])
    raw_prediction = np.array([2.0, 2.05, 1.95, 2.02, 1.88, 1.90])

    result = evaluate_models_on_cell(
        models={"pinn": FakeModel(raw_prediction), "lstm": FakeModel(raw_prediction)},
        cell_id="CELL_X",
        cycles=cycles,
        capacity=capacity,
        noise_level=0.5,
        seed=42,
        conditions=("noisy",),
    )

    pinn_pred = result.evaluations["pinn"]["noisy"].predictions
    lstm_pred = result.evaluations["lstm"]["noisy"].predictions
    assert np.allclose(pinn_pred, lstm_pred)
    assert np.all(np.diff(pinn_pred) <= 1e-10)


def test_validation_scripts_use_protocol_specific_titles():
    same_cell_source = (ROOT / "scripts" / "validate_real_data.py").read_text(encoding="utf-8")
    logo_source = (ROOT / "scripts" / "validate_real_data_logo.py").read_text(encoding="utf-8")

    assert "Same-Cell Noise Robustness Report" in same_cell_source
    assert "Cross-Cell Results" not in same_cell_source
    assert "LOGO Cross-Cell Robustness Report" in logo_source


def test_readme_real_data_sections_match_current_evidence_state():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    full_results = (ROOT / "docs" / "comprehensive_experimental_results.md").read_text(encoding="utf-8")
    claim_matrix = (ROOT / "docs" / "claim_evidence_matrix.md").read_text(encoding="utf-8")

    assert "### Same-Cell Noise Robustness" in readme
    assert "### LOGO Cross-Cell Validation" in readme
    assert "### Multi-Seed Corruption Stress Suite" in readme
    assert "docs/comprehensive_experimental_results.md" in readme
    assert "docs/claim_evidence_matrix.md" in readme

    same_cell_section = readme.split("### Same-Cell Noise Robustness", 1)[1].split(
        "### LOGO Cross-Cell Validation", 1
    )[0]
    assert "cross-cell generalization" not in same_cell_section.lower()
    assert "47.97%" not in same_cell_section
    assert "49.95%" not in same_cell_section
    assert "0.2160" in same_cell_section
    assert "pinn-specific real-data safety advantage" in same_cell_section.lower()

    assert "results pending" not in readme.lower()
    assert "section v." not in readme.lower()
    assert "fairness validation (section v.k)" not in readme.lower()
    assert "does not support cross-cell generalization claims" in full_results.lower()
    assert "multi-seed corruption stress suite" in full_results.lower()
    assert "executed logo calce outputs are present" in claim_matrix.lower()
    assert "bounded safety claim, not a superiority claim" in claim_matrix.lower()
    assert "seeded multi-corruption real-data stress suite" in claim_matrix.lower()


def test_readme_reproducibility_distinguishes_same_cell_logo_and_stress_suite_outputs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python scripts/validate_real_data.py" in readme
    assert "python scripts/validate_real_data_logo.py" in readme
    assert "python scripts/validate_real_data_stress_suite.py" in readme
    assert "LOGO validated" not in readme
    assert "LOGO protocol included" not in readme
    assert "pending empirical summary" not in readme.lower()
    assert "real_data_logo_validation.png" in readme
    assert "real_data_logo_validation_report.md" in readme
    assert "real_data_stress_suite_report.md" in readme
    assert "real_data_stress_suite_summary.csv" in readme
    assert "lags lstm on rmse" in readme.lower()
    assert "held-out cells, bounded conclusion" in readme.lower()

    logo_figure = ROOT / "robustness_results" / "real_data_logo_validation.png"
    logo_report = ROOT / "robustness_results" / "real_data_logo_validation_report.md"
    stress_report = ROOT / "robustness_results" / "real_data_stress_suite_report.md"
    stress_summary = ROOT / "robustness_results" / "real_data_stress_suite_summary.csv"
    assert logo_figure.exists()
    assert logo_report.exists()
    assert stress_report.exists()
    assert stress_summary.exists()
