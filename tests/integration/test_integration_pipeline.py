import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.pinn_model import PINNModel
from src.training.pipeline import TrainingPipeline


def create_mock_data(n_cycles=40, n_batteries=2):
    """Create a small mock dataset for integration testing."""
    data = []
    for b_id in range(n_batteries):
        base_cap = 2.0
        for i in range(n_cycles):
            cap = base_cap - 0.005 * i + 0.001 * np.random.randn()
            data.append({
                "cycle": i + 1,
                "capacity": cap,
                "battery_id": f"BATT_{b_id:03d}",
                "rul": n_cycles - i - 1
            })
    return pd.DataFrame(data)

def test_pinn_pipeline_integration(workspace_tmp_path):
    """
    Integration Test:
    Data Generation -> PINN Training with Physics Calibration -> Evaluation -> Checkpoint.
    """
    df = create_mock_data()

    # Setup directories in a workspace-local temporary path
    ckpt_dir = workspace_tmp_path / "checkpoints"
    log_dir = workspace_tmp_path / "logs"

    # Initialize Pipeline — use capacity as target (not RUL) because PINN's
    # physics model Q(n)=Q0-a√n-b·n is designed for capacity fade. (Expert #6 fix)
    pipeline = TrainingPipeline(
        features=["cycle"],
        target="capacity",
        group_col="battery_id",
        checkpoint_dir=str(ckpt_dir),
        log_dir=str(log_dir)
    )

    # Initialize PINN Model (small epochs for speed)
    model = PINNModel(
        input_dim=1,
        epochs=5,
        patience=2,
        device="cpu"
    )

    # Run full training flow
    results = pipeline.train_and_evaluate(df, model)

    # Assertions
    assert "RMSE_mean" in results
    assert results["n_folds"] == 2
    assert (ckpt_dir / "pinn_best.pt").exists()
    assert (log_dir / "pinn_seed42.json").exists()

    print("\nIntegration test successful: Model trained and saved.")


def test_pinn_pipeline_rejects_rul_target(workspace_tmp_path):
    """PINN pipelines must train on capacity targets, not RUL."""
    df = create_mock_data()

    pipeline = TrainingPipeline(
        features=["cycle"],
        target="rul",
        group_col="battery_id",
        checkpoint_dir=str(workspace_tmp_path / "checkpoints"),
        log_dir=str(workspace_tmp_path / "logs"),
    )

    model = PINNModel(
        input_dim=1,
        epochs=1,
        patience=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="target='capacity'"):
        pipeline.train_and_evaluate(df, model)

if __name__ == "__main__":
    # For manual debugging
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_pinn_pipeline_integration(Path(tmp))
