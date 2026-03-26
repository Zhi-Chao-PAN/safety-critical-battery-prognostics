import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

def test_pinn_pipeline_integration(tmp_path):
    """
    Integration Test:
    Data Generation -> PINN Training with Physics Calibration -> Evaluation -> Checkpoint.
    """
    df = create_mock_data()

    # Setup directories in tmp_path
    ckpt_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"

    # Initialize Pipeline
    pipeline = TrainingPipeline(
        features=["cycle"],
        target="rul",
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

if __name__ == "__main__":
    # For manual debugging
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_pinn_pipeline_integration(Path(tmp))
