import numpy as np
import pandas as pd
import pytest
import torch

from src.data.chronos_dataset import ChronosFineTuningDataset, get_chronos_dataloaders


@pytest.fixture
def dummy_battery_df():
    """Create a dummy dataframe with 2 batteries."""
    # Battery A: 100 cycles
    df_a = pd.DataFrame({
        "battery_id": ["BatA"] * 100,
        "cycle": list(range(1, 101)),
        "capacity": np.linspace(2.0, 1.0, 100),
    })
    # Battery B: 50 cycles (too short for 128 context)
    df_b = pd.DataFrame({
        "battery_id": ["BatB"] * 50,
        "cycle": list(range(1, 51)),
        "capacity": np.linspace(2.0, 1.5, 50),
    })
    # Battery C: 150 cycles
    df_c = pd.DataFrame({
        "battery_id": ["BatC"] * 150,
        "cycle": list(range(1, 151)),
        "capacity": np.linspace(2.0, 0.5, 150),
    })

    return pd.concat([df_a, df_b, df_c], ignore_index=True)


def test_chronos_dataset_initialization(dummy_battery_df):
    """Test standard initialization and battery isolation."""
    # window_size = 50 + 10 = 60
    dataset = ChronosFineTuningDataset(
        df=dummy_battery_df,
        context_length=50,
        prediction_length=10,
        stride=5,
    )

    # Expected windows:
    # BatA (100 cycles): N=100, W=60. (100 - 60) // 5 + 1 = 8 + 1 = 9 windows
    # BatB (50 cycles): N=50, W=60. Smaller than window, 0 windows
    # BatC (150 cycles): N=150, W=60. (150 - 60) // 5 + 1 = 18 + 1 = 19 windows
    # Total = 9 + 0 + 19 = 28

    assert len(dataset) == 28


def test_chronos_dataset_shapes(dummy_battery_df):
    dataset = ChronosFineTuningDataset(
        df=dummy_battery_df,
        context_length=64,
        prediction_length=16,
        stride=10,
    )

    # BatA (100): W=80. (100 - 80) // 10 + 1 = 3
    # BatB (50): 0
    # BatC (150): W=80. (150 - 80) // 10 + 1 = 8
    assert len(dataset) == 11

    sample = dataset[0]
    assert "past_values" in sample
    assert "future_values" in sample

    past = sample["past_values"]
    future = sample["future_values"]

    assert isinstance(past, torch.Tensor)
    assert isinstance(future, torch.Tensor)

    assert past.shape == (64,)
    assert future.shape == (16,)
    assert past.dtype == torch.float32


def test_get_chronos_dataloaders(dummy_battery_df):
    # Split dummy data
    train_df = dummy_battery_df[dummy_battery_df["battery_id"].isin(["BatA", "BatB"])]
    val_df = dummy_battery_df[dummy_battery_df["battery_id"] == "BatC"]

    train_loader, val_loader = get_chronos_dataloaders(
        train_df=train_df,
        val_df=val_df,
        context_length=40,
        prediction_length=10,
        stride=5,
        batch_size=4,
    )

    assert iter(train_loader)
    assert iter(val_loader)

    # Check batch shapes from train loader
    batch = next(iter(train_loader))
    assert batch["past_values"].shape == (4, 40)
    assert batch["future_values"].shape == (4, 10)
