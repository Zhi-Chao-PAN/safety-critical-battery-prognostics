import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ChronosFineTuningDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning Chronos models.

    Converts battery continuous lifespan data into sliding windows of
    (past_values, future_values) preventing cross-contamination between batteries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int = 128,
        prediction_length: int = 20,
        stride: int = 5,
        target_col: str = "capacity",
        id_col: str = "battery_id",
        cycle_col: str = "cycle",
    ):
        """
        Args:
            df: DataFrame containing the battery data (must have capacity, battery_id).
            context_length: Number of time steps in the historical context window.
            prediction_length: Number of time steps to predict into the future.
            stride: Step size for the sliding window across the sequence.
            target_col: Name of the column containing the time-series values to predict.
            id_col: Name of the column identifying distinct batteries.
            cycle_col: Name of the column containing the cycle index (for sorting).
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.window_size = context_length + prediction_length

        self.samples = []
        self._build_windows(df, target_col, id_col, cycle_col)

    def _build_windows(self, df: pd.DataFrame, target_col: str, id_col: str, cycle_col: str) -> None:
        """Sliding window generator ensuring boundaries don't cross different batteries."""
        # Ensure data is correctly structured and sorted
        required_cols = [target_col, id_col, cycle_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

        for bat_id, group in df.groupby(id_col):
            # Sort strictly by cycle to ensure temporal contiguousness
            group = group.sort_values(by=cycle_col)
            # Extract raw 1D numpy array
            series = group[target_col].values.astype(np.float32)

            n_points = len(series)
            if n_points < self.window_size:
                logger.warning(
                    f"Battery {bat_id} has only {n_points} cycles, which is shorter "
                    f"than the required window size of {self.window_size} (ctx {self.context_length} + pred {self.prediction_length}). "
                    f"Skipping this battery."
                )
                continue

            # Slide the window
            num_windows = 0
            for i in range(0, n_points - self.window_size + 1, self.stride):
                window = series[i : i + self.window_size]
                past_values = window[:self.context_length]
                future_values = window[self.context_length:]

                self.samples.append((past_values, future_values))
                num_windows += 1

            logger.debug(f"Battery {bat_id}: extracted {num_windows} valid windows.")

        logger.info(f"ChronosFineTuningDataset built with {len(self.samples)} total sliding windows.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a single training example as a dictionary of tensors.

        Format matches what Hugging Face trainer expects for custom models, or
        can be intercepted by a custom `collate_fn`.
        """
        past_values, future_values = self.samples[idx]

        return {
            "past_values": torch.tensor(past_values, dtype=torch.float32),
            "future_values": torch.tensor(future_values, dtype=torch.float32),
        }


def get_chronos_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    context_length: int = 128,
    prediction_length: int = 20,
    stride: int = 5,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Factory function to create train and validation PyTorch DataLoaders.
    
    The returned dataloaders yield dictionary batches with 'past_values' 
    and 'future_values'. Since Chronos uses a pseudo-quantization tokenizer,
    the actual string-to-token or float-to-token mapping should happen
    either in a custom collate_fn inside the fine-tuning script, or 
    directly within the training step leveraging `ChronosPipeline.tokenizer`.
    """
    train_dataset = ChronosFineTuningDataset(
        df=train_df,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )

    val_dataset = ChronosFineTuningDataset(
        df=val_df,
        context_length=context_length,
        prediction_length=prediction_length,
        stride=stride,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle windows across different batteries
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
