"""
Data Splitter - Multiple splitting strategies for battery RUL evaluation.

Strategies:
  1. Nested CV (outer LOGO + inner hold-out)
  2. Cross-dataset OOD split
  3. Temporal split (within-battery)
  4. Few-shot transfer split
"""

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

logger = logging.getLogger(__name__)


class DataSplitter:
    """Flexible data splitting for battery evaluation."""

    @staticmethod
    def logo_cv(
        df: pd.DataFrame, group_col: str = "battery_id"
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
        """
        Leave-One-Group-Out CV. Yields (train_df, test_df, test_group_id).
        """
        logo = LeaveOneGroupOut()
        groups = df[group_col].values
        X_dummy = np.zeros(len(df))

        for train_idx, test_idx in logo.split(X_dummy, groups=groups):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()
            test_id = test_df[group_col].iloc[0]
            logger.info(f"LOGO fold: test={test_id}, train={len(train_df)}, test={len(test_df)}")
            yield train_df, test_df, test_id

    @staticmethod
    def nested_cv(
        df: pd.DataFrame,
        group_col: str = "battery_id",
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]]:
        """
        Nested CV: Outer LOGO for test, inner hold-out for validation.
        Yields (train_df, val_df, test_df, test_group_id).
        """
        rng = np.random.default_rng(seed)

        for train_df, test_df, test_id in DataSplitter.logo_cv(df, group_col):
            # Inner split: Hold out one battery from training for validation
            train_batteries = train_df[group_col].unique()
            if len(train_batteries) > 1:
                val_bat = rng.choice(train_batteries)
                val_df = train_df[train_df[group_col] == val_bat].copy()
                inner_train = train_df[train_df[group_col] != val_bat].copy()
            else:
                # Only 1 training battery: Use temporal split for val
                inner_train, val_df = DataSplitter._temporal_split_single(
                    train_df, train_fraction=1.0 - val_fraction
                )

            logger.info(
                f"Nested fold: test={test_id}, val={len(val_df)}, "
                f"train={len(inner_train)}"
            )
            yield inner_train, val_df, test_df, test_id

    @staticmethod
    def cross_dataset_split(
        df: pd.DataFrame,
        train_sources: list[str],
        test_sources: list[str],
        source_col: str = "dataset_source",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by dataset source for OOD evaluation.
        E.g., train on NASA+CALCE, test on Oxford.
        """
        train_df = df[df[source_col].isin(train_sources)].copy()
        test_df = df[df[source_col].isin(test_sources)].copy()

        if len(train_df) == 0:
            raise ValueError(f"No training data for sources: {train_sources}")
        if len(test_df) == 0:
            raise ValueError(f"No test data for sources: {test_sources}")

        logger.info(
            f"Cross-dataset: train={len(train_df)} ({train_sources}), "
            f"test={len(test_df)} ({test_sources})"
        )
        return train_df, test_df

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        train_fraction: float = 0.7,
        group_col: str = "battery_id",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal split: First N% cycles for training, rest for testing.
        Applied per battery to avoid data leakage.
        """
        train_parts, test_parts = [], []

        for bat_id in df[group_col].unique():
            sub = df[df[group_col] == bat_id].sort_values("cycle")
            split_idx = int(len(sub) * train_fraction)
            train_parts.append(sub.iloc[:split_idx])
            test_parts.append(sub.iloc[split_idx:])

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)

        logger.info(f"Temporal split: train={len(train_df)}, test={len(test_df)}")
        return train_df, test_df

    @staticmethod
    def few_shot_split(
        df: pd.DataFrame,
        target_battery: str,
        n_shots: int = 20,
        group_col: str = "battery_id",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Few-shot transfer: Pre-train on all others, fine-tune on N cycles of target.
        Returns (pretrain_df, finetune_df, test_df).
        """
        pretrain_df = df[df[group_col] != target_battery].copy()
        target_df = df[df[group_col] == target_battery].sort_values("cycle")

        finetune_df = target_df.iloc[:n_shots].copy()
        test_df = target_df.iloc[n_shots:].copy()

        logger.info(
            f"Few-shot: pretrain={len(pretrain_df)}, "
            f"finetune={len(finetune_df)} ({n_shots} shots), "
            f"test={len(test_df)}"
        )
        return pretrain_df, finetune_df, test_df

    @staticmethod
    def _temporal_split_single(
        df: pd.DataFrame, train_fraction: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a single battery's data temporally."""
        df = df.sort_values("cycle")
        split_idx = int(len(df) * train_fraction)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
