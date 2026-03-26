"""
High-Frequency Mock DataLoader (MVP Phase)

Generates synthetic decoupled datasets:
1. Macro-level attributes for the cycle (e.g., target capacity)
2. Micro-level sequences (high-frequency I_app_cycle over time)

Used strictly for validating the VRAM stability of the PyTorch Computational Graph 
before building the massive engineering pipeline for CALCE raw .txt parsing.
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class MockDecoupledBatteryDataset(Dataset):
    def __init__(self, num_samples=100, micro_steps=100, seq_len=10):
        """
        Args:
            num_samples: Total number of cycles to mock
            micro_steps: Number of high-frequency samples within ONE cycle
            seq_len: Number of historical cycles fed to the macro-forecaster (e.g., Chronos)
        """
        super().__init__()
        self.num_samples = num_samples
        self.micro_steps = micro_steps
        self.seq_len = seq_len

        # Micro Data: Sequence of currents for the current cycle
        # Simulating a dynamic discharge profile (e.g., 1A to 3A fluctuations)
        self.i_app_micro = torch.rand((num_samples, micro_steps), dtype=torch.float32) * 2.0 + 1.0

        # Macro Data: Historical RUL features (mocking inputs to standard TCN)
        self.macro_features = torch.randn((num_samples, seq_len, 2), dtype=torch.float32)

        # Target: Remaining Useful Life or Capacity
        self.target_rul = torch.linspace(100, 0, num_samples, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "i_app_micro": self.i_app_micro[idx],      # Shape: [micro_steps]
            "macro_features": self.macro_features[idx], # Shape: [seq_len, 2]
            "target": self.target_rul[idx]             # Scalar
        }

def get_mock_decoupled_loaders(batch_size=16, micro_steps=100, seq_len=10):
    dataset = MockDecoupledBatteryDataset(num_samples=200, micro_steps=micro_steps, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
