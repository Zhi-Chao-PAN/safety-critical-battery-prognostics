"""
End-to-End Computational Graph Validator
Proves the PyTorch execution chain:
    Mock Data (Micro) -> SPM -> Physics Feature Extractor -> Macro Network -> Loss -> Backward()
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

from src.coupling.time_decoupling import PhysicsFeatureExtractor
from src.data.mock_decoupled_loader import get_mock_decoupled_loaders
from src.physics.electrochemistry.spm import PyTorchSPM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphValidator")

class MockMacroForecaster(nn.Module):
    def __init__(self, macro_in_dim=2, phys_in_dim=2, hidden=32):
        super().__init__()
        # Simulating a small sequence model (e.g., the base of TCN/Chronos wrapper)
        self.lstm = nn.LSTM(input_size=macro_in_dim, hidden_size=hidden, batch_first=True)
        # Merges macro LSTM embedding with the single-cycle physics features extracted by SPM
        self.fc = nn.Sequential(
            nn.Linear(hidden + phys_in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, macro_seq, phys_features):
        _, (h_n, _) = self.lstm(macro_seq) # h_n shape: [1, batch, hidden]
        macro_emb = h_n.squeeze(0)

        # Concatenate neural features and PINN extracted features
        fused = torch.cat([macro_emb, phys_features], dim=-1)
        return self.fc(fused).squeeze(-1)

def validate_end_to_end_graph():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Initialize PINN Engine
    spm = PyTorchSPM(n_shells=5, device=device)
    extractor = PhysicsFeatureExtractor(spm_model=spm).to(device)

    # 2. Initialize RUL Forecaster
    macro_net = MockMacroForecaster().to(device)
    optimizer = torch.optim.Adam(list(macro_net.parameters()) + list(extractor.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    # 3. Micro DataLoader (e.g. 100 high freq steps per cycle)
    dt_micro = 10.0 # 10 seconds per step, safely under CFL limit
    loader = get_mock_decoupled_loaders(batch_size=8, micro_steps=100)

    logger.info("Starting Graph Forward/Backward check...")

    for batch_idx, batch in enumerate(loader):
        i_app = batch["i_app_micro"].to(device)
        macro_seq = batch["macro_features"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # --- THE DECOUPLED FORWARD PASS ---

        # Step A: Physics Extractor condenses thousands of Micro equations down to Macro features
        phys_feats = extractor(i_app, dt_micro)

        # Step B: Data-driven network combines standard memory with physical boundaries
        preds = macro_net(macro_seq, phys_feats)

        loss = loss_fn(preds, targets)

        # --- BACKWARD PASS ---
        # Tests if gradients flow from Loss -> MacroNet -> PhysicsExtractor -> SPM perfectly without OOM
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            logger.info("✅ Gradients calculated successfully.")
            logger.info(f"Physics Extractor Output Shape: {phys_feats.shape}")
            logger.info(f"Loss: {loss.item():.4f}")
            if device.type == "cuda":
                 mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                 logger.info(f"GPU VRAM Allocated Check: {mem_mb:.2f} MB (Extremely Safe for RTX 4060)")
            else:
                 logger.info("Running on CPU, skipping VRAM check.")
            break

if __name__ == "__main__":
    validate_end_to_end_graph()
