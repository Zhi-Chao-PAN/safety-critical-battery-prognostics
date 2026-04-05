"""
CVAE-based Virtual Battery Data Generator.
Generates synthetic capacity degradation time series using Conditional Variational Autoencoder.
Includes physics-informed monotonicity penalty to ensure physically realistic capacity fade.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CapacityTimeSeriesDataset(Dataset):
    def __init__(
        self,
        capacity_sequences: list[np.ndarray],
        conditions: list[dict],
        sequence_length: int = 200,
    ):
        self.capacity_sequences = capacity_sequences
        self.conditions = conditions
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.capacity_sequences)

    def __getitem__(self, idx):
        cap_seq = self.capacity_sequences[idx]
        if len(cap_seq) < self.sequence_length:
            cap_seq = np.pad(cap_seq, (0, self.sequence_length - len(cap_seq)), mode='edge')
        else:
            cap_seq = cap_seq[:self.sequence_length]

        cond = self.conditions[idx]
        condition = torch.tensor([
            cond.get('initial_capacity', 2.0),
            cond.get('discharge_rate', 1.0),
            cond.get('temperature', 25.0),
            cond.get('soh_initial', 1.0),
        ], dtype=torch.float32)

        return torch.tensor(cap_seq, dtype=torch.float32), condition


class CVAEEncoder(nn.Module):
    def __init__(self, input_dim: int = 200, condition_dim: int = 4, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_condition = nn.Linear(condition_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc_input(x))
        hc = F.relu(self.fc_condition(condition))
        h = torch.cat([h, hc], dim=-1)
        h = F.relu(self.fc_hidden(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CVAEDecoder(nn.Module):
    def __init__(self, output_dim: int = 200, condition_dim: int = 4, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.fc_condition = nn.Linear(condition_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_latent(z))
        hc = F.relu(self.fc_condition(condition))
        h = torch.cat([h, hc], dim=-1)
        h = F.relu(self.fc_hidden(h))
        output = self.fc_output(h)
        return output


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 200,
        condition_dim: int = 4,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = CVAEEncoder(input_dim, condition_dim, latent_dim, hidden_dim)
        self.decoder = CVAEDecoder(input_dim, condition_dim, latent_dim, hidden_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, condition)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z, condition)
        return reconstructed, mu, logvar


def monotonicity_penalty(predicted_sequence: torch.Tensor) -> torch.Tensor:
    """
    Physics-informed monotonicity penalty.
    Penalizes capacity rebound (increase) which violates physical degradation.
    """
    diff = predicted_sequence[:, 1:] - predicted_sequence[:, :-1]
    rebound = F.relu(diff)
    penalty = torch.mean(rebound ** 2)
    return penalty


def capacity_range_penalty(predicted_sequence: torch.Tensor, initial_capacity: torch.Tensor) -> torch.Tensor:
    """
    Penalize unrealistic capacity values (negative or exceeds initial by too much).
    """
    lower_violation = F.relu(0.1 - predicted_sequence)
    upper_violation = F.relu(predicted_sequence - initial_capacity.unsqueeze(1) * 1.1)
    penalty = torch.mean(lower_violation ** 2 + upper_violation ** 2)
    return penalty


def cvae_loss(
    reconstructed: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    condition: torch.Tensor,
    beta: float = 1.0,
    lambda_mono: float = 10.0,
    lambda_range: float = 5.0,
) -> dict[str, torch.Tensor]:
    """
    Combined CVAE loss with physics-informed penalties.
    """
    recon_loss = F.mse_loss(reconstructed, x, reduction='mean')

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    mono_penalty = monotonicity_penalty(reconstructed)

    initial_cap = condition[:, 0]
    range_penalty = capacity_range_penalty(reconstructed, initial_cap)

    total_loss = recon_loss + beta * kl_loss + lambda_mono * mono_penalty + lambda_range * range_penalty

    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss,
        'mono': mono_penalty,
        'range': range_penalty,
    }


class SyntheticBatteryGenerator:
    """
    CVAE-based generator for synthetic battery degradation data.
    """

    def __init__(
        self,
        sequence_length: int = 200,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ):
        self.sequence_length = sequence_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CVAE(
            input_dim=sequence_length,
            condition_dim=4,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.is_trained = False

    def train(
        self,
        capacity_sequences: list[np.ndarray],
        conditions: list[dict],
        epochs: int = 100,
        batch_size: int = 32,
        beta: float = 1.0,
        lambda_mono: float = 10.0,
        lambda_range: float = 5.0,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train the CVAE model on battery capacity degradation sequences.
        """
        dataset = CapacityTimeSeriesDataset(
            capacity_sequences, conditions, self.sequence_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {'total': [], 'recon': [], 'kl': [], 'mono': [], 'range': []}

        self.model.train()

        for epoch in range(epochs):
            epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0, 'mono': 0.0, 'range': 0.0}
            n_batches = 0

            for batch_x, batch_cond in dataloader:
                batch_x = batch_x.to(self.device)
                batch_cond = batch_cond.to(self.device)

                self.optimizer.zero_grad()

                reconstructed, mu, logvar = self.model(batch_x, batch_cond)

                losses = cvae_loss(
                    reconstructed, batch_x, mu, logvar, batch_cond,
                    beta=beta, lambda_mono=lambda_mono, lambda_range=lambda_range
                )

                losses['total'].backward()
                self.optimizer.step()

                for k, v in losses.items():
                    epoch_losses[k] += v.item()

                n_batches += 1

            for k in epoch_losses:
                epoch_losses[k] /= n_batches
                history[k].append(epoch_losses[k])

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Total: {epoch_losses['total']:.4f}, "
                    f"Recon: {epoch_losses['recon']:.4f}, "
                    f"KL: {epoch_losses['kl']:.4f}, "
                    f"Mono: {epoch_losses['mono']:.4f}"
                )

        self.is_trained = True
        return history

    def generate(
        self,
        conditions: list[dict],
        n_samples: int = 1,
    ) -> list[np.ndarray]:
        """
        Generate synthetic capacity degradation sequences given conditions.
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Generating random samples.")

        self.model.eval()
        generated_sequences = []

        with torch.no_grad():
            for i in range(n_samples):
                cond = conditions[i % len(conditions)]
                condition = torch.tensor([
                    cond.get('initial_capacity', 2.0),
                    cond.get('discharge_rate', 1.0),
                    cond.get('temperature', 25.0),
                    cond.get('soh_initial', 1.0),
                ], dtype=torch.float32).to(self.device)

                z = torch.randn(1, self.model.latent_dim).to(self.device)
                generated = self.model.decoder(z, condition)
                generated_seq = generated.cpu().numpy()[0]

                if cond.get('apply_physics_postprocessing', True):
                    generated_seq = self._apply_physics_constraints(generated_seq, cond)

                generated_sequences.append(generated_seq)

        return generated_sequences

    def _apply_physics_constraints(
        self,
        sequence: np.ndarray,
        condition: dict,
    ) -> np.ndarray:
        """
        Post-process generated sequence to ensure physical validity.
        """
        initial_capacity = condition.get('initial_capacity', 2.0)
        min_capacity = initial_capacity * 0.7

        sequence = np.maximum(sequence, min_capacity * 0.5)
        sequence = np.minimum(sequence, initial_capacity * 1.02)

        for i in range(1, len(sequence)):
            if sequence[i] > sequence[i-1]:
                sequence[i] = sequence[i-1] * 0.999

        sequence = np.clip(sequence, min_capacity * 0.5, initial_capacity)

        return sequence


def generate_virtual_fleet(
    n_batteries: int = 100,
    sequence_length: int = 200,
    initial_capacity_range: tuple[float, float] = (1.8, 2.2),
    discharge_rate_range: tuple[float, float] = (0.5, 2.0),
    temperature_range: tuple[float, float] = (20.0, 45.0),
    seed: Optional[int] = 42,
    trained_model: Optional[SyntheticBatteryGenerator] = None,
    generate_realistic: bool = True,
) -> pd.DataFrame:
    """
    Generate a fleet of virtual batteries with synthetic capacity degradation data.

    Args:
        n_batteries: Number of virtual batteries to generate
        sequence_length: Number of cycles/degradation time steps per battery
        initial_capacity_range: Range for initial capacity (Ah)
        discharge_rate_range: Range for discharge rate (C-rate)
        temperature_range: Range for operating temperature (°C)
        seed: Random seed for reproducibility
        trained_model: Pre-trained SyntheticBatteryGenerator (if None, uses physics-based model)
        generate_realistic: If True, use trained model; otherwise use physics-based approximation

    Returns:
        DataFrame with columns: battery_id, cycle, capacity, resistance, temperature, rul, etc.
        Compatible with BatteryDataset format.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    data_records = []

    if generate_realistic and trained_model is not None and trained_model.is_trained:
        conditions = []
        for i in range(n_batteries):
            conditions.append({
                'initial_capacity': np.random.uniform(*initial_capacity_range),
                'discharge_rate': np.random.uniform(*discharge_rate_range),
                'temperature': np.random.uniform(*temperature_range),
                'soh_initial': 1.0,
                'apply_physics_postprocessing': True,
            })

        generated_sequences = trained_model.generate(conditions, n_samples=n_batteries)

        for i, seq in enumerate(generated_sequences):
            battery_id = f"vsim_B{i:04d}"
            initial_cap = conditions[i]['initial_capacity']
            eol_threshold = initial_cap * 0.8

            for cycle_idx, capacity in enumerate(seq):
                rul = 0
                for j in range(cycle_idx, len(seq)):
                    if seq[j] <= eol_threshold:
                        rul = j - cycle_idx
                        break
                else:
                    rul = len(seq) - cycle_idx

                resistance = 0.02 + (cycle_idx / sequence_length) * 0.03 + np.random.normal(0, 0.002)
                temperature = conditions[i]['temperature'] + np.random.normal(0, 2)

                data_records.append({
                    'battery_id': battery_id,
                    'cycle': cycle_idx,
                    'capacity': float(capacity),
                    'resistance': float(max(resistance, 0.015)),
                    'temperature': float(temperature),
                    'rul': rul,
                })

    else:
        for i in range(n_batteries):
            battery_id = f"vsim_B{i:04d}"

            initial_capacity = np.random.uniform(*initial_capacity_range)
            discharge_rate = np.random.uniform(*discharge_rate_range)
            temperature = np.random.uniform(*temperature_range)

            degradation_rate = 0.001 + 0.002 * discharge_rate
            if temperature > 35:
                degradation_rate *= 1.5

            q0 = initial_capacity
            a = degradation_rate * np.sqrt(initial_capacity) * 0.5
            b = degradation_rate * initial_capacity * 0.1

            for cycle in range(sequence_length):
                n = cycle + 1
                theoretical_capacity = q0 - a * np.sqrt(n) - b * n
                noise = np.random.normal(0, 0.005)
                capacity = max(theoretical_capacity + noise, q0 * 0.6)

                eol_threshold = initial_capacity * 0.8
                if capacity <= eol_threshold:
                    rul = 0
                else:
                    future_n = n
                    while future_n < sequence_length:
                        future_cap = q0 - a * np.sqrt(future_n) - b * future_n
                        if future_cap <= eol_threshold:
                            break
                        future_n += 1
                    rul = future_n - n

                resistance = 0.02 + (cycle / sequence_length) * 0.03 + np.random.normal(0, 0.001)
                temp = temperature + np.random.normal(0, 1.5)

                data_records.append({
                    'battery_id': battery_id,
                    'cycle': cycle,
                    'capacity': float(capacity),
                    'resistance': float(max(resistance, 0.015)),
                    'temperature': float(temp),
                    'rul': int(rul),
                })

    df = pd.DataFrame(data_records)
    df = df.sort_values(['battery_id', 'cycle']).reset_index(drop=True)

    logger.info(f"Generated virtual fleet: {n_batteries} batteries, {len(df)} total records")

    return df


def train_cvae_on_battery_data(
    df: pd.DataFrame,
    sequence_length: int = 200,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    epochs: int = 100,
    batch_size: int = 32,
    beta: float = 1.0,
    lambda_mono: float = 10.0,
    lambda_range: float = 5.0,
    verbose: bool = True,
) -> SyntheticBatteryGenerator:
    """
    Train CVAE model on existing battery data DataFrame.

    Args:
        df: DataFrame with battery data (must have 'battery_id', 'cycle', 'capacity')
        sequence_length: Length of capacity sequences
        latent_dim: Dimension of latent space
        hidden_dim: Hidden layer dimension
        epochs: Training epochs
        batch_size: Batch size
        beta: KL divergence weight
        lambda_mono: Monotonicity penalty weight
        lambda_range: Capacity range penalty weight
        verbose: Print training progress

    Returns:
        Trained SyntheticBatteryGenerator model
    """
    capacity_sequences = []
    conditions = []

    for battery_id in df['battery_id'].unique():
        battery_df = df[df['battery_id'] == battery_id].sort_values('cycle')
        capacity_seq = battery_df['capacity'].values

        if len(capacity_seq) < 10:
            continue

        capacity_sequences.append(capacity_seq)

        initial_capacity = float(capacity_seq[0])
        discharge_rate = battery_df.get('discharge_rate', pd.Series([1.0] * len(battery_df))).mean()
        temperature = battery_df['temperature'].mean() if 'temperature' in battery_df.columns else 25.0
        soh_initial = 1.0

        conditions.append({
            'initial_capacity': initial_capacity,
            'discharge_rate': float(discharge_rate),
            'temperature': float(temperature),
            'soh_initial': soh_initial,
        })

    if len(capacity_sequences) < 5:
        logger.warning("Insufficient batteries for CVAE training. Using physics-based generation.")
        generator = SyntheticBatteryGenerator(
            sequence_length=sequence_length,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        generator.is_trained = False
        return generator

    generator = SyntheticBatteryGenerator(
        sequence_length=sequence_length,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )

    generator.train(
        capacity_sequences=capacity_sequences,
        conditions=conditions,
        epochs=epochs,
        batch_size=batch_size,
        beta=beta,
        lambda_mono=lambda_mono,
        lambda_range=lambda_range,
        verbose=verbose,
    )

    return generator