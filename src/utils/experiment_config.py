"""
Experiment Configuration System.
YAML-based configs for reproducible experiments.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    data_dir: str = "data/battery_data"
    datasets: list[str] = field(default_factory=lambda: ["nasa"])
    eol_fraction: float = 0.7
    rated_capacity: float = 2.0
    val_fraction: float = 0.2


@dataclass
class FeatureConfig:
    rolling_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    include_ic_dv: bool = True
    include_frequency: bool = True
    include_trends: bool = True


@dataclass
class ModelConfig:
    name: str = "lstm"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    seq_length: int = 30
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    mc_samples: int = 100
    # TCN specific
    num_channels: list[int] = field(default_factory=lambda: [32, 32, 64, 64])
    kernel_size: int = 3
    # Transformer specific
    d_model: int = 64
    nhead: int = 4
    # PINN specific
    lambda_physics: float = 0.1
    lambda_mono: float = 0.05
    # BNN specific
    kl_weight: float = 0.01
    # Ensemble specific
    n_members: int = 5


@dataclass
class SafetyConfig:
    rul_critical: float = 10.0
    rul_warning: float = 30.0
    epistemic_threshold_low: float = 5.0
    epistemic_threshold_high: float = 15.0


@dataclass
class ExperimentConfig:
    name: str = "default"
    seed: int = 42
    seeds: list[int] = field(default_factory=lambda: [42])
    device: str = "cpu"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict) -> "ExperimentConfig":
        cfg = cls()
        for key, val in d.items():
            if key == "data" and isinstance(val, dict):
                cfg.data = DataConfig(**val)
            elif key == "features" and isinstance(val, dict):
                cfg.features = FeatureConfig(**val)
            elif key == "model" and isinstance(val, dict):
                cfg.model = ModelConfig(**val)
            elif key == "safety" and isinstance(val, dict):
                cfg.safety = SafetyConfig(**val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg

    def to_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import dataclasses
        d = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Config saved: {path}")
