from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field, ValidationError

class DatasetConfig(BaseModel):
    path: Path

class TargetConfig(BaseModel):
    name: str

class GroupConfig(BaseModel):
    name: str

class FeaturesConfig(BaseModel):
    numeric: List[str]

class LSTMConfig(BaseModel):
    window_size: int = 30
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100

class BayesianModelConfig(BaseModel):
    pooled: dict
    hierarchical: dict

class ModelingConfig(BaseModel):
    lstm: LSTMConfig
    bayesian: BayesianModelConfig

class AppConfig(BaseModel):
    dataset: DatasetConfig
    target: TargetConfig
    group: GroupConfig
    features: FeaturesConfig
    modeling: ModelingConfig

def load_config(config_path: Path = Path("config/schema.yaml")) -> AppConfig:
    """
    Load and validate configuration using Pydantic.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
        
    try:
        config = AppConfig(**raw_config)
        return config
    except ValidationError as e:
        print(f"Configuration Validation Error: {e}")
        raise
