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

class ModelingConfig(BaseModel):
    standardize: List[str]

class PooledConfig(BaseModel):
    slope: str

class HierarchicalConfig(BaseModel):
    slope: List[str]
    group: str

class BayesianConfig(BaseModel):
    pooled: PooledConfig
    hierarchical: HierarchicalConfig

class AppConfig(BaseModel):
    dataset: DatasetConfig
    target: TargetConfig
    group: GroupConfig
    features: FeaturesConfig
    modeling: ModelingConfig
    bayesian: BayesianConfig

def load_config(config_path: Path = Path("config/schema.yaml")) -> AppConfig:
    """
    Load and validate configuration using Pydantic.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Validated AppConfig object.
        
    Raises:
        FileNotFoundError: If config file is missing.
        ValidationError: If config structure is invalid.
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
