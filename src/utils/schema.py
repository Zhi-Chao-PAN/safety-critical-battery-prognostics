# src/utils/schema.py
"""
Schema loading utilities for configuration-driven modeling.

This module provides functions to load and validate the project's
schema configuration file (config/schema.yaml), ensuring consistent
feature definitions across all models.
"""

from typing import Dict, Any
import yaml
from pathlib import Path

SCHEMA_PATH = Path("config/schema.yaml")


def load_schema(schema_path: Path = SCHEMA_PATH) -> Dict[str, Any]:
    """
    Load the project schema configuration from YAML file.
    
    The schema defines:
        - Dataset path
        - Target variable
        - Feature lists (numeric, categorical)
        - Modeling configurations (standardization, Bayesian priors)
    
    Args:
        schema_path: Path to the schema YAML file. Defaults to 'config/schema.yaml'.
        
    Returns:
        Dictionary containing the parsed schema configuration.
        
    Raises:
        FileNotFoundError: If the schema file does not exist.
        
    Example:
        >>> schema = load_schema()
        >>> schema["target"]["name"]
        'RUL'
        >>> schema["features"]["numeric"]
        ['Discharge_Time', 'Internal_Resistance', ...]
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path, "r") as f:
        schema = yaml.safe_load(f)

    return schema
