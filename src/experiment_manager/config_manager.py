"""
Experiment Manager - Config Manager
Parses and validates experiment YAML configurations.
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

class ExperimentConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load()
        self._validate()

    def _load(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _validate(self):
        """Ensure critical parameters exist."""
        required = ['project_name', 'model', 'data', 'tracking']
        for req in required:
            if req not in self.config:
                raise ValueError(f"Missing required config key: {req}")

        logger.info(f"Validated config: {self.config.get('experiment_name', 'unnamed')}")

    def get(self, key: str, default=None):
        return self.config.get(key, default)
