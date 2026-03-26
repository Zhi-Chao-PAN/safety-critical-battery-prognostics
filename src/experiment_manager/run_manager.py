"""
Experiment Manager - Run Manager
Orchestrates experiment execution runs.
"""

import logging

from .config_manager import ExperimentConfig

logger = logging.getLogger(__name__)

class RunManager:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def execute(self):
        logger.info(f"Starting run for {self.config.get('experiment_name')}")
        # In a real environment, this would call training scripts via subprocess or internal module
        # For our architecture it links with experiments/pipelines/runner.py

        try:
             # Just an orchestration stub for now
             logger.info("Run executing successfully.")
             return True
        except Exception as e:
             logger.error(f"Run failed: {e}")
             return False
