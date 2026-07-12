# src/core/mock_worker.py
from typing import Any
from pathlib import Path
from loguru import logger
from dataclasses import dataclass

@dataclass
class MockModel:
    """Dummy model object to pass through the evaluator."""
    pass

class MockModelWorker:
    """
    Simulates a model worker for Dry Run verification.
    Logs the parameters received to verify HPO injection logic.
    """
    def __init__(self, config: Any, run_id: str):
        self.config = config
        self.run_id = run_id
        self.model: MockModel = MockModel()
        self.path = Path("data/models/mock_model_dry_run.pkl")

    def train(self, X: Any, y: Any) -> MockModel:
        # This is the critical check: verify the HPO params reached the worker
        logger.info(f"[DRY RUN] Training Model with Params: {self.config.params}")
        return self.model

    def get_path(self) -> Path:
        return self.path
