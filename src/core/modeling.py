# src/core/modeling.py
from abc import ABC, abstractmethod
from typing import Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.core.exceptions import ModelTrainingError
from loguru import logger

class BaseModelWorker(ABC):
    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> Any:
        pass

class SklearnRandomForestClassifierWorker(BaseModelWorker):
    def __init__(self, model_cfg, run_id: str):
        self.model_cfg = model_cfg
        self.run_id = run_id
        self.model_kind = model_cfg.model_kind
        self.model = RandomForestClassifier(**model_cfg.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=model_cfg.params
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            X_shape=X_train.shape,
            y_shape=y_train.shape
        ).info(f"Training {self.model_kind}")

        try:
            self.model.fit(X_train, y_train)
            logger.bind(module="ModelWorker", run_id=self.run_id).info(f"{self.model_kind} training complete")
            return self.model
        except Exception as e:
            logger.bind(module="ModelWorker", run_id=self.run_id, error=str(e)).error(f"{self.model_kind} training failure")
            raise ModelTrainingError("Model training failed", context={
                "model_kind": self.model_kind,
                "hyperparameters": self.model_cfg.params,
                "X_shape": X_train.shape,
                "y_shape": y_train.shape
            }) from e

class SklearnRandomForestRegressorWorker(BaseModelWorker):
    def __init__(self, model_cfg, run_id: str):
        self.model_cfg = model_cfg
        self.run_id = run_id
        self.model_kind = model_cfg.model_kind
        self.model = RandomForestRegressor(**model_cfg.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=model_cfg.params
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            X_shape=X_train.shape,
            y_shape=y_train.shape
        ).info(f"Training {self.model_kind}")

        try:
            self.model.fit(X_train, y_train)
            logger.bind(module="ModelWorker", run_id=self.run_id).info(f"{self.model_kind} training complete")
            return self.model
        except Exception as e:
            logger.bind(module="ModelWorker", run_id=self.run_id, error=str(e)).error(f"{self.model_kind} training failure")
            raise ModelTrainingError("Model training failed", context={
                "model_kind": self.model_kind,
                "hyperparameters": self.model_cfg.params,
                "X_shape": X_train.shape,
                "y_shape": y_train.shape
            }) from e

class ModelWorkerFactory:
    _workers = {
        "random_forest_classifier": SklearnRandomForestClassifierWorker,
        "random_forest_regressor": SklearnRandomForestRegressorWorker,
    }

    @staticmethod
    def get_worker(model_cfg, run_id: str) -> BaseModelWorker:
        # --- DRY RUN LOGIC START ---
        # Use getattr to safely check if dry_run exists on the config object
        is_dry_run = getattr(model_cfg, "dry_run", False)
        
        if is_dry_run:
            logger.bind(module="ModelWorkerFactory", run_id=run_id).info("Dry run mode detected. Injecting MockModelWorker.")
            from src.core.mock_worker import MockModelWorker # Local import to avoid circularity
            return MockModelWorker(model_cfg, run_id)
        # --- DRY RUN LOGIC END ---

        model_kind = model_cfg.model_kind
        worker_class = ModelWorkerFactory._workers.get(model_kind)

        if not worker_class:
            raise ModelTrainingError(
                f"Unsupported model kind: {model_kind}",
                context={"available_types": list(ModelWorkerFactory._workers.keys())}
            )

        logger.bind(module="ModelWorkerFactory", run_id=run_id, model_kind=model_kind).info(f"Instantiating {model_kind} worker")
        return worker_class(model_cfg, run_id)
