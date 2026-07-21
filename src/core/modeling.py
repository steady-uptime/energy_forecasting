# src/core/modeling.py
from abc import ABC, abstractmethod
from typing import Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from loguru import logger

from src.core.exceptions import ModelTrainingError
from src.core.config.schemas import ModelDefinition


# -------------------------
# Base Worker Contract
# -------------------------
class BaseModelWorker(ABC):
    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> Any:
        pass


# -------------------------
# Concrete Workers
# -------------------------
class SklearnRandomForestClassifierWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, run_id: str):
        self.definition = definition
        self.run_id = run_id
        self.model_kind = definition.model_kind
        self.model = RandomForestClassifier(**definition.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=definition.params,
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            X_shape=X_train.shape,
            y_shape=y_train.shape,
        ).info(f"Training {self.model_kind}")

        try:
            self.model.fit(X_train, y_train)
            logger.bind(module="ModelWorker", run_id=self.run_id).info(
                f"{self.model_kind} training complete"
            )
            return self.model
        except Exception as e:
            logger.bind(
                module="ModelWorker",
                run_id=self.run_id,
                error=str(e),
            ).error(f"{self.model_kind} training failure")

            raise ModelTrainingError(
                "Model training failed",
                context={
                    "model_kind": self.model_kind,
                    "hyperparameters": self.definition.params,
                    "X_shape": X_train.shape,
                    "y_shape": y_train.shape,
                },
            ) from e


class SklearnRandomForestRegressorWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, run_id: str):
        self.definition = definition
        self.run_id = run_id
        self.model_kind = definition.model_kind
        self.model = RandomForestRegressor(**definition.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=definition.params,
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            X_shape=X_train.shape,
            y_shape=y_train.shape,
        ).info(f"Training {self.model_kind}")

        try:
            self.model.fit(X_train, y_train)
            logger.bind(module="ModelWorker", run_id=self.run_id).info(
                f"{self.model_kind} training complete"
            )
            return self.model
        except Exception as e:
            logger.bind(
                module="ModelWorker",
                run_id=self.run_id,
                error=str(e),
            ).error(f"{self.model_kind} training failure")

            raise ModelTrainingError(
                "Model training failed",
                context={
                    "model_kind": self.model_kind,
                    "hyperparameters": self.definition.params,
                    "X_shape": X_train.shape,
                    "y_shape": y_train.shape,
                },
            ) from e


# -------------------------
# Unified Worker Wrapper
# -------------------------
class ModelWorker(BaseModelWorker):
    """
    Wraps the correct concrete worker based on model_kind.
    """

    def __init__(self, definition: ModelDefinition, artifact_manager, run_id: str):
        self.definition = definition
        self.artifact_manager = artifact_manager
        self.run_id = run_id

        model_kind = definition.model_kind

        if model_kind == "random_forest_classifier":
            self.worker = SklearnRandomForestClassifierWorker(definition, run_id)
        elif model_kind == "random_forest_regressor":
            self.worker = SklearnRandomForestRegressorWorker(definition, run_id)
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")

    def train(self, X_train, y_train):
        return self.worker.train(X_train, y_train)


# -------------------------
# Factory
# -------------------------
class ModelWorkerFactory:
    def __init__(self, artifact_manager):
        self.artifact_manager = artifact_manager

    def get_worker(self, definition: ModelDefinition, run_id: str) -> ModelWorker:
        return ModelWorker(
            definition=definition,
            artifact_manager=self.artifact_manager,
            run_id=run_id,
        )
