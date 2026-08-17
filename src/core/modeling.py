# src/core/modeling.py
from abc import ABC, abstractmethod
from typing import Any
from loguru import logger

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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
# Concrete Workers (train only)
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


class GradientBoostingWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, run_id: str):
        self.definition = definition
        self.run_id = run_id
        self.model_kind = definition.model_kind
        self.model = GradientBoostingRegressor(**definition.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=definition.params,
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return self.model
        except Exception as e:
            raise ModelTrainingError("Model training failed") from e


class XGBoostWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, run_id: str):
        self.definition = definition
        self.run_id = run_id
        self.model_kind = definition.model_kind
        self.model = XGBRegressor(**definition.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=definition.params,
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return self.model
        except Exception as e:
            raise ModelTrainingError("Model training failed") from e


class LightGBMWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, run_id: str):
        self.definition = definition
        self.run_id = run_id
        self.model_kind = definition.model_kind
        self.model = LGBMRegressor(**definition.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=definition.params,
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return self.model
        except Exception as e:
            raise ModelTrainingError("Model training failed") from e


class ElasticNetWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, run_id: str):
        self.definition = definition
        self.run_id = run_id
        self.model_kind = definition.model_kind
        self.model = ElasticNet(**definition.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_kind=self.model_kind,
            hyperparameters=definition.params,
        ).info(f"Initialized {self.model_kind} worker")

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            return self.model
        except Exception as e:
            raise ModelTrainingError("Model training failed") from e


# -------------------------
# Unified Worker Wrapper
# -------------------------
class ModelWorker(BaseModelWorker):
    def __init__(self, definition: ModelDefinition, artifact_manager, run_id: str):
        self.definition = definition
        self.artifact_manager = artifact_manager
        self.run_id = run_id

        kind = definition.model_kind

        if kind == "random_forest_classifier":
            self.worker = SklearnRandomForestClassifierWorker(definition, run_id)
        elif kind == "random_forest_regressor":
            self.worker = SklearnRandomForestRegressorWorker(definition, run_id)
        elif kind == "gradient_boosting":
            self.worker = GradientBoostingWorker(definition, run_id)
        elif kind == "xgboost":
            self.worker = XGBoostWorker(definition, run_id)
        elif kind == "lightgbm":
            self.worker = LightGBMWorker(definition, run_id)
        elif kind == "elastic_net":
            self.worker = ElasticNetWorker(definition, run_id)
        else:
            raise ValueError(f"Unsupported model_kind: {kind}")

    def train(self, X_train, y_train):
        return self.worker.train(X_train, y_train)

    def save(self, model, run_id):
        from joblib import dump
        model_path = self.artifact_manager.model_path(self.definition.name, run_id)
        dump(model, model_path)
        return model_path


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
