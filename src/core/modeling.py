# src/core/modeling.py
from abc import ABC, abstractmethod
from typing import Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from src.core.exceptions import ModelTrainingError
from loguru import logger

# --- Abstraction Layer ---
class BaseModelWorker(ABC):
    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> Any:
        pass

# --- Concrete Implementations ---

class SklearnRandomForestClassifierWorker(BaseModelWorker):
    def __init__(self, model_cfg, run_id: str):
        self.model_cfg = model_cfg
        self.run_id = run_id
        self.model = RandomForestClassifier(**model_cfg.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_type="RandomForestClassifier",
            hyperparameters=model_cfg.params
        ).info("Initialized RandomForestClassifier worker")

    def train(self, X_train, y_train):
        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            X_shape=X_train.shape,
            y_shape=y_train.shape
        ).info("Training RandomForestClassifier")

        try:
            self.model.fit(X_train, y_train)

            logger.bind(
                module="ModelWorker",
                run_id=self.run_id
            ).info("RandomForestClassifier training complete")

            return self.model

        except Exception as e:
            logger.bind(
                module="ModelWorker",
                run_id=self.run_id,
                error=str(e)
            ).error("RandomForestClassifier training failure")

            raise ModelTrainingError(
                "Model training failed",
                context={
                    "model_type": "RandomForestClassifier",
                    "hyperparameters": self.model_cfg.params,
                    "X_shape": X_train.shape,
                    "y_shape": y_train.shape
                }
            ) from e


class SklearnRandomForestRegressorWorker(BaseModelWorker):
    def __init__(self, model_cfg, run_id: str):
        self.model_cfg = model_cfg
        self.run_id = run_id
        self.model = RandomForestRegressor(**model_cfg.params)

        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            model_type="RandomForestRegressor",
            hyperparameters=model_cfg.params
        ).info("Initialized RandomForestRegressor worker")

    def train(self, X_train, y_train):
        logger.bind(
            module="ModelWorker",
            run_id=self.run_id,
            X_shape=X_train.shape,
            y_shape=y_train.shape
        ).info("Training RandomForestRegressor")

        try:
            self.model.fit(X_train, y_train)

            logger.bind(
                module="ModelWorker",
                run_id=self.run_id
            ).info("RandomForestRegressor training complete")

            return self.model

        except Exception as e:
            logger.bind(
                module="ModelWorker",
                run_id=self.run_id,
                error=str(e)
            ).error("RandomForestRegressor training failure")

            raise ModelTrainingError(
                "Model training failed",
                context={
                    "model_type": "RandomForestRegressor",
                    "hyperparameters": self.model_cfg.params,
                    "X_shape": X_train.shape,
                    "y_shape": y_train.shape
                }
            ) from e


# --- Factory Pattern ---
class ModelWorkerFactory:
    """
    Handles the instantiation of model workers based on configuration.
    This decouples the Orchestrator from specific Model implementations.
    """
    _workers = {
        "random_forest_classifier": SklearnRandomForestClassifierWorker,
        "random_forest_regressor": SklearnRandomForestRegressorWorker,
    }

    @staticmethod
    def get_worker(model_cfg, run_id: str) -> BaseModelWorker:
        model_type = model_cfg.model_type

        worker_class = ModelWorkerFactory._workers.get(model_type)

        if not worker_class:
            raise ModelTrainingError(
                f"Unsupported model_type: {model_type}",
                context={"available_types": list(ModelWorkerFactory._workers.keys())}
            )

        logger.bind(
            module="ModelWorkerFactory",
            run_id=run_id,
            model_type=model_type
        ).info("Instantiating model worker")

        return worker_class(model_cfg, run_id)
