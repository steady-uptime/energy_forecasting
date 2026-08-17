# src/core/hpo_service.py
from dataclasses import dataclass
from typing import Any, Dict, List
from loguru import logger

from src.core.config_loader import config
from src.core.config.schemas import ModelDefinition, ModelSearchConfig
from src.core.modeling import ModelWorkerFactory
from src.core.evaluator import ModelEvaluator
from src.core.exceptions import ModelTrainingError


@dataclass
class TrialResult:
    model_name: str
    model_kind: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    status: str  # "success" or "failed"
    model_path: str  # NEW: saved model artifact path


@dataclass
class ModelSearchResult:
    trials: List[TrialResult]
    best_model: TrialResult


class HPOService:
    """
    Lightweight AutoML-style model search:
    - Iterates through ModelDefinition entries
    - Trains each model once
    - Evaluates each model
    - Saves each trial model + metrics
    - Selects best model based on scoring metric
    """

    def __init__(self, search_config: ModelSearchConfig, artifact_manager):
        self.search_config = search_config
        self.artifact_manager = artifact_manager
        self.worker_factory = ModelWorkerFactory(artifact_manager)
        self.evaluator = ModelEvaluator(eval_cfg=config.evaluation)

    def run(self, X_train, y_train, X_val, y_val) -> ModelSearchResult:
        trials: List[TrialResult] = []
        best_trial: TrialResult | None = None

        for definition in self.search_config.models:
            run_id = self.artifact_manager.generate_run_id()

            logger.bind(
                module="HPOService",
                run_id=run_id,
                model_name=definition.name,
                model_kind=definition.model_kind,
            ).info("Starting model trial")

            try:
                # Instantiate worker
                worker = self.worker_factory.get_worker(definition, run_id)

                # Train model
                model = worker.train(X_train, y_train)

                # Save model artifact
                model_path = worker.save(model, run_id)

                # Evaluate model
                metrics = self.evaluator.evaluate(model, X_val, y_val)

                # Save trial metrics
                self.artifact_manager.save_metrics(
                    metrics=metrics,
                    model_name=definition.name,
                    model_uri=model_path,
                    training_params=config.training,
                    hyperparameters=definition.params,
                    run_id=run_id,
                )

                # Build trial result
                trial = TrialResult(
                    model_name=definition.name,
                    model_kind=definition.model_kind,
                    params=definition.params,
                    metrics=metrics,
                    status="success",
                    model_path=model_path,
                )

                trials.append(trial)

                # Select best model
                scoring_metric = self.search_config.scoring
                if (
                    best_trial is None
                    or metrics[scoring_metric] < best_trial.metrics[scoring_metric]
                ):
                    best_trial = trial

            except ModelTrainingError as e:
                trial = TrialResult(
                    model_name=definition.name,
                    model_kind=definition.model_kind,
                    params=definition.params,
                    metrics={},
                    status="failed",
                    model_path="N/A",
                )
                trials.append(trial)

                logger.bind(
                    module="HPOService",
                    run_id=run_id,
                    error=str(e),
                ).error("Model trial failed")

        return ModelSearchResult(trials=trials, best_model=best_trial)
