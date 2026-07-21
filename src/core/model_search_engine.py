# src/core/model_search_engine.py
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

from src.core.config.schemas import (
    ModelDefinition,
    ModelSearchConfig,
    CandidateModel,
    SearchResults,
)
from src.core.exceptions import ModelTrainingError


class ModelSearchEngine:
    """
    Multi-model search engine.
    Trains each model in the search space, evaluates it, and selects the champion.
    """

    def __init__(
        self,
        cfg: ModelSearchConfig,
        model_factory,
        evaluator,
        artifact_manager,
    ):
        self.cfg = cfg
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.artifact_manager = artifact_manager

    def run(self, X_train, y_train, X_val, y_val, run_id: str) -> SearchResults:
        logger.bind(module="ModelSearchEngine", run_id=run_id).info(
            f"Starting model search with strategy={self.cfg.strategy}, scoring={self.cfg.scoring}"
        )

        candidates: List[CandidateModel] = []

        for definition in self.cfg.models:
            logger.bind(
                module="ModelSearchEngine",
                run_id=run_id,
                model_name=definition.name,
                model_kind=definition.model_kind,
            ).info(f"Evaluating candidate model: {definition.name}")

            try:
                # 1. Instantiate model worker
                worker = self.model_factory.get_worker(definition, run_id)

                # 2. Train model
                model = worker.train(X_train, y_train)

                # 3. Evaluate model on validation set
                metrics = self.evaluator.evaluate(model, X_val, y_val)

                # 4. Persist model artifact
                artifact_path = self.artifact_manager.save_model(
                    model=model,
                    model_name=definition.name,
                    run_id=run_id,
                )

                # 5. Build candidate object
                candidate = CandidateModel(
                    definition=definition,
                    metrics=metrics,
                    artifact_path=artifact_path,
                )
                candidates.append(candidate)

                logger.bind(
                    module="ModelSearchEngine",
                    run_id=run_id,
                    model_name=definition.name,
                    metrics=metrics,
                ).info(f"Candidate model {definition.name} evaluated successfully")

            except Exception as e:
                logger.bind(
                    module="ModelSearchEngine",
                    run_id=run_id,
                    model_name=definition.name,
                    error=str(e),
                ).error(f"Candidate model {definition.name} failed during search")
                raise ModelTrainingError(
                    f"Model search failed for candidate {definition.name}",
                    context={"model_name": definition.name, "error": str(e)},
                ) from e

        # 6. Select champion based on scoring metric
        scoring_key = self.cfg.scoring

        champion = min(
            candidates,
            key=lambda c: c.metrics.get(scoring_key, float("inf")),
        )

        logger.bind(
            module="ModelSearchEngine",
            run_id=run_id,
            champion=champion.definition.name,
            metrics=champion.metrics,
        ).success(f"Champion selected: {champion.definition.name}")

        return SearchResults(
            candidates=candidates,
            champion=champion,
        )
