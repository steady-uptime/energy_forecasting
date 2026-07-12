from typing import List, Dict, Any
from dataclasses import replace
from loguru import logger
from src.core.exceptions import PipelineError

class HyperparameterSearchService:
    def __init__(self, model_factory: Any, evaluator: Any, artifact_manager: Any):
        self.model_factory = model_factory
        self.evaluator = evaluator
        self.artifact_manager = artifact_manager

    def search(self, app_config: Any, data_split: Dict[str, Any], run_id: str) -> List[Dict[str, Any]]:
        hpo_cfg = app_config.model.hpo
        if not hpo_cfg:
            return []

        combinations = self._generate_combinations(hpo_cfg.strategy, hpo_cfg.parameters)
        candidates = []

        for i, params in enumerate(combinations):
            trial_id = f"trial_{i:03d}"
            try:
                # Contract-Safe update using dataclasses.replace
                trial_model_cfg = replace(app_config.model, params=params)

                worker = self.model_factory.get_worker(
                    model_cfg=trial_model_cfg,
                    run_id=run_id
                )

                model_artifact = worker.train(data_split["train"], data_split["test"])
                metrics = self.evaluator.evaluate(model_artifact, data_split["test"])
                
                candidates.append({
                    "trial_id": trial_id,
                    "params": params,
                    "metrics": metrics,
                    "model_artifact": model_artifact,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Trial {trial_id} failed: {str(e)}")
                candidates.append({"trial_id": trial_id, "params": params, "status": "failed", "error": str(e)})

        return candidates

    def _generate_combinations(self, strategy: str, params_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        if strategy == "grid":
            import itertools
            keys = params_space.keys()
            values = params_space.values()
            return [dict(zip(keys, v)) for v in itertools.product(*values)]
        elif strategy == "random":
            import random
            return [{k: random.choice(v) for k, v in params_space.items()} for _ in range(10)]
        else:
            raise ValueError(f"Unsupported HPO strategy: {strategy}")
