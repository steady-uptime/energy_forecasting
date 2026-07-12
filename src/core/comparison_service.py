# src/core/comparison_service.py
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from pathlib import Path
from src.core.exceptions import ComparisonError
from loguru import logger

@dataclass
class CandidateResult:
    model_id: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    artifact_path: str

@dataclass
class ComparisonReport:
    run_id: str
    timestamp: str
    baseline: Dict[str, Any]
    candidates: List[CandidateResult]
    champion_id: str
    primary_metric: str

class ModelComparisonService:
    """
    Handles the logic of picking a winner and generating a 
    structured JSON report for the experiment tracker.
    """
    def __init__(self, primary_metric: str, report_dir: str):
        self.primary_metric = primary_metric
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def select_champion(
        self, 
        baseline: CandidateResult, 
        candidates: List[CandidateResult]
    ) -> CandidateResult:
        best_candidate = None
        
        for candidate in candidates:
            # Logic: Is candidate better than baseline on primary metric?
            if candidate.metrics[self.primary_metric] < baseline.metrics[self.primary_metric]:
                if best_candidate is None or \
                   candidate.metrics[self.primary_metric] < best_candidate.metrics[self.primary_metric]:
                    best_candidate = candidate
        
        # Fallback to baseline if no candidate was better
        winner = best_candidate if best_candidate else baseline
        logger.success(f"Champion selected: {winner.model_id}")
        return winner

    def save_report(self, run_id: str, baseline: CandidateResult, candidates: List[CandidateResult], winner: CandidateResult):
        report = ComparisonReport(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            baseline=asdict(baseline),
            candidates=[asdict(c) for c in candidates],
            champion_id=winner.model_id,
            primary_metric=self.primary_metric
        )
        
        report_path = self.report_dir / f"comparison_{run_id}.json"
        with open(report_path, "w") as f:
            json.dump(report.__dict__, f, indent=4)
            
        logger.info(f"Comparison report saved to {report_path}")
