# src/core/run_metadata.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class PhaseMetadata:
    name: str
    status: str  # PENDING | RUNNING | SUCCESS | FAILED
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # e.g. {"raw_path": "...", "clean_path": "...", "features_path": "..."}
    artifact_paths: Dict[str, str] = field(default_factory=dict)

    # populated only on failure
    error_message: Optional[str] = None


@dataclass
class PipelineRunMetadata:
    run_id: str
    pipeline_name: str  # "train_pipeline"
    status: str  # RUNNING | SUCCESS | FAILED

    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # global artifacts
    config_snapshot_path: Optional[str] = None
    model_artifact_path: Optional[str] = None
    metrics_artifact_path: Optional[str] = None
    registry_version: Optional[str] = None

    # per-phase metadata
    phases: Dict[str, PhaseMetadata] = field(default_factory=dict)

    # optional tags (git SHA, environment, data snapshot ID)
    tags: Dict[str, str] = field(default_factory=dict)
