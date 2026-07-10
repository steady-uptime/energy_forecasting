# /src/core/registry.py
from dataclasses import dataclass, asdict
from src.core.exceptions import ArtifactError
from datetime import datetime, UTC
from pathlib import Path
import json


@dataclass
class RegistryRecord:
    version: str
    model_name: str
    model_path: str
    metrics: dict
    stage: str
    created_at: str


class ModelRegistry:
    """
    Lightweight, file-based model registry.

    Responsibilities:
    - Assign version numbers (v1, v2, v3, ...)
    - Persist registry records to metadata directory
    - Maintain a 'champion' pointer for inference API
    """

    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.records_file = self.registry_dir / "model_registry.json"
        self.champion_file = self.registry_dir / "champion.json"

        # Initialize registry file if missing
        if not self.records_file.exists():
            with self.records_file.open("w") as f:
                json.dump([], f)

    def load_champion(self):
        """
        Load the current champion model record.
        """
        if not self.champion_file.exists():
            raise ArtifactError(
                "Champion model not found. Train a model before running monitoring.",
                context={"champion_file": str(self.champion_file)}
            )

        try:
            with self.champion_file.open("r") as f:
                return json.load(f)

        except Exception as e:
            raise ArtifactError(
                "Failed to load champion model",
                context={"champion_file": str(self.champion_file)}
            ) from e

    def _load_records(self):
        with self.records_file.open("r") as f:
            return json.load(f)

    def _save_records(self, records):
        with self.records_file.open("w") as f:
            json.dump(records, f, indent=2)

    def _next_version(self):
        records = self._load_records()
        return f"v{len(records) + 1}"

    def register(self, model_name: str, model_path: str, metrics: dict):
        version = self._next_version()

        record = RegistryRecord(
            version=version,
            model_name=model_name,
            model_path=model_path,
            metrics=metrics,
            stage="champion",
            created_at=datetime.now(UTC).isoformat()
        )

        # Append to registry
        records = self._load_records()
        records.append(asdict(record))
        self._save_records(records)

        # Update champion pointer
        with self.champion_file.open("w") as f:
            json.dump(asdict(record), f, indent=2)

        return asdict(record)
