# src/core/config/factories/artifacts_factory.py
from ..schemas import ArtifactsConfig
def build_artifact_config(raw: dict) -> ArtifactsConfig:
    return ArtifactsConfig(**raw)