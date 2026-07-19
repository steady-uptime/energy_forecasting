# src/core/config/factories/paths_factory.py
from ..schemas import PathsConfig

def build_paths_config(raw: dict, project_root: str) -> PathsConfig:
    """
    Constructs PathsConfig. 
    Note: project_root is injected as a dependency to ensure 
    deterministic path resolution.
    """
    # We use **raw to map primitives automatically.
    # project_root is not in the YAML, it is provided by the orchestrator.
    return PathsConfig(**raw)