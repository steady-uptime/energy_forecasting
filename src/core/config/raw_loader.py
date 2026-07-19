# src/core/config/raw_loader.py
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

class RawLoader:
    """
    Responsible for loading, merging, and overriding raw configuration data.
    This is the ONLY place where raw Dicts/YAML are handled.
    """

    @staticmethod
    def load(path: Optional[str] = None, project_root: Path = None) -> Dict[str, Any]:
        """
        Loads all YAML configs from the project root, merges them, 
        applies environment overrides, and returns a raw dictionary.
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        logger.info(f"System: Loading raw configuration from {project_root}/configs")

        config_dir = project_root / "configs"
        source_files: List[Path] = []

        # 1. Identify source files
        if path:
            source_files.append(Path(path))
        
        # Add all yaml files in the configs directory
        if config_dir.exists():
            source_files.extend(list(config_dir.glob("*.yaml")))

        combined_raw_data: Dict[str, Any] = {}

        # 2. Read and Deep Merge YAML files
        for config_file in source_files:
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        file_content = yaml.safe_load(f)
                        if file_content:
                            RawLoader._deep_merge(combined_raw_data, file_content)
                            logger.debug(f"System: Merged config from {config_file.name}")
                except Exception as e:
                    logger.error(f"System: Failed to parse {config_file}: {e}")
                    raise

        # 3. Inject project root into the dict for use by factories
        combined_raw_data["project_root"] = str(project_root)

        # 4. Apply Environment Overrides
        if "env_mapping" in combined_raw_data:
            combined_raw_data = RawLoader._apply_env_overrides(combined_raw_data)

        return combined_raw_data

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Recursively merges two dictionaries.
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                RawLoader._deep_merge(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _apply_env_overrides(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifies environment variables mapped in the YAML and overrides 
        the corresponding keys in the raw dictionary.
        """
        env_mapping = raw_data.get("env_mapping", {})
        for env_var, mapping_values in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # mapping_values is expected to be a tuple: (section, key)
                section, key = mapping_values
                if section in raw_data and key in raw_data[section]:
                    current_val = raw_data[section][key]
                    target_type = type(current_val)
                    
                    # Type casting logic
                    if target_type == bool:
                        overridden_value = env_value.lower() in ("true", "1", "yes")
                    elif target_type in (int, float):
                        overridden_value = target_type(env_value)
                    else:
                        overridden_value = env_value
                    
                    raw_data[section][key] = overridden_value
                    logger.info(f"System: Overriding {section}.{key} with EnvVar {env_var}")
        return raw_data
