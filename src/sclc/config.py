"""
Configuration management for SCLC pipeline.
All paths and parameters must be loaded from config files - no hardcoding.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config(config_name: str = "pipeline") -> Dict[str, Any]:
    """
    Load a configuration file from configs/ directory.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Dictionary containing configuration
    """
    root = get_project_root()
    config_path = root / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    return load_yaml(config_path)


def load_cohorts() -> Dict[str, Any]:
    """Load cohort definitions from configs/cohorts.yaml."""
    return load_config("cohorts")


def load_signatures() -> Dict[str, Any]:
    """Load gene signatures from configs/signatures.yaml."""
    return load_config("signatures")


def load_stage0_assets() -> Dict[str, Any]:
    """Load stage 0 asset definitions."""
    return load_config("stage0_assets")


def get_env_var(var_name: str, required: bool = False) -> Optional[str]:
    """
    Get environment variable, optionally required.
    Never log or store the actual value in code.

    Args:
        var_name: Name of environment variable
        required: If True, raise error if not set

    Returns:
        Value of environment variable or None
    """
    value = os.environ.get(var_name)

    if required and value is None:
        raise EnvironmentError(
            f"Required environment variable {var_name} is not set. "
            f"See .env.example for required variables."
        )

    return value


def get_paths(config: Optional[Dict] = None) -> Dict[str, Path]:
    """
    Get all configured paths as Path objects.

    Args:
        config: Pipeline config dict, loaded if not provided

    Returns:
        Dictionary of path names to Path objects
    """
    if config is None:
        config = load_config("pipeline")

    root = get_project_root()
    paths = {}

    for key, value in config.get("paths", {}).items():
        paths[key] = root / value

    return paths


def ensure_paths_exist(paths: Dict[str, Path]) -> None:
    """Create directories for all configured paths."""
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)


class PipelineConfig:
    """
    Pipeline configuration manager.
    Provides structured access to all configuration.
    """

    def __init__(self):
        self.root = get_project_root()
        self._pipeline = None
        self._cohorts = None
        self._signatures = None
        self._assets = None

    @property
    def pipeline(self) -> Dict[str, Any]:
        if self._pipeline is None:
            self._pipeline = load_config("pipeline")
        return self._pipeline

    @property
    def cohorts(self) -> Dict[str, Any]:
        if self._cohorts is None:
            self._cohorts = load_cohorts()
        return self._cohorts

    @property
    def signatures(self) -> Dict[str, Any]:
        if self._signatures is None:
            self._signatures = load_signatures()
        return self._signatures

    @property
    def assets(self) -> Dict[str, Any]:
        if self._assets is None:
            self._assets = load_stage0_assets()
        return self._assets

    @property
    def paths(self) -> Dict[str, Path]:
        return get_paths(self.pipeline)

    def is_stage_enabled(self, stage_name: str) -> bool:
        """Check if a pipeline stage is enabled."""
        stages = self.pipeline.get("stages", {})
        stage = stages.get(stage_name, {})
        return stage.get("enabled", False)

    def get_open_access_cohorts(self) -> Dict[str, list]:
        """Get only open-access cohorts for automated download."""
        cohorts = self.cohorts.get("cohorts", {})
        open_cohorts = {}

        for data_type, datasets in cohorts.items():
            if data_type == "controlled":
                continue
            open_cohorts[data_type] = [
                d for d in datasets
                if d.get("access") == "open"
            ]

        return open_cohorts

    def get_controlled_cohorts(self) -> list:
        """Get controlled-access cohorts for scaffold generation."""
        return self.cohorts.get("cohorts", {}).get("controlled", [])
