"""Portable project paths with optional environment-variable overrides."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_from_env(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_DIR = _path_from_env("CAFLOW_DATA_DIR", PROJECT_ROOT / "data")
RESULTS_DIR = _path_from_env("CAFLOW_RESULTS_DIR", PROJECT_ROOT / "results")
LOG_DIR = _path_from_env("CAFLOW_LOG_DIR", PROJECT_ROOT / ".log")
OUTPUT_DIR = _path_from_env("CAFLOW_OUTPUT_DIR", PROJECT_ROOT / "outputs")
CONFIG_DIR = _path_from_env("CAFLOW_CONFIG_DIR", PROJECT_ROOT / "configs")
VQMODEL_DIR = _path_from_env("CAFLOW_VQMODEL_DIR", PROJECT_ROOT / "VQmodel")
