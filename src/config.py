"""Loads config/config.yaml so no module hardcodes paths or hyperparameters."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the project configuration file."""
    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str) -> Path:
    """Turn a config path (recorded relative to the repo root) into an
    absolute path, so scripts work regardless of the working directory.
    """
    return PROJECT_ROOT / relative
