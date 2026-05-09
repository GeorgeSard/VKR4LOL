"""Загрузка путей и параметров проекта."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict[str, Any]:
    with PARAMS_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str) -> Path:
    return PROJECT_ROOT / rel


def get_mlflow_tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
