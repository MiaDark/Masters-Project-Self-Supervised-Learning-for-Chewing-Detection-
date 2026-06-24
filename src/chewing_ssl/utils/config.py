from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict."""
    path = Path(path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[2] / path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config(path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    """Load the data config and apply env-var overrides for S3 settings."""
    cfg = load_yaml(path)

    # Environment variables take precedence over the YAML defaults.
    bucket = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_DEFAULT_REGION")
    if bucket:
        cfg.setdefault("s3", {})["bucket"] = bucket
    if region:
        cfg.setdefault("s3", {})["region"] = region

    return cfg
