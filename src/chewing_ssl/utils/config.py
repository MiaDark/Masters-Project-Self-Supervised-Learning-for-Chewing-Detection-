"""Lightweight configuration loading.

Loads a YAML file, then lets specific environment variables override selected
fields so secrets and machine-specific values never live in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load variables from a local .env file if present (no-op if it isn't).
load_dotenv()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_data_config(path: str | Path = "configs/data.yaml") -> dict[str, Any]:
    """Load the data config and apply env-var overrides for S3 settings."""
    cfg = load_yaml(path)

    # Environment variables take precedence over the YAML defaults.
    bucket = os.getenv("S3_BUCKET")
    prefix = os.getenv("S3_PREFIX")
    if bucket:
        cfg.setdefault("s3", {})["bucket"] = bucket
    if prefix:
        cfg.setdefault("s3", {})["prefix"] = prefix

    return cfg
