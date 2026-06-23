"""Thin wrappers around MLflow so every experiment logs the same way.

The tracking URI and experiment name come from the environment (.env), with
sensible local-file defaults so things work out of the box:

    MLFLOW_TRACKING_URI   default: file:./mlruns
    MLFLOW_EXPERIMENT_NAME default: chewing-ssl

Typical usage in a training script:

    from chewing_ssl.tracking.mlflow_utils import start_run, log_params
    with start_run(run_name="pretrain_masked"):
        log_params({"mask_ratio": 0.5, "lr": 1e-3, "seed": 42})
        ...  # train
        mlflow.log_metric("val_f1", f1)
        log_pytorch_model(model, "encoder")
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import mlflow


def _configure() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "chewing-ssl"))


@contextmanager
def start_run(run_name: str | None = None, tags: dict[str, str] | None = None):
    """Context manager that configures MLflow and opens a run."""
    _configure()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log a dict of hyper-parameters / config values."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics, optionally at a given training step."""
    mlflow.log_metrics(metrics, step=step)


def log_pytorch_model(model, artifact_path: str = "model", registered_name: str | None = None):
    """Log a PyTorch model as an MLflow artifact (and optionally register it).

    Passing `registered_name` adds the model to the MLflow Model Registry so you
    can version and compare your best checkpoints across runs.
    """
    import mlflow.pytorch  # imported lazily so the data stage needs no torch

    mlflow.pytorch.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_name,
    )
