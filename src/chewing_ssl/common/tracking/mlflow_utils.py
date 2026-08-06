"""Thin wrappers around MLflow so every experiment logs the same way.

The tracking URI and experiment name come from the environment (.env), with
sensible local-file defaults so things work out of the box:

    MLFLOW_TRACKING_URI   default: file:./mlruns
    MLFLOW_EXPERIMENT_NAME default: chewing-ssl

"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import mlflow
from dotenv import load_dotenv

load_dotenv()


def _configure() -> None:
    # mlflow>=3 deprecated the plain filesystem backend by default; this restores
    # the documented zero-setup local file_store unless the user opts into a
    # database backend instead (see MLFLOW_TRACKING_URI in .env).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "chewing-ssl"))


@contextmanager
def start_run(
    run_name: str | None = None, tags: dict[str, str] | None = None, nested: bool = False
):
    """Context manager that configures MLflow and opens a run.

    `nested=True` opens this as a CHILD of the currently-active run (e.g. one
    child run per hyperparameter-sweep combination, grouped under one parent
    run for the whole sweep) -- MLflow requires this flag explicitly rather
    than inferring it, to avoid accidentally nesting runs from an unrelated
    already-open run elsewhere in the call stack.
    """
    _configure()
    with mlflow.start_run(run_name=run_name, tags=tags, nested=nested) as run:
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log a dict of hyper-parameters / config values."""
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics, optionally at a given training step."""
    mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log a local file (e.g. a results CSV, a feature-importance plot) as
    an MLflow artifact, attached to the current run."""
    mlflow.log_artifact(local_path, artifact_path=artifact_path)


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
