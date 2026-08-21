"""Shared entropy/uncertainty-decomposition math, used by both MC-Dropout
(mc_dropout.py, the masked-reconstruction SSL encoder) and tree-ensemble
(rf_uncertainty.py, the classical-ML Random Forest) uncertainty. Same
decomposition either way -- only the source of the stochastic "passes"
being decomposed differs -- so both pipelines produce directly comparable
mean/std/entropy/mutual_info outputs into one shared summary CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def binary_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decompose_uncertainty(passes: np.ndarray) -> tuple[np.ndarray, ...]:
    """passes: (n_predictors, n_items) probabilities -- T stochastic dropout
    passes of one window, or one row per tree from a Random Forest -- ->
    (mean, std, total_entropy, aleatoric_entropy, epistemic_mutual_info),
    each (n_items,).

    Standard decomposition (Gal 2016's BALD; Kwon et al. 2020): total
    predictive uncertainty (entropy of the MEAN prediction) splits into
    aleatoric (the MEAN of each individual predictor's own entropy -- noise
    inherent to the input, irreducible by more training) and epistemic (the
    remainder -- disagreement BETWEEN predictors, i.e. what more/better-
    supported data would reduce). For MC-Dropout the "predictors" are
    stochastic passes of the SAME weights; for a Random Forest they are
    genuinely different trees fit on different bootstrap samples/feature
    subsets -- arguably a cleaner match to the disagreement intuition than
    dropout's same-weights re-masking, and needs no retraining either,
    since the forest already contains all `n_estimators` trees.
    """
    mean = passes.mean(axis=0)
    std = passes.std(axis=0)
    h_total = binary_entropy(mean)
    h_aleatoric = binary_entropy(passes).mean(axis=0)
    mutual_info = h_total - h_aleatoric
    return mean, std, h_total, h_aleatoric, mutual_info


def infer_config_label(path: str | Path) -> dict:
    """Best-effort (norm, channels) label parsed from a checkpoint/model
    path, using this project's directory-naming convention
    (*_clipped[_per_file]* / *_{nav,accel}_only). Used to auto-fill the
    identity columns of the shared uncertainty-summary CSV across every
    checkpoint family instead of hand-typing them per run -- see
    checkpoints/cross_sensor_evaluation/ for the naming this mirrors."""
    s = str(path)
    norm = "per_file" if "per_file" in s else "global"
    if "nav_only" in s:
        channels = "nav"
    elif "accel_only" in s:
        channels = "accel"
    else:
        channels = "all"
    return {"norm": norm, "channels": channels}
