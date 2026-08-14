"""Tree-ensemble uncertainty for the classical-ML Random Forest pipeline --
the tree-ensemble analogue of mc_dropout.py's stochastic-pass uncertainty.

A RandomForestClassifier already IS an ensemble of ~300 trees
(train.py's n_estimators=300), each fit on an independent bootstrap sample
with a random feature subset. Where MC-Dropout re-runs the SAME weights
with different dropout masks, here each of the `n_estimators` trees is a
genuinely different model fit on different data -- if anything a cleaner
match to "epistemic disagreement" than dropout's same-weights re-masking --
and no retraining is needed either, since the forest already contains
every tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from chewing_ssl.uncertainty.decompose import decompose_uncertainty


def tree_ensemble_predict(pipeline: Pipeline, X: pd.DataFrame) -> dict:
    """X: (n_windows, n_features) -> per-WINDOW mean/std/entropy/
    exp_entropy/mutual_info, each (n_windows,), decomposed across every
    tree in the forest (same role as mc_dropout's T stochastic passes,
    except each "pass" here is a different tree, not a different mask)."""
    scaler = pipeline.named_steps["scaler"]
    forest = pipeline.named_steps["model"]
    Xs = scaler.transform(X)
    passes = np.stack([tree.predict_proba(Xs)[:, 1] for tree in forest.estimators_])  # (n_trees, n_windows)
    mean, std, h_tot, h_exp, mi = decompose_uncertainty(passes)
    return {"mean": mean, "std": std, "entropy": h_tot, "exp_entropy": h_exp, "mutual_info": mi}


def upscale_columns(cols: dict[str, np.ndarray], starts: np.ndarray, window_len: int) -> dict[str, np.ndarray]:
    """Broadcast several per-window arrays (all aligned to the same
    `starts`, e.g. tree_ensemble_predict's mean/std/entropy/mutual_info) to
    per-timestep resolution together, in one pass -- same averaging-over-
    overlapping-windows rule as
    classical_ml.upscale_predictions.upscale_recording (mean across every
    window covering a timestep), generalized to many columns at once
    instead of one call per column. Returns only COVERED rows, keyed by
    `row_idx` -- the caller scatters into a full-length array itself,
    exactly like upscale_predictions.compute_upscaled_recordings does for
    the single-column case."""
    n_rows = int(starts.max()) + window_len
    coverage = np.zeros(n_rows, dtype=np.int32)
    sums = {name: np.zeros(n_rows, dtype=np.float64) for name in cols}
    for i, start in enumerate(starts):
        sl = slice(int(start), int(start) + window_len)
        coverage[sl] += 1
        for name, arr in cols.items():
            sums[name][sl] += arr[i]
    covered = coverage > 0
    row_idx = np.nonzero(covered)[0]
    out: dict[str, np.ndarray] = {"row_idx": row_idx}
    for name, s in sums.items():
        out[name] = (s[covered] / coverage[covered]).astype(np.float32)
    return out
