from __future__ import annotations

import numpy as np


def reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Equal-width binning of `probs` into `n_bins`. Per bin: mean predicted
    confidence, observed positive rate (fraction of labels==1 within the
    bin -- the correct calibration target for a P(class=1) output; a
    well-calibrated model has bin_accuracy ~= bin_confidence in every bin),
    and count. Also returns the scalar Expected Calibration Error (ECE):
    the count-weighted average |confidence - accuracy| across bins.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)

    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_conf[b] = probs[mask].mean()
            bin_acc[b] = labels[mask].mean()

    valid = bin_count > 0
    ece = float(
        np.sum(bin_count[valid] * np.abs(bin_conf[valid] - bin_acc[valid])) / bin_count.sum()
    )
    return {
        "ece": ece,
        "bin_edges": edges,
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
        "bin_count": bin_count,
        "n_total": int(bin_count.sum()),
    }
