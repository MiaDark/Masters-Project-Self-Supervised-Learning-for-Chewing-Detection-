"""Broadcast train.py's window-level predictions back to per-timestep
resolution -- a library module (no CLI, like `common/evaluation/postprocess.py`/
`pipeline.py`), not a standalone script.

train.py's model predicts one probability per WINDOW (e.g. 200 timesteps for
the default non-overlapping 4s/50Hz segmentation) -- but a recording's raw
signal has one row per timestep. `compute_upscaled_recordings` re-predicts on
a feature table (data/processed/ml_features/<purpose>.parquet, which carries
dataset/participant/tag/start alongside the features) and, for each distinct
recording, "paints" each window's probability across every original-row index
it covers, returning one dict per recording in the same shape
`ssl/masked/evaluate_bouts.py`'s recordings use --
`{dataset, participant, tag, segment, gt, probs, envelope}` -- ready for
`chewing_ssl.common.evaluation.pipeline.evaluate_recording`.

If windows overlap (segment.py --stride-seconds < --window-seconds), some
timesteps are covered by more than one window -- their probability is the
mean across every window covering them. Non-overlapping segmentation is the
special case where every timestep has exactly one contributor, so this
reduces to a plain broadcast automatically -- no separate code path needed.

Always computed FRESH from --model-path, never cached to disk. This used to
be a separate CLI script (`upscale_predictions.py`) that wrote its output to
`checkpoints/ml_predictions_upscaled/`, which every downstream script
(`evaluate_bouts.py`, `plot_predictions.py`, `compare_bites.py`) then read
back -- a real reproducibility risk: retraining the model without
remembering to re-run that script left every downstream script silently
evaluating the OLD model's predictions, with no error. `evaluate_bouts.py`/
`plot_predictions.py`/`compare_bites.py` now call this function directly,
the same way SSL's scripts always run inference fresh from whatever
checkpoint they're pointed at -- there is no cached prediction artifact left
to go stale. See docs/bite_evaluation.md.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chewing_ssl.classical_ml.train import META_COLUMNS
from chewing_ssl.common.evaluation.postprocess import ENVELOPE_CHANNELS, FS, compute_envelope


def upscale_recording(
    probs: np.ndarray, starts: np.ndarray, window_len: int, threshold: float = 0.5
) -> pd.DataFrame:
    """probs/starts: (n_windows,) for ONE recording's windows -> per-timestep
    DataFrame (only covered rows; row_idx is the 0-based index into the
    original segment). threshold only affects the diagnostic pred_label
    column -- evaluation itself uses the continuous pred_prob."""
    n_rows = int(starts.max()) + window_len
    prob_sum = np.zeros(n_rows, dtype=np.float64)
    coverage = np.zeros(n_rows, dtype=np.int32)

    for start, p in zip(starts, probs):
        prob_sum[start : start + window_len] += p
        coverage[start : start + window_len] += 1

    covered = coverage > 0
    pred_prob = np.full(n_rows, np.nan)
    pred_prob[covered] = prob_sum[covered] / coverage[covered]

    return (
        pd.DataFrame(
            {
                "row_idx": np.arange(n_rows),
                "pred_prob": pred_prob,
                "pred_label": (pred_prob >= threshold).astype(float),
                "n_windows_covering": coverage,
            }
        )
        .loc[covered]
        .reset_index(drop=True)
    )


def compute_upscaled_recordings(
    model_path: Path,
    features_path: Path,
    interim_root: Path,
    window_len: int,
    envelope_window_s: float = 1.5,
) -> list[dict]:
    """Run `model_path` fresh against `features_path` and return one dict per
    recording: {dataset, participant, tag, segment, gt, probs, envelope} --
    probs is NaN outside the window(s) that cover a given row, matching
    `ssl/masked/evaluate_bouts.py`'s recording shape. `gt` and `envelope`
    come from `interim_root` (ground truth is dense per-timestep there
    already; no upscaling needed for it)."""
    pipeline = joblib.load(model_path)
    df = pd.read_parquet(features_path)

    meta_cols = [c for c in META_COLUMNS if c in df.columns and c != "label"]
    X = df.drop(columns=[c for c in META_COLUMNS if c in df.columns])
    probs_all = pipeline.predict_proba(X)[:, 1]

    recordings = []
    # Group by segment too, not just (dataset, participant, tag): a recording
    # can have multiple segment files (gap-segmentation splits dropouts into
    # separate segments under the same tag), each with independent `start`
    # numbering -- merging them would silently mix unrelated windows.
    for (dataset, participant, tag, segment), group in (
        df[meta_cols].assign(_prob=probs_all).groupby(["dataset", "participant", "tag", "segment"])
    ):
        seg_path = interim_root / dataset / tag / f"{segment}.parquet"
        if not seg_path.exists():
            print(f"  [skip] no interim segment for {dataset}/{tag}/{segment}")
            continue

        upscaled = upscale_recording(
            group["_prob"].to_numpy(), group["start"].to_numpy(), window_len
        )

        seg = pd.read_parquet(seg_path, columns=["label", *ENVELOPE_CHANNELS])
        gt = seg["label"].to_numpy(dtype=bool)

        probs = np.full(len(gt), np.nan, dtype=np.float32)
        probs[upscaled["row_idx"].to_numpy()] = upscaled["pred_prob"].to_numpy()

        signal_mag = np.sqrt((seg[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
        envelope = compute_envelope(signal_mag, fs=FS, window_s=envelope_window_s)

        recordings.append(
            {
                "dataset": dataset,
                "participant": participant,
                "tag": tag,
                "segment": segment,
                "gt": gt,
                "probs": probs,
                "envelope": envelope,
            }
        )
    return recordings
