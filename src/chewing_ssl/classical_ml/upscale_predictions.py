"""Broadcast train_ml.py's window-level predictions back to per-timestep
resolution, for overlaying on the raw signal (see plot_predictions.py).

train_ml.py's model predicts one probability per WINDOW (e.g. 200 timesteps
for the default non-overlapping 4s/50Hz segmentation) -- but a recording's
raw signal has one row per timestep. This script re-predicts on a feature
table (data/processed/ml_features/<purpose>.parquet, which carries
dataset/participant/tag/start alongside the features) and, for each distinct
(dataset, participant, tag) recording, "paints" each window's probability
across every original-row index it covers.

If windows overlap (segment_ml.py --stride-seconds < --window-seconds), some
timesteps are covered by more than one window -- their probability is the
mean across every window covering them (n_windows_covering records how
many, so a downstream reader isn't left guessing). Non-overlapping
segmentation is the special case where every timestep has exactly one
contributor, so this reduces to a plain broadcast automatically -- no
separate code path needed.

Ground truth is NOT touched here -- it's already dense per-timestep in
data/interim_normalized, no upscaling needed; plot_predictions.py reads it
directly from there and merges in this script's output by row index.

Output: one parquet per recording, data/processed/ml_predictions_upscaled/<dataset>_<participant>_<tag>.parquet
  row_idx             int, 0-based index into the ORIGINAL segment's rows
  pred_prob           float, mean predicted probability covering this row
  pred_label           0/1, pred_prob >= --threshold
  n_windows_covering  int, how many windows contributed (diagnostic)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.classical_ml.train import META_COLUMNS


def upscale_recording(
    probs: np.ndarray, starts: np.ndarray, window_len: int, threshold: float
) -> pd.DataFrame:
    """probs/starts: (n_windows,) for ONE recording's windows -> per-timestep DataFrame."""
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


def run(
    model_path: Path, features_path: Path, out_dir: Path, window_len: int, threshold: float
) -> None:
    pipeline = joblib.load(model_path)
    df = pd.read_parquet(features_path)

    meta_cols = [c for c in META_COLUMNS if c in df.columns and c != "label"]
    X = df.drop(columns=[c for c in META_COLUMNS if c in df.columns])
    probs_all = pipeline.predict_proba(X)[:, 1]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    # Group by segment too, not just (dataset, participant, tag): a recording
    # can have multiple segment files (gap-segmentation splits dropouts into
    # separate segments under the same tag), each with independent `start`
    # numbering -- merging them would silently mix unrelated windows.
    for i, ((dataset, participant, tag, segment), group) in enumerate(
        df[meta_cols].assign(_prob=probs_all).groupby(["dataset", "participant", "tag", "segment"])
    ):
        probs = group["_prob"].to_numpy()
        starts = group["start"].to_numpy()
        upscaled = upscale_recording(probs, starts, window_len, threshold)

        overlap_frac = (upscaled["n_windows_covering"] > 1).mean()
        # Filename is a plain index, not dataset/participant/tag/segment
        # stitched together -- that string is not reliably reversible (tag
        # and segment can both already contain underscores). The manifest
        # below is the source of truth for identity; readers should use it
        # rather than parsing filenames.
        out_path = out_dir / f"recording_{i:04d}.parquet"
        upscaled.to_parquet(out_path, index=False)
        manifest_rows.append(
            {
                "dataset": dataset,
                "participant": participant,
                "tag": tag,
                "segment": segment,
                "n_windows": len(group),
                "n_rows": len(upscaled),
                "out_path": str(out_path),
            }
        )
        print(
            f"  {dataset}/{participant}/{tag}/{segment}: {len(group)} windows -> {len(upscaled):,} rows  "
            f"({overlap_frac:.1%} overlapping)  -> {out_path}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(
        f"\nDone. {len(manifest)} recording(s) written to {out_dir}/  (see {manifest_path} for identities)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="From train_ml.py, e.g. checkpoints/train_ml/random_forest.joblib",
    )
    parser.add_argument(
        "--features-path", type=Path, default=Path("data/processed/ml_features/test.parquet")
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Must match whatever segment_ml.py --window-seconds was used to build --features-path. "
        "Default: configs/preprocessing.yaml's value (segment_ml.py's own default).",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("data/processed/ml_predictions_upscaled"))
    args = parser.parse_args()

    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN

    print("\nUpscaling window-level predictions to per-timestep")
    print("=" * 60)
    print(f"  Model    : {args.model_path}")
    print(f"  Features : {args.features_path}")
    print(f"  Window   : {window_len} samples ({window_len / TARGET_HZ:.1f}s)")
    print(f"  Threshold: {args.threshold}")
    print("=" * 60)

    run(args.model_path, args.features_path, args.out, window_len, args.threshold)


if __name__ == "__main__":
    main()
