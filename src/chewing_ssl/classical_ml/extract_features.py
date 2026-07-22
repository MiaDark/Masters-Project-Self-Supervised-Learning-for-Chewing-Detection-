"""Tabular feature extraction for classical-ML chewing-bout detection, over
the materialized windows from segment_ml.py. Uses the emteqai package's
feature drivers (already a project dependency) rather than reimplementing
standard HAR features.
  - statistical  (9/channel):  mean, std, min, max, range, IQR, kurtosis, skew, RMS
  - time-domain  (14/channel): IAV, AAC, DASDV, log-detector, MAV/MAV1/MAV2,
                                SSI, MHW, MTW, TM, variance, V-order, waveform length
  - frequency    (35/channel): top-5 PSD frequencies, 17 normalized PSD
                                amplitudes, FFT energy/entropy, PSD
                                kurtosis/skew, 9-bin PSD distribution (0-20Hz)
  - wavelet      (27/channel @ dec_level=3, 'haar'): statistical features over
                                each of the 3 coarser wavelet-decomposition
                                levels (finest detail level is skipped)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from emteqai.utils.features.frequency_features import freq_driver
from emteqai.utils.features.statistical_features import stat_driver
from emteqai.utils.features.time_features import time_driver
from emteqai.utils.features.timefrequency_features import wavelet_driver

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS

DRIVERS = {
    "statistical": lambda data, ch, fs, parallel, n_jobs: stat_driver(
        data, sensor_name=ch, parallel=parallel, n_jobs=n_jobs
    ),
    "time": lambda data, ch, fs, parallel, n_jobs: time_driver(
        data, sensor_name=ch, parallel=parallel, n_jobs=n_jobs
    ),
    "frequency": lambda data, ch, fs, parallel, n_jobs: freq_driver(
        data, sensor_name=ch, fs=fs, parallel=parallel, n_jobs=n_jobs
    ),
    "wavelet": lambda data, ch, fs, parallel, n_jobs: wavelet_driver(
        data, sensor_name=ch, parallel=parallel, n_jobs=n_jobs
    ),
}


def extract_features(
    X: np.ndarray, families: list[str], fs: float, parallel: bool, n_jobs: int
) -> pd.DataFrame:
    """X: (n_windows, n_channels, window_len) -> (n_windows, n_features) DataFrame."""
    per_channel_frames = []
    for ch_idx, channel in enumerate(SENSOR_CHANNELS):
        data = X[:, ch_idx, :]  # (n_windows, window_len)
        for family in families:
            per_channel_frames.append(DRIVERS[family](data, channel, fs, parallel, n_jobs))
    return pd.concat(per_channel_frames, axis=1)


def sanitize(features: pd.DataFrame, medians: pd.Series | None) -> tuple[pd.DataFrame, pd.Series]:
    """Replace +-inf with NaN, then fill NaN with the given per-column medians
    (computed from this same data if medians is None -- i.e. the train split)."""
    out = features.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = out.median(numeric_only=True)
    n_bad = out.isna().sum().sum()
    if n_bad:
        print(f"    sanitized {n_bad} NaN/Inf value(s) across {out.isna().any().sum()} column(s)")
    out = out.fillna(medians)
    return out, medians


def run(
    segments_dir: Path, out_dir: Path, families: list[str], fs: float, parallel: bool, n_jobs: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    medians = None

    for purpose in ("train", "val", "test"):
        x_path = segments_dir / f"{purpose}_X.npy"
        if not x_path.exists():
            print(f"  {purpose}: no segments found at {x_path} -- skipping")
            continue

        X = np.load(x_path)
        y = np.load(segments_dir / f"{purpose}_y.npy")
        meta = pd.read_parquet(segments_dir / f"{purpose}_meta.parquet")
        print(
            f"  {purpose}: extracting features for {len(y):,} windows x {len(SENSOR_CHANNELS)} channels ..."
        )

        features = extract_features(X, families, fs, parallel, n_jobs)
        features, medians = sanitize(features, medians if purpose != "train" else None)

        out = pd.concat(
            [
                meta.reset_index(drop=True),
                pd.Series(y, name="label"),
                features.reset_index(drop=True),
            ],
            axis=1,
        )
        out_path = out_dir / f"{purpose}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"    -> {out_path}  ({out.shape[0]} rows, {features.shape[1]} features)")

        if purpose == "train":
            (out_dir / "feature_medians.json").write_text(json.dumps(medians.to_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=Path("data/processed/ml_segments"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/ml_features"))
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DRIVERS),
        choices=list(DRIVERS),
        help="Which feature families to compute. Default: all 4.",
    )
    parser.add_argument(
        "--fs", type=float, default=50.0, help="Sampling rate (Hz) for frequency-domain features."
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use emteqai's joblib/loky parallelism inside each driver call. Default off: this "
        "script makes many small per-channel-per-family calls (4 families x 10 channels), so "
        "per-call worker-pool spawn overhead dominates and made a real run ~77x SLOWER in "
        "testing (162s vs 2.1s for the same 80 windows) -- only worth trying on a machine "
        "where multiprocessing overhead is cheaper relative to array size.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    n_per_channel = sum(
        {"statistical": 9, "time": 14, "frequency": 35, "wavelet": 27}[f] for f in args.features
    )
    print("\nML feature extraction")
    print(f"  Families      : {', '.join(args.features)}")
    print(
        f"  Features/win  : {len(SENSOR_CHANNELS)} channels x {n_per_channel}/channel = {len(SENSOR_CHANNELS) * n_per_channel}"
    )
    print(f"  Parallel      : {args.parallel}  (n_jobs={args.n_jobs})")

    run(args.segments_dir, args.out, args.features, args.fs, args.parallel, args.n_jobs)

    print(f"\nDone. Feature tables written to {args.out}/")


if __name__ == "__main__":
    main()
