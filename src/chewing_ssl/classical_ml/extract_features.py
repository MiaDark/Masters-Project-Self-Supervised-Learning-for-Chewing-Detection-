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
import warnings
from concurrent.futures import ProcessPoolExecutor
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


def _extract_one(task: tuple[str, np.ndarray, str, float]) -> pd.DataFrame:
    """Top-level, picklable worker for ProcessPoolExecutor: one (family,
    channel) pair's features. emteqai's own internal parallelism is always
    OFF here (parallel=False, n_jobs=1) -- this function call itself IS the
    unit of parallelism (see extract_features' docstring); nesting emteqai's
    finer-grained internal joblib parallelism inside each worker process
    would reintroduce the exact worker-pool-spawn-overhead pathology that
    made the old --parallel path ~77x slower."""
    family, data, channel, fs = task
    with warnings.catch_warnings():
        # Expected, already safety-netted: near-constant-variance windows
        # make scipy's kurtosis/skew (stat_driver) numerically unstable
        # (catastrophic cancellation dividing by a tiny variance). Any
        # resulting NaN/Inf is replaced by the train-split median in
        # sanitize() downstream -- this is a diagnostic, not a correctness
        # bug, and would otherwise spam output across thousands of windows.
        warnings.filterwarnings(
            "ignore", message="Precision loss occurred in moment calculation.*"
        )
        return DRIVERS[family](data, channel, fs, False, 1)


def extract_features(
    X: np.ndarray, families: list[str], fs: float, parallel: bool, n_jobs: int
) -> pd.DataFrame:
    """X: (n_windows, n_channels, window_len) -> (n_windows, n_features) DataFrame.

    `parallel`/`n_jobs` control OUTER parallelism across the channel x
    family task grid (e.g. 10 channels x 4 families = 40 independent tasks,
    each processing the FULL window batch for one channel/family) via a
    process pool -- not emteqai's own internal per-call parallelism (see
    _extract_one). Each of these 40 tasks is substantial (thousands of
    windows), unlike the fine-grained chunks emteqai's own internal
    parallelism splits a single call into, so worker-pool spawn overhead is
    paid once per task instead of repeatedly per tiny chunk -- this is what
    actually benefits from concurrency at this workload's scale, where the
    previous (emteqai-internal) --parallel option measured 77x slower.
    """
    tasks = [
        (family, X[:, ch_idx, :], channel, fs)
        for ch_idx, channel in enumerate(SENSOR_CHANNELS)
        for family in families
    ]
    if not parallel:
        per_channel_frames = [_extract_one(t) for t in tasks]
    else:
        max_workers = None if n_jobs is None or n_jobs <= 0 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            per_channel_frames = list(pool.map(_extract_one, tasks))
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

    n_per_channel = sum(
        {"statistical": 9, "time": 14, "frequency": 35, "wavelet": 27}[f] for f in families
    )
    segment_meta_path = segments_dir / "segment_meta.json"
    segmentation = (
        json.loads(segment_meta_path.read_text()) if segment_meta_path.exists() else None
    )
    feature_meta = {
        "segments_dir": str(segments_dir),
        "families": families,
        "fs": fs,
        "channels": SENSOR_CHANNELS,
        "n_features_per_channel": n_per_channel,
        "n_features_total": len(SENSOR_CHANNELS) * n_per_channel,
        # Best-effort: only present if segment.py wrote it (older segment.py
        # runs, or a segments_dir built some other way, won't have one) --
        # this is what lets train.py log "how the input was processed" (window
        # length, train/eval stride, overlap) without needing its own
        # knowledge of the segmentation step, several stages upstream of it.
        "segmentation": segmentation,
    }
    (out_dir / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2))
    print(f"  Wrote provenance -> {out_dir / 'feature_meta.json'}"
          + ("" if segmentation else "  (segment_meta.json not found upstream -- segmentation info omitted)"))


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
        help="Parallelize across the channel x family task grid (e.g. 10 channels x 4 "
        "families = 40 tasks) using a process pool -- one task per worker, each processing "
        "the full window batch for one channel/family. Default off (sequential). NOT the "
        "same as an earlier version's --parallel, which enabled emteqai's own internal "
        "per-call parallelism and was measured ~77x SLOWER (worker-pool spawn overhead "
        "dominating many small per-call chunks); emteqai's internal parallelism stays "
        "permanently disabled regardless of this flag, to avoid nesting the two.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Worker processes for --parallel. -1 = all cores."
    )
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
