"""Raw -> interim preprocessing: channel selection, gap segmentation,
resampling, and filtering + derivative.

Pipeline position (see docs/architecture.md, docs/preprocessing.md):
    data/raw  --[this module]-->  data/interim/<dataset>/<tag>/  (raw-scale)
        --[windowing.py, splits only]--> data/processed/splits.json
        --[refit_global_norm.py]--> data/interim_normalized/<dataset>/<tag>/
        --[windowing.py, index]--> data/processed/index/<purpose>.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from chewing_ssl.common.data.eda import SIGNAL_CLIP_RANGES, clip_signal_ranges, discover_recordings
from chewing_ssl.common.data.loaders import ACCEL_COLS, load_recording

DEFAULT_TARGET_HZ = 50.0
DEFAULT_GAP_THRESHOLD_S = 1.0
DEFAULT_MIN_SEGMENT_S = 10.0

BUTTER_CUTOFF_HZ = 8.0
BUTTER_ORDER = 4

DERIVATIVE_SUFFIX = "_d1"

# The only 5 raw channels this model uses. Proximity and Nav RightTemple are
# both deliberately excluded -- see module docstring.
RAW_CHANNELS = [*ACCEL_COLS, "nav_x_left_temple", "nav_y_left_temple"]
DERIVATIVE_CHANNELS = [f"{c}{DERIVATIVE_SUFFIX}" for c in RAW_CHANNELS]
SENSOR_CHANNELS = RAW_CHANNELS + DERIVATIVE_CHANNELS

# Single-modality channel subsets (raw channels only -- see channels_for_modality
# for the raw+derivative expansion), shared by every script that needs to
# build a nav-only or accel-only model or ablation: classical_ml's
# segment.py/extract_features.py, ssl/masked's pretrain.py/finetune.py, and
# predict_new_recording.py's inference-time zero-ablation.
MODALITY_CHANNELS: dict[str, list[str]] = {
    "nav": ["nav_x_left_temple", "nav_y_left_temple"],
    "accel": list(ACCEL_COLS),
}


def channels_for_modality(only: str) -> list[str]:
    """'all' -> the full SENSOR_CHANNELS list. 'nav'/'accel' -> just that
    modality's raw + `_d1` derivative channels (a genuine channel-count
    reduction, for training a real single-modality model -- contrast with
    predict_new_recording.py's zero_out_modality, which keeps all 10
    channels and zeroes the excluded modality's values instead, for
    ablating an already-trained joint model without retraining)."""
    if only == "all":
        return list(SENSOR_CHANNELS)
    raw = MODALITY_CHANNELS[only]
    return raw + [f"{c}{DERIVATIVE_SUFFIX}" for c in raw]


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only datetime, the 5 raw sensor channels, and label (if present)."""
    cols = ["datetime", *[c for c in RAW_CHANNELS if c in df.columns]]
    if "label" in df.columns:
        cols.append("label")
    return df[cols].copy()


def apply_lowpass_filter(
    df: pd.DataFrame,
    channels: list[str] = RAW_CHANNELS,
    cutoff_hz: float = BUTTER_CUTOFF_HZ,
    order: int = BUTTER_ORDER,
    fs: float = DEFAULT_TARGET_HZ,
) -> pd.DataFrame:
    """Zero-phase Butterworth low-pass, applied independently per channel."""
    out = df.copy()
    b, a = butter(order, cutoff_hz / (fs / 2.0), btype="low")
    for c in channels:
        if c in out.columns:
            out[c] = filtfilt(b, a, out[c].to_numpy())
    return out


def add_derivative_channels(
    df: pd.DataFrame, channels: list[str] = RAW_CHANNELS, fs: float = DEFAULT_TARGET_HZ
) -> pd.DataFrame:
    """First time-derivative (per second) of each channel, as new `<channel>_d1` columns.

    Computed on the already-filtered signal -- differentiating raw/noisy
    data would amplify high-frequency noise, so filter first (see
    apply_lowpass_filter, always called before this in resample_recording).
    """
    out = df.copy()
    dt = 1.0 / fs
    for c in channels:
        if c in out.columns:
            out[f"{c}{DERIVATIVE_SUFFIX}"] = np.gradient(out[c].to_numpy(), dt)
    return out


def segment_at_gaps(
    df: pd.DataFrame, gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S
) -> list[pd.DataFrame]:
    """Split into contiguous chunks wherever the timestamp gap exceeds the threshold."""
    dt = np.diff(df["datetime"].to_numpy()) / np.timedelta64(1, "s")
    breaks = np.where(dt > gap_threshold_s)[0] + 1
    bounds = [0, *breaks.tolist(), len(df)]
    return [
        df.iloc[bounds[i] : bounds[i + 1]].reset_index(drop=True) for i in range(len(bounds) - 1)
    ]


def resample_segment(df: pd.DataFrame, target_hz: float = DEFAULT_TARGET_HZ) -> pd.DataFrame:
    """Resample one contiguous segment onto a uniform grid at target_hz.

    Sensor channels are linearly interpolated. The label, if present, is
    carried over by nearest-neighbour -- linear interpolation would invent
    fractional labels, which makes no sense for a binary chewing flag.
    """
    start, end = df["datetime"].iloc[0], df["datetime"].iloc[-1]
    n_samples = max(int(round((end - start).total_seconds() * target_hz)) + 1, 1)
    grid = pd.date_range(start, end, periods=n_samples)

    t_src = (df["datetime"] - start).dt.total_seconds().to_numpy()
    t_grid = (grid - start).total_seconds().to_numpy()

    out = pd.DataFrame({"datetime": grid})
    for c in RAW_CHANNELS:
        if c in df.columns:
            out[c] = np.interp(t_grid, t_src, df[c].to_numpy())

    if "label" in df.columns:
        out = pd.merge_asof(out, df[["datetime", "label"]], on="datetime", direction="nearest")

    return out


def fit_global_stats(segments: list[pd.DataFrame], channels: list[str] = SENSOR_CHANNELS) -> dict:
    """Population mean/std per channel across many resampled-but-unnormalized segments.

    For the global-normalization ablation: fit on training-split segments
    only, to avoid leaking validation/test statistics (decision 003).
    """
    stats = {}
    for c in channels:
        values = np.concatenate([seg[c].to_numpy() for seg in segments if c in seg.columns])
        stats[c] = {"mean": float(values.mean()), "std": float(values.std())}
    return stats


def apply_global_stats(df: pd.DataFrame, stats: dict, eps: float = 1e-8) -> pd.DataFrame:
    out = df.copy()
    for c, s in stats.items():
        if c in out.columns:
            out[c] = (out[c] - s["mean"]) / (s["std"] + eps)
    return out


def fit_minmax_stats(
    segments: list[pd.DataFrame],
    channels: list[str] = SENSOR_CHANNELS,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict:
    """Per-channel (lo, hi) bounds, pooled across every segment of ONE
    recording (all its gap-bounded segments combined) -- the fit population
    for per-recording min-max normalization to [-1, 1].

    Unlike `fit_global_stats`, this is never pooled ACROSS recordings/
    participants -- each recording supplies and consumes its own bounds, so
    there's no train/val/test split dependency and no decision-003 leakage
    concern to guard against (nothing crosses a file boundary).

    `lower_pct`/`upper_pct` (default 1st/99th percentile), not the literal
    min/max: a direct check across all 2650 (recording, channel) pairs in
    this project's raw data found some recordings (concentrated in
    Spectacle_NonMeal's nav channels, both raw and derivative) where a
    handful of samples -- a single sensor glitch amplified by np.gradient
    into a derivative spike -- inflate the true range up to ~460x beyond
    where 98% of the actual signal sits. Literal min/max would compress the
    entire real signal in those recordings to near-zero after scaling;
    percentile-based bounds are robust to that while still being a plain
    min-max-style scaling in spirit. See `apply_minmax_stats` for the
    matching clip (percentile bounds don't cover the tails by construction,
    so values beyond them must be clipped to stay within [-1, 1]).
    """
    stats = {}
    for c in channels:
        values = np.concatenate([seg[c].to_numpy() for seg in segments if c in seg.columns])
        lo, hi = np.percentile(values, [lower_pct, upper_pct])
        stats[c] = {"lo": float(lo), "hi": float(hi), "lower_pct": lower_pct, "upper_pct": upper_pct}
    return stats


def apply_minmax_stats(df: pd.DataFrame, stats: dict, eps: float = 1e-8) -> pd.DataFrame:
    """Rescale each channel to [-1, 1] via its own (lo, hi) bounds:
    `2*(x-lo)/(hi-lo+eps) - 1`, then clip to [-1, 1] -- values outside the
    fitted [lo, hi] range (expected: `fit_minmax_stats` uses percentile
    bounds, not the literal extremes, so ~1% of samples on each tail fall
    outside by construction) are clamped rather than left to blow past the
    intended range. `eps` guards a degenerate (hi == lo) channel, though
    none were observed in this project's data."""
    out = df.copy()
    for c, s in stats.items():
        if c in out.columns:
            span = s["hi"] - s["lo"]
            scaled = 2.0 * (out[c] - s["lo"]) / (span + eps) - 1.0
            out[c] = scaled.clip(-1.0, 1.0)
    return out


def resample_recording(
    dataset: str,
    path: Path,
    target_hz: float = DEFAULT_TARGET_HZ,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    cutoff_hz: float = BUTTER_CUTOFF_HZ,
    filter_order: int = BUTTER_ORDER,
) -> list[tuple[pd.DataFrame, dict]]:
    """Load one recording and return its gap-bounded, resampled + filtered +
    differentiated (segment_df, metadata) pairs, at raw scale -- i.e.
    everything except normalization. Shared by both the 'personal' and
    'global' normalization variants so the (expensive) raw CSV read,
    resampling, filtering and differentiation only happen once per recording.
    """
    df = select_columns(load_recording(dataset, path))

    results = []
    for raw_segment in segment_at_gaps(df, gap_threshold_s):
        duration_s = (
            raw_segment["datetime"].iloc[-1] - raw_segment["datetime"].iloc[0]
        ).total_seconds()
        if duration_s < min_segment_s:
            continue
        resampled = resample_segment(raw_segment, target_hz)
        resampled = clip_signal_ranges(resampled)
        filtered = apply_lowpass_filter(resampled, RAW_CHANNELS, cutoff_hz, filter_order, target_hz)
        with_derivative = add_derivative_channels(filtered, RAW_CHANNELS, target_hz)
        meta = {
            "dataset": dataset,
            "source_path": str(path),
            "target_hz": target_hz,
            "duration_s": duration_s,
            "n_samples": len(with_derivative),
            "has_label": "label" in with_derivative.columns,
            "clip": {"ranges": SIGNAL_CLIP_RANGES},
            "filter": {
                "type": "butterworth_lowpass",
                "cutoff_hz": cutoff_hz,
                "order": filter_order,
                "zero_phase": True,
            },
            "derivative": {
                "method": "np.gradient",
                "suffix": DERIVATIVE_SUFFIX,
                "channels": RAW_CHANNELS,
            },
        }
        results.append((with_derivative, meta))
    return results


def run_batch(datasets: list[str], root: Path, out_root: Path, **kwargs) -> None:
    """For every recording: resample, filter, differentiate, and write the
    raw-scale (unnormalized) segments to out_root/<dataset>/<tag>/segment_NN.*.

    Deliberately does NOT normalize anything here. Normalization needs to be
    fit on training-split participants only (decision 003), and those splits
    can't be assigned until this raw-scale interim tree exists for
    windowing.py to discover tags/participants from -- so fitting+applying
    global stats is chewing_ssl.common.data.refit_global_norm's job, run after this
    and after windowing.py, not this module's.
    """
    for dataset in datasets:
        recordings = discover_recordings(dataset, root)
        print(f"\n=== {dataset}: {len(recordings)} file(s) ===")
        for i, (path, tag) in enumerate(recordings, 1):
            print(f"[{i}/{len(recordings)}] {dataset}/{tag}")
            try:
                segments = resample_recording(dataset, path, **kwargs)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                continue

            out_dir = out_root / dataset / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, (resampled, meta) in enumerate(segments):
                resampled.to_parquet(out_dir / f"segment_{j:02d}.parquet", index=False)
                (out_dir / f"segment_{j:02d}.json").write_text(json.dumps(meta, default=str))
            print(f"  -> {len(segments)} segment(s) written to {out_dir}")


def main() -> None:
    from chewing_ssl.common.utils.config import load_preprocessing_config

    cfg = load_preprocessing_config()

    parser = argparse.ArgumentParser(
        description="Raw -> interim preprocessing (raw-scale, unnormalized). "
        "Defaults come from configs/preprocessing.yaml; CLI flags override it for one-off runs."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Use a different preprocessing config file."
    )
    parser.add_argument(
        "--dataset",
        choices=cfg["datasets"],
        help="Only this dataset; default: all 5 in the config.",
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--target-hz", type=float, default=None)
    parser.add_argument("--gap-threshold-s", type=float, default=None)
    parser.add_argument("--min-segment-s", type=float, default=None)
    parser.add_argument(
        "--cutoff-hz", type=float, default=None, help="Butterworth low-pass cutoff."
    )
    parser.add_argument("--filter-order", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        cfg = load_preprocessing_config(args.config)

    datasets = [args.dataset] if args.dataset else cfg["datasets"]
    root = args.root or Path(cfg["root"])
    out = args.out or Path(cfg["paths"]["interim"])
    run_batch(
        datasets,
        root,
        out,
        target_hz=args.target_hz or cfg["resample"]["target_hz"],
        gap_threshold_s=args.gap_threshold_s or cfg["resample"]["gap_threshold_s"],
        min_segment_s=args.min_segment_s or cfg["resample"]["min_segment_s"],
        cutoff_hz=args.cutoff_hz or cfg["filter"]["cutoff_hz"],
        filter_order=args.filter_order or cfg["filter"]["order"],
    )


if __name__ == "__main__":
    main()
