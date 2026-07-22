from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from chewing_ssl.common.data.eda import discover_recordings, find_bouts
from chewing_ssl.common.data.loaders import load_recording
from chewing_ssl.common.data.preprocess import (
    BUTTER_CUTOFF_HZ,
    BUTTER_ORDER,
    DEFAULT_TARGET_HZ,
    DERIVATIVE_SUFFIX,
    RAW_CHANNELS,
    add_derivative_channels,
    apply_lowpass_filter,
    resample_segment,
    segment_at_gaps,
    select_columns,
)

MAX_PLOT_POINTS = 200_000  # downsample the drawn line only, never the underlying signal


def load_global_stats(global_stats_path: Path, channel: str, filtered: pd.DataFrame) -> tuple[dict, str]:
    if global_stats_path.exists():
        return json.loads(global_stats_path.read_text())[channel], "global stats"
    stats = {"mean": float(filtered[channel].mean()), "std": float(filtered[channel].std())}
    return stats, "segment-level fallback -- no global_stats.json found, run refit_global_norm.py first"


def _thin(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    stride = len(df) // max_points + 1
    return df.iloc[::stride]


def _shade_bouts(ax, bouts: pd.DataFrame) -> None:
    for _, b in bouts.iterrows():
        ax.axvspan(b["start"], b["end"], color="orange", alpha=0.25, linewidth=0)


def plot_one(
    dataset: str, tag: str, path: Path, channel: str, global_stats_path: Path, out_path: Path,
    max_points: int = MAX_PLOT_POINTS,
) -> None:
    # Stage 0: raw, exactly as loaded (post scaling-revert + clip upstream,
    # before anything in preprocess.py touches it). The longest gap-free
    # segment, in full -- not a zoomed cutout.
    df = select_columns(load_recording(dataset, path))
    raw = max(segment_at_gaps(df), key=len)
    bouts = find_bouts(raw["label"], raw["datetime"]) if "label" in raw.columns else pd.DataFrame()

    # Stage 1: resampled to the uniform grid.
    resampled = resample_segment(raw, DEFAULT_TARGET_HZ)

    # Stage 2: Butterworth low-pass.
    filtered = apply_lowpass_filter(resampled, RAW_CHANNELS, BUTTER_CUTOFF_HZ, BUTTER_ORDER, DEFAULT_TARGET_HZ)

    # Stage 3: first derivative of the filtered signal.
    with_derivative = add_derivative_channels(filtered, RAW_CHANNELS, DEFAULT_TARGET_HZ)

    # Stage 4: normalized (global stats if available, else this segment's own
    # mean/std as a clearly-labelled fallback -- NOT the real global model input).
    stats, norm_label = load_global_stats(global_stats_path, channel, filtered)
    normalized = (filtered[channel] - stats["mean"]) / (stats["std"] + 1e-8)

    raw_plot = _thin(raw, max_points)
    filtered_plot = _thin(filtered, max_points)
    deriv_plot = _thin(with_derivative, max_points)
    normalized_plot = _thin(pd.DataFrame({"datetime": filtered["datetime"], "value": normalized}), max_points)

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(raw_plot["datetime"], raw_plot[channel], color="steelblue", linewidth=0.6)
    axes[0].set_title(f"1. Raw -- {channel}")

    axes[1].plot(filtered_plot["datetime"], filtered_plot[channel], color="darkorange", linewidth=0.6)
    axes[1].set_title(f"2. Filtered (Butterworth low-pass, {BUTTER_CUTOFF_HZ} Hz cutoff, order {BUTTER_ORDER})")

    axes[2].plot(
        deriv_plot["datetime"], deriv_plot[f"{channel}{DERIVATIVE_SUFFIX}"],
        color="seagreen", linewidth=0.6,
    )
    axes[2].set_title(f"3. First derivative -- {channel}{DERIVATIVE_SUFFIX} (per second, of the filtered signal)")

    axes[3].plot(normalized_plot["datetime"], normalized_plot["value"], color="firebrick", linewidth=0.6)
    axes[3].set_title(f"4. Normalized ({norm_label})")
    axes[3].set_xlabel("Time")

    for ax in axes:
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        _shade_bouts(ax, bouts)

    fig.suptitle(f"{dataset}/{tag} preprocessing pipeline -- {channel} (orange = chewing bout)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SIPMEatingStudy", help="Ignored if --all is given.")
    parser.add_argument("--root", type=Path, default=Path("src/chewing_ssl/data/data/raw_unscaled"))
    parser.add_argument("--tag", default=None, help="Recording tag; default: first found. Ignored if --all.")
    parser.add_argument("--channel", choices=RAW_CHANNELS, default="accel_z", help="Ignored if --all.")
    parser.add_argument(
        "--global-stats", type=Path, default=Path("data/processed/global_stats.json"),
        help="From refit_global_norm.py. Falls back to this segment's own mean/std (labelled as such) if missing.",
    )
    parser.add_argument("--out", type=Path, default=Path("src/chewing_ssl/data/reports/preprocessing_steps.png"))
    parser.add_argument(
        "--all", action="store_true",
        help="Batch mode: every recording in every dataset (or --datasets), for --channels, instead of one plot.",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated; --all only. Default: all 5 datasets from configs/preprocessing.yaml.",
    )
    parser.add_argument(
        "--channels", default="accel_x,nav_y_left_temple",
        help="Comma-separated; --all only.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("src/chewing_ssl/data/reports/preprocessing_steps"),
        help="--all only. Written as <out-dir>/<dataset>/<tag>/<channel>.png",
    )
    args = parser.parse_args()

    if not args.all:
        recordings = discover_recordings(args.dataset, args.root)
        if not recordings:
            raise SystemExit(f"No recordings found for {args.dataset} under {args.root}")
        if args.tag:
            path = next(p for p, t in recordings if t == args.tag)
        else:
            path, args.tag = recordings[0]
        print(f"Using {args.dataset}/{args.tag}: {path}")
        plot_one(args.dataset, args.tag, path, args.channel, args.global_stats, args.out)
        print(f"Wrote {args.out}")
        return

    from chewing_ssl.common.utils.config import load_preprocessing_config

    datasets = args.datasets.split(",") if args.datasets else load_preprocessing_config()["datasets"]
    channels = args.channels.split(",")

    total = failed = 0
    for dataset in datasets:
        recordings = discover_recordings(dataset, args.root)
        print(f"\n=== {dataset}: {len(recordings)} recording(s) ===")
        for i, (path, tag) in enumerate(recordings, 1):
            for channel in channels:
                out_path = args.out_dir / dataset / tag / f"{channel}.png"
                print(f"[{i}/{len(recordings)}] {dataset}/{tag} -- {channel}")
                try:
                    plot_one(dataset, tag, path, channel, args.global_stats, out_path)
                    total += 1
                except Exception as exc:
                    print(f"  FAILED: {exc}")
                    failed += 1

    print(f"\nWrote {total} plot(s) under {args.out_dir} ({failed} failed).")


if __name__ == "__main__":
    main()
