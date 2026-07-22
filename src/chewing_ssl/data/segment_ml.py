"""Materialized (label-carrying) windowing for classical-ML chewing-bout
detection, as opposed to windowing.py's lazy index used by the deep-learning
(TF-C/TF-FC/masked-reconstruction) training loops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.data.windowing import WINDOW_LEN, extract_participant, list_tags


def window_label(label_slice: np.ndarray, threshold: float = 0.5) -> float:
    """(window_len,) dense per-timestep labels -> single window-level target,
    majority vote -- same convention as finetune_window.window_labels."""
    return float(label_slice.mean() >= threshold)


def segment_recording(seg_path: Path, window_len: int, stride: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One segment parquet -> (X, y, starts) for every full, labeled window in it.
    X: (n, len(SENSOR_CHANNELS), window_len). Segments without a label column,
    or shorter than one window, contribute nothing (empty arrays)."""
    df = pd.read_parquet(seg_path)
    if "label" not in df.columns or len(df) < window_len:
        return np.empty((0, len(SENSOR_CHANNELS), window_len), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    channels = df[SENSOR_CHANNELS].to_numpy(dtype=np.float32)  # (n_samples, n_channels)
    labels = df["label"].to_numpy(dtype=np.float32)

    xs, ys, starts = [], [], []
    for start in range(0, len(df) - window_len + 1, stride):
        end = start + window_len
        xs.append(channels[start:end].T)  # (n_channels, window_len)
        ys.append(window_label(labels[start:end]))
        starts.append(start)

    if not xs:
        return np.empty((0, len(SENSOR_CHANNELS), window_len), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    return np.stack(xs), np.array(ys, dtype=np.float32), starts


def run(interim_root: Path, splits_path: Path, out_dir: Path, window_len: int, stride: int) -> None:
    splits = json.loads(splits_path.read_text())
    datasets = [d for d in splits if d != "pretrain_val_holdout"]

    by_purpose: dict[str, dict[str, list]] = {
        p: {"X": [], "y": [], "meta": []} for p in ("train", "val", "test")
    }

    for dataset in datasets:
        for tag in list_tags(interim_root, dataset):
            participant = extract_participant(dataset, tag)
            purpose = splits[dataset].get(participant)
            if purpose not in by_purpose:
                continue

            seg_dir = interim_root / dataset / tag
            for seg_path in sorted(seg_dir.glob("segment_*.parquet")):
                X, y, starts = segment_recording(seg_path, window_len, stride)
                if len(y) == 0:
                    continue
                by_purpose[purpose]["X"].append(X)
                by_purpose[purpose]["y"].append(y)
                # segment (e.g. "segment_00") is part of the row identity, not
                # just tag: a recording can have multiple segment files
                # (gap-segmentation upstream splits dropouts into separate
                # segments under the same tag), each with its own `start`
                # numbering starting back at 0 -- without this, windows from
                # genuinely different segments can collide on an identical
                # (dataset, participant, tag, start) key.
                segment = seg_path.stem
                by_purpose[purpose]["meta"].extend(
                    {"dataset": dataset, "participant": participant, "tag": tag, "segment": segment, "start": s}
                    for s in starts
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    for purpose, parts in by_purpose.items():
        if not parts["X"]:
            print(f"  {purpose}: 0 windows -- skipping")
            continue
        X = np.concatenate(parts["X"], axis=0)
        y = np.concatenate(parts["y"], axis=0)
        meta = pd.DataFrame(parts["meta"])

        np.save(out_dir / f"{purpose}_X.npy", X)
        np.save(out_dir / f"{purpose}_y.npy", y)
        meta.to_parquet(out_dir / f"{purpose}_meta.parquet", index=False)

        n_participants = meta[["dataset", "participant"]].drop_duplicates().shape[0]
        print(
            f"  {purpose}: {len(y):>7,} windows  |  {n_participants} participants  |  "
            f"{y.mean():.1%} positive  |  shape {X.shape}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/ml_segments"))
    parser.add_argument("--window-seconds", type=float, default=None, help="Default: configs/preprocessing.yaml.")
    parser.add_argument(
        "--stride-seconds", type=float, default=None,
        help="Default: same as --window-seconds (non-overlapping). Set smaller for overlapping windows.",
    )
    args = parser.parse_args()

    from chewing_ssl.data.windowing import TARGET_HZ

    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN
    stride = int(args.stride_seconds * TARGET_HZ) if args.stride_seconds else window_len

    print("\nMaterialized ML segmentation")
    print("=" * 60)
    print(f"  Window length : {window_len} samples ({window_len / TARGET_HZ:.1f}s)")
    print(f"  Stride        : {stride} samples ({'non-overlapping' if stride >= window_len else 'overlapping'})")
    print(f"  Channels      : {len(SENSOR_CHANNELS)} ({', '.join(SENSOR_CHANNELS)})")
    print("=" * 60)

    run(args.interim_root, args.splits, args.out, window_len, stride)

    print(f"\nDone. Segments written to {args.out}/")


if __name__ == "__main__":
    main()
