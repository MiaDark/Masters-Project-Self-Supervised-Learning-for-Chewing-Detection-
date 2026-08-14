from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS, channels_for_modality
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN, extract_participant, list_tags


def window_label(label_slice: np.ndarray, threshold: float = 0.5) -> float:
    """(window_len,) dense per-timestep labels -> single window-level target,
    majority vote -- same convention as finetune_window.window_labels."""
    return float(label_slice.mean() >= threshold)


def segment_recording(
    seg_path: Path, window_len: int, stride: int, channels: list[str] = SENSOR_CHANNELS
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """One segment parquet -> (X, y, starts) for every full, labeled window in it.
    X: (n, len(channels), window_len). Segments without a label column,
    or shorter than one window, contribute nothing (empty arrays)."""
    df = pd.read_parquet(seg_path)
    if "label" not in df.columns or len(df) < window_len:
        return (
            np.empty((0, len(channels), window_len), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )

    channel_arr = df[channels].to_numpy(dtype=np.float32)  # (n_samples, n_channels)
    labels = df["label"].to_numpy(dtype=np.float32)

    xs, ys, starts = [], [], []
    for start in range(0, len(df) - window_len + 1, stride):
        end = start + window_len
        xs.append(channel_arr[start:end].T)  # (n_channels, window_len)
        ys.append(window_label(labels[start:end]))
        starts.append(start)

    if not xs:
        return (
            np.empty((0, len(channels), window_len), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )
    return np.stack(xs), np.array(ys, dtype=np.float32), starts


def run(
    interim_root: Path,
    splits_path: Path,
    out_dir: Path,
    window_len: int,
    purpose_stride: dict[str, int],
    train_overlap: float | None = None,
    channels: list[str] = SENSOR_CHANNELS,
) -> None:
    """purpose_stride: {"train": ..., "val": ..., "test": ...} -- each purpose
    gets its own stride (see main()'s --train-overlap: train-purpose windows
    may overlap, val/test never do, mirroring windowing.py's own
    purpose_stride_for convention for the deep-learning pipelines).
    `train_overlap` is recorded into segment_meta.json purely for provenance
    (e.g. so extract_features.py/train.py can log "how the input was
    processed" to MLflow) -- it plays no role in the windowing itself,
    which only uses purpose_stride."""
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
                X, y, starts = segment_recording(seg_path, window_len, purpose_stride[purpose], channels)
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
                    {
                        "dataset": dataset,
                        "participant": participant,
                        "tag": tag,
                        "segment": segment,
                        "start": s,
                    }
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

    meta_path = out_dir / "segment_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "interim_root": str(interim_root),
                "splits_path": str(splits_path),
                "window_len": window_len,
                "window_seconds": window_len / TARGET_HZ,
                "purpose_stride": purpose_stride,
                "train_overlap": train_overlap,
                "channels": channels,
            },
            indent=2,
        )
    )
    print(f"  Wrote provenance -> {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/ml_segments"))
    parser.add_argument(
        "--window-seconds", type=float, default=None, help="Default: configs/preprocessing.yaml."
    )
    parser.add_argument(
        "--train-overlap",
        type=float,
        default=None,
        help="Fraction overlap for TRAIN-purpose windows only (default: "
        "configs/preprocessing.yaml's window.train_overlap, currently 0.5 -- same "
        "value and same asymmetric convention windowing.py uses for the SSL "
        "pipelines' pretrain_train/finetune_train). val/test windows are ALWAYS "
        "non-overlapping regardless of this flag -- reported metrics must stay "
        "computed on independent windows; only the training set's density changes. "
        "Pass 0.0 to reproduce the fully-non-overlapping behavior this script used "
        "before train-purpose overlap was reintroduced.",
    )
    parser.add_argument(
        "--channels", choices=["all", "nav", "accel"], default="all",
        help="Restrict windows to one modality's raw+derivative channels (4 for nav, 6 for "
        "accel) instead of the full 10 -- for training a genuine single-modality model. "
        "Default 'all' (unchanged behavior).",
    )
    args = parser.parse_args()

    from chewing_ssl.common.data.windowing import TRAIN_OVERLAP

    channels = channels_for_modality(args.channels)
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN
    train_overlap = args.train_overlap if args.train_overlap is not None else TRAIN_OVERLAP
    # Asymmetric by purpose, matching windowing.py's purpose_stride_for: train
    # windows may overlap (denser training set, same rationale the SSL
    # pipelines already use for pretrain_train/finetune_train), val/test
    # windows never do (independent, non-correlated windows are what makes
    # the reported window-level metrics trustworthy -- this is the one
    # invariant that must never be reintroduced as a footgun default).
    train_stride = max(int(window_len * (1 - train_overlap)), 1)
    eval_stride = window_len
    purpose_stride = {"train": train_stride, "val": eval_stride, "test": eval_stride}

    print("\nMaterialized ML segmentation")
    print("=" * 60)
    print(f"  Window length : {window_len} samples ({window_len / TARGET_HZ:.1f}s)")
    print(f"  Train stride  : {train_stride} samples ({train_overlap:.0%} overlap)")
    print(f"  Val/test stride: {eval_stride} samples (non-overlapping, always)")
    print(f"  Channels      : {len(channels)} ({', '.join(channels)})")
    print("=" * 60)

    run(args.interim_root, args.splits, args.out, window_len, purpose_stride, train_overlap, channels)

    print(f"\nDone. Segments written to {args.out}/")


if __name__ == "__main__":
    main()
