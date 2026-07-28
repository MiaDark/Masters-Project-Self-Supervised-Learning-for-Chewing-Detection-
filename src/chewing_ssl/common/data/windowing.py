"""Interim -> processed: subject-disjoint splits and a lightweight window index.

Run in two passes, since normalization (refit_global_norm.py) needs splits.json
to know which participants are pretrain-eligible, but splits.json needs this
module's tag/participant discovery, which in turn needs *some* interim tree to
already exist:
  1. `--splits-only --interim-root data/interim` (the raw-scale tree) -- writes
     splits.json and stops, without touching the window index.
  2. run refit_global_norm.py (needs splits.json from step 1).
  3. `--interim-root data/interim_normalized` (now that it exists) -- reuses
     the splits.json already on disk and builds the window index against the
     normalized segments.

Design choices, and why:

- Windows are NOT materialized to disk as a giant tensor file. Storing every
  overlapping window as a dense float array would mean tens of GB of
  duplicated data for no benefit, since the underlying signal is already on
  disk as interim segments. Instead this module writes a lightweight INDEX --
  one row per window, recording which segment file it comes from and its
  [start, end) sample range -- and `WindowDataset` slices the real window out
  of the (cached) segment array at training time.
- Window length / stride are configurable (configs/preprocessing.yaml,
  `window.window_seconds` / `window.train_overlap`) for experimentation --
  see docs/preprocessing.md. Default: 4 seconds (200 samples @ 50 Hz),
  50% overlap for "train"-purpose windows (pretrain_train, finetune_train --
  more training signal.
- Labels: per-timestep (dense), carried straight through from the interim
  segment -- no majority-vote collapse. Pretraining windows never carry a
  label even when the source segment has one (decision 001: pretext tasks
  are self-supervised by construction).
- Splits are subject-disjoint (decision 003), applied with the SAME ratio
  (default 80/10/10, configs/preprocessing.yaml `splits.ratios`) uniformly
  across every dataset in the config -- SIPM/Sussex/FEIT/Office/Spectacle_NonMeal
  are all labeled, meal-or-explicitly-non-meal recordings.
  Participant IDs are different namespaces per dataset (never the same
  physical person across datasets), so splits are computed independently
  per dataset. Only TRAIN-split participants contribute to pretraining
  (conservative reading of decision 003 -- val/test participants are held
  out of pretraining too, not just fine-tuning).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.utils.config import load_preprocessing_config
from chewing_ssl.common.utils.seed import set_seed

_CFG = load_preprocessing_config()
DATASETS: list[str] = _CFG["datasets"]
SPLIT_RATIOS: dict[str, float] = _CFG["splits"]["ratios"]
DEFAULT_SEED: int = _CFG["splits"]["seed"]
PRETRAIN_VAL_FRACTION: float = _CFG["splits"]["pretrain_val_fraction"]

TARGET_HZ: float = _CFG["resample"]["target_hz"]
WINDOW_SECONDS: float = _CFG["window"]["window_seconds"]
WINDOW_LEN: int = int(WINDOW_SECONDS * TARGET_HZ)
TRAIN_OVERLAP: float = _CFG["window"]["train_overlap"]
TRAIN_STRIDE: int = max(int(WINDOW_LEN * (1 - TRAIN_OVERLAP)), 1)
EVAL_STRIDE: int = WINDOW_LEN  # no overlap


def purpose_stride_for(window_len: int, train_overlap: float = TRAIN_OVERLAP) -> dict[str, int]:
    """purpose -> stride, for a given window length. "train"-flavoured purposes
    get `train_overlap` overlap; eval ones never overlap (stride == window_len).

    Computed from the ACTUAL window_len in use, not the config/module-level
    default -- calling this instead of hardcoding the module-level
    PURPOSE_STRIDE is required whenever window_len can differ from
    configs/preprocessing.yaml's default (e.g. --window-seconds), since a
    fixed train/eval stride computed once at import time would silently
    produce a wrong (over- or under-lapping) index at any other window size.
    """
    train_stride = max(int(window_len * (1 - train_overlap)), 1)
    eval_stride = window_len
    return {
        "pretrain_train": train_stride,
        "pretrain_val": eval_stride,
        "finetune_train": train_stride,
        "finetune_val": eval_stride,
        "finetune_test": eval_stride,
        "meal_test": eval_stride,
    }


# purpose -> stride, at the config's default window length. Only valid at
# WINDOW_LEN itself -- build_window_index recomputes this from the actual
# window_len in use when purpose_stride isn't passed explicitly.
PURPOSE_STRIDE = purpose_stride_for(WINDOW_LEN)
# purposes that require a per-timestep label to exist in the segment.
LABELLED_PURPOSES = {"finetune_train", "finetune_val", "finetune_test", "meal_test"}


def list_tags(interim_root: Path, dataset: str) -> list[str]:
    """Every recording tag with preprocessed segments for this dataset.

    Unlike chewing_ssl.common.data.eda.discover_recordings (which globs the raw
    per-dataset folder layouts), interim segments are uniformly laid out as
    <dataset>/<tag>/segment_NN.{parquet,json}, so a single directory listing
    covers every dataset. Tag/participant structure is identical whether
    interim_root points at the raw-scale tree (preprocess.py's output) or the
    normalized tree (refit_global_norm.py's output) -- only the parquet
    contents differ, so this works against either.
    """
    dataset_dir = interim_root / dataset
    if not dataset_dir.is_dir():
        return []
    return sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())


def extract_participant(dataset: str, tag: str) -> str:
    """Recover the participant id from a recording tag.

    One rule for every dataset: the tag up to (not including) the first
    underscore. For SIPMEatingStudy/SussexEatingUK/FEIT_eating_S05/
    Office_eating_s05 the tag IS the participant folder name and never
    contains an underscore, so this is a no-op identity. For
    Spectacle_NonMeal, tags look like "p01_week2_p01_year_..._snippet004"
    (participant folder name + source recording stem), so this correctly
    recovers "p01".
    """
    return tag.split("_", 1)[0]


def _split_list(items: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Deterministically shuffle items and partition them by ratio.

    Returns {item: label}. The last label absorbs any rounding remainder so
    every item is assigned exactly once.
    """
    items = sorted(items)  # sort first so the shuffle is reproducible regardless of glob order
    rng = random.Random(seed)
    rng.shuffle(items)

    labels = list(ratios.keys())
    counts = [int(len(items) * ratios[label]) for label in labels[:-1]]
    counts.append(len(items) - sum(counts))

    assignment = {}
    i = 0
    for label, count in zip(labels, counts):
        for item in items[i : i + count]:
            assignment[item] = label
        i += count
    return assignment


def assign_splits(
    interim_root: Path,
    datasets: list[str] = DATASETS,
    ratios: dict[str, float] = SPLIT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Build the participant -> split assignment for every dataset, uniformly."""
    set_seed(seed)
    splits: dict[str, dict[str, str]] = {}
    for dataset in datasets:
        tags = list_tags(interim_root, dataset)
        participants = sorted({extract_participant(dataset, tag) for tag in tags})
        splits[dataset] = _split_list(participants, ratios, seed)
    return splits


def purposes_for(dataset: str, participant: str, splits: dict) -> list[str]:
    """Which window purposes a participant's segments should feed.

    "pretrain" here is the coarse, pre-monitoring-split label; build_window_index
    refines it into "pretrain_train" / "pretrain_val" via pretrain_val_holdout.
    Uniform across every dataset: train -> pretrain + finetune_train,
    val -> finetune_val, test -> finetune_test.
    """
    label = splits[dataset][participant]
    return {
        "train": ["pretrain", "finetune_train"],
        "val": ["finetune_val"],
        "test": ["finetune_test"],
    }[label]


def assign_pretrain_val_holdout(
    splits: dict,
    datasets: list[str] = DATASETS,
    seed: int = DEFAULT_SEED,
    val_fraction: float = PRETRAIN_VAL_FRACTION,
) -> dict[str, list[str]]:
    """Within the pretrain-eligible participants of each dataset, hold out a small
    slice purely for monitoring masked-reconstruction loss during pretraining.

    Independent, seed-offset shuffle from the main train/val/test split, so it
    isn't correlated with which participants ended up in which main split.
    """
    holdout: dict[str, list[str]] = {}
    for dataset in datasets:
        eligible = sorted(
            p for p in splits[dataset] if "pretrain" in purposes_for(dataset, p, splits)
        )
        sub = _split_list(
            eligible, {"pretrain_train": 1 - val_fraction, "pretrain_val": val_fraction}, seed + 1
        )
        holdout[dataset] = sorted(p for p, label in sub.items() if label == "pretrain_val")
    return holdout


def build_window_index(
    interim_root: Path,
    splits: dict,
    pretrain_val_holdout: dict,
    datasets: list[str] = DATASETS,
    window_len: int = WINDOW_LEN,
    purpose_stride: dict[str, int] | None = None,
) -> pd.DataFrame:
    """One row per window: which segment it comes from, its sample range, and purpose."""
    # Recompute from window_len when the caller doesn't pass an explicit
    # override -- PURPOSE_STRIDE alone is only correct at the config's
    # default window_len (see purpose_stride_for's docstring).
    purpose_stride = purpose_stride or purpose_stride_for(window_len)
    rows = []
    for dataset in datasets:
        for tag in list_tags(interim_root, dataset):
            participant = extract_participant(dataset, tag)
            for purpose in purposes_for(dataset, participant, splits):
                if purpose == "pretrain":
                    purpose = (
                        "pretrain_val"
                        if participant in pretrain_val_holdout[dataset]
                        else "pretrain_train"
                    )
                seg_dir = interim_root / dataset / tag
                for meta_path in sorted(seg_dir.glob("segment_*.json")):
                    meta = json.loads(meta_path.read_text())
                    if purpose in LABELLED_PURPOSES and not meta["has_label"]:
                        continue
                    n_samples = meta["n_samples"]
                    if n_samples < window_len:
                        continue
                    stride = purpose_stride[purpose]
                    segment_path = meta_path.with_suffix(".parquet")
                    for start in range(0, n_samples - window_len + 1, stride):
                        rows.append(
                            {
                                "dataset": dataset,
                                "participant": participant,
                                "tag": tag,
                                "purpose": purpose,
                                "segment_path": str(segment_path),
                                "start": start,
                                "end": start + window_len,
                            }
                        )
    return pd.DataFrame(rows)


def run(
    interim_root: Path,
    out_root: Path,
    datasets: list[str] = DATASETS,
    seed: int = DEFAULT_SEED,
    splits_only: bool = False,
    window_len: int = WINDOW_LEN,
    train_overlap: float = TRAIN_OVERLAP,
) -> None:
    splits_path = out_root / "splits.json"
    pretrain_val_holdout = None
    if splits_path.exists():
        on_disk = json.loads(splits_path.read_text())
        splits = {k: v for k, v in on_disk.items() if k != "pretrain_val_holdout"}
        pretrain_val_holdout = on_disk.get("pretrain_val_holdout")
        print(f"Reusing existing splits at {splits_path}")
    else:
        splits = assign_splits(interim_root, datasets, seed=seed)
        out_root.mkdir(parents=True, exist_ok=True)
        print("Computed new splits")

    if pretrain_val_holdout is None:
        pretrain_val_holdout = assign_pretrain_val_holdout(splits, datasets, seed)

    splits_path.write_text(
        json.dumps({**splits, "pretrain_val_holdout": pretrain_val_holdout}, indent=2)
    )
    print(f"Wrote {splits_path} (including pretrain_val_holdout)")

    if splits_only:
        print(
            "--splits-only: skipping window index (run again against the normalized tree once it exists)."
        )
        return

    index = build_window_index(
        interim_root,
        splits,
        pretrain_val_holdout,
        datasets,
        window_len,
        purpose_stride_for(window_len, train_overlap),
    )
    index_dir = out_root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    for purpose, sub in index.groupby("purpose"):
        sub.to_parquet(index_dir / f"{purpose}.parquet", index=False)
        n_participants = sub["participant"].nunique()
        print(f"  {purpose}: {len(sub)} windows from {n_participants} participant(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build subject-disjoint splits and a window index."
    )
    parser.add_argument(
        "--interim-root",
        type=Path,
        default=Path("data/interim"),
        help=(
            "Pass the raw-scale tree (data/interim) with --splits-only first; "
            "once refit_global_norm.py has produced data/interim_normalized, "
            "pass that instead (without --splits-only) to build the real index."
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--splits-only",
        action="store_true",
        help="Assign+write splits.json and stop, without building the window index.",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=None, help="Override configs/preprocessing.yaml."
    )
    parser.add_argument(
        "--train-overlap",
        type=float,
        default=None,
        help="Override configs/preprocessing.yaml's window.train_overlap for "
        "train-purpose windows (pretrain_train, finetune_train). Eval-purpose "
        "windows never overlap regardless of this value.",
    )
    args = parser.parse_args()

    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN
    train_overlap = args.train_overlap if args.train_overlap is not None else TRAIN_OVERLAP
    run(args.interim_root, args.out, DATASETS, args.seed, args.splits_only, window_len, train_overlap)


class WindowDataset:
    """Slices windows out of cached interim segments at access time, per a window index.

    A tiny bounded LRU cache avoids re-reading a segment's parquet file for
    every window drawn from it (consecutive index rows usually share a
    segment), while keeping memory bounded regardless of how many distinct
    segments the dataset spans.
    """

    def __init__(
        self,
        index: pd.DataFrame,
        with_label: bool,
        channels: list[str] = SENSOR_CHANNELS,
        cache_size: int = 32,
    ):
        self.index = index.reset_index(drop=True)
        self.with_label = with_label
        self.channels = channels
        self.cache_size = cache_size
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load_segment(self, path: str) -> pd.DataFrame:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        df = pd.read_parquet(path)
        self._cache[path] = df
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return df

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]
        segment = self._load_segment(row["segment_path"])
        window = segment.iloc[row["start"] : row["end"]]
        x = window[self.channels].to_numpy(dtype=np.float32).T  # (C, window_len)
        if self.with_label:
            y = window["label"].to_numpy(dtype=np.float32)  # (window_len,)
            return x, y
        return x


if __name__ == "__main__":
    main()
