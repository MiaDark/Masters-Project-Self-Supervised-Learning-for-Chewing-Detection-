"""Fit + apply the one global normalization variant, using only
pretrain-eligible participants (decision 003: never let val/test statistics
leak into what the model is normalized with).

Runs after windowing.py has produced data/processed/splits.json (which in
turn needs preprocess.py's raw-scale interim tree to exist, to discover
tags/participants from -- see windowing.py's module docstring for the full
three-step order). Reads directly from preprocess.py's raw-scale segments
(no normalization has been applied to them yet, so no inversion step is
needed here, unlike when a "personal" variant existed), computes population
mean/std per channel from pretrain-eligible segments only (every dataset's
train split -- see windowing.py's uniform 60/20/20 split), and writes every
segment -- train, val, test, meal-test alike -- normalized with those same
fixed stats to data/interim_normalized/<dataset>/<tag>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS, apply_global_stats
from chewing_ssl.common.data.windowing import DATASETS, extract_participant, list_tags, purposes_for


def iter_segments(interim_root: Path, splits: dict, only_pretrain_eligible: bool, datasets: list[str] = DATASETS):
    """Yield (dataset, tag, segment_parquet_path, meta_dict) for every raw-scale segment."""
    for dataset in datasets:
        for tag in list_tags(interim_root, dataset):
            participant = extract_participant(dataset, tag)
            if only_pretrain_eligible and "pretrain" not in purposes_for(dataset, participant, splits):
                continue
            seg_dir = interim_root / dataset / tag
            for meta_path in sorted(seg_dir.glob("segment_*.json")):
                meta = json.loads(meta_path.read_text())
                yield dataset, tag, meta_path.with_suffix(".parquet"), meta


def fit_train_only_global_stats(interim_root: Path, splits: dict, channels: list[str] = SENSOR_CHANNELS) -> dict:
    """Streaming (bounded-memory) accumulation -- never holds more than one segment in memory."""
    acc = {c: {"n": 0, "sum": 0.0, "sumsq": 0.0} for c in channels}
    n_segments = 0
    for _, _, path, _meta in iter_segments(interim_root, splits, only_pretrain_eligible=True):
        df = pd.read_parquet(path, columns=channels)
        for c in channels:
            v = df[c].to_numpy()
            acc[c]["n"] += len(v)
            acc[c]["sum"] += float(v.sum())
            acc[c]["sumsq"] += float(np.square(v).sum())
        n_segments += 1

    stats = {}
    for c, a in acc.items():
        mean = a["sum"] / a["n"]
        variance = max(a["sumsq"] / a["n"] - mean**2, 0.0)
        stats[c] = {"mean": float(mean), "std": float(np.sqrt(variance))}
    print(f"Fit global stats from {n_segments} pretrain-eligible segment(s).")
    return stats


def apply_and_write_normalized(interim_root: Path, out_root: Path, splits: dict, global_stats: dict) -> None:
    n_written = 0
    for dataset, tag, path, meta in iter_segments(interim_root, splits, only_pretrain_eligible=False):
        df = pd.read_parquet(path)
        normalized = apply_global_stats(df, global_stats)
        g_meta = {
            **meta,
            "normalization": "global",
            "channel_stats": global_stats,
            "global_fit_scope": "pretrain_eligible_participants_only",
        }

        out_dir = out_root / dataset / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        normalized.to_parquet(out_dir / path.name, index=False)
        (out_dir / path.with_suffix(".json").name).write_text(json.dumps(g_meta, indent=2, default=str))
        n_written += 1
    print(f"Wrote normalized segment(s) for {n_written} segment(s) to {out_root}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit + apply the global (train-only) normalization variant.")
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"), help="Raw-scale segments.")
    parser.add_argument("--out", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    splits = json.loads((args.processed_root / "splits.json").read_text())

    global_stats = fit_train_only_global_stats(args.interim_root, splits)
    stats_path = args.processed_root / "global_stats.json"
    stats_path.write_text(json.dumps(global_stats, indent=2))
    print(f"Wrote {stats_path}")
    print(json.dumps(global_stats, indent=2))

    apply_and_write_normalized(args.interim_root, args.out, splits, global_stats)


if __name__ == "__main__":
    main()
