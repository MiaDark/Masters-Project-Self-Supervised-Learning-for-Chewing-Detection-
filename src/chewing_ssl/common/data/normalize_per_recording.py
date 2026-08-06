from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS, apply_minmax_stats, fit_minmax_stats
from chewing_ssl.common.data.windowing import DATASETS, list_tags


def normalize_recording(
    interim_root: Path,
    out_root: Path,
    dataset: str,
    tag: str,
    channels: list[str] = SENSOR_CHANNELS,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict | None:
    """Fit + apply per-recording min-max bounds for one (dataset, tag),
    pooled across all of its gap-bounded segments, written back segment by
    segment. Returns the fitted stats (for reporting), or None if this
    recording has no segments (shouldn't happen against a well-formed
    interim tree, but guarded rather than assumed)."""
    seg_dir = interim_root / dataset / tag
    seg_paths = sorted(seg_dir.glob("segment_*.parquet"))
    if not seg_paths:
        return None

    segments = [pd.read_parquet(p) for p in seg_paths]
    stats = fit_minmax_stats(segments, channels, lower_pct, upper_pct)

    out_dir = out_root / dataset / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for seg_path, seg_df in zip(seg_paths, segments):
        normalized = apply_minmax_stats(seg_df, stats)
        normalized.to_parquet(out_dir / seg_path.name, index=False)

        meta_path = seg_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        g_meta = {
            **meta,
            "normalization": "minmax_per_recording",
            "channel_stats": stats,
            "fit_scope": f"this_recording_only ({len(seg_paths)} segment(s) pooled)",
        }
        (out_dir / meta_path.name).write_text(json.dumps(g_meta, indent=2, default=str))

    return stats


def run_batch(
    interim_root: Path,
    out_root: Path,
    datasets: list[str] = DATASETS,
    channels: list[str] = SENSOR_CHANNELS,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> None:
    n_recordings, n_segments = 0, 0
    for dataset in datasets:
        tags = list_tags(interim_root, dataset)
        print(f"\n=== {dataset}: {len(tags)} recording(s) ===")
        for i, tag in enumerate(tags, 1):
            stats = normalize_recording(interim_root, out_root, dataset, tag, channels, lower_pct, upper_pct)
            if stats is None:
                print(f"  [{i}/{len(tags)}] {tag}: no segments -- skipped")
                continue
            n_segs_this = len(list((interim_root / dataset / tag).glob("segment_*.parquet")))
            n_recordings += 1
            n_segments += n_segs_this
            print(f"  [{i}/{len(tags)}] {tag}: {n_segs_this} segment(s) normalized")
    print(f"\nDone. {n_recordings} recording(s), {n_segments} segment(s) written to {out_root}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--out", type=Path, default=Path("data/interim_normalized_per_file"))
    parser.add_argument("--dataset", choices=DATASETS, default=None, help="Default: all 5.")
    parser.add_argument("--lower-pct", type=float, default=1.0)
    parser.add_argument("--upper-pct", type=float, default=99.0)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else DATASETS
    run_batch(args.interim_root, args.out, datasets, lower_pct=args.lower_pct, upper_pct=args.upper_pct)


if __name__ == "__main__":
    main()
