"""
Clip accelerometer/navigation columns in the raw sensor_data CSVs.
Values outside the valid range for a sensor type are clamped to the nearest
bound (not dropped), matching chewing_ssl.data.eda.clip_signal_ranges but
applied directly to the on-disk CSVs (in their original, per-dataset column
naming) instead of only the in-memory frame used for plotting.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from chewing_ssl.data.analyze_data import is_nav_debug_column, prefix_of
from chewing_ssl.data.eda import SIGNAL_CLIP_RANGES

ACCEL_RANGE = SIGNAL_CLIP_RANGES["accel_"]
NAV_RANGE = SIGNAL_CLIP_RANGES["nav_"]
DEBUG_KEYWORDS_EMTEQAI = ["iq", "shutter", "frame avg", "bright", "frameavg", "white", "als"]
DEBUG_KEYWORDS_SPACED = ["iq", "shutter", "frame avg"]

DATASET_CLIP_SPECS: dict[str, dict] = {
    "SIPMEatingStudy": {
        "root": Path("src/chewing_ssl/data/data/raw_unscaled"),
        "accel_prefix": "Acc",
        "nav_prefix": "Nav",
        "prox_prefix": "Prox",
        "exclude_keywords": DEBUG_KEYWORDS_EMTEQAI,
    },
    "SussexEatingUK": {
        "root": Path("src/chewing_ssl/data/data/raw_unscaled"),
        "accel_prefix": "Acc",
        "nav_prefix": "Nav",
        "prox_prefix": "Prox",
        "exclude_keywords": DEBUG_KEYWORDS_EMTEQAI,
    },
    "FEIT_eating_S05": {
        "root": Path("src/chewing_ssl/data/data/raw_unscaled"),
        "accel_prefix": "Accelerometer",
        "nav_prefix": "Navigation",
        "prox_prefix": "Proximity",
        "exclude_keywords": DEBUG_KEYWORDS_SPACED,
    },
    "Office_eating_s05": {
        "root": Path("src/chewing_ssl/data/data/raw_unscaled"),
        "accel_prefix": "Accelerometer",
        "nav_prefix": "Navigation",
        "prox_prefix": "Proximity",
        "exclude_keywords": DEBUG_KEYWORDS_SPACED,
    },
    "Spectacle_NonMeal": {
        "root": Path("src/chewing_ssl/data/data/raw"),
        "canonical": True,  # mixed naming per file; match via prefix_of() instead
    },
}


def clip_file(path: Path, spec: dict, dry_run: bool = False) -> tuple[list[str], list[str], list[str]]:
    df = pd.read_csv(path, low_memory=False)

    if spec.get("canonical"):
        accel_cols = [c for c in df.columns if prefix_of(c) == "Accelerometer"]
        nav_cols = [c for c in df.columns if prefix_of(c) == "Navigation" and not is_nav_debug_column(c)]
        prox_cols = [c for c in df.columns if prefix_of(c) == "Proximity"]
    else:
        def matched(prefix: str) -> list[str]:
            return [
                c
                for c in df.columns
                if prefix in c and not any(kw in c.lower() for kw in spec["exclude_keywords"])
            ]

        accel_cols = matched(spec["accel_prefix"])
        nav_cols = matched(spec["nav_prefix"])
        prox_cols = matched(spec["prox_prefix"])

    if not dry_run:
        if accel_cols:
            df[accel_cols] = df[accel_cols].clip(lower=ACCEL_RANGE[0], upper=ACCEL_RANGE[1])
        if nav_cols:
            df[nav_cols] = df[nav_cols].clip(lower=NAV_RANGE[0], upper=NAV_RANGE[1])
        df.to_csv(path, index=False)

    return accel_cols, nav_cols, prox_cols


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CLIP_SPECS),
        help="Only process this dataset; default is all 4.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report matched columns without writing."
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_CLIP_SPECS)

    total = 0
    for dataset in datasets:
        spec = DATASET_CLIP_SPECS[dataset]
        src_dir = spec["root"] / dataset
        if not src_dir.exists():
            print(f"skip (missing): {src_dir}")
            continue

        csv_paths = sorted(p for p in src_dir.rglob("*.csv") if p.name != "manifest.csv")
        print(f"{dataset}: {len(csv_paths)} file(s) under {src_dir}")

        for csv_path in csv_paths:
            accel_cols, nav_cols, prox_cols = clip_file(csv_path, spec, dry_run=args.dry_run)
            rel = csv_path.relative_to(spec["root"])
            print(
                f"  {rel}: accel={len(accel_cols)} nav={len(nav_cols)} prox={len(prox_cols)}"
                + (" (dry-run)" if args.dry_run else "")
            )
            total += 1

    print(f"Done. {total} file(s) {'checked' if args.dry_run else 'clipped in place'}.")


if __name__ == "__main__":
    main()
