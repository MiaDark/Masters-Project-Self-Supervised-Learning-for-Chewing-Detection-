"""Revert accelerometer/navigation scaling applied by emteqai's load_data().

emteqai.scale_data() scales in-place via `df[col] = df[col] / divisor`. This
script undoes that for the Acc/Nav columns by multiplying back by the same
divisor, and writes the result to a parallel output tree (originals are left
untouched).
"""
import argparse
from pathlib import Path

import pandas as pd

# accelerometer/navigation divisors are identical for oco_v1 and oco_v2 in
# metadata_oco.yml, so glasses version doesn't change these two.
ACCEL_DIVISOR = 417.53312946
NAV_DIVISOR = 240

# Mirrors emteqai's SENSOR_COLUMNS_MAP + DEBUG_COLUMNS_MAP for these sensors.
ACCEL_PREFIX = "Acc"
NAV_PREFIX = "Nav"
NAV_DEBUG_KEYWORDS = ["iq", "shutter", "frame avg", "bright", "frameavg"]

DATASETS = ["SIPMEatingStudy", "SussexEatingUK"]


def revert_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    accel_cols = [c for c in df.columns if ACCEL_PREFIX in c]
    nav_cols = [
        c
        for c in df.columns
        if NAV_PREFIX in c
        and not any(kw in c.lower() for kw in NAV_DEBUG_KEYWORDS)
    ]

    if accel_cols:
        df[accel_cols] = df[accel_cols] * ACCEL_DIVISOR
    if nav_cols:
        df[nav_cols] = df[nav_cols] * NAV_DIVISOR

    return df, accel_cols, nav_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("src/chewing_ssl/data/data/raw"),
        help="Directory containing SIPMEatingStudy/ and SussexEatingUK/",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("src/chewing_ssl/data/data/raw_reverted"),
        help="Output directory (mirrors the input tree structure)",
    )
    args = parser.parse_args()

    total_files = 0
    for dataset in DATASETS:
        src_dir = args.raw_root / dataset
        if not src_dir.exists():
            print(f"skip (missing): {src_dir}")
            continue

        csv_paths = sorted(src_dir.rglob("*.csv"))
        print(f"{dataset}: {len(csv_paths)} files")

        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, low_memory=False)
            df, accel_cols, nav_cols = revert_columns(df)

            rel_path = csv_path.relative_to(args.raw_root)
            out_path = args.out_root / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)

            print(
                f"  {rel_path}: reverted {len(accel_cols)} accel col(s), "
                f"{len(nav_cols)} nav col(s) -> {out_path}"
            )
            total_files += 1

    print(f"Done. {total_files} file(s) written under {args.out_root}")


if __name__ == "__main__":
    main()
