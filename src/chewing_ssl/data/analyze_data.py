from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Raw sensor streams -> candidate model inputs.
RAW_PREFIXES = {"Accelerometer", "Gyroscope", "Magnetometer", "Euler", "Nav", "Prox", "Pressure"}
# Outputs of Emteq's own on-device pipeline -> exclude from SSL input (leaky / not in all datasets).
DERIVED_PREFIXES = {"Expression", "Activity", "Stats", "Oco"}
# Device housekeeping / metadata -> exclude.
HOUSEKEEPING = {
    "Battery", "RSSI", "DebugBits", "MicBits", "SoftwareTimestamp",
    "LSLinletMarkers", "Annotations",
}


def prefix_of(col: str) -> str:
    """Return the modality key of a column, e.g. 'Nav/Raw.X[LeftTemple]' -> 'Nav'."""
    return col.split("/")[0].split(".")[0].split("[")[0].strip()


def site_of(col: str) -> str | None:
    """Return the bracketed sensor site, e.g. 'Nav/Raw.X[LeftTemple]' -> 'LeftTemple'."""
    m = re.search(r"\[([^\]]+)\]", col)
    return m.group(1) if m else None


def category_of(col: str) -> str:
    """Classify a column into one of: time / raw / derived / housekeeping / label."""
    if col == "Timestamp":
        return "time"
    p = prefix_of(col)
    if p in RAW_PREFIXES:
        return "raw"
    if p in DERIVED_PREFIXES:
        return "derived"
    if p in HOUSEKEEPING:
        return "housekeeping"
    return "label"


def read_header(path: Path) -> list[str]:
    """Read just the first line of a CSV and return cleaned column names."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        row = next(csv.reader(f))
    return [c.strip() for c in row]


def estimate_rate(path: Path, n_rows: int = 500) -> float | None:
    """Estimate sampling rate (Hz) from the median timestamp delta of the first rows."""
    import numpy as np
    import pandas as pd

    try:
        ts = pd.read_csv(path, usecols=["Timestamp"], nrows=n_rows)["Timestamp"].to_numpy()
        dt = np.diff(ts)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else None
    except Exception:
        return None


def collect(root: Path) -> dict[str, list[Path]]:
    """Map each top-level dataset folder to its list of CSV files."""
    datasets: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            datasets[child.name] = sorted(child.rglob("*.csv"))
    return datasets


def analyse(root: Path, do_rate: bool, rate_sample: int) -> dict:
    datasets = collect(root)
    if not datasets:
        raise SystemExit(f"No dataset folders found under {root}. Did you download the data?")

    report: dict = {"datasets": {}, "matrix": {}}
    col_presence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for ds, files in datasets.items():
        if not files:
            report["datasets"][ds] = {"n_files": 0}
            continue

        schemas: Counter = Counter()       # distinct header tuples -> file count
        col_counts: Counter = Counter()    # column -> n files containing it
        nav_sites: set[str] = set()
        prox_sites: set[str] = set()

        for fp in files:
            header = read_header(fp)
            schemas[tuple(header)] += 1
            for c in header:
                col_counts[c] += 1
                col_presence[c][ds] += 1
                if prefix_of(c) == "Nav" and site_of(c):
                    nav_sites.add(site_of(c))
                if prefix_of(c) == "Prox" and site_of(c):
                    prox_sites.add(site_of(c))

        by_cat = defaultdict(list)
        for c in col_counts:
            by_cat[category_of(c)].append(c)

        rates = []
        if do_rate:
            for fp in files[:rate_sample]:
                r = estimate_rate(fp)
                if r:
                    rates.append(r)

        report["datasets"][ds] = {
            "n_files": len(files),
            "n_distinct_schemas": len(schemas),
            "schema_file_counts": sorted(schemas.values(), reverse=True),
            "n_columns_union": len(col_counts),
            "columns_by_category": {k: sorted(v) for k, v in by_cat.items()},
            "nav_sites": sorted(nav_sites),
            "prox_sites": sorted(prox_sites),
            "sampling_rate_hz": (
                f"{min(rates):.1f}-{max(rates):.1f}" if rates else None
            ),
        }

    report["matrix"] = {c: dict(d) for c, d in col_presence.items()}
    report["dataset_names"] = list(datasets.keys())
    return report


def write_outputs(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump of the full report.
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    # column x dataset presence matrix (file counts per dataset).
    names = report["dataset_names"]
    matrix_path = out_dir / "column_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "category", "modality", *names])
        for col in sorted(report["matrix"]):
            counts = report["matrix"][col]
            w.writerow(
                [col, category_of(col), prefix_of(col), *[counts.get(n, 0) for n in names]]
            )
    print(f"Wrote {matrix_path} and {out_dir/'summary.json'}")


def print_summary(report: dict) -> None:
    names = report["dataset_names"]
    print("\n================ SCHEMA SUMMARY ================")
    for ds, info in report["datasets"].items():
        if not info.get("n_files"):
            print(f"\n[{ds}] no CSV files found")
            continue
        print(f"\n[{ds}]")
        print(f"  files: {info['n_files']} | distinct header layouts: {info['n_distinct_schemas']}")
        print(f"  columns (union): {info['n_columns_union']} | sampling rate: {info['sampling_rate_hz']}")
        for cat in ("raw", "derived", "housekeeping", "label"):
            cols = info["columns_by_category"].get(cat, [])
            print(f"    {cat:13s} ({len(cols):2d}): {', '.join(cols) if len(cols) <= 12 else ', '.join(cols[:12]) + ' ...'}")
        print(f"    Nav (OMG) sites : {info['nav_sites']}")
        print(f"    Prox sites      : {info['prox_sites']}")

    # Raw-sensor columns present in every dataset = the cross-dataset SSL input.
    present_in_all = [
        c for c, counts in report["matrix"].items()
        if category_of(c) == "raw" and all(n in counts for n in names)
    ]
    print("\n---- RAW SENSOR COLUMNS COMMON TO ALL DATASETS (SSL input candidates) ----")
    print(f"  ({len(present_in_all)}): {', '.join(sorted(present_in_all))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CSV schemas across datasets.")
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--out", default="reports/schema")
    parser.add_argument("--rate", action="store_true", help="Estimate sampling rate (slower).")
    parser.add_argument("--rate-sample", type=int, default=3, help="Files per dataset to sample for rate.")
    args = parser.parse_args()

    report = analyse(Path(args.root), do_rate=args.rate, rate_sample=args.rate_sample)
    print_summary(report)
    write_outputs(report, Path(args.out))


if __name__ == "__main__":
    main()