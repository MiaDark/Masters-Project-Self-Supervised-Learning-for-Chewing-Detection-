from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.data.loaders import DATASET_SPECS

MODALITY_ALIASES: dict[str, list[str]] = {
    "Accelerometer": ["Accelerometer"],
    "Gyroscope": ["Gyroscope"],
    "Magnetometer": ["Magnetometer"],
    "Euler": ["Euler"],
    "Rotation": ["Rotation Vector", "Rotation"],
    "Navigation": ["Navigation", "Nav"],
    "Proximity": ["Proximity", "Prox"],
    "Pressure": ["Pressure"],
}
_ALIAS_LOOKUP: list[tuple[str, str]] = sorted(
    ((token, canonical) for canonical, tokens in MODALITY_ALIASES.items() for token in tokens),
    key=lambda kv: -len(kv[0]),
)

RAW_PREFIXES = set(MODALITY_ALIASES)
DERIVED_PREFIXES = {"Expression", "Activity", "Stats", "Oco"}
HOUSEKEEPING = {
    "Battery", "RSSI", "DebugBits", "MicBits", "SoftwareTimestamp",
    "LSLinletMarkers", "Annotations",
}
TIMESTAMP_COLUMNS = {"Timestamp", "Software Timestamp", "Hardware Timestamp"}
NAV_DEBUG_KEYWORDS = ("iq", "shutter", "frame avg", "bright", "frameavg")
DEAD_STD_THRESHOLD = 1e-6


def _matches_token(col: str, token: str) -> bool:
    """True if `col` starts with `token` at a real field boundary (not mid-word)."""
    if col == token:
        return True
    if not col.startswith(token):
        return False
    return col[len(token)] in " /.["


def prefix_of(col: str) -> str:
    """Return the canonical modality of a column. Handles both naming
    conventions: 'Nav/Raw.X[LeftTemple]' and 'Navigation Left Temple X'
    both -> 'Navigation'.
    """
    for token, canonical in _ALIAS_LOOKUP:
        if _matches_token(col, token):
            return canonical
    return col.split("/")[0].split(".")[0].split("[")[0].strip()


def is_nav_debug_column(col: str) -> bool:
    return any(kw in col.lower() for kw in NAV_DEBUG_KEYWORDS)


def site_of(col: str) -> str | None:
    """Return the sensor site. 'Nav/Raw.X[LeftTemple]' -> 'LeftTemple',
    'Navigation Left Temple X' -> 'LeftTemple', 'Proximity Left Temple' ->
    'LeftTemple'.
    """
    m = re.search(r"\[([^\]]+)\]", col)
    if m:
        return m.group(1)

    modality = prefix_of(col)
    if modality not in ("Navigation", "Proximity"):
        return None

    rest = None
    for token in MODALITY_ALIASES[modality]:
        if _matches_token(col, token) and col != token:
            rest = col[len(token):].strip()
            break
    if rest is None:
        return None

    for suffix in sorted(("IQ", "Shutter", "Frame Avg", "X", "Y", "Z"), key=len, reverse=True):
        if rest == suffix or rest.endswith(" " + suffix):
            rest = rest[: len(rest) - len(suffix)].strip()
            break

    return rest.replace(" ", "") or None


def category_of(col: str) -> str:
    """Classify a column into one of: time / raw / derived / housekeeping / label."""
    if col in TIMESTAMP_COLUMNS:
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


def _timestamp_spec(dataset: str) -> tuple[str, str]:
    """(raw timestamp column, unit) for a dataset, from loaders.DATASET_SPECS
    (the same mapping chewing_ssl.data.loaders uses), falling back to a
    literal 'Timestamp' column in seconds for unknown datasets.
    """
    spec = DATASET_SPECS.get(dataset)
    if not spec:
        return "Timestamp", "s"
    ts_col = next(
        (raw for raw, canon in spec["columns"].items() if canon == "timestamp_raw"),
        "Timestamp",
    )
    return ts_col, spec.get("timestamp_unit", "s")


def estimate_rate(path: Path, dataset: str, n_rows: int = 500) -> float | None:
    """Estimate sampling rate (Hz) from the median timestamp delta of the first rows."""
    ts_col, unit = _timestamp_spec(dataset)
    try:
        ts = pd.read_csv(path, usecols=[ts_col], nrows=n_rows)[ts_col].to_numpy(dtype=float)
        if unit == "ms":
            ts = ts / 1000.0
        dt = np.diff(ts)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else None
    except Exception:
        return None


NON_DATA_CSV_NAMES = {"manifest.csv", "excluded_windows.csv"}


def collect(root: Path) -> dict[str, list[Path]]:
    """Map each top-level dataset folder to its list of CSV files.

    Excludes the Spectacle_NonMeal extraction metadata files (manifest.csv,
    excluded_windows.csv, see scripts/extract_spectacle_nonmeal.py /
    topup_spectacle_nonmeal.py) -- these sit alongside the real per-participant
    CSVs but aren't sensor data, and would otherwise corrupt schema/rate stats.
    """
    datasets: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir():
            datasets[child.name] = sorted(
                p for p in child.rglob("*.csv") if p.name not in NON_DATA_CSV_NAMES
            )
    return datasets


def _nav_side(site: str) -> str:
    if site.startswith("Left"):
        return "Left"
    if site.startswith("Right"):
        return "Right"
    return "Centre"


def nav_activity(files: list[Path], columns: set[str]) -> dict[str, dict]:
    """For every non-debug Navigation site, check the *data* (not just the
    header) across every file to see whether that site is genuinely live
    (non-constant) in every single recording, vs. present-but-dead in some.

    Returns {site: {side, columns, n_files, n_files_dead, always_active}}.
    """
    nav_data_cols = sorted(
        c
        for c in columns
        if prefix_of(c) == "Navigation" and not is_nav_debug_column(c) and "Temple" in (site_of(c) or "")
    )
    if not nav_data_cols:
        return {}

    site_cols: dict[str, list[str]] = defaultdict(list)
    for c in nav_data_cols:
        site = site_of(c)
        if site:
            site_cols[site].append(c)

    counts = {site: {"n_files": 0, "n_files_dead": 0} for site in site_cols}
    for fp in files:
        try:
            df = pd.read_csv(fp, usecols=nav_data_cols, low_memory=False)
        except (ValueError, FileNotFoundError):
            # ValueError: usecols columns missing from this file's header
            # (a schema variant) -> can't assess this file's liveness.
            continue
        for site, cols in site_cols.items():
            present = [c for c in cols if c in df.columns]
            if not present:
                continue
            stds = [float(df[c].std()) for c in present]
            is_dead = not any(s > DEAD_STD_THRESHOLD for s in stds)
            counts[site]["n_files"] += 1
            if is_dead:
                counts[site]["n_files_dead"] += 1

    return {
        site: {
            "side": _nav_side(site),
            "columns": cols,
            "n_files": counts[site]["n_files"],
            "n_files_dead": counts[site]["n_files_dead"],
            "always_active": counts[site]["n_files"] > 0 and counts[site]["n_files_dead"] == 0,
        }
        for site, cols in site_cols.items()
    }


def summarize_nav_sides(datasets_report: dict) -> dict:
    """Per dataset, roll individual nav sites up into Left/Right/Centre:
    which sites exist, and whether *every* site on that side is live in
    *every* recording (not just present in the header).
    """
    out = {}
    for ds, info in datasets_report.items():
        nav = info.get("nav_activity", {})
        by_side: dict[str, list[str]] = defaultdict(list)
        for site, s in nav.items():
            by_side[s["side"]].append(site)

        side_summary = {}
        for side in ("Left", "Right", "Centre"):
            sites = sorted(by_side.get(side, []))
            side_summary[side] = {
                "sites": sites,
                "present": bool(sites),
                "always_active": bool(sites) and all(nav[s]["always_active"] for s in sites),
            }
        out[ds] = side_summary
    return out


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
                if prefix_of(c) == "Navigation" and site_of(c):
                    nav_sites.add(site_of(c))
                if prefix_of(c) == "Proximity" and site_of(c):
                    prox_sites.add(site_of(c))

        by_cat = defaultdict(list)
        for c in col_counts:
            by_cat[category_of(c)].append(c)

        rates = []
        if do_rate:
            for fp in files[:rate_sample]:
                r = estimate_rate(fp, ds)
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
            "nav_activity": nav_activity(files, set(col_counts)),
            "sampling_rate_hz": (
                f"{min(rates):.1f}-{max(rates):.1f}" if rates else None
            ),
        }

    report["matrix"] = {c: dict(d) for c, d in col_presence.items()}
    report["dataset_names"] = list(datasets.keys())
    report["nav_left_right"] = summarize_nav_sides(report["datasets"])
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

    # Per-site nav liveness (X/Y position channels only, debug cols excluded).
    nav_path = out_dir / "nav_activity.csv"
    with open(nav_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "site", "side", "n_files", "n_files_dead", "always_active"])
        for ds, info in report["datasets"].items():
            for site, s in sorted(info.get("nav_activity", {}).items()):
                w.writerow([ds, site, s["side"], s["n_files"], s["n_files_dead"], s["always_active"]])

    print(f"Wrote {matrix_path}, {nav_path} and {out_dir/'summary.json'}")


def print_summary(report: dict) -> None:
    names = report["dataset_names"]
    print("\n SCHEMA SUMMARY: ")
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

    present_in_all = [
        c for c, counts in report["matrix"].items()
        if category_of(c) == "raw" and all(n in counts for n in names)
    ]
    print("\n---- RAW SENSOR COLUMNS COMMON TO ALL DATASETS (literal name match; SSL input candidates) ----")
    print(f"  ({len(present_in_all)}): {', '.join(sorted(present_in_all))}")

    print_nav_side_comparison(report)


def print_nav_side_comparison(report: dict) -> None:
    print("\nNAVIGATION: LEFT vs RIGHT (active = never a dead/constant channel, in every recording)")
    col_w = (20, 28, 14, 28, 14)
    header = (
        f"{'dataset':<{col_w[0]}} {'Left sites':<{col_w[1]}} {'Left active?':<{col_w[2]}} "
        f"{'Right sites':<{col_w[3]}} {'Right active?':<{col_w[4]}}"
    )
    print(header)
    print("-" * len(header))
    for ds, sides in report.get("nav_left_right", {}).items():
        left, right = sides.get("Left", {}), sides.get("Right", {})

        def status(side: dict) -> str:
            if not side.get("present"):
                return "absent"
            return "yes" if side.get("always_active") else "present, dead"

        print(
            f"{ds:<{col_w[0]}} {','.join(left.get('sites', [])) or '-':<{col_w[1]}} "
            f"{status(left):<{col_w[2]}} {','.join(right.get('sites', [])) or '-':<{col_w[3]}} "
            f"{status(right):<{col_w[4]}}"
        )


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
