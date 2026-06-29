from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chewing_ssl.data.loaders import ACCEL_COLS, NAV_TEMPLE_COLS, PROX_CHEEK_COLS

TARGET_CHANNELS = NAV_TEMPLE_COLS + PROX_CHEEK_COLS + ACCEL_COLS
DATASETS = ["SIPMEatingStudy", "SussexEatingUK", "Spectacle"]


def load_summaries(eda_root: Path, dataset: str) -> list[tuple[str, dict]]:
    """Every (tag, summary dict) pair already written under reports/eda/<dataset>/*/summary.json."""
    return [
        (f.parent.name, json.loads(f.read_text()))
        for f in sorted((eda_root / dataset).glob("*/summary.json"))
    ]


def channel_status(summary: dict, channel: str) -> str:
    """Classify one channel in one recording: 'absent' / 'dead' / 'active' / 'unknown'.

    'unknown' covers the rare case where the column was loaded but had no
    non-NaN values to compute variance from -- treated as not usable, same
    as 'dead', when scoring reliability.
    """
    if channel not in summary.get("columns", []):
        return "absent"
    dead_info = summary.get("dead_channels", {}).get(channel)
    if dead_info is None:
        return "unknown"
    return "dead" if dead_info.get("likely_dead_or_disconnected") else "active"


def build_per_file_table(eda_root: Path) -> pd.DataFrame:
    rows = [
        {"dataset": dataset, "tag": tag, "channel": channel, "status": channel_status(summary, channel)}
        for dataset in DATASETS
        for tag, summary in load_summaries(eda_root, dataset)
        for channel in TARGET_CHANNELS
    ]
    if not rows:
        raise SystemExit(
            f"No summary.json files found under {eda_root}. "
            "Run `python -m chewing_ssl.data.eda --all` first."
        )
    return pd.DataFrame(rows)


def summarize_by_dataset(per_file: pd.DataFrame) -> pd.DataFrame:
    n_files = per_file.groupby("dataset")["tag"].nunique()

    counts = per_file.groupby(["dataset", "channel", "status"]).size().unstack(fill_value=0)
    for status in ("absent", "dead", "active", "unknown"):
        if status not in counts.columns:
            counts[status] = 0
    counts = counts.reset_index()

    counts["n_files"] = counts["dataset"].map(n_files)
    n_present = counts["active"] + counts["dead"] + counts["unknown"]
    counts["n_present"] = n_present
    counts["pct_present"] = n_present / counts["n_files"]
    counts["pct_active_of_present"] = (counts["active"] / n_present.replace(0, pd.NA)).fillna(0.0)
    counts["pct_dead_of_present"] = (counts["dead"] / n_present.replace(0, pd.NA)).fillna(0.0)

    cols = [
        "dataset", "channel", "n_files", "n_present", "pct_present",
        "active", "dead", "unknown", "pct_active_of_present", "pct_dead_of_present",
    ]
    return counts[cols].sort_values(["channel", "dataset"]).reset_index(drop=True)


def reliable_channels(by_dataset: pd.DataFrame) -> list[str]:
    """Channels present in (~)100% of files AND active in (~)100% of present files, in every dataset."""
    reliable = []
    for channel in TARGET_CHANNELS:
        sub = by_dataset[by_dataset["channel"] == channel]
        if len(sub) < len(DATASETS):
            continue
        if (sub["pct_present"] >= 0.999).all() and (sub["pct_active_of_present"] >= 0.999).all():
            reliable.append(channel)
    return reliable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report which Nav Temple / Proximity Cheek / Accelerometer channels are "
        "present and live across every recording in every dataset."
    )
    parser.add_argument("--eda-root", type=Path, default=Path("src/chewing_ssl/data/reports/eda"))
    parser.add_argument("--out", type=Path, default=Path("src/chewing_ssl/data/reports/sensor_activity"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    per_file = build_per_file_table(args.eda_root)
    per_file.to_csv(args.out / "per_file_channel_status.csv", index=False)

    by_dataset = summarize_by_dataset(per_file)
    by_dataset.to_csv(args.out / "channel_summary_by_dataset.csv", index=False)

    pd.set_option("display.width", 140)
    print(by_dataset.to_string(index=False, float_format=lambda v: f"{v:.1%}" if 0 <= v <= 1 else str(v)))

    reliable = reliable_channels(by_dataset)
    print(f"\nChannels present & active in ~100% of files in ALL {len(DATASETS)} datasets:")
    print(f"  ({len(reliable)}/{len(TARGET_CHANNELS)}): {reliable}")
    unreliable = [c for c in TARGET_CHANNELS if c not in reliable]
    print(f"\nChannels NOT reliable everywhere ({len(unreliable)}/{len(TARGET_CHANNELS)}): {unreliable}")

    print(f"\nWrote {args.out / 'per_file_channel_status.csv'} and channel_summary_by_dataset.csv")


if __name__ == "__main__":
    main()
