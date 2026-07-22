"""
Extract non-meal snippets from Spectacle to balance the eating-only
datasets (SIPM/Sussex/FEIT/Office)

  1. Target duration = total meal recording time across SIPMEatingStudy,
     SussexEatingUK, FEIT_eating_S05, Office_eating_s05.
  2. "Non-meal" = outside every chewing bout (chewing_ssl.common.data.eda.find_bouts
     on the unified `label` column), padded by --pad-minutes on each side to
     also exclude pre/post-meal handling that isn't itself labeled chewing.
  3. Snippets are drawn round-robin across participants (not just days), each
     day capped at --cap-minutes per pass, so the result isn't dominated by
     a handful of people who happen to have long non-meal stretches.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from chewing_ssl.common.data.eda import discover_recordings, find_bouts
from chewing_ssl.common.data.loaders import read_header

MEAL_DATASETS = ["SIPMEatingStudy", "SussexEatingUK", "FEIT_eating_S05", "Office_eating_s05"]
MEAL_ROOT = Path("src/chewing_ssl/data/data/raw_unscaled")
SPECTACLE_ROOT = Path("src/chewing_ssl/data/data/raw")
OUT_ROOT = Path("src/chewing_ssl/data/data/raw/Spectacle_NonMeal")


def meal_target_seconds() -> float:
    total = 0.0
    for ds in MEAL_DATASETS:
        for path, _tag in discover_recordings(ds, MEAL_ROOT):
            ts_col = "Timestamp" if ds in ("SIPMEatingStudy", "SussexEatingUK") else "Software Timestamp"
            ts = pd.read_csv(path, usecols=[ts_col], low_memory=False)[ts_col]
            dur = float(ts.max() - ts.min())
            if ds in ("FEIT_eating_S05", "Office_eating_s05"):
                dur /= 1000.0
            total += dur
    return total


def non_meal_intervals(
    datetime: pd.Series, label: pd.Series, pad_minutes: float
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Gaps between padded, merged chewing bouts, clipped to the recording span."""
    bouts = find_bouts(label, datetime)
    day_start, day_end = datetime.min(), datetime.max()
    if bouts.empty:
        return [(day_start, day_end)]

    pad = pd.Timedelta(minutes=pad_minutes)
    windows = sorted(
        (max(day_start, b["start"] - pad), min(day_end, b["end"] + pad)) for _, b in bouts.iterrows()
    )
    merged = [windows[0]]
    for s, e in windows[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    intervals = []
    cursor = day_start
    for s, e in merged:
        if s > cursor:
            intervals.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < day_end:
        intervals.append((cursor, day_end))
    return intervals


def scan_spectacle(pad_minutes: float) -> dict[str, list[dict]]:
    """participant -> [{path, tag, intervals, available}, ...] for every
    labeled Spectacle day with nonzero non-meal time available."""
    by_participant: dict[str, list[dict]] = defaultdict(list)
    recordings = discover_recordings("Spectacle", SPECTACLE_ROOT)
    print(f"Spectacle: {len(recordings)} day(s) found; scanning for Label column + non-meal time...")
    for path, tag in recordings:
        if "Label" not in read_header(path):
            continue
        raw = pd.read_csv(path, usecols=["Software Timestamp", "Label"], low_memory=False)
        datetime = pd.to_datetime(raw["Software Timestamp"], unit="ms")
        label = (raw["Label"] == 1).astype(int)
        intervals = non_meal_intervals(datetime, label, pad_minutes)
        available = sum((e - s).total_seconds() for s, e in intervals)
        if available <= 0:
            continue
        participant = path.parent.parent.name
        by_participant[participant].append(
            {"path": path, "tag": tag, "intervals": intervals, "available": available}
        )
        print(f"  {tag}: {available / 60:.1f} min non-meal available")
    return by_participant


def select_snippets(
    by_participant: dict[str, list[dict]], target_seconds: float, cap_minutes: float, seed: int
) -> list[dict]:
    """Round-robin across participants (then across their days) until the
    target duration is reached or every participant is exhausted."""
    rng = random.Random(seed)
    participants = list(by_participant)
    rng.shuffle(participants)
    for p in participants:
        rng.shuffle(by_participant[p])

    cap = cap_minutes * 60
    day_idx: dict[str, int] = defaultdict(int)
    chosen: list[dict] = []
    remaining = target_seconds
    progressed = True

    while remaining > 0 and progressed:
        progressed = False
        for p in participants:
            if remaining <= 0:
                break
            days = by_participant[p]
            while day_idx[p] < len(days):
                day = days[day_idx[p]]
                taken = False
                for iv in day["intervals"]:
                    s, e = iv
                    span = (e - s).total_seconds()
                    if span <= 0:
                        continue
                    take = min(span, cap, remaining)
                    if take <= 0:
                        continue
                    snippet_end = s + pd.Timedelta(seconds=take)
                    chosen.append(
                        {
                            "path": day["path"],
                            "tag": day["tag"],
                            "participant": p,
                            "start": s,
                            "end": snippet_end,
                            "duration_s": take,
                        }
                    )
                    remaining -= take
                    progressed = taken = True
                    remainder = (e - snippet_end).total_seconds()
                    day["intervals"] = [x for x in day["intervals"] if x != iv]
                    if remainder > 60:
                        day["intervals"].append((snippet_end, e))
                    break
                if taken:
                    break
                day_idx[p] += 1
            if remaining <= 0:
                break

    return chosen


def extract_snippets(chosen: list[dict], out_root: Path) -> list[dict]:
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, c in enumerate(chosen, 1):
        df = pd.read_csv(c["path"], low_memory=False)
        dt = pd.to_datetime(df["Software Timestamp"], unit="ms")
        snippet = df.loc[(dt >= c["start"]) & (dt < c["end"])]

        out_dir = out_root / c["participant"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{c['tag']}_snippet{i:03d}.csv"
        snippet.to_csv(out_path, index=False)

        manifest.append(
            {
                "participant": c["participant"],
                "source_tag": c["tag"],
                "source_path": str(c["path"]),
                "start": c["start"],
                "end": c["end"],
                "duration_s": c["duration_s"],
                "n_rows": len(snippet),
                "out_path": str(out_path),
            }
        )
        print(
            f"  [{i}/{len(chosen)}] {c['tag']} {c['start']}..{c['end']} "
            f"({c['duration_s'] / 60:.1f} min, {len(snippet)} rows) -> {out_path}"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-minutes", type=float, default=None, help="Override the computed target.")
    parser.add_argument("--pad-minutes", type=float, default=5.0)
    parser.add_argument("--cap-minutes", type=float, default=20.0, help="Max minutes drawn from one day per pass.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Select snippets and report, but don't write files.")
    args = parser.parse_args()

    target_seconds = (
        args.target_minutes * 60 if args.target_minutes is not None else meal_target_seconds()
    )
    print(f"Target non-meal duration: {target_seconds / 60:.1f} min ({target_seconds / 3600:.2f} h)\n")

    by_participant = scan_spectacle(args.pad_minutes)
    chosen = select_snippets(by_participant, target_seconds, args.cap_minutes, args.seed)

    total_chosen = sum(c["duration_s"] for c in chosen)
    n_participants = len({c["participant"] for c in chosen})
    print(
        f"\nSelected {len(chosen)} snippet(s) from {n_participants} participant(s) "
        f"(of {len(by_participant)} eligible), total {total_chosen / 60:.1f} min "
        f"(target {target_seconds / 60:.1f} min)"
    )

    if args.dry_run:
        print("Dry run: not writing snippet files.")
        return

    manifest = extract_snippets(chosen, OUT_ROOT)
    manifest_path = OUT_ROOT / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"\nWrote {len(manifest)} snippet file(s) + manifest to {manifest_path}")


if __name__ == "__main__":
    main()
