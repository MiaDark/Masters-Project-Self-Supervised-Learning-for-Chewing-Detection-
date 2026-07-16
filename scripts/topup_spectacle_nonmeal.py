from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from chewing_ssl.data.analyze_data import is_nav_debug_column, prefix_of
from chewing_ssl.data.eda import discover_recordings
from chewing_ssl.data.loaders import read_header
from scripts.extract_spectacle_nonmeal import OUT_ROOT, SPECTACLE_ROOT, non_meal_intervals, scan_spectacle

ACCEL_DIVISOR = 417.53312946
NAV_DIVISOR = 240
FLAT_STD_THRESHOLD = 0.05  # in already-scaled (pre-revert) units, same convention the bad snippets were caught in
EXCLUDED_LEDGER = OUT_ROOT / "excluded_windows.csv"


def load_consumed_windows(manifest_path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Union of everything currently in the manifest AND everything ever
    deleted (EXCLUDED_LEDGER), so a deleted-then-retried window is never
    re-selected just because its manifest row is gone."""
    consumed: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for path in (manifest_path, EXCLUDED_LEDGER):
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["start", "end"])
        for _, row in df.iterrows():
            consumed[row["source_path"]].append((row["start"], row["end"]))
    return consumed


def next_snippet_index(out_root: Path) -> int:
    """Highest existing snippetNNN index across every participant folder, + 1.
    Guarantees uniqueness across separate top-up runs, unlike restarting at 1."""
    indices = [
        int(m.group(1))
        for p in out_root.glob("p*/*_snippet*.csv")
        if (m := re.search(r"_snippet(\d+)\.csv$", p.name))
    ]
    return max(indices) + 1 if indices else 1


def delete_snippets(manifest_path: Path, out_paths: list[str]) -> None:
    """Delete snippet file(s) + their manifest row(s), logging the exact
    (source_path, start, end) to EXCLUDED_LEDGER first so a future top-up
    run can never re-select the same window."""
    mani = pd.read_csv(manifest_path)
    mask = mani["out_path"].isin(out_paths)
    removed = mani[mask]

    ledger = pd.read_csv(EXCLUDED_LEDGER) if EXCLUDED_LEDGER.exists() else pd.DataFrame(
        columns=["source_path", "start", "end"]
    )
    ledger = pd.concat([ledger, removed[["source_path", "start", "end"]]], ignore_index=True)
    ledger.to_csv(EXCLUDED_LEDGER, index=False)

    for p in removed["out_path"]:
        Path(p).unlink(missing_ok=True)

    mani.loc[~mask].to_csv(manifest_path, index=False)
    print(f"Deleted {mask.sum()} snippet(s), logged to {EXCLUDED_LEDGER}, manifest now has {(~mask).sum()} row(s)")


def subtract_consumed(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    consumed: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not consumed:
        return intervals
    out = list(intervals)
    for cs, ce in consumed:
        next_out = []
        for s, e in out:
            if ce <= s or cs >= e:
                next_out.append((s, e))
                continue
            if cs > s:
                next_out.append((s, cs))
            if ce < e:
                next_out.append((ce, e))
        out = next_out
    return out


def candidate_is_valid(df: pd.DataFrame) -> bool:
    """Same defects flagged manually: any NaN in accel/nav data columns, or
    an accel axis that's essentially flat (disconnected/stationary-off-body).

    Also rejects an empty slice outright: std()/isna() on 0 rows are
    vacuously "fine" (NaN-vs-threshold comparisons are always False), so
    without this check an interval-availability/actual-data mismatch (the
    light scan says a window is free, but the real row slice comes back
    empty) silently produces a corrupted 0-row snippet instead of being
    caught here.
    """
    if len(df) == 0:
        return False
    accel_cols = [c for c in df.columns if prefix_of(c) == "Accelerometer"]
    nav_cols = [c for c in df.columns if prefix_of(c) == "Navigation" and not is_nav_debug_column(c)]
    check_cols = accel_cols + nav_cols
    if not check_cols:
        return False
    if df[check_cols].isna().any().any():
        return False
    if accel_cols and df[accel_cols].std().max() < FLAT_STD_THRESHOLD:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-minutes", type=float, help="Minutes of replacement snippets to select.")
    parser.add_argument(
        "--delete", nargs="+", metavar="OUT_PATH",
        help="Delete these snippet file(s) (by out_path) instead of topping up; logs to the exclusion ledger.",
    )
    parser.add_argument("--pad-minutes", type=float, default=5.0)
    parser.add_argument("--cap-minutes", type=float, default=20.0)
    args = parser.parse_args()

    manifest_path = OUT_ROOT / "manifest.csv"
    if args.delete:
        delete_snippets(manifest_path, args.delete)
        return
    if args.target_minutes is None:
        parser.error("--target-minutes is required unless --delete is given.")

    consumed = load_consumed_windows(manifest_path)
    print(f"{sum(len(v) for v in consumed.values())} already-consumed window(s) across {len(consumed)} file(s)")

    by_participant = scan_spectacle(args.pad_minutes)
    # Subtract already-used time from every day's available intervals.
    for days in by_participant.values():
        for day in days:
            day["intervals"] = subtract_consumed(day["intervals"], consumed.get(str(day["path"]), []))
            day["available"] = sum((e - s).total_seconds() for s, e in day["intervals"])
    for p in list(by_participant):
        by_participant[p] = [d for d in by_participant[p] if d["available"] > 0]
        if not by_participant[p]:
            del by_participant[p]

    target_seconds = args.target_minutes * 60
    cap = args.cap_minutes * 60
    participants = sorted(by_participant)  # deterministic order for a top-up run
    day_idx: dict[str, int] = defaultdict(int)
    remaining = target_seconds
    manifest_rows = []
    existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    next_idx = next_snippet_index(OUT_ROOT)

    progressed = True
    while remaining > 0 and progressed:
        progressed = False
        for p in participants:
            if remaining <= 0:
                break
            days = by_participant[p]
            while day_idx[p] < len(days):
                day = days[day_idx[p]]
                taken_this_day = False
                for iv in list(day["intervals"]):
                    s, e = iv
                    span = (e - s).total_seconds()
                    if span <= 0:
                        continue
                    take = min(span, cap, remaining)
                    snippet_end = s + pd.Timedelta(seconds=take)

                    df = pd.read_csv(day["path"], low_memory=False)
                    dt = pd.to_datetime(df["Software Timestamp"], unit="ms")
                    snippet = df.loc[(dt >= s) & (dt < snippet_end)]

                    if not candidate_is_valid(snippet):
                        print(f"  REJECTED {day['tag']} {s}..{snippet_end} (failed validation)")
                        day["intervals"] = [x for x in day["intervals"] if x != iv]
                        remainder = (e - snippet_end).total_seconds()
                        if remainder > 60:
                            day["intervals"].append((snippet_end, e))
                        continue

                    accel_cols = [c for c in snippet.columns if prefix_of(c) == "Accelerometer"]
                    nav_cols = [c for c in snippet.columns if prefix_of(c) == "Navigation" and not is_nav_debug_column(c)]
                    snippet = snippet.copy()
                    if accel_cols:
                        snippet[accel_cols] = snippet[accel_cols] * ACCEL_DIVISOR
                    if nav_cols:
                        snippet[nav_cols] = snippet[nav_cols] * NAV_DIVISOR

                    out_dir = OUT_ROOT / p
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{day['tag']}_snippet{next_idx:03d}.csv"
                    snippet.to_csv(out_path, index=False)
                    print(f"  [+{next_idx}] {day['tag']} {s}..{snippet_end} ({take/60:.1f} min) -> {out_path}")
                    next_idx += 1

                    manifest_rows.append(
                        {
                            "participant": p, "source_tag": day["tag"], "source_path": str(day["path"]),
                            "start": s, "end": snippet_end, "duration_s": take,
                            "n_rows": len(snippet), "out_path": str(out_path),
                        }
                    )
                    remaining -= take
                    progressed = taken_this_day = True
                    day["intervals"] = [x for x in day["intervals"] if x != iv]
                    remainder = (e - snippet_end).total_seconds()
                    if remainder > 60:
                        day["intervals"].append((snippet_end, e))
                    break
                if taken_this_day:
                    break
                day_idx[p] += 1
            if remaining <= 0:
                break

    added = pd.DataFrame(manifest_rows)
    combined = pd.concat([existing, added], ignore_index=True) if len(existing) else added
    combined.to_csv(manifest_path, index=False)

    total_added = added["duration_s"].sum() if len(added) else 0.0
    print(
        f"\nAdded {len(added)} snippet(s), {total_added/60:.1f} min "
        f"(target {args.target_minutes:.1f} min). Manifest now has {len(combined)} row(s)."
    )


if __name__ == "__main__":
    main()
