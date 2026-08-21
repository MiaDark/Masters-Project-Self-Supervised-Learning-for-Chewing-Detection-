from __future__ import annotations

import statistics
from pathlib import Path

from chewing_ssl.common.data.eda import discover_recordings
from chewing_ssl.common.data.loaders import load_recording
from chewing_ssl.common.data.preprocess import segment_at_gaps, select_columns

RAW_ROOT = Path("data/raw_unscaled/raw_unscaled")
S11_ROOT = Path("data/raw/s11_feit_eating")
S11_USED_DIR = Path("checkpoints/predict_new/s11_feit_eating_masked_ssl")


def longest_segment_duration_s(dataset: str, path: Path) -> float | None:
    df = select_columns(load_recording(dataset, path))
    segments = segment_at_gaps(df)
    if not segments:
        return None
    longest = max(segments, key=len)
    return (longest["datetime"].iloc[-1] - longest["datetime"].iloc[0]).total_seconds()


def report(label: str, durations_s: list[float]) -> None:
    mins = [d / 60 for d in durations_s]
    total_h = sum(mins) / 60
    median_min = statistics.median(mins)
    print(
        f"{label:4s}  n={len(mins):3d}  median={median_min:6.2f} min  "
        f"total={total_h:6.2f} h  (min={min(mins):.2f}, max={max(mins):.2f})"
    )


def main() -> None:
    for label, dataset in [
        ("M1", "SIPMEatingStudy"),
        ("M2", "SussexEatingUK"),
        ("M3", "FEIT_eating_S05"),
        ("M4", "Office_eating_s05"),
    ]:
        recordings = discover_recordings(dataset, RAW_ROOT)
        durations = []
        for path, tag in recordings:
            d = longest_segment_duration_s(dataset, path)
            if d is not None:
                durations.append(d)
            else:
                print(f"  WARNING: no usable segment for {dataset}/{tag}")
        report(label, durations)

    # NB: per-snippet-file, not per-participant.
    recordings = discover_recordings("Spectacle_NonMeal", RAW_ROOT)
    durations = []
    for path, tag in recordings:
        d = longest_segment_duration_s("Spectacle_NonMeal", path)
        if d is not None:
            durations.append(d)
    report("NB", durations)

    # S11: only the 39 participants actually used (per the existing output dir),
    # loaded via the project's already-defined right-temple-substitute dataset key.
    used_participants = {p.name.removesuffix("_eating") for p in S11_USED_DIR.iterdir()}
    durations = []
    for participant_dir in sorted(S11_ROOT.iterdir()):
        if participant_dir.name not in used_participants:
            continue
        csv = participant_dir / f"{participant_dir.name}_eating.csv"
        if not csv.exists():
            print(f"  WARNING: expected {csv} for a participant in the used set, not found")
            continue
        d = longest_segment_duration_s("FEIT_eating_S11_RightTempleSubstitute_UNVALIDATED", csv)
        if d is not None:
            durations.append(d)
    report("S11", durations)


if __name__ == "__main__":
    main()
