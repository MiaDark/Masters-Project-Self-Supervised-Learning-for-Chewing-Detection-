from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from chewing_ssl.common.data.eda import discover_recordings, find_bouts
from chewing_ssl.common.data.loaders import load_recording
from chewing_ssl.common.data.preprocess import (
    BUTTER_CUTOFF_HZ,
    BUTTER_ORDER,
    DEFAULT_TARGET_HZ,
    RAW_CHANNELS,
    apply_lowpass_filter,
    resample_segment,
    segment_at_gaps,
    select_columns,
)

NAV_X, NAV_Y = "nav_x_left_temple", "nav_y_left_temple"
PALETTE = {"nav_x": "#1565C0", "nav_y": "#6A1B9A", "bout": "#2E7D32", "baseline": "#757575"}


def find_bite_window(
    bouts: pd.DataFrame, n_bites: int, max_span_s: float, pre_pad_s: float, post_pad_s: float
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]:
    """Earliest run of `n_bites` consecutive bouts spanning <= max_span_s
    (a visually tight, legible cluster) -- falls back to just the first
    `n_bites` bouts, however spread out, if no tight run exists."""
    if len(bouts) < n_bites:
        raise SystemExit(
            f"Only {len(bouts)} ground-truth bite(s) in this recording, need {n_bites}."
        )

    for i in range(len(bouts) - n_bites + 1):
        window = bouts.iloc[i : i + n_bites]
        span_s = (window["end"].iloc[-1] - window["start"].iloc[0]).total_seconds()
        if span_s <= max_span_s:
            break
    else:
        window = bouts.iloc[:n_bites]  # fallback: first n_bites, whatever the spacing

    win_start = window["start"].iloc[0] - pd.Timedelta(seconds=pre_pad_s)
    win_end = window["end"].iloc[-1] + pd.Timedelta(seconds=post_pad_s)
    return win_start, win_end, window


def meal_groups(bouts: pd.DataFrame, meal_gap_s: float) -> list[pd.DataFrame]:
    """Split bites into meal groups wherever the gap since the previous bite
    exceeds meal_gap_s -- a coarser split than the inter-bite pause within a
    single meal, for recordings that contain more than one eating episode."""
    if bouts.empty:
        return []
    gap_before = (bouts["start"] - bouts["end"].shift(1)).dt.total_seconds()
    new_meal = (gap_before > meal_gap_s).fillna(False)
    meal_id = new_meal.cumsum()
    return [g.reset_index(drop=True) for _, g in bouts.groupby(meal_id)]


def bouts_in_window(
    bouts: pd.DataFrame, win_start: pd.Timestamp, win_end: pd.Timestamp
) -> pd.DataFrame:
    """Ground-truth bites overlapping [win_start, win_end] at all (even
    partially -- a manually-chosen row range won't necessarily land on bite
    boundaries), in time order."""
    mask = (bouts["end"] >= win_start) & (bouts["start"] <= win_end)
    return bouts.loc[mask].reset_index(drop=True)


def plot_one(
    dataset: str,
    tag: str,
    path: Path,
    n_bites: int,
    max_span_s: float,
    pre_pad_s: float,
    post_pad_s: float,
    out_path: Path,
    start_row: int | None = None,
    end_row: int | None = None,
    meal: int | None = None,
    meal_gap_s: float = 60.0,
) -> None:
    df = select_columns(load_recording(dataset, path))
    raw = max(segment_at_gaps(df), key=len)
    resampled = resample_segment(raw, DEFAULT_TARGET_HZ)
    filtered = apply_lowpass_filter(
        resampled, RAW_CHANNELS, BUTTER_CUTOFF_HZ, BUTTER_ORDER, DEFAULT_TARGET_HZ
    )

    bouts = find_bouts(filtered["label"], filtered["datetime"])

    if start_row is not None or end_row is not None:
        # Manual override: zoom to an exact row range instead of
        # auto-picking a cluster of bites. Rows index into `filtered` --
        # the resampled-to-50Hz, filtered signal (i.e. row i is i/50 s into
        # the longest gap-free segment of this recording).
        s = start_row if start_row is not None else 0
        e = min(end_row, len(filtered) - 1) if end_row is not None else len(filtered) - 1
        if s < 0 or s >= len(filtered):
            raise SystemExit(f"--start-row {s} out of range (recording has {len(filtered)} rows).")
        if s >= e:
            raise SystemExit(f"--start-row ({s}) must be less than --end-row ({e}).")
        win_start = filtered["datetime"].iloc[s]
        win_end = filtered["datetime"].iloc[e]
        shown_bites = bouts_in_window(bouts, win_start, win_end)
        if shown_bites.empty:
            print("Warning: no ground-truth bites overlap this row range -- plotting signal only.")
    elif meal is not None:
        # Zoom to the first --n-bites bites of the --meal'th eating episode
        # (1-indexed), where episodes are split by gaps > --meal-gap-s.
        groups = meal_groups(bouts, meal_gap_s)
        if meal < 1 or meal > len(groups):
            raise SystemExit(
                f"--meal {meal} out of range: found {len(groups)} meal(s) in this recording "
                f"(gaps > {meal_gap_s}s split meals; use --meal-gap-s to change the threshold)."
            )
        group = groups[meal - 1]
        shown_bites = group.iloc[:n_bites].reset_index(drop=True)
        if len(shown_bites) < n_bites:
            print(
                f"Warning: meal {meal} only has {len(shown_bites)} bite(s), "
                f"fewer than --n-bites {n_bites}."
            )
        win_start = shown_bites["start"].iloc[0] - pd.Timedelta(seconds=pre_pad_s)
        win_end = shown_bites["end"].iloc[-1] + pd.Timedelta(seconds=post_pad_s)
    else:
        win_start, win_end, shown_bites = find_bite_window(
            bouts, n_bites, max_span_s, pre_pad_s, post_pad_s
        )

    view = filtered[(filtered["datetime"] >= win_start) & (filtered["datetime"] <= win_end)].copy()
    t0 = view["datetime"].iloc[0]
    view["t_s"] = (view["datetime"] - t0).dt.total_seconds()
    shown_bites = shown_bites.copy()
    shown_bites["start_s"] = (shown_bites["start"] - t0).dt.total_seconds()
    shown_bites["end_s"] = (shown_bites["end"] - t0).dt.total_seconds()

    # "At rest" baseline = mean signal outside any shown bite, in this view.
    at_rest = pd.Series(True, index=view.index)
    for _, b in shown_bites.iterrows():
        at_rest &= ~((view["t_s"] >= b["start_s"]) & (view["t_s"] <= b["end_s"]))
    baseline_x = view.loc[at_rest, NAV_X].mean()
    baseline_y = view.loc[at_rest, NAV_Y].mean()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        view["t_s"], view[NAV_X], color=PALETTE["nav_x"], linewidth=0.9, label="nav_x_left_temple"
    )
    ax.plot(
        view["t_s"], view[NAV_Y], color=PALETTE["nav_y"], linewidth=0.9, label="nav_y_left_temple"
    )
    ax.axhline(baseline_x, color=PALETTE["nav_x"], linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(baseline_y, color=PALETTE["nav_y"], linestyle="--", linewidth=0.7, alpha=0.5)

    ymin, ymax = ax.get_ylim()
    view_end_s = view["t_s"].iloc[-1]
    label_y = ymax - 0.08 * (ymax - ymin)
    for i, (_, b) in enumerate(shown_bites.iterrows(), start=1):
        ax.axvspan(b["start_s"], b["end_s"], color=PALETTE["bout"], alpha=0.25, linewidth=0)
        # Clip to the visible window for label placement only (a bite at the
        # edge of a manually-chosen row range may extend past it).
        label_x = (max(b["start_s"], 0) + min(b["end_s"], view_end_s)) / 2
        ax.text(
            label_x,
            label_y,
            f"Bite {i}",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=PALETTE["bout"],
        )

    # Label one inter-bite pause explicitly, between the first two shown bites.
    if len(shown_bites) >= 2:
        # Annotate the widest inter-bite gap among those shown, not just the
        # first -- a narrow gap leaves no room for the label without
        # overlapping the bite shading either side of it.
        gaps = [
            (shown_bites["end_s"].iloc[i], shown_bites["start_s"].iloc[i + 1])
            for i in range(len(shown_bites) - 1)
        ]
        gap_start, gap_end = max(gaps, key=lambda g: g[1] - g[0])
        if gap_end > gap_start:
            arrow_y = ymin + 0.08 * (ymax - ymin)
            ax.annotate(
                "",
                xy=(gap_start, arrow_y),
                xytext=(gap_end, arrow_y),
                arrowprops={"arrowstyle": "<->", "color": PALETTE["baseline"], "linewidth": 1.1},
            )
            ax.text(
                (gap_start + gap_end) / 2,
                arrow_y + 0.03 * (ymax - ymin),
                "inter-bite pause\n(return to baseline)",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["baseline"],
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temple navigation signal (filtered)")
    ax.set_title("Left-temple navigation signal during chewing")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="SIPMEatingStudy")
    parser.add_argument("--root", type=Path, default=Path("data/raw_unscaled/raw_unscaled"))
    parser.add_argument("--tag", default=None, help="Recording tag; default: first found.")
    parser.add_argument(
        "--n-bites",
        type=int,
        default=3,
        help="How many consecutive bites to auto-select. Ignored if --start-row/--end-row given.",
    )
    parser.add_argument(
        "--max-span-s",
        type=float,
        default=60.0,
        help="Prefer a run of --n-bites bites spanning no more than this (a legible, tight cluster). "
        "Ignored if --start-row/--end-row given.",
    )
    parser.add_argument(
        "--pre-pad-s",
        type=float,
        default=3.0,
        help="Baseline shown before the first bite. Ignored if --start-row/--end-row given.",
    )
    parser.add_argument(
        "--post-pad-s",
        type=float,
        default=3.0,
        help="Baseline shown after the last bite. Ignored if --start-row/--end-row given.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=None,
        help="Zoom to this exact row range instead of auto-selecting bites. Row i is i/50 s into "
        "the recording's longest gap-free segment (post-resample-to-50Hz). Overrides --n-bites etc.",
    )
    parser.add_argument(
        "--end-row", type=int, default=None, help="See --start-row. Default: last row."
    )
    parser.add_argument(
        "--meal",
        type=int,
        default=None,
        help="Zoom to the first --n-bites bites of the Nth eating episode in this recording "
        "(1-indexed; episodes are split by gaps > --meal-gap-s between bites). "
        "Overrides --max-span-s; ignored if --start-row/--end-row given.",
    )
    parser.add_argument(
        "--meal-gap-s",
        type=float,
        default=60.0,
        help="Gap between bites (seconds) large enough to count as a new meal/eating episode. --meal only.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/chewing_ssl/data/reports/temple_chewing_signature.pdf"),
    )
    args = parser.parse_args()

    recordings = discover_recordings(args.dataset, args.root)
    if not recordings:
        raise SystemExit(f"No recordings found for {args.dataset} under {args.root}")
    if args.tag:
        path = next(p for p, t in recordings if t == args.tag)
    else:
        path, args.tag = recordings[0]
    print(f"Using {args.dataset}/{args.tag}: {path}")

    plot_one(
        args.dataset,
        args.tag,
        path,
        args.n_bites,
        args.max_span_s,
        args.pre_pad_s,
        args.post_pad_s,
        args.out,
        start_row=args.start_row,
        end_row=args.end_row,
        meal=args.meal,
        meal_gap_s=args.meal_gap_s,
    )
    print(f"Wrote {args.out} (and matching .png)")


if __name__ == "__main__":
    main()
