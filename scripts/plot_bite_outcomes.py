"""Plot every MISSED bite (a real bite with no overlapping prediction) and
every FALSE ALARM bite (a predicted bout with no overlapping real bite) from
an already-computed bout_ledger.csv, one zoomed PNG per bite.

Needs no inference/postprocessing of its own: a bite ledger already records
every GT bite's and every unmatched predicted bout's span
(chewing_ssl.common.evaluation.pipeline.ledger_rows_for_recording), so this
just reads the raw nav signal for context and reconstructs each segment's
full GT/predicted-bout set from the ledger rows themselves (matched
predicted bouts aren't stored as their own ledger row, but their span is on
the matching "detected" GT row's matched_start_s/matched_end_s).

Companion to scripts/plot_taxonomy_examples.py (same shading convention),
but ledger-driven and exhaustive rather than six hand-picked examples.

Run from the project root:
  python -m scripts.plot_bite_outcomes \\
    --bout-ledger checkpoints/cross_sensor_evaluation/native_test/masked_ssl_global_all/bout_ledger.csv \\
    --interim-root data/interim_normalized_80_10_10_clipped
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from chewing_ssl.common.evaluation.postprocess import FS

PALETTE = {
    "nav_x": "#1565C0",
    "nav_y": "#6A1B9A",
    "gt": "#757575",
    "pred": "#2E7D32",
    "box": "#C62828",
}

GLOSS = {
    "missed": "MISSED -- a real bite with no overlapping prediction at all",
    "false_alarm": "FALSE ALARM -- a predicted bout with no overlapping real bite",
}


def _safe(s: str) -> str:
    return str(s).replace("/", "_")


def _shade(ax, bouts: list[dict], color: str, alpha: float, y_lo: float, y_hi: float) -> None:
    for b in bouts:
        ax.axvspan(b["start_s"], b["end_s"], y_lo, y_hi, color=color, alpha=alpha, linewidth=0)


def segment_bouts(seg_rows: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Reconstruct one segment's full GT-bout and predicted-bout lists from
    its ledger rows (missed/detected/false_alarm all included) -- gives the
    same neighbouring-bout context plot_taxonomy_examples.py shows, without
    needing to re-run inference."""
    gt_rows = seg_rows[seg_rows["bout_type"] == "gt"]
    gt_bouts = [{"start_s": r.start_s, "end_s": r.end_s} for r in gt_rows.itertuples()]

    fa_rows = seg_rows[seg_rows["status"] == "false_alarm"]
    pred_bouts = [{"start_s": r.start_s, "end_s": r.end_s} for r in fa_rows.itertuples()]
    matched_rows = gt_rows[gt_rows["status"] == "detected"]
    pred_bouts += [
        {"start_s": r.matched_start_s, "end_s": r.matched_end_s} for r in matched_rows.itertuples()
    ]
    return gt_bouts, pred_bouts


def plot_bite(
    df: pd.DataFrame,
    t,
    row,
    gt_bouts: list[dict],
    pred_bouts: list[dict],
    left_pad_s: float,
    right_pad_s: float,
    out_path: Path,
    ylabel: str,
) -> None:
    lo = max(0.0, row.start_s - left_pad_s)
    hi = min(t[-1], row.end_s + right_pad_s)
    mask = (t >= lo) & (t <= hi)
    t_win, df_win = t[mask], df.iloc[mask]

    gt_win = [b for b in gt_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
    pred_win = [b for b in pred_bouts if b["end_s"] >= lo and b["start_s"] <= hi]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        t_win,
        df_win["nav_x_left_temple"],
        color=PALETTE["nav_x"],
        linewidth=0.8,
        label="nav_x_left_temple",
    )
    ax.plot(
        t_win,
        df_win["nav_y_left_temple"],
        color=PALETTE["nav_y"],
        linewidth=0.8,
        label="nav_y_left_temple",
    )
    _shade(ax, gt_win, PALETTE["gt"], 0.30, 0.0, 0.5)
    _shade(ax, pred_win, PALETTE["pred"], 0.30, 0.5, 1.0)

    box = mpatches.Rectangle(
        (row.start_s, 0.0),
        row.end_s - row.start_s,
        1.0,
        transform=ax.get_xaxis_transform(),
        fill=False,
        edgecolor=PALETTE["box"],
        linewidth=1.8,
        linestyle="--",
        zorder=5,
    )
    ax.add_patch(box)

    ax.set_xlim(lo, hi)
    ax.set_title(
        f"{GLOSS[row.status]}\n"
        f"{row.dataset}/{row.tag}/{row.segment}  --  duration={row.duration_s:.2f}s  "
        f"cluster={row.cluster_label} ({row.cluster_n_gt} GT, {row.cluster_n_pred} pred)",
        fontsize=9,
        loc="left",
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(alpha=0.2)

    legend_handles = [
        mpatches.Patch(color=PALETTE["gt"], alpha=0.30, label="Ground truth (lower half)"),
        mpatches.Patch(color=PALETTE["pred"], alpha=0.30, label="Predicted bout (upper half)"),
        mpatches.Patch(
            facecolor="none",
            edgecolor=PALETTE["box"],
            linewidth=1.8,
            linestyle="--",
            label="Flagged bite",
        ),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bout-ledger", type=Path, required=True)
    parser.add_argument(
        "--interim-root",
        type=Path,
        required=True,
        help="Root the ledger's dataset/tag/segment resolve under, as "
        "<interim-root>/<dataset>/<tag>/<segment>.parquet.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("checkpoints/plots"))
    parser.add_argument("--left-pad-s", type=float, default=3.0)
    parser.add_argument("--right-pad-s", type=float, default=3.0)
    parser.add_argument("--ylabel", default="nav (normalized)")
    args = parser.parse_args()

    ledger = pd.read_csv(args.bout_ledger)
    flagged = ledger[ledger["status"].isin(["missed", "false_alarm"])]
    if flagged.empty:
        print(f"No missed/false_alarm rows in {args.bout_ledger}.")
        return

    out_dirs = {"missed": args.out_root / "missed", "false_alarm": args.out_root / "false_alarms"}
    n_written = n_skipped = 0

    for (dataset, tag, segment), seg_rows in ledger.groupby(["dataset", "tag", "segment"]):
        seg_flagged = flagged[
            (flagged["dataset"] == dataset)
            & (flagged["tag"] == tag)
            & (flagged["segment"] == segment)
        ]
        if seg_flagged.empty:
            continue

        seg_path = args.interim_root / dataset / tag / f"{segment}.parquet"
        if not seg_path.exists():
            print(
                f"  [skip] {dataset}/{tag}/{segment}: {seg_path} not found ({len(seg_flagged)} bite(s))"
            )
            n_skipped += len(seg_flagged)
            continue

        df = pd.read_parquet(seg_path, columns=["nav_x_left_temple", "nav_y_left_temple"])
        t = (df.index / FS).to_numpy()
        gt_bouts, pred_bouts = segment_bouts(seg_rows)

        for row in seg_flagged.itertuples():
            fname = f"{_safe(dataset)}__{_safe(tag)}__{_safe(segment)}__{row.start_s:.2f}s.png"
            plot_bite(
                df,
                t,
                row,
                gt_bouts,
                pred_bouts,
                args.left_pad_s,
                args.right_pad_s,
                out_dirs[row.status] / fname,
                args.ylabel,
            )
            n_written += 1

    print(
        f"\nWrote {n_written} plot(s) ({(flagged['status'] == 'missed').sum()} missed -> {out_dirs['missed']}/, "
        f"{(flagged['status'] == 'false_alarm').sum()} false_alarm -> {out_dirs['false_alarm']}/)"
        + (f"  [{n_skipped} skipped, segment not found]" if n_skipped else "")
    )


if __name__ == "__main__":
    main()
