"""Plot navigation signal + chewing labels for one recording: ground truth
on top, the classical-ML model's upscaled predictions (upscale_predictions.py)
directly below on the same axes, so the two are visually comparable.

Bout matching (extract_bouts/match_bouts, from common.evaluation.bouts) is
the same IoU-based logic evaluate_bouts.py uses for its aggregate/per-bout
numbers -- this plot is that same verdict rendered visually, rather than a
single flat "chewing" color you have to eyeball against the top panel
yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts

FS = 50.0
PALETTE = {
    "nav_x": "#1565C0", "nav_y": "#6A1B9A",
    "detected": "#2E7D32", "missed": "#C62828", "false_alarm": "#EF6C00",
}


def _shade_bout(ax: plt.Axes, bout: dict, t: pd.Series, color: str, alpha: float = 0.28) -> None:
    end = min(bout["end_idx"], len(t)) - 1
    ax.axvspan(t.iloc[bout["start_idx"]], t.iloc[end], color=color, alpha=alpha, linewidth=0)


def load_manifest(upscaled_dir: Path) -> pd.DataFrame:
    manifest_path = upscaled_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path} -- run upscale_predictions.py first")
    return pd.read_csv(manifest_path)


def find_recording(
    upscaled_dir: Path, dataset: str | None, participant: str | None, tag: str | None, segment: str | None
) -> pd.Series:
    """Resolve which recording to plot, via upscale_predictions.py's
    manifest.csv (the source of truth for identity -- filenames are opaque
    indices, not reliably parseable back into dataset/participant/tag/segment,
    since tag and segment can themselves contain underscores). If
    dataset/participant/tag/segment are all given, look up that exact row;
    otherwise pick whichever recording has the most rows (longest)."""
    manifest = load_manifest(upscaled_dir)

    if dataset and participant and tag and segment:
        match = manifest[
            (manifest["dataset"] == dataset) & (manifest["participant"] == participant)
            & (manifest["tag"] == tag) & (manifest["segment"] == segment)
        ]
        if match.empty:
            raise ValueError(
                f"No recording found for {dataset}/{participant}/{tag}/{segment} "
                f"in {upscaled_dir / 'manifest.csv'}"
            )
        return match.iloc[0]

    return manifest.loc[manifest["n_rows"].idxmax()]


def plot_predictions(
    interim_root: Path, row: pd.Series, out_path: Path,
    iou_thresh: float = 0.3, min_bout_dur: float = 0.5,
) -> None:
    dataset, participant, tag, segment = row["dataset"], row["participant"], row["tag"], row["segment"]
    seg_path = interim_root / dataset / tag / f"{segment}.parquet"
    if not seg_path.exists():
        raise FileNotFoundError(f"No interim segment at {seg_path}")

    signal = pd.read_parquet(seg_path, columns=["datetime", "nav_x_left_temple", "nav_y_left_temple", "label"])
    signal["row_idx"] = range(len(signal))

    preds = pd.read_parquet(row["out_path"])

    merged = signal.merge(preds, on="row_idx", how="left")
    n_unpredicted = merged["pred_label"].isna().sum()

    # Bout-level match, same logic evaluate_ml_bouts.py uses -- this plot is
    # that verdict rendered visually instead of a flat "chewing" shade.
    gt_binary = merged["label"].fillna(0).to_numpy(dtype=bool)
    pred_binary = merged["pred_label"].fillna(0).to_numpy(dtype=bool)
    gt_bouts = extract_bouts(gt_binary, fs=FS, min_dur=min_bout_dur)
    pred_bouts = extract_bouts(pred_binary, fs=FS, min_dur=min_bout_dur)
    result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
    matched_gt_ids = {id(m["gt"]) for m in result["matches"]}
    matched_pred_ids = {id(m["pred"]) for m in result["matches"]}
    n_detected, n_missed = len(matched_gt_ids), len(gt_bouts) - len(matched_gt_ids)
    n_false_alarm = len(pred_bouts) - len(matched_pred_ids)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"Navigation signal vs. chewing labels -- {dataset} / {participant} / {tag} / {segment}\n"
        f"({len(merged):,} rows, {n_unpredicted:,} not covered)   "
        f"bouts: {n_detected}/{len(gt_bouts)} GT detected, {n_false_alarm} false alarms  "
        f"(IoU >= {iou_thresh}, min bout {min_bout_dur}s)",
        fontsize=11,
    )

    panels = [
        (axes[0], gt_bouts, matched_gt_ids, {True: "detected", False: "missed"}, "Ground truth"),
        (axes[1], pred_bouts, matched_pred_ids, {True: "detected", False: "false_alarm"}, "Model prediction"),
    ]
    for ax, bouts, matched_ids, status_of, title in panels:
        ax.plot(merged["datetime"], merged["nav_x_left_temple"], color=PALETTE["nav_x"], linewidth=0.8, label="nav_x_left_temple")
        ax.plot(merged["datetime"], merged["nav_y_left_temple"], color=PALETTE["nav_y"], linewidth=0.8, label="nav_y_left_temple")
        for bout in bouts:
            status = status_of[id(bout) in matched_ids]
            _shade_bout(ax, bout, merged["datetime"], PALETTE[status])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("nav (normalized)", fontsize=9)
        ax.grid(alpha=0.2)

    legend_handles = [
        mpatches.Patch(color=PALETTE["detected"], alpha=0.28, label="detected"),
        mpatches.Patch(color=PALETTE["missed"], alpha=0.28, label="missed (GT only)"),
        mpatches.Patch(color=PALETTE["false_alarm"], alpha=0.28, label="false alarm (pred only)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  ->  {out_path}  ({n_detected}/{len(gt_bouts)} detected, {n_false_alarm} false alarms)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument("--upscaled-dir", type=Path, default=Path("data/processed/ml_predictions_upscaled"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--participant", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--segment", default=None, help="e.g. 'segment_00'. Must be given together with the other three.")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/plots/ml_predictions/predictions.png"))
    parser.add_argument(
        "--all", action="store_true",
        help="Plot every recording in manifest.csv (i.e. every test-set recording, since that's "
             "upscale_predictions.py's default --features-path), one PNG each under --out-dir "
             "instead of a single --out file.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("checkpoints/plots/ml_predictions/all"),
        help="Used only with --all: one PNG per recording is written here, named "
             "<dataset>_<participant>_<tag>_<segment>.png.",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.3, help="Bout-match threshold, same default as evaluate_ml_bouts.py.")
    parser.add_argument("--min-bout-dur", type=float, default=0.5, help="Minimum bout duration (s), same default as evaluate_ml_bouts.py.")
    args = parser.parse_args()

    if args.all:
        manifest = load_manifest(args.upscaled_dir)
        print(f"\nPlotting {len(manifest)} recording(s) -> {args.out_dir}/")
        for _, row in manifest.iterrows():
            out_path = args.out_dir / f"{row['dataset']}_{row['participant']}_{row['tag']}_{row['segment']}.png"
            plot_predictions(args.interim_root, row, out_path, args.iou_thresh, args.min_bout_dur)
        print(f"\nDone. {len(manifest)} plot(s) written to {args.out_dir}/")
        return

    row = find_recording(args.upscaled_dir, args.dataset, args.participant, args.tag, args.segment)
    print(f"\nPlotting {row['dataset']} / {row['participant']} / {row['tag']} / {row['segment']}")

    plot_predictions(args.interim_root, row, args.out, args.iou_thresh, args.min_bout_dur)


if __name__ == "__main__":
    main()
