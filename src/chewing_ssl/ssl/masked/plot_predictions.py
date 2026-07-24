"""Plot navigation signal + chewing labels for one recording: ground truth
on top, the masked-reconstruction model's predictions directly below on the
same axes -- the SSL analogue of chewing_ssl.classical_ml.plot_predictions,
same visual convention and (by default) the same tuned postprocessing
(hysteresis -> gap-fill -> duration/valley split -> raw-signal envelope
split, chewing_ssl.common.evaluation.postprocess) and tolerance-based bout
matching, so the two are visually and numerically comparable.

Unlike the classical-ML path, there's no separate "upscale predictions"
step here: chewing_ssl.ssl.masked.finetune.ChewingDetector already outputs
one probability per timestep directly, so this script runs inference on the
fly (chewing_ssl.ssl.masked.evaluate_bouts.infer_recording) rather than
reading a saved upscaled-predictions parquet.

Run from the project root, after fine-tuning a checkpoint:
  python -m chewing_ssl.ssl.masked.plot_predictions --all \\
      --checkpoint checkpoints/finetune_masked_4s/bigru_finetune_best.pt
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

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import TARGET_HZ
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.postprocess import (
    compute_envelope,
    postprocess,
    remove_short_bouts,
    split_bouts_by_envelope,
)
from chewing_ssl.ssl.masked.evaluate_bouts import ENVELOPE_CHANNELS, FS, infer_recording, load_finetuned_model

PALETTE = {
    "nav_x": "#1565C0",
    "nav_y": "#6A1B9A",
    "detected": "#2E7D32",
    "missed": "#C62828",
    "false_alarm": "#EF6C00",
}


def _shade_bout(ax: plt.Axes, bout: dict, t: pd.Series, color: str, alpha: float = 0.28) -> None:
    end = min(bout["end_idx"], len(t)) - 1
    ax.axvspan(t.iloc[bout["start_idx"]], t.iloc[end], color=color, alpha=alpha, linewidth=0)


def plot_recording(
    model: ChewingDetector,
    seg_path: Path,
    label: str,
    out_path: Path,
    window_len: int,
    min_bout_dur: float = 0.5,
    raw: bool = False,
    threshold: float = 0.5,
    post_high: float = 0.75,
    post_low: float = 0.52,
    post_max_gap: float = 1.0,
    post_min_dur: float = 2.0,
    post_max_dur: float | None = 35.0,
    post_valley: float | None = 0.35,
    envelope_split: bool = True,
    envelope_window_s: float = 1.5,
    envelope_min_piece_dur: float = 8.0,
    envelope_prominence: float = 2.5,
    match_mode: str = "tolerance",
    tolerance_s: float = 5.0,
    iou_thresh: float = 0.3,
) -> None:
    df = pd.read_parquet(seg_path)
    if "label" not in df.columns:
        print(f"  [skip] no label column in {seg_path}")
        return

    t = pd.Series(np.arange(len(df)) / FS)  # seconds, used as the x-axis
    x = df[SENSOR_CHANNELS].to_numpy(dtype=np.float32).T
    gt_binary = df["label"].to_numpy(dtype=bool)

    probs = infer_recording(model, x, window_len=window_len)
    n_unpredicted = int(np.isnan(probs).sum())

    if raw:
        pred_binary = np.nan_to_num(probs, nan=0.0) >= threshold
        match_label = f"raw >= {threshold}, IoU >= {iou_thresh}"
    else:
        pred_binary = postprocess(
            probs,
            high=post_high,
            low=post_low,
            max_gap_s=post_max_gap,
            min_dur_s=post_min_dur,
            max_dur_s=post_max_dur,
            valley_thresh=post_valley,
        )
        if envelope_split:
            signal_mag = np.sqrt(
                (df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1)
            )
            envelope = compute_envelope(signal_mag, fs=FS, window_s=envelope_window_s)
            pred_binary = split_bouts_by_envelope(
                pred_binary,
                envelope,
                fs=FS,
                min_piece_dur_s=envelope_min_piece_dur,
                prominence_frac=envelope_prominence,
            )
            pred_binary = remove_short_bouts(pred_binary, fs=FS, min_dur_s=post_min_dur)
        match_label = (
            f"tolerance={tolerance_s}s" if match_mode == "tolerance" else f"IoU >= {iou_thresh}"
        )

    gt_bouts = extract_bouts(gt_binary, fs=FS, min_dur=min_bout_dur)
    pred_bouts = extract_bouts(pred_binary, fs=FS, min_dur=min_bout_dur)
    if not raw and match_mode == "tolerance":
        result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=tolerance_s)
    else:
        result = match_bouts(gt_bouts, pred_bouts, iou_thresh=iou_thresh)
    matched_gt_ids = {id(m["gt"]) for m in result["matches"]}
    matched_pred_ids = {id(m["pred"]) for m in result["matches"]}
    n_detected = len(matched_gt_ids)
    n_false_alarm = len(pred_bouts) - len(matched_pred_ids)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"Navigation signal vs. chewing labels -- {label}  (masked-reconstruction)\n"
        f"({len(df):,} rows, {n_unpredicted:,} not covered)   "
        f"bouts: {n_detected}/{len(gt_bouts)} GT detected, {n_false_alarm} false alarms  "
        f"({match_label}, min bout {min_bout_dur}s)",
        fontsize=11,
    )

    panels = [
        (axes[0], gt_bouts, matched_gt_ids, {True: "detected", False: "missed"}, "Ground truth"),
        (
            axes[1],
            pred_bouts,
            matched_pred_ids,
            {True: "detected", False: "false_alarm"},
            "Model prediction (raw)" if raw else "Model prediction (postprocessed)",
        ),
    ]
    for ax, bouts, matched_ids, status_of, title in panels:
        ax.plot(t, df["nav_x_left_temple"], color=PALETTE["nav_x"], linewidth=0.8, label="nav_x_left_temple")
        ax.plot(t, df["nav_y_left_temple"], color=PALETTE["nav_y"], linewidth=0.8, label="nav_y_left_temple")
        for bout in bouts:
            status = status_of[id(bout) in matched_ids]
            _shade_bout(ax, bout, t, PALETTE[status])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("nav (normalized)", fontsize=9)
        ax.grid(alpha=0.2)

    legend_handles = [
        mpatches.Patch(color=PALETTE["detected"], alpha=0.28, label="detected"),
        mpatches.Patch(color=PALETTE["missed"], alpha=0.28, label="missed (GT only)"),
        mpatches.Patch(color=PALETTE["false_alarm"], alpha=0.28, label="false alarm (pred only)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  ->  {out_path}  ({n_detected}/{len(gt_bouts)} detected, {n_false_alarm} false alarms)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--segment", default=None, help="e.g. 'segment_00'. Must be given together with --tag."
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Window length the checkpoint was fine-tuned on, if different from the default.",
    )
    parser.add_argument("--out", type=Path, default=Path("checkpoints/plots/masked_predictions/predictions.png"))
    parser.add_argument("--all", action="store_true", help="Plot every finetune_test recording.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("checkpoints/plots/masked_predictions/all")
    )
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-mode", choices=["tolerance", "iou"], default="tolerance")
    parser.add_argument("--tolerance-s", type=float, default=5.0)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--post-high", type=float, default=0.75)
    parser.add_argument("--post-low", type=float, default=0.52)
    parser.add_argument("--post-max-gap", type=float, default=1.0)
    parser.add_argument("--post-min-dur", type=float, default=2.0)
    parser.add_argument("--post-max-dur", type=float, default=35.0)
    parser.add_argument("--post-valley", type=float, default=0.35)
    parser.add_argument("--envelope-split", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envelope-window-s", type=float, default=1.5)
    parser.add_argument("--envelope-min-piece-dur", type=float, default=8.0)
    parser.add_argument("--envelope-prominence", type=float, default=2.5)
    args = parser.parse_args()

    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()
    run_name = f"{meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}"
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else meta["window_len"]
    print(
        f"  Loaded [{run_name}]  architecture={meta['architecture']}  "
        f"pretext_task={meta['pretext_task']}  window={window_len} samples  val_f1={meta['val_f1']:.3f}"
    )

    kwargs = dict(
        window_len=window_len,
        min_bout_dur=args.min_bout_dur,
        raw=args.raw,
        threshold=args.threshold,
        post_high=args.post_high,
        post_low=args.post_low,
        post_max_gap=args.post_max_gap,
        post_min_dur=args.post_min_dur,
        post_max_dur=args.post_max_dur if args.post_max_dur >= 0 else None,
        post_valley=args.post_valley if args.post_valley >= 0 else None,
        envelope_split=args.envelope_split,
        envelope_window_s=args.envelope_window_s,
        envelope_min_piece_dur=args.envelope_min_piece_dur,
        envelope_prominence=args.envelope_prominence,
        match_mode=args.match_mode,
        tolerance_s=args.tolerance_s,
        iou_thresh=args.iou_thresh,
    )

    idx = pd.read_parquet(args.processed_root / "index" / "finetune_test.parquet")
    if args.dataset:
        idx = idx[idx["dataset"] == args.dataset]

    if args.all:
        recordings = sorted(idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False))
        print(f"\nPlotting {len(recordings)} recording(s) -> {args.out_dir}/")
        for dataset, tag in recordings:
            seg_dir = args.interim_root / dataset / tag
            seg_paths = sorted(seg_dir.glob("segment_*.parquet"))
            for seg_path in seg_paths:
                label = f"{dataset}/{tag}" + (f"/{seg_path.stem}" if len(seg_paths) > 1 else "")
                out_path = args.out_dir / f"{dataset}_{tag}_{seg_path.stem}.png"
                plot_recording(model, seg_path, label, out_path, **kwargs)
        print(f"\nDone. {len(recordings)} recording(s) processed -> {args.out_dir}/")
        return

    if not (args.dataset and args.tag and args.segment):
        # Default: the longest test recording, for a quick single-file look.
        seg_lens = []
        for dataset, tag in idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False):
            for seg_path in sorted((args.interim_root / dataset / tag).glob("segment_*.parquet")):
                seg_lens.append((seg_path.stat().st_size, dataset, tag, seg_path))
        _, dataset, tag, seg_path = max(seg_lens)
    else:
        dataset, tag = args.dataset, args.tag
        seg_path = args.interim_root / dataset / tag / f"{args.segment}.parquet"

    label = f"{dataset}/{tag}/{seg_path.stem}"
    print(f"\nPlotting {label}")
    plot_recording(model, seg_path, label, args.out, **kwargs)


if __name__ == "__main__":
    main()
