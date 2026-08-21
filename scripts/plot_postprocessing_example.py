from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chewing_ssl.common.evaluation.bouts import extract_bouts
from chewing_ssl.common.evaluation.pipeline import (
    add_postprocess_args,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing, compute_envelope
from chewing_ssl.ssl.masked.evaluate_bouts import (
    ENVELOPE_CHANNELS,
    FS,
    infer_recording,
    load_finetuned_model,
)

PALETTE = {"nav_x": "#1565C0", "nav_y": "#6A1B9A", "gt": "#757575", "pred": "#2E7D32"}


def _shade(ax, bouts: list[dict], color: str, alpha: float, y_lo: float, y_hi: float) -> None:
    # start_s/end_s are absolute-recording time values (seconds), independent of
    # any later zoom-slicing of the plotted t/df arrays -- unlike start_idx/end_idx,
    # which index into the full, unzoomed array and would silently misalign after a zoom.
    for b in bouts:
        ax.axvspan(b["start_s"], b["end_s"], y_lo, y_hi, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt"),
    )
    parser.add_argument(
        "--interim-root", type=Path, default=Path("data/interim_normalized_80_10_10")
    )
    parser.add_argument("--dataset", default="FEIT_eating_S05")
    parser.add_argument("--tag", default="p22")
    parser.add_argument("--segment", default="segment_00")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Raw-panel binarisation threshold."
    )
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument(
        "--start-s",
        type=float,
        default=None,
        help="Zoom window start (s). Default: whole recording.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Zoom window length (s). Default: whole recording.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("checkpoints/plots/postprocessing_example.png")
    )
    add_postprocess_args(parser)
    # Same override plot_predictions.py applies: this pipeline's smoothing default is 2.0s, not the shared 0.0s.
    for action in parser._actions:
        if action.dest == "smooth_window_s":
            action.default = 2.0
    args = parser.parse_args()
    cfg = postprocess_config_from_args(args)

    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()
    print(
        f"Loaded checkpoint: {meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}, val_f1={meta['val_f1']:.3f}"
    )

    seg_path = args.interim_root / args.dataset / args.tag / f"{args.segment}.parquet"
    df = pd.read_parquet(seg_path)
    t = np.arange(len(df)) / FS
    x = df[meta["channels"]].to_numpy(dtype=np.float32).T
    gt_binary = df["label"].to_numpy(dtype=bool)

    probs = infer_recording(model, x, window_len=meta["window_len"])
    raw_binary = np.nan_to_num(probs, nan=0.0) >= args.threshold

    signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
    envelope = (
        compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)
        if cfg.envelope_split
        else None
    )
    post_binary = apply_full_postprocessing(probs, envelope, cfg)

    gt_bouts = extract_bouts(gt_binary, fs=FS, min_dur=args.min_bout_dur)
    raw_bouts = extract_bouts(raw_binary, fs=FS, min_dur=args.min_bout_dur)
    post_bouts = extract_bouts(post_binary, fs=FS, min_dur=args.min_bout_dur)
    print(
        f"GT bites: {len(gt_bouts)}  |  raw predicted bouts: {len(raw_bouts)}  |  postprocessed predicted bouts: {len(post_bouts)}"
    )

    if args.start_s is not None and args.duration_s is not None:
        lo, hi = args.start_s, args.start_s + args.duration_s
        mask = (t >= lo) & (t <= hi)
        t, df = t[mask], df.iloc[mask]
        gt_bouts = [b for b in gt_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
        raw_bouts = [b for b in raw_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
        post_bouts = [b for b in post_bouts if b["end_s"] >= lo and b["start_s"] <= hi]
        print(
            f"Zoomed to [{lo}, {hi}]s: GT={len(gt_bouts)}  raw={len(raw_bouts)}  post={len(post_bouts)}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    titles = ["Raw prediction (thresholded only)", "Postprocessed prediction"]
    for ax, bouts, title in zip(axes, [raw_bouts, post_bouts], titles):
        ax.plot(
            t,
            df["nav_x_left_temple"],
            color=PALETTE["nav_x"],
            linewidth=0.7,
            label="nav_x_left_temple",
        )
        ax.plot(
            t,
            df["nav_y_left_temple"],
            color=PALETTE["nav_y"],
            linewidth=0.7,
            label="nav_y_left_temple",
        )
        _shade(ax, gt_bouts, PALETTE["gt"], 0.30, 0.0, 0.5)
        _shade(ax, bouts, PALETTE["pred"], 0.30, 0.5, 1.0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("nav (normalized)", fontsize=9)
        ax.grid(alpha=0.2)
        if args.start_s is not None and args.duration_s is not None:
            # Hard clip to exactly [lo, hi] -- axvspan for a bout straddling the
            # window edge would otherwise bleed the shading past it, since bouts
            # overlapping the edge are kept (not truncated) for correct counts.
            ax.set_xlim(lo, hi)

    legend_handles = [
        mpatches.Patch(color=PALETTE["gt"], alpha=0.30, label="Ground truth (lower half)"),
        mpatches.Patch(color=PALETTE["pred"], alpha=0.30, label="Predicted bout (upper half)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(
        "Example prediction and postprocessed predictions -- masked-SSL detection", fontsize=13
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
