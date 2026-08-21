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

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import argparse

import numpy as np
import pandas as pd

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.windowing import TARGET_HZ
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.pipeline import (
    add_postprocess_args,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.plotting import gt_panel, plot_bout_panels, pred_panel
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing, compute_envelope
from chewing_ssl.ssl.masked.evaluate_bouts import (
    ENVELOPE_CHANNELS,
    FS,
    infer_recording,
    load_finetuned_model,
)


def plot_recording(
    model,
    seg_path: Path,
    label: str,
    out_path: Path,
    window_len: int,
    cfg,
    min_bout_dur: float = 0.5,
    raw: bool = False,
    threshold: float = 0.5,
    match_mode: str = "tolerance",
    tolerance_s: float = 5.0,
    iou_thresh: float = 0.3,
    ylim: tuple[float, float] | None = None,
    raw_seg_path: Path | None = None,
    channels: list[str] = SENSOR_CHANNELS,
) -> bool:
    """Returns True if a plot was written, False if this segment was skipped
    (e.g. no label column) -- callers use this to report an accurate count.

    `channels` must match the checkpoint's own training channels (auto-detected
    from its metadata by the caller) -- a 4/6-channel checkpoint fed the full
    10-channel SENSOR_CHANNELS default would crash on a shape mismatch.

    `raw_seg_path`, if given, is read purely for the PLOTTED nav_x/nav_y
    trace (e.g. a raw-scale/clipped interim tree) -- model inference and the
    envelope/label columns still come from `seg_path` (the normalized tree
    the checkpoint was actually trained on). Without it, both plotting and
    inference use `seg_path`, as before."""
    df = pd.read_parquet(seg_path)
    if "label" not in df.columns:
        print(f"  [skip] no label column in {seg_path}")
        return False
    plot_df = pd.read_parquet(raw_seg_path) if raw_seg_path is not None else df

    t = pd.Series(np.arange(len(df)) / FS)  # seconds, used as the x-axis
    x = df[channels].to_numpy(dtype=np.float32).T
    gt_binary = df["label"].to_numpy(dtype=bool)

    probs = infer_recording(model, x, window_len=window_len)
    n_unpredicted = int(np.isnan(probs).sum())

    if raw:
        pred_binary = np.nan_to_num(probs, nan=0.0) >= threshold
        match_label = f"raw >= {threshold}, IoU >= {iou_thresh}"
    else:
        envelope = None
        if cfg.envelope_split:
            signal_mag = np.sqrt(
                (df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1)
            )
            envelope = compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)
        pred_binary = apply_full_postprocessing(probs, envelope, cfg)
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

    suptitle = (
        f"Navigation signal vs. chewing labels -- {label}  (masked-reconstruction)\n"
        f"({len(df):,} rows, {n_unpredicted:,} not covered)   "
        f"bouts: {n_detected}/{len(gt_bouts)} GT detected, {n_false_alarm} false alarms  "
        f"({match_label}, min bout {min_bout_dur}s)"
    )
    panels = [
        gt_panel(gt_bouts),
        pred_panel(
            pred_bouts,
            title="Model prediction (raw)" if raw else "Model prediction (postprocessed)",
        ),
    ]
    if raw_seg_path is not None:
        ylabel = "nav (raw, clipped)"
    elif "normalized" in str(seg_path):
        ylabel = "nav (normalized)"
    else:
        ylabel = "nav (raw, clipped)"
    plot_bout_panels(
        t,
        plot_df["nav_x_left_temple"],
        plot_df["nav_y_left_temple"],
        panels,
        suptitle,
        out_path,
        ylim=ylim,
        ylabel=ylabel,
    )
    print(
        f"  Saved  ->  {out_path}  ({n_detected}/{len(gt_bouts)} detected, {n_false_alarm} false alarms)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument(
        "--raw-interim-root",
        type=Path,
        default=None,
        help="If given, the plotted nav_x/nav_y trace is read from this tree instead of "
        "--interim-root (e.g. data/interim_clipped for raw-scale/clipped units) -- model "
        "inference still uses --interim-root, since the checkpoint needs normalized input. "
        "Must have the identical <dataset>/<tag>/segment_NN.parquet layout as --interim-root.",
    )
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
    parser.add_argument(
        "--out", type=Path, default=Path("checkpoints/plots/masked_predictions/predictions.png")
    )
    parser.add_argument("--all", action="store_true", help="Plot every finetune_test recording.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("checkpoints/plots/masked_predictions/all")
    )
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--match-mode", choices=["tolerance", "iou"], default="iou")
    parser.add_argument("--tolerance-s", type=float, default=5.0)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Fix every panel's y-axis to [LO, HI] instead of matplotlib's per-panel "
        "auto-scale, e.g. --ylim -2 2. Default: auto-scale per panel.",
    )
    add_postprocess_args(parser)
    # Override the shared --smooth-window-s default (0.0) for this pipeline --
    # see ssl/masked/evaluate_bouts.py's module docstring for why 2.0s is
    # validated here specifically (classical ML's own default stays 0.0).
    for action in parser._actions:
        if action.dest == "smooth_window_s":
            action.default = 2.0
    args = parser.parse_args()

    cfg = postprocess_config_from_args(args)
    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()
    run_name = f"{meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}"
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else meta["window_len"]
    print(
        f"  Loaded [{run_name}]  architecture={meta['architecture']}  "
        f"pretext_task={meta['pretext_task']}  window={window_len} samples  val_f1={meta['val_f1']:.3f}"
    )

    kwargs = {
        "window_len": window_len,
        "cfg": cfg,
        "min_bout_dur": args.min_bout_dur,
        "raw": args.raw,
        "threshold": args.threshold,
        "match_mode": args.match_mode,
        "tolerance_s": args.tolerance_s,
        "iou_thresh": args.iou_thresh,
        "ylim": tuple(args.ylim) if args.ylim else None,
        "channels": meta["channels"],
    }

    idx = pd.read_parquet(args.processed_root / "index" / "finetune_test.parquet")
    if args.dataset:
        idx = idx[idx["dataset"] == args.dataset]

    if args.all:
        tags = sorted(idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False))
        print(f"\nPlotting recordings for {len(tags)} participant(s) -> {args.out_dir}/")
        n_plotted = 0
        for dataset, tag in tags:
            seg_dir = args.interim_root / dataset / tag
            seg_paths = sorted(seg_dir.glob("segment_*.parquet"))
            if not seg_paths:
                print(
                    f"  [warn] no segment_*.parquet found under {seg_dir} -- check --interim-root"
                )
            for seg_path in seg_paths:
                label = f"{dataset}/{tag}" + (f"/{seg_path.stem}" if len(seg_paths) > 1 else "")
                out_path = args.out_dir / f"{dataset}_{tag}_{seg_path.stem}.png"
                raw_seg_path = (
                    args.raw_interim_root / dataset / tag / seg_path.name
                    if args.raw_interim_root
                    else None
                )
                if plot_recording(
                    model, seg_path, label, out_path, raw_seg_path=raw_seg_path, **kwargs
                ):
                    n_plotted += 1
        print(
            f"\nDone. {n_plotted} plot(s) written (across {len(tags)} participant(s)) -> {args.out_dir}/"
        )
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
    raw_seg_path = (
        args.raw_interim_root / dataset / tag / seg_path.name if args.raw_interim_root else None
    )
    plot_recording(model, seg_path, label, args.out, raw_seg_path=raw_seg_path, **kwargs)


if __name__ == "__main__":
    main()
