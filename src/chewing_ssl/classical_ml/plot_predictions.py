"""Plot navigation signal + chewing labels for one recording: ground truth
on top, the classical-ML model's predictions directly below on the same
axes, so the two are visually comparable.

By default the bottom panel shows POSTPROCESSED predictions -- the shared
recipe in `chewing_ssl.common.evaluation.postprocess.apply_full_postprocessing`
(hysteresis -> gap-fill -> bout splitting -> duration bounds -> raw-signal
envelope split, see evaluate_bouts.py's module docstring and
docs/classical_ml.md §6b for why both splitting mechanisms exist) -- with
onset/offset-tolerance bout matching (chewing_ssl.common.evaluation.bouts.
match_bouts_tolerant) -- the same defaults as evaluate_bouts.py, so this plot
is that script's verdict rendered visually rather than a flat "chewing" color
you have to eyeball against the top panel yourself. Pass --raw to instead
plot the raw thresholded prediction with strict IoU matching, for comparison.

Predictions are computed FRESH from --model-path every run (same
`chewing_ssl.classical_ml.upscale_predictions.compute_upscaled_recordings`
call `evaluate_bouts.py` uses) -- no cached prediction artifact, so this
always reflects whatever model --model-path currently points at.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import argparse

import numpy as np
import pandas as pd

from chewing_ssl.classical_ml.upscale_predictions import compute_upscaled_recordings
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.pipeline import add_postprocess_args, postprocess_config_from_args
from chewing_ssl.common.evaluation.plotting import gt_panel, plot_bout_panels, pred_panel
from chewing_ssl.common.evaluation.postprocess import ENVELOPE_CHANNELS, FS, apply_full_postprocessing


def plot_predictions(
    rec: dict,
    interim_root: Path,
    out_path: Path,
    cfg,
    min_bout_dur: float = 0.5,
    raw: bool = False,
    threshold: float = 0.5,
    match_mode: str = "tolerance",
    tolerance_s: float = 5.0,
    iou_thresh: float = 0.3,
    ylim: tuple[float, float] | None = None,
) -> None:
    dataset, participant, tag, segment = rec["dataset"], rec["participant"], rec["tag"], rec["segment"]
    seg_path = interim_root / dataset / tag / f"{segment}.parquet"
    if not seg_path.exists():
        raise FileNotFoundError(f"No interim segment at {seg_path}")

    signal = pd.read_parquet(seg_path, columns=["datetime", *ENVELOPE_CHANNELS])
    probs = rec["probs"]
    n_unpredicted = int(np.isnan(probs).sum())

    if raw:
        pred_binary = np.nan_to_num(probs, nan=0.0) >= threshold
        match_label = f"raw >= {threshold}, IoU >= {iou_thresh}"
    else:
        pred_binary = apply_full_postprocessing(probs, rec.get("envelope"), cfg)
        match_label = (
            f"tolerance={tolerance_s}s" if match_mode == "tolerance" else f"IoU >= {iou_thresh}"
        )

    # Bout-level match, same logic evaluate_bouts.py uses -- this plot is
    # that verdict rendered visually instead of a flat "chewing" shade.
    gt_bouts = extract_bouts(rec["gt"], fs=FS, min_dur=min_bout_dur)
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
        f"Navigation signal vs. chewing labels -- {dataset} / {participant} / {tag} / {segment}\n"
        f"({len(signal):,} rows, {n_unpredicted:,} not covered)   "
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
    ylabel = "nav (normalized)" if "normalized" in str(interim_root) else "nav (raw, clipped)"
    plot_bout_panels(
        signal["datetime"], signal["nav_x_left_temple"], signal["nav_y_left_temple"],
        panels, suptitle, out_path, ylim=ylim, ylabel=ylabel,
    )
    print(
        f"  Saved  ->  {out_path}  ({n_detected}/{len(gt_bouts)} detected, {n_false_alarm} false alarms)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="From train.py, e.g. checkpoints/train_ml/best_model.joblib. Predictions are always "
        "computed fresh against this file -- no cached prediction artifact.",
    )
    parser.add_argument(
        "--features-path", type=Path, default=Path("data/processed/ml_features/test.parquet")
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Must match whatever segment.py --window-seconds built --features-path with. "
        "Default: configs/preprocessing.yaml's value (segment.py's own default).",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--participant", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--segment",
        default=None,
        help="e.g. 'segment_00'. Filters --dataset/--participant/--tag/--segment jointly; "
        "if none given and not --all, plots the single longest test recording.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("checkpoints/plots/ml_predictions/predictions.png")
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every matching recording, one PNG each under --out-dir instead of a single "
        "--out file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/plots/ml_predictions/all"),
        help="Used only with --all: one PNG per recording is written here, named "
        "<dataset>_<participant>_<tag>_<segment>.png.",
    )
    parser.add_argument(
        "--min-bout-dur", type=float, default=0.5, help="Minimum bout duration (s)."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Plot the raw thresholded prediction with strict IoU matching, instead of "
        "postprocessed predictions with tolerance matching (the default).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="--raw only: binarization threshold."
    )
    parser.add_argument(
        "--match-mode",
        choices=["tolerance", "iou"],
        default="tolerance",
        help="How a predicted bout is credited against a ground-truth bout (default: tolerance).",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=5.0,
        help="--match-mode tolerance: max onset/offset deviation in seconds. Default 5.0, "
        "not 2.0 -- see evaluate_bouts.py / docs/classical_ml.md §6a.",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.3, help="--match-mode iou (or --raw): minimum IoU."
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Fix every panel's y-axis to [LO, HI] instead of matplotlib's per-panel "
        "auto-scale, e.g. --ylim -2 2. Makes bout shape/position comparable across "
        "panels and across plots; default: auto-scale per panel.",
    )
    add_postprocess_args(parser)
    args = parser.parse_args()

    cfg = postprocess_config_from_args(args)
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN

    print(f"  Computing predictions fresh from {args.model_path} ...")
    recordings = compute_upscaled_recordings(
        args.model_path, args.features_path, args.interim_root, window_len,
        envelope_window_s=args.envelope_window_s,
    )

    def matches(rec: dict) -> bool:
        return (
            (not args.dataset or rec["dataset"] == args.dataset)
            and (not args.participant or rec["participant"] == args.participant)
            and (not args.tag or rec["tag"] == args.tag)
            and (not args.segment or rec["segment"] == args.segment)
        )

    recordings = [r for r in recordings if matches(r)]
    if not recordings:
        raise ValueError("No recording matches the given --dataset/--participant/--tag/--segment filters.")

    kwargs = dict(
        interim_root=args.interim_root,
        cfg=cfg,
        min_bout_dur=args.min_bout_dur,
        raw=args.raw,
        threshold=args.threshold,
        match_mode=args.match_mode,
        tolerance_s=args.tolerance_s,
        iou_thresh=args.iou_thresh,
        ylim=tuple(args.ylim) if args.ylim else None,
    )

    if args.all:
        print(f"\nPlotting {len(recordings)} recording(s) -> {args.out_dir}/")
        for rec in recordings:
            out_path = (
                args.out_dir
                / f"{rec['dataset']}_{rec['participant']}_{rec['tag']}_{rec['segment']}.png"
            )
            plot_predictions(rec, out_path=out_path, **kwargs)
        print(f"\nDone. {len(recordings)} plot(s) written to {args.out_dir}/")
        return

    # Default (no --all, filters didn't narrow to exactly one): the longest
    # matching recording, for a quick single-file look.
    rec = max(recordings, key=lambda r: len(r["gt"]))
    print(f"\nPlotting {rec['dataset']} / {rec['participant']} / {rec['tag']} / {rec['segment']}")
    plot_predictions(rec, out_path=args.out, **kwargs)


if __name__ == "__main__":
    main()
