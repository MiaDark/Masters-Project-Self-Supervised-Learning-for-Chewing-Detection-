"""Sweep the shared postprocessing recipe's hyperparameters against the
masked-reconstruction SSL pipeline's predictions.

Same idea as `chewing_ssl.classical_ml.sweep_postprocess`: inference
(`load_test_recordings`) runs ONCE -- the expensive step -- then every combo
in the sweep only re-runs the cheap postprocessing+matching step
(`chewing_ssl.common.evaluation.sweep`).

Run from the project root:
  python -m chewing_ssl.ssl.masked.sweep_postprocess \
    --checkpoint checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt \
    --interim-root data/interim_normalized_80_10_10 \
    --processed-root data/processed_80_10_10 \
    --out-csv checkpoints/postprocess_sweep_masked_80_10_10.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chewing_ssl.common.evaluation.postprocess import PostprocessConfig
from chewing_ssl.common.evaluation.sweep import (
    COORDINATE_DESCENT_GRID,
    coordinate_descent_sweep,
    sweep_postprocessing,
    summarize_best,
)
from chewing_ssl.ssl.masked.evaluate_bouts import load_finetuned_model, load_test_recordings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", default=None, help="Restrict to one dataset. Default: all.")
    parser.add_argument(
        "--envelope-window-s",
        type=float,
        default=1.5,
        help="Fixed for the whole sweep -- not itself swept, see sweep.py's module docstring.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Override the window length auto-detected from the checkpoint (rarely needed).",
    )
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    parser.add_argument("--tolerance-s", type=float, default=5.0)
    parser.add_argument(
        "--metric",
        choices=["bout_iou_f1", "bout_tolerance_f1"],
        default="bout_tolerance_f1",
        help="--mode coordinate only: which metric each step's winner is picked by.",
    )
    parser.add_argument(
        "--mode",
        choices=["coordinate", "grid"],
        default="coordinate",
        help="coordinate: tune one field at a time (cheap, default). grid: full "
        "cross-product of COORDINATE_DESCENT_GRID (huge -- only sensible if you've "
        "narrowed the grid down first, e.g. by editing this script).",
    )
    parser.add_argument(
        "--smooth-window-s-default",
        type=float,
        default=2.0,
        help="Starting point for smooth_window_s (this pipeline's own tuned default is "
        "2.0, not classical ML's 0.0) -- only matters if smooth_window_s isn't itself "
        "in the sweep grid, or as the seed value before it's tuned.",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("checkpoints/postprocess_sweep_masked.csv"))
    args = parser.parse_args()

    from chewing_ssl.common.data.windowing import TARGET_HZ

    model, meta = load_finetuned_model(args.checkpoint)
    model.eval()
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else meta["window_len"]

    print(f"\nRunning inference fresh from {args.checkpoint} ...")
    recordings = load_test_recordings(
        model,
        args.interim_root,
        args.processed_root,
        window_len,
        envelope_window_s=args.envelope_window_s,
        dataset=args.dataset,
        channels=meta["channels"],
    )
    print(f"  {len(recordings)} recordings loaded.")

    base_cfg = PostprocessConfig(smooth_window_s=args.smooth_window_s_default)

    if args.mode == "coordinate":
        best_cfg, history = coordinate_descent_sweep(
            recordings,
            COORDINATE_DESCENT_GRID,
            base_cfg,
            min_bout_dur=args.min_bout_dur,
            iou_thresh=args.iou_thresh,
            tolerance_s=args.tolerance_s,
            metric=args.metric,
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.out_csv, index=False)
        print(f"\nBest config found (optimizing {args.metric}):")
        print(json.dumps(best_cfg.__dict__, indent=2, default=str))
        print(f"\nFull sweep history ({len(history)} combos tried) -> {args.out_csv}")
    else:
        df = sweep_postprocessing(
            recordings,
            COORDINATE_DESCENT_GRID,
            base_cfg,
            min_bout_dur=args.min_bout_dur,
            iou_thresh=args.iou_thresh,
            tolerance_s=args.tolerance_s,
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        summarize_best(df)
        print(f"\nFull grid ({len(df)} combos) -> {args.out_csv}")


if __name__ == "__main__":
    main()
