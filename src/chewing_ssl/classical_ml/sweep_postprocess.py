"""Sweep the shared postprocessing recipe's hyperparameters against the
classical-ML pipeline's predictions.

Predictions (`compute_upscaled_recordings`) are computed ONCE from
--model-path -- the expensive step, same fresh-every-run computation
evaluate_bouts.py does, no cached artifact -- then every combo in the sweep
only re-runs the cheap postprocessing+matching step
(`chewing_ssl.common.evaluation.sweep`).

Run from the project root:
  python -m chewing_ssl.classical_ml.sweep_postprocess \
    --model-path checkpoints/train_ml_80_10_10/best_model.joblib \
    --interim-root data/interim_normalized_80_10_10 \
    --features-path data/processed_80_10_10/ml_features/test.parquet \
    --out-csv checkpoints/postprocess_sweep_rf_4s_80_10_10.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chewing_ssl.classical_ml.upscale_predictions import compute_upscaled_recordings
from chewing_ssl.common.evaluation.postprocess import PostprocessConfig
from chewing_ssl.common.evaluation.sweep import (
    COORDINATE_DESCENT_GRID,
    coordinate_descent_sweep,
    summarize_best,
    sweep_postprocessing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="From train.py.")
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized"))
    parser.add_argument(
        "--features-path", type=Path, default=Path("data/processed/ml_features/test.parquet")
    )
    parser.add_argument(
        "--window-seconds", type=float, default=None, help="Default: configs/preprocessing.yaml."
    )
    parser.add_argument(
        "--envelope-window-s",
        type=float,
        default=1.5,
        help="Fixed for the whole sweep -- not itself swept, see sweep.py's module docstring.",
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
    parser.add_argument("--out-csv", type=Path, default=Path("checkpoints/postprocess_sweep.csv"))
    args = parser.parse_args()

    from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN

    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN

    print(f"\nComputing predictions fresh from {args.model_path} ...")
    recordings = compute_upscaled_recordings(
        args.model_path,
        args.features_path,
        args.interim_root,
        window_len,
        envelope_window_s=args.envelope_window_s,
    )
    print(f"  {len(recordings)} recordings loaded.")

    base_cfg = PostprocessConfig()  # current tuned defaults -- the sweep's starting point

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
