"""Illustrate the Round-1 zero-ablation finding directly: with navigation
channels zeroed at inference time on an already jointly-trained model
(accelerometer-only input), both pipelines predict essentially zero bouts
-- even though the accelerometer signal itself still visibly tracks real
chewing activity. Companion figure for the accelerometer-only row of
table:results-channel-round1 (dissertation.tex, sec:results-channels).

Shows the ACCELEROMETER trace (not navigation, unlike every earlier figure
in this project) for one real test recording, both pipelines' predictions
side by side, ground truth shaded in each panel's lower half and predicted
bouts (if any) in the upper half.

Reuses zero_out_modality/predict_classical_ml/predict_masked_ssl UNCHANGED
from predict_new_recording.py -- the same mechanism and the same jointly-
trained checkpoints behind table:results-bout-level and
table:results-channel-round1's headline numbers, just pointed at one
recording and rendered instead of aggregated.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chewing_ssl.classical_ml.extract_features import DRIVERS
from chewing_ssl.common.data.windowing import WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts
from chewing_ssl.common.evaluation.pipeline import add_postprocess_args, postprocess_config_from_args
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing
from chewing_ssl.predict_new_recording import (
    DEFAULT_CLASSICAL_ML_MODEL,
    DEFAULT_FEATURE_MEDIANS,
    DEFAULT_SSL_CHECKPOINT,
    predict_classical_ml,
    predict_masked_ssl,
    zero_out_modality,
)
from chewing_ssl.ssl.masked.evaluate_bouts import load_finetuned_model

PALETTE = {"accel_x": "#B71C1C", "accel_y": "#E65100", "accel_z": "#F9A825", "gt": "#757575", "pred": "#2E7D32"}


def _shade(ax, bouts: list[dict], color: str, alpha: float, y_lo: float, y_hi: float) -> None:
    for b in bouts:
        ax.axvspan(b["start_s"], b["end_s"], y_lo, y_hi, color=color, alpha=alpha, linewidth=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_normalized_80_10_10"))
    parser.add_argument("--dataset", default="FEIT_eating_S05")
    parser.add_argument("--tag", default="p10")
    parser.add_argument("--segment", default="segment_00")
    parser.add_argument("--classical-ml-model-path", type=Path, default=DEFAULT_CLASSICAL_ML_MODEL)
    parser.add_argument("--ssl-checkpoint", type=Path, default=DEFAULT_SSL_CHECKPOINT)
    parser.add_argument("--feature-medians", type=Path, default=DEFAULT_FEATURE_MEDIANS)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/plots/accel_only_zero_predictions.png"))
    parser.add_argument("--min-bout-dur", type=float, default=0.5)
    add_postprocess_args(parser)
    args = parser.parse_args()
    # accel-only: ENVELOPE_CHANNELS is nav-only, which is zeroed here -- disable
    # envelope-split (same override predict_new_recording.py applies under --only accel).
    args.envelope_split = False

    seg_path = args.interim_root / args.dataset / args.tag / f"{args.segment}.parquet"
    df_full = pd.read_parquet(seg_path)
    t = np.arange(len(df_full)) / 50.0
    gt_bouts_full = extract_bouts(df_full["label"].to_numpy(dtype=bool), fs=50.0, min_dur=0.5)

    feature_medians = pd.Series(json.loads(args.feature_medians.read_text()))
    ssl_model, ssl_meta = load_finetuned_model(args.ssl_checkpoint)
    ssl_model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_seg_path = Path(tmpdir) / "segment.parquet"
        df_zeroed = zero_out_modality(df_full.copy(), only="accel")
        df_zeroed.to_parquet(tmp_seg_path, index=False)

        # classical ML: smooth_window_s default 0.0 (already smoothed by upscaling averaging)
        cfg_ml_args = argparse.Namespace(**vars(args))
        cfg_ml_args.smooth_window_s = 0.0
        cfg_ml = postprocess_config_from_args(cfg_ml_args)
        rec_ml = predict_classical_ml(
            tmp_seg_path, args.classical_ml_model_path, feature_medians, WINDOW_LEN, list(DRIVERS), 50.0
        )
        binary_ml = apply_full_postprocessing(rec_ml["probs"], rec_ml.get("envelope"), cfg_ml)
        pred_ml = extract_bouts(binary_ml, fs=50.0, min_dur=args.min_bout_dur)

        # masked SSL: smooth_window_s default 2.0s (no equivalent overlap-averaging smoothing)
        cfg_ssl_args = argparse.Namespace(**vars(args))
        cfg_ssl_args.smooth_window_s = 2.0
        cfg_ssl = postprocess_config_from_args(cfg_ssl_args)
        rec_ssl = predict_masked_ssl(tmp_seg_path, ssl_model, ssl_meta["window_len"])
        binary_ssl = apply_full_postprocessing(rec_ssl["probs"], rec_ssl.get("envelope"), cfg_ssl)
        pred_ssl = extract_bouts(binary_ssl, fs=50.0, min_dur=args.min_bout_dur)

    print(f"GT bites: {len(gt_bouts_full)}")
    print(f"Classical ML (accel-only, nav zeroed): {len(pred_ml)} predicted bout(s)")
    print(f"Masked SSL   (accel-only, nav zeroed): {len(pred_ssl)} predicted bout(s)")

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    titles = [
        f"Classical ML -- accelerometer-only (navigation zeroed), {len(pred_ml)} predicted bout(s)",
        f"Self-supervised -- accelerometer-only (navigation zeroed), {len(pred_ssl)} predicted bout(s)",
    ]
    for ax, bouts, title in zip(axes, [pred_ml, pred_ssl], titles):
        for ch, color in zip(["accel_x", "accel_y", "accel_z"], [PALETTE["accel_x"], PALETTE["accel_y"], PALETTE["accel_z"]]):
            ax.plot(t, df_full[ch], color=color, linewidth=0.6, alpha=0.85, label=ch)
        _shade(ax, gt_bouts_full, PALETTE["gt"], 0.30, 0.0, 0.5)
        _shade(ax, bouts, PALETTE["pred"], 0.30, 0.5, 1.0)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel("accel (normalized)", fontsize=9)
        ax.grid(alpha=0.2)

    legend_handles = [
        mpatches.Patch(color=PALETTE["gt"], alpha=0.30, label="Ground truth (lower half)"),
        mpatches.Patch(color=PALETTE["pred"], alpha=0.30, label="Predicted bout, if any (upper half)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(
        "Accelerometer-only zero-ablation: real chewing activity, essentially no predicted bouts",
        fontsize=13,
    )
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
