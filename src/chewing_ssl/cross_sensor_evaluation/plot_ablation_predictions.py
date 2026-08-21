from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chewing_ssl.classical_ml.extract_features import DRIVERS
from chewing_ssl.common.data.windowing import WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.plotting import gt_panel, plot_bout_panels, pred_panel
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing
from chewing_ssl.predict_new_recording import (
    DEFAULT_CLASSICAL_ML_MODEL,
    DEFAULT_FEATURE_MEDIANS,
    DEFAULT_SSL_CHECKPOINT,
    predict_classical_ml,
)
from chewing_ssl.ssl.masked.evaluate_bouts import load_finetuned_model
from chewing_ssl.ssl.masked.plot_predictions import plot_recording

IN_ROOT = Path("checkpoints/predict_new/ablation")
OUT_ROOT = Path("checkpoints/plots/predict_new/ablation")
SCOPES = ["SIPMEatingStudy", "SussexEatingUK", "FEIT_eating_S05", "Office_eating_s05", "s11"]
ONLYS = ["nav", "accel"]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")
    args = parser.parse_args([])
    feature_medians = pd.Series(json.loads(DEFAULT_FEATURE_MEDIANS.read_text()))

    ssl_model, ssl_meta = load_finetuned_model(DEFAULT_SSL_CHECKPOINT)
    ssl_model.eval()
    window_len = ssl_meta["window_len"]

    n_plotted = 0
    for only in ONLYS:
        ml_args = argparse.Namespace(**vars(args))
        ssl_args = argparse.Namespace(**vars(args))
        if only == "accel":
            ml_args.envelope_split = False
            ssl_args.envelope_split = False
        ssl_args.smooth_window_s = 2.0
        ml_cfg = postprocess_config_from_args(ml_args)
        ssl_cfg = postprocess_config_from_args(ssl_args)

        for scope in SCOPES:
            ml_dir = IN_ROOT / f"classical_ml_{scope}_only_{only}"
            ssl_dir = IN_ROOT / f"masked_ssl_{scope}_only_{only}"

            for seg_path in sorted(ml_dir.glob("*/interim_normalized/segment_*.parquet")):
                participant = seg_path.parents[1].name
                rec = predict_classical_ml(
                    seg_path,
                    DEFAULT_CLASSICAL_ML_MODEL,
                    feature_medians,
                    WINDOW_LEN,
                    list(DRIVERS),
                    50.0,
                )
                signal = pd.read_parquet(
                    seg_path, columns=["datetime", "nav_x_left_temple", "nav_y_left_temple"]
                )
                probs = rec["probs"]
                pred_binary = apply_full_postprocessing(probs, rec.get("envelope"), ml_cfg)
                gt_bouts = extract_bouts(rec["gt"], fs=50.0, min_dur=ml_args.min_bout_dur)
                pred_bouts = extract_bouts(pred_binary, fs=50.0, min_dur=ml_args.min_bout_dur)
                result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=ml_args.tolerance_s)
                n_detected = len({id(m["gt"]) for m in result["matches"]})
                n_fa = len(pred_bouts) - len({id(m["pred"]) for m in result["matches"]})
                suptitle = (
                    f"classical_ml -- {scope}/{participant} (--only {only}, zero-ablation)\n"
                    f"bouts: {n_detected}/{len(gt_bouts)} GT detected, {n_fa} false alarms  "
                    f"(tolerance={ml_args.tolerance_s}s)"
                )
                out_dir = OUT_ROOT / f"classical_ml_{scope}_only_{only}"
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_bout_panels(
                    signal["datetime"],
                    signal["nav_x_left_temple"],
                    signal["nav_y_left_temple"],
                    [
                        gt_panel(gt_bouts),
                        pred_panel(pred_bouts, title="Model prediction (postprocessed)"),
                    ],
                    suptitle,
                    out_dir / f"{participant}.png",
                    ylim=(-2.0, 2.0),
                )
                n_plotted += 1

            for seg_path in sorted(ssl_dir.glob("*/interim_normalized/segment_*.parquet")):
                participant = seg_path.parents[1].name
                label = f"masked_ssl -- {scope}/{participant} (--only {only}, zero-ablation)"
                out_dir = OUT_ROOT / f"masked_ssl_{scope}_only_{only}"
                out_dir.mkdir(parents=True, exist_ok=True)
                plot_recording(
                    ssl_model,
                    seg_path,
                    label,
                    out_dir / f"{participant}.png",
                    window_len=window_len,
                    cfg=ssl_cfg,
                    min_bout_dur=ssl_args.min_bout_dur,
                    match_mode=ssl_args.match_mode,
                    tolerance_s=ssl_args.tolerance_s,
                    iou_thresh=ssl_args.iou_thresh,
                    ylim=(-2.0, 2.0),
                )
                n_plotted += 1

    print(f"\nDone. {n_plotted} plot(s) -> {OUT_ROOT}/")


if __name__ == "__main__":
    main()
