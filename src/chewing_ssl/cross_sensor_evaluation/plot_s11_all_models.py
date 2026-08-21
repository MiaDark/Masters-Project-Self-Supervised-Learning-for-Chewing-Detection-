from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from chewing_ssl.classical_ml.extract_features import DRIVERS
from chewing_ssl.common.data.preprocess import channels_for_modality, resample_recording
from chewing_ssl.common.data.windowing import WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.pipeline import add_match_args, add_postprocess_args, postprocess_config_from_args
from chewing_ssl.common.evaluation.plotting import gt_panel, plot_bout_panels, pred_panel
from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing
from chewing_ssl.cross_sensor_evaluation.model_grid import CLASSICAL_ML, MASKED_SSL
from chewing_ssl.predict_new_recording import predict_classical_ml, predict_masked_ssl
from chewing_ssl.ssl.masked.evaluate_bouts import load_finetuned_model

DATASET = "FEIT_eating_S11_RightTempleSubstitute_UNVALIDATED"
EVAL_ROOT = Path("checkpoints/cross_sensor_evaluation/s11")
FILES = sorted(Path("data/raw/s11_feit_eating").glob("*/*_eating.csv"))


def _modality(model_key: str) -> str:
    return "nav" if "_nav" in model_key else "accel" if "_accel" in model_key else "all"


def plot_one(model_key, participant, signal, rec, ml_cfg, gt_bouts, base_args, out_root, ylabel):
    pred_binary = apply_full_postprocessing(rec["probs"], rec.get("envelope"), ml_cfg)
    pred_bouts = extract_bouts(pred_binary, fs=50.0, min_dur=base_args.min_bout_dur)
    result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=base_args.tolerance_s)
    n_detected = len({id(m["gt"]) for m in result["matches"]})
    n_fa = len(pred_bouts) - len({id(m["pred"]) for m in result["matches"]})
    suptitle = (
        f"{model_key} -- {participant}  (s11, UNVALIDATED)\n"
        f"bouts: {n_detected}/{len(gt_bouts)} GT detected, {n_fa} false alarms  "
        f"(tolerance={base_args.tolerance_s}s)"
    )
    out_dir = out_root / model_key / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_bout_panels(
        signal["datetime"], signal["nav_x_left_temple"], signal["nav_y_left_temple"],
        [gt_panel(gt_bouts), pred_panel(pred_bouts, title="Model prediction (postprocessed)")],
        suptitle, out_dir / f"{participant}.png", ylim=(-50.0, 50.0), ylabel=ylabel,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")
    base_args = parser.parse_args([])

    ssl_args = argparse.Namespace(**vars(base_args))
    ssl_args.smooth_window_s = 2.0  # masked_ssl's validated default (vs classical_ml's 0.0)
    ssl_cfg = postprocess_config_from_args(ssl_args)
    ml_cfg = postprocess_config_from_args(base_args)

    print("Loading 6 masked_ssl checkpoints ...")
    ssl_models = {}
    for key, (checkpoint, _interim_root, _processed_root, _norm) in MASKED_SSL.items():
        model, meta = load_finetuned_model(checkpoint)
        model.eval()
        ssl_models[key] = (model, meta["window_len"], meta["channels"])

    ml_medians_cache = {
        key: pd.Series(json.loads(medians_path.read_text()))
        for key, (_model_path, _test_features, medians_path, _norm) in CLASSICAL_ML.items()
    }

    n_plotted = 0
    for input_path in FILES:
        participant = input_path.stem
        raw_segments = resample_recording(DATASET, input_path)
        if not raw_segments:
            print(f"  [skip] {participant}: no usable segments")
            continue
        raw_df = raw_segments[0][0]  # single-segment recordings throughout this cohort
        signal = raw_df[["datetime", "nav_x_left_temple", "nav_y_left_temple"]]

        for model_key, (model_path, _test_features, _medians, _norm) in CLASSICAL_ML.items():
            seg_path = EVAL_ROOT / model_key / participant / "interim_normalized" / "segment_00.parquet"
            if not seg_path.exists():
                print(f"  [skip] {model_key}/{participant}: {seg_path} not found (run eval first)")
                continue
            channels = channels_for_modality(_modality(model_key))
            rec = predict_classical_ml(
                seg_path, model_path, ml_medians_cache[model_key], WINDOW_LEN, list(DRIVERS), 50.0,
                channels=channels,
            )
            gt_bouts = extract_bouts(rec["gt"], fs=50.0, min_dur=base_args.min_bout_dur)
            plot_one(model_key, participant, signal, rec, ml_cfg, gt_bouts, base_args, EVAL_ROOT, "nav (raw, clipped)")
            n_plotted += 1

        for model_key, (model, window_len, channels) in ssl_models.items():
            seg_path = EVAL_ROOT / model_key / participant / "interim_normalized" / "segment_00.parquet"
            if not seg_path.exists():
                print(f"  [skip] {model_key}/{participant}: {seg_path} not found (run eval first)")
                continue
            rec = predict_masked_ssl(seg_path, model, window_len, channels)
            gt_bouts = extract_bouts(rec["gt"], fs=50.0, min_dur=base_args.min_bout_dur)
            plot_one(model_key, participant, signal, rec, ssl_cfg, gt_bouts, base_args, EVAL_ROOT, "nav (raw, clipped)")
            n_plotted += 1

        print(f"  {participant}: plotted for 12 models")

    print(f"\nDone. {n_plotted} plot(s) written across {len(FILES)} recording(s) x 12 models -> {EVAL_ROOT}/<model_key>/plots/")


if __name__ == "__main__":
    main()
