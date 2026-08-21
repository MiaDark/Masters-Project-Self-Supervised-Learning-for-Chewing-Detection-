"""Run inference on a BRAND-NEW raw recording (not part of the training
corpus), with either pipeline's best 80/10/10, global-normalization model:

  classical_ml   checkpoints/train_ml_80_10_10/best_model.joblib
  masked_ssl     checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt

This is the inference-time analogue of the training pipeline: same raw ->
interim -> normalized preprocessing (chewing_ssl.common.data.preprocess),
but normalization REUSES the already-fitted global_stats.json from the
80/10/10 run (data/processed_80_10_10/global_stats.json) rather than
refitting -- exactly as a genuinely new recording must be handled (decision
003: normalization statistics are fit once, on training data, then frozen).
Classical ML likewise reuses the saved feature_medians.json for
NaN/Inf sanitization rather than recomputing medians from this one new
recording.

Everything downstream of "get probs/gt/envelope for this recording" is the
same shared code every other evaluation script in this project uses
(chewing_ssl.common.evaluation.postprocess/pipeline) -- nothing new is
reimplemented here, this module is orchestration only.

If the new recording has a `label` column (ground truth available),
bout-level evaluation runs automatically (both match modes, the match/
merge/split/tangle/missed taxonomy, a per-bite ledger). If it doesn't,
those steps are skipped and only the predicted bouts are reported --
this is what genuine, label-free deployment inference looks like.

Run from the project root:
  python -m chewing_ssl.predict_new_recording \\
    --model classical_ml \\
    --input /path/to/new_recording.csv --dataset FEIT_eating_S05 \\
    --out-dir checkpoints/predict_new/my_new_recording
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chewing_ssl.classical_ml.extract_features import DRIVERS, extract_features, sanitize
from chewing_ssl.classical_ml.segment import segment_recording
from chewing_ssl.classical_ml.upscale_predictions import upscale_recording
from chewing_ssl.common.data.eda import discover_recordings
from chewing_ssl.common.data.loaders import ACCEL_COLS, DATASET_SPECS
from chewing_ssl.common.data.preprocess import (
    DERIVATIVE_SUFFIX,
    RAW_CHANNELS,
    SENSOR_CHANNELS,
    apply_global_stats,
    apply_minmax_stats,
    fit_minmax_stats,
    resample_recording,
)
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_alignment,
    aggregate_dual_mode,
    append_summary_row,
    build_summary_row,
    evaluate_recording,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.postprocess import ENVELOPE_CHANNELS, FS, compute_envelope
from chewing_ssl.common.evaluation.reporting import print_alignment_summary, print_summary_table
from chewing_ssl.ssl.masked.evaluate_bouts import infer_recording, load_finetuned_model

DEFAULT_CLASSICAL_ML_MODEL = Path("checkpoints/train_ml_80_10_10/best_model.joblib")
DEFAULT_SSL_CHECKPOINT = Path("checkpoints/finetune_masked_80_10_10/span_bigru_w200_best.pt")
DEFAULT_GLOBAL_STATS = Path("data/processed_80_10_10/global_stats.json")
DEFAULT_FEATURE_MEDIANS = Path("data/processed_80_10_10/ml_features/feature_medians.json")

# Cross-sensor ablation on ALREADY-TRAINED models (no retraining): --only nav
# or --only accel zeroes the other modality's raw + derivative columns before
# either pipeline sees them, so both models can be probed for how much they
# rely on each modality without touching segment.py/extract_features.py/
# pretrain.py/finetune.py at all.
MODALITY_CHANNELS = {
    "nav": ["nav_x_left_temple", "nav_y_left_temple"],
    "accel": list(ACCEL_COLS),
}


def zero_out_modality(df: pd.DataFrame, only: str) -> pd.DataFrame:
    """--only 'nav'/'accel' zeroes the OTHER modality's raw + _d1 derivative
    columns, in already-normalized space (0.0 == that channel's fitted mean
    for global z-score normalization, a legitimate 'no signal' value).
    --only 'all' (default) is a no-op."""
    if only == "all":
        return df
    keep = set(MODALITY_CHANNELS[only])
    exclude_raw = [c for c in RAW_CHANNELS if c not in keep]
    exclude_cols = exclude_raw + [f"{c}{DERIVATIVE_SUFFIX}" for c in exclude_raw]
    df = df.copy()
    df[exclude_cols] = 0.0
    return df


# ── Stage 1: shared preprocessing (both pipelines) ──────────────────────────


def preprocess_new_recording(
    dataset: str,
    input_path: Path,
    global_stats: dict | None,
    out_dir: Path,
    only: str = "all",
    normalization: str = "global",
) -> list[tuple[pd.DataFrame, dict]]:
    """Raw CSV -> list of (normalized_segment_df, meta) pairs, one per
    gap-bounded segment -- reuses preprocess.py's resample_recording (gap
    segmentation, resample to 50Hz, Butterworth filter, derivative channels)
    UNCHANGED, then normalizes one of two ways:

    `normalization="global"` (default): applies the ALREADY-FITTED
    `global_stats` (never refit on this new recording -- decision 003) via
    apply_global_stats, exactly as refit_global_norm.py does for every other
    recording in this project. `global_stats` is required in this mode.

    `normalization="per_file"`: fits fresh percentile-based min-max bounds
    from THIS recording's own gap-segments (pooled together), via
    fit_minmax_stats/apply_minmax_stats -- the exact same per-recording,
    no-cross-file-leakage convention normalize_per_recording.py uses for the
    training corpus. Unlike global normalization this has no train-split
    dependency at all, so it's procedurally identical whether the recording
    is in-corpus or genuinely new. `global_stats` is ignored in this mode.

    Writes each normalized segment to out_dir/interim_normalized/ too, both
    for debuggability and because the classical-ML path (segment_recording)
    needs a real file path to read from, not just an in-memory frame.
    """
    raw_segments = resample_recording(dataset, input_path)
    if not raw_segments:
        raise ValueError(
            f"{input_path} produced no usable segments (too short, or every "
            f"gap-bounded piece was under the minimum segment duration)."
        )

    norm_dir = out_dir / "interim_normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    if normalization == "per_file":
        minmax_stats = fit_minmax_stats([seg_df for seg_df, _ in raw_segments], SENSOR_CHANNELS)

    results = []
    for i, (seg_df, meta) in enumerate(raw_segments):
        if normalization == "per_file":
            normalized = apply_minmax_stats(seg_df, minmax_stats)
        else:
            normalized = apply_global_stats(seg_df, global_stats)
        normalized = zero_out_modality(normalized, only)
        has_label = "label" in normalized.columns
        if not has_label:
            # segment_recording (classical ML) requires a label column to
            # produce non-empty windows at all -- a genuinely unlabeled new
            # recording still needs *some* column there structurally. A
            # dummy all-zero label lets windowing/feature-extraction run
            # unchanged; evaluation is skipped separately based on
            # `has_label`, so this placeholder never affects any reported
            # metric.
            normalized = normalized.copy()
            normalized["label"] = 0
        seg_path = norm_dir / f"segment_{i:02d}.parquet"
        normalized.to_parquet(seg_path, index=False)
        results.append((seg_path, {**meta, "has_label": has_label}))
        print(
            f"  segment_{i:02d}: {len(normalized):,} samples "
            f"({meta['duration_s']:.1f}s)  ->  {seg_path}"
            + ("" if has_label else "  [no ground truth -- evaluation will be skipped]")
        )
    return results


# ── Stage 2a: classical ML prediction + upscaling ───────────────────────────


def predict_classical_ml(
    seg_path: Path,
    model_path: Path,
    feature_medians: pd.Series,
    window_len: int,
    families: list[str],
    fs: float,
    channels: list[str] = SENSOR_CHANNELS,
) -> dict:
    """One normalized segment -> {gt, probs, envelope} via the classical-ML
    path: materialize non-overlapping windows (segment_recording, same
    convention eval/test windows always use) -> hand-crafted features
    (extract_features) -> sanitize with the SAVED train-split medians (never
    refit here) -> predict_proba -> upscale (paint each window's single
    probability across the timesteps it covers; non-overlapping here so
    this is a plain broadcast, not an average). `channels` must match
    model_path's own training channels (see model_meta.json next to it)."""
    X, y, starts = segment_recording(seg_path, window_len, stride=window_len, channels=channels)
    df = pd.read_parquet(seg_path)
    gt = df["label"].to_numpy(dtype=bool)

    if len(y) == 0:
        probs = np.full(len(gt), np.nan, dtype=np.float32)
    else:
        features = extract_features(X, families, fs, parallel=False, n_jobs=1, channels=channels)
        features, _ = sanitize(features, feature_medians)
        pipeline = joblib.load(model_path)
        window_probs = pipeline.predict_proba(features)[:, 1]
        upscaled = upscale_recording(window_probs, np.array(starts), window_len)
        probs = np.full(len(gt), np.nan, dtype=np.float32)
        probs[upscaled["row_idx"].to_numpy()] = upscaled["pred_prob"].to_numpy()

    signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
    envelope = compute_envelope(signal_mag, fs=FS)
    return {"gt": gt, "probs": probs, "envelope": envelope}


# ── Stage 2b: masked-SSL prediction (no upscaling needed) ──────────────────


def predict_masked_ssl(
    seg_path: Path, model, window_len: int, channels: list[str] = SENSOR_CHANNELS
) -> dict:
    """One normalized segment -> {gt, probs, envelope} via the masked-SSL
    path: the model predicts one probability per TIMESTEP directly
    (ChewingDetector.forward returns (B, T)) by sliding non-overlapping
    windows over the raw (C, T) tensor -- no separate upscaling step
    exists for this pipeline, this per-timestep output already is the
    full-resolution trace. `channels` must match the checkpoint's own
    training channels (auto-detected from its metadata by the caller)."""
    df = pd.read_parquet(seg_path)
    x = df[channels].to_numpy(dtype=np.float32).T
    gt = df["label"].to_numpy(dtype=bool)
    probs = infer_recording(model, x, window_len=window_len)

    signal_mag = np.sqrt((df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1))
    envelope = compute_envelope(signal_mag, fs=FS)
    return {"gt": gt, "probs": probs, "envelope": envelope}


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["classical_ml", "masked_ssl"],
        help="Which pipeline's best 80/10/10 model to run inference with.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=None,
        help="One or more raw recording CSV(s) to run inference on. Mutually exclusive "
        "with --from-split.",
    )
    parser.add_argument(
        "--from-split",
        type=Path,
        default=None,
        help="Path to a splits.json; auto-discovers --dataset's raw files for --purpose "
        "(via chewing_ssl.common.data.eda.discover_recordings) instead of --input. Since "
        "splits.json spans multiple source datasets each with their own column convention, "
        "this still runs one --dataset at a time.",
    )
    parser.add_argument(
        "--purpose",
        choices=["train", "val", "test"],
        default="test",
        help="Only used with --from-split.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Only used with --from-split. Root passed to discover_recordings -- e.g. "
        "data/raw_unscaled/raw_unscaled for the 5 datasets whose original raw CSVs live "
        "there (data/raw/<dataset> no longer has them for most of the 5).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_SPECS),
        help="Which raw column-naming convention this file follows (selects the "
        "loader's column-mapping table) -- a new recording still needs to be in "
        "one of the project's known raw formats.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "nav", "accel"],
        default="all",
        help="Cross-sensor ablation on the ALREADY-TRAINED model: zero out the other "
        "modality's channels before prediction. 'all' (default) is the normal, unablated "
        "run.",
    )
    parser.add_argument(
        "--normalization",
        choices=["global", "per_file"],
        default="global",
        help="'global' (default): apply the already-fitted --global-stats (never refit, "
        "decision 003) -- required for a checkpoint trained on globally-normalized data. "
        "'per_file': fit fresh percentile-based min-max bounds from THIS recording's own "
        "segments (no train-split dependency at all, same convention "
        "normalize_per_recording.py uses) -- required for a checkpoint trained on "
        "per-file-normalized data. --global-stats is ignored in this mode.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--classical-ml-model-path", type=Path, default=DEFAULT_CLASSICAL_ML_MODEL)
    parser.add_argument("--ssl-checkpoint", type=Path, default=DEFAULT_SSL_CHECKPOINT)
    parser.add_argument("--global-stats", type=Path, default=DEFAULT_GLOBAL_STATS)
    parser.add_argument("--feature-medians", type=Path, default=DEFAULT_FEATURE_MEDIANS)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Classical ML only. Default: configs/preprocessing.yaml (segment.py's own default).",
    )
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write the per-bite ledger here (only produced if the recording has "
        "ground truth). Default: <out-dir>/bout_ledger.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional: append this run's aggregate dual-mode metrics as one row "
        "(schema shared with classical_ml/evaluate_bouts.py and "
        "ssl/masked/evaluate_bouts.py) -- only produced if at least one recording "
        "has ground truth.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Identity for --summary-csv's row. Defaults to --out-dir's name.",
    )
    args = parser.parse_args()
    argv_dests = [a.split("=")[0] for a in __import__("sys").argv]

    if (args.input is None) == (args.from_split is None):
        parser.error("Exactly one of --input or --from-split is required.")
    if args.input is None:
        splits = json.loads(args.from_split.read_text())
        if args.dataset not in splits:
            parser.error(f"--dataset {args.dataset!r} not found in {args.from_split}")
        wanted_tags = {
            tag for tag, purpose in splits[args.dataset].items() if purpose == args.purpose
        }
        discovered = discover_recordings(args.dataset, args.raw_root)
        args.input = [path for path, tag in discovered if tag in wanted_tags]
        print(
            f"  --from-split: {len(args.input)} {args.dataset} recording(s) for purpose={args.purpose!r} "
            f"(raw-root={args.raw_root})"
        )

    # smooth_window_s's validated default differs by pipeline (0.0 classical
    # ML, already smoothed by upscaling's averaging; 2.0s masked SSL, which
    # has no equivalent) -- same override-the-shared-default pattern used in
    # ssl/masked/evaluate_bouts.py, only applied here iff the user didn't
    # pass --smooth-window-s explicitly themselves.
    if "--smooth-window-s" not in argv_dests:
        for action in parser._actions:
            if action.dest == "smooth_window_s" and args.model == "masked_ssl":
                args.smooth_window_s = 2.0

    # ENVELOPE_CHANNELS (postprocess.py) is nav-only -- under --only accel, nav
    # is zeroed, so the raw-signal envelope-split step would run on a
    # degenerate all-zero envelope and silently confound the accel-only
    # result. Disable it, unless the user explicitly passed the flag
    # themselves (same override-iff-not-explicit pattern as smooth_window_s
    # above).
    envelope_flag_passed = any(d in argv_dests for d in ("--envelope-split", "--no-envelope-split"))
    if args.only == "accel" and not envelope_flag_passed:
        args.envelope_split = False
        print("  --only accel: --envelope-split forced off (ENVELOPE_CHANNELS is nav-only)")

    cfg = postprocess_config_from_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n{'=' * 70}\n  Inference on new recording(s) -- {args.model}  (--only {args.only})\n{'=' * 70}"
    )
    print(
        f"  Postprocessing: hysteresis({cfg.high}/{cfg.low})  gap={cfg.max_gap_s}s  "
        f"dur=[{cfg.min_dur_s},{cfg.max_dur_s}]s  valley={cfg.valley_thresh}  "
        f"envelope-split={cfg.envelope_split}  smooth={cfg.smooth_window_s}s"
    )

    global_stats = (
        json.loads(args.global_stats.read_text()) if args.normalization == "global" else None
    )

    ssl_model, ssl_meta, window_len = None, None, None
    feature_medians = None
    model_channels = SENSOR_CHANNELS
    if args.model == "masked_ssl":
        ssl_model, ssl_meta = load_finetuned_model(args.ssl_checkpoint)
        ssl_model.eval()
        window_len = ssl_meta["window_len"]
        model_channels = ssl_meta["channels"]
        print(
            f"  Loaded checkpoint: {args.ssl_checkpoint}  "
            f"(architecture={ssl_meta['architecture']}, pretext_task={ssl_meta['pretext_task']}, "
            f"window={window_len} samples, val_f1={ssl_meta['val_f1']:.3f}, "
            f"channels={len(model_channels)})"
        )
    else:
        window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN
        feature_medians = pd.Series(json.loads(args.feature_medians.read_text()))
        model_meta_path = args.classical_ml_model_path.parent / "model_meta.json"
        if model_meta_path.exists():
            model_channels = (
                json.loads(model_meta_path.read_text()).get("channels") or SENSOR_CHANNELS
            )
        print(
            f"  Loaded model: {args.classical_ml_model_path}  (window={window_len} samples, "
            f"channels={len(model_channels)})"
        )

    all_results = []
    recordings_with_gt = []
    for input_path in args.input:
        print(f"\n--- {input_path} (dataset={args.dataset}) ---")
        rec_out_dir = args.out_dir / input_path.stem
        segments = preprocess_new_recording(
            args.dataset,
            input_path,
            global_stats,
            rec_out_dir,
            only=args.only,
            normalization=args.normalization,
        )

        for seg_path, meta in segments:
            if args.model == "masked_ssl":
                rec = predict_masked_ssl(seg_path, ssl_model, window_len, model_channels)
            else:
                rec = predict_classical_ml(
                    seg_path,
                    args.classical_ml_model_path,
                    feature_medians,
                    window_len,
                    list(DRIVERS),
                    50.0,
                    model_channels,
                )
            rec["participant"] = f"{input_path.stem}/{seg_path.stem}"
            rec["dataset"], rec["tag"], rec["segment"] = (
                args.dataset,
                input_path.stem,
                seg_path.stem,
            )
            rec["_has_label"] = meta["has_label"]

            binary = None
            if meta["has_label"]:
                result = evaluate_recording(
                    rec, cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
                )
                result["_tolerance_s"], result["_iou_thresh"], result["_min_dur"] = (
                    args.tolerance_s,
                    args.iou_thresh,
                    args.min_bout_dur,
                )
                all_results.append(result)
                recordings_with_gt.append(
                    {
                        "dataset": rec["dataset"],
                        "participant": rec["participant"],
                        "tag": rec["tag"],
                        "segment": rec["segment"],
                    }
                )
                print(
                    f"  {seg_path.stem}: GT={result['n_gt_bouts']} bouts  Pred={result['n_pred_bouts']} bouts  "
                    f"TP={result['tp']} FP={result['fp']} FN={result['fn']}  F1={result['f1']:.3f}"
                )
                binary = result["binary"]
            else:
                binary = None
                from chewing_ssl.common.evaluation.postprocess import apply_full_postprocessing

                binary = apply_full_postprocessing(rec["probs"], rec.get("envelope"), cfg)
                pred_bouts = extract_bouts(binary, fs=FS, min_dur=args.min_bout_dur)
                print(
                    f"  {seg_path.stem}: {len(pred_bouts)} predicted bout(s), no ground truth to evaluate against:"
                )
                for b in pred_bouts:
                    print(
                        f"    [{b['start_s']:.2f}s, {b['end_s']:.2f}s]  duration={b['duration_s']:.2f}s"
                    )
                bouts_path = rec_out_dir / f"{seg_path.stem}_predicted_bouts.csv"
                pd.DataFrame(pred_bouts).to_csv(bouts_path, index=False)
                print(f"    -> {bouts_path}")

    if all_results:
        print(f"\n{'=' * 70}\n  Aggregate (recordings with ground truth only)\n{'=' * 70}")
        print_summary_table(all_results, match_mode=args.match_mode)
        print_alignment_summary(aggregate_alignment(all_results))
        dual = aggregate_dual_mode(all_results)
        print(
            f"\n  bout_iou_f1={dual['iou']['f1']:.3f} (tp={dual['iou']['tp']:.0f} fp={dual['iou']['fp']:.0f} "
            f"fn={dual['iou']['fn']:.0f})  bout_tolerance_f1={dual['tolerance']['f1']:.3f} "
            f"(tp={dual['tolerance']['tp']:.0f} fp={dual['tolerance']['fp']:.0f} fn={dual['tolerance']['fn']:.0f})  "
            f"(n_gt_bouts={dual['n_gt_bouts']}, n_pred_bouts={dual['n_pred_bouts']})"
        )

        out_csv = args.out_csv or (args.out_dir / "bout_ledger.csv")
        ledger_rows = []
        from chewing_ssl.common.evaluation.pipeline import ledger_rows_for_recording

        for ident, result in zip(recordings_with_gt, all_results):
            ledger_rows += ledger_rows_for_recording(ident, result, args.match_mode)
        pd.DataFrame(ledger_rows).to_csv(out_csv, index=False)
        print(f"\n  Per-bite ledger -> {out_csv}")

        if args.summary_csv:
            row = build_summary_row(
                {
                    "run_name": args.run_name or args.out_dir.name,
                    "model_type": args.model,
                    "normalization": args.normalization,
                    "only": args.only,
                },
                dual,
            )
            append_summary_row(args.summary_csv, row)
            print(f"  Summary row -> {args.summary_csv}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
