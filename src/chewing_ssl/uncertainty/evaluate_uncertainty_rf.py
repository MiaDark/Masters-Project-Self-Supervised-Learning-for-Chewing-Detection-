"""Tree-ensemble uncertainty evaluation for a trained classical-ML (Random
Forest) chewing detector -- the classical-ML counterpart of
uncertainty.evaluate_uncertainty (MC-Dropout, masked-reconstruction SSL).
Same three questions, same shared postprocessing/calibration code, so rows
from this script and evaluate_uncertainty.py land in ONE comparable
checkpoints/uncertainty_summary.csv (a `pipeline` column distinguishes
them, `norm`/`channels` -- auto-parsed from --model-path via
decompose.infer_config_label -- distinguish the 6 configs each way):

  1. Bout-level sanity check: does the ensemble-MEAN probability trace
     reproduce classical_ml/evaluate_bouts.py's numbers (already on record
     in checkpoints/classical_ml_summary_80_10_10.csv / the cross-sensor
     study's checkpoints/cross_sensor_evaluation/summary.csv)?
  2. Calibration (ECE + reliability diagram) of the ensemble-mean prob.
  3. Does tree disagreement (epistemic) / total entropy predict error?

Run from the project root:
  python -m chewing_ssl.uncertainty.evaluate_uncertainty_rf \
    --model-path checkpoints/train_ml_80_10_10_clipped/best_model.joblib \
    --features-path data/processed_80_10_10_clipped/ml_features/test.parquet \
    --interim-root data/interim_normalized_80_10_10_clipped
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from chewing_ssl.classical_ml.train import META_COLUMNS
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_dual_mode,
    append_summary_row,
    evaluate_recording,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.postprocess import ENVELOPE_CHANNELS, FS, compute_envelope
from chewing_ssl.uncertainty.calibration import reliability_diagram
from chewing_ssl.uncertainty.decompose import infer_config_label
from chewing_ssl.uncertainty.rf_uncertainty import tree_ensemble_predict, upscale_columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tree-ensemble uncertainty evaluation (classical ML)."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--features-path", type=Path, required=True)
    parser.add_argument("--interim-root", type=Path, required=True)
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Must match whatever segment.py --window-seconds built --features-path with. "
        "Default: configs/preprocessing.yaml's value (segment.py's own default).",
    )
    parser.add_argument("--n-bins", type=int, default=15, help="Reliability-diagram bin count.")
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/uncertainty"))
    parser.add_argument(
        "--summary-csv", type=Path, default=Path("checkpoints/uncertainty_summary.csv")
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN
    cfg = postprocess_config_from_args(args)
    label = infer_config_label(args.model_path)
    run_name = args.run_name or f"rf_{label['norm']}_{label['channels']}"

    print(f"\n{'=' * 60}")
    print(f"  Tree-ensemble uncertainty evaluation  [{run_name}]")
    print(f"  Model         : {args.model_path}")
    print(f"  Norm/channels : {label['norm']} / {label['channels']}")
    print(f"{'=' * 60}\n")

    pipeline = joblib.load(args.model_path)
    n_trees = len(pipeline.named_steps["model"].estimators_)
    df = pd.read_parquet(args.features_path)
    meta_cols = [c for c in META_COLUMNS if c in df.columns and c != "label"]
    X = df.drop(columns=[c for c in META_COLUMNS if c in df.columns])

    print(f"  Trees (ensemble size) : {n_trees}")
    print(f"  Windows                : {len(df):,}\n")

    mc = tree_ensemble_predict(pipeline, X)  # per-WINDOW mean/std/entropy/exp_entropy/mutual_info
    col_names = ["mean", "std", "entropy", "exp_entropy", "mutual_info"]
    df_meta = df[meta_cols].copy()
    for name in col_names:
        df_meta[name] = mc[name]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_probs, all_labels, all_mi, all_entropy = [], [], [], []

    for (dataset, participant, tag, segment), group in df_meta.groupby(
        ["dataset", "participant", "tag", "segment"]
    ):
        seg_path = args.interim_root / dataset / tag / f"{segment}.parquet"
        if not seg_path.exists():
            print(f"  [skip] no interim segment for {dataset}/{tag}/{segment}")
            continue

        upscaled = upscale_columns(
            {name: group[name].to_numpy() for name in col_names},
            group["start"].to_numpy(),
            window_len,
        )

        seg = pd.read_parquet(seg_path, columns=["label", *ENVELOPE_CHANNELS])
        y = seg["label"].to_numpy(dtype=np.float32)
        n = len(y)
        idx = upscaled["row_idx"]
        in_range = idx < n  # guard: predictions can't extend past the segment's real length

        def _scatter(name: str, n=n, idx=idx, in_range=in_range, upscaled=upscaled) -> np.ndarray:
            full = np.full(n, np.nan, dtype=np.float32)
            full[idx[in_range]] = upscaled[name][in_range]
            return full

        mean_full = _scatter("mean")
        participant_id = f"{dataset}/{tag}/{segment}"

        envelope = None
        if cfg.envelope_split:
            signal_mag = np.sqrt(
                (seg[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1)
            )
            envelope = compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)

        rec = {
            "participant": participant_id,
            "gt": y.astype(bool),
            "probs": mean_full,
            "envelope": envelope,
        }
        bout_result = evaluate_recording(
            rec, cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
        )
        bout_result["dataset"], bout_result["tag"], bout_result["segment"] = dataset, tag, segment
        results.append(bout_result)

        covered = ~np.isnan(mean_full)
        all_probs.append(mean_full[covered])
        all_labels.append(y[covered])
        all_mi.append(_scatter("mutual_info")[covered])
        all_entropy.append(_scatter("entropy")[covered])

        np.savez(
            args.out_dir / f"{run_name}_{participant_id.replace('/', '_')}.npz",
            mean=mean_full,
            std=_scatter("std"),
            entropy=_scatter("entropy"),
            exp_entropy=_scatter("exp_entropy"),
            mutual_info=_scatter("mutual_info"),
            label=y,
        )
        print(
            f"  {participant_id} ... GT={bout_result['n_gt_bouts']} bouts  "
            f"Pred={bout_result['n_pred_bouts']} bouts  F1={bout_result['f1']:.3f}  "
            f"mean_std={np.nanmean(_scatter('std')):.4f}  mean_MI={np.nanmean(_scatter('mutual_info')):.4f}"
        )

    # -- Bout-level sanity check (ensemble-mean vs. evaluate_bouts.py's point prediction) --
    dual = aggregate_dual_mode(results)
    print(
        f"\n{'-' * 60}\n  Bout-level (from ensemble-mean probability trace, {len(results)} recordings)"
    )
    print(
        f"  IoU F1        : {dual['iou']['f1']:.3f}  "
        f"(precision {dual['iou']['precision']:.3f}, recall {dual['iou']['recall']:.3f})"
    )
    print(
        f"  Tolerance F1  : {dual['tolerance']['f1']:.3f}  "
        f"(precision {dual['tolerance']['precision']:.3f}, recall {dual['tolerance']['recall']:.3f})"
    )

    # -- Calibration --
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    mi = np.concatenate(all_mi)
    entropy = np.concatenate(all_entropy)
    rel = reliability_diagram(probs, labels, n_bins=args.n_bins)
    print(f"\n{'-' * 60}\n  Calibration ({rel['n_total']:,} timesteps)")
    print(f"  ECE : {rel['ece']:.4f}")
    for b in range(args.n_bins):
        if rel["bin_count"][b] > 0:
            print(
                f"    [{rel['bin_edges'][b]:.2f},{rel['bin_edges'][b + 1]:.2f}) "
                f"n={rel['bin_count'][b]:>7,}  conf={rel['bin_confidence'][b]:.3f}  "
                f"acc={rel['bin_accuracy'][b]:.3f}"
            )

    # -- Does uncertainty predict error? --
    preds = (probs >= 0.5).astype(int)
    correct = preds == labels.astype(int)
    has_both = len(np.unique(correct)) > 1
    auroc_mi = float(roc_auc_score(~correct, mi)) if has_both else float("nan")
    auroc_ent = float(roc_auc_score(~correct, entropy)) if has_both else float("nan")
    print(f"\n{'-' * 60}\n  Uncertainty vs. correctness")
    print(
        f"  Mean epistemic (mutual_info) | correct={mi[correct].mean():.4f}  wrong={mi[~correct].mean():.4f}"
    )
    print(
        f"  Mean total (entropy)         | correct={entropy[correct].mean():.4f}  wrong={entropy[~correct].mean():.4f}"
    )
    print(f"  AUROC(mutual_info -> misclassified) : {auroc_mi:.3f}")
    print(f"  AUROC(entropy -> misclassified)     : {auroc_ent:.3f}")

    # -- Persist --
    summary = {
        "run_name": run_name,
        "pipeline": "classical_ml",
        "norm": label["norm"],
        "channels": label["channels"],
        "checkpoint": str(args.model_path),
        "n_trees": n_trees,
        "n_timesteps": int(rel["n_total"]),
        "bout_iou_f1": dual["iou"]["f1"],
        "bout_tolerance_f1": dual["tolerance"]["f1"],
        "ece": rel["ece"],
        "mean_mi_correct": float(mi[correct].mean()),
        "mean_mi_wrong": float(mi[~correct].mean()),
        "auroc_mi_vs_error": auroc_mi,
        "auroc_entropy_vs_error": auroc_ent,
    }
    (args.out_dir / f"{run_name}_summary.json").write_text(json.dumps(summary, indent=2))
    if args.summary_csv:
        append_summary_row(args.summary_csv, summary)
        print(f"\n  Summary row -> {args.summary_csv}")
    print(
        f"  Per-recording arrays (mean/std/entropy/mutual_info/label) -> {args.out_dir}/{run_name}_*.npz"
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
