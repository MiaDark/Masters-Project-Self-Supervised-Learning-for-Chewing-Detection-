from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_dual_mode,
    append_summary_row,
    evaluate_recording,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.postprocess import FS, compute_envelope
from chewing_ssl.ssl.masked.evaluate_bouts import ENVELOPE_CHANNELS, load_finetuned_model
from chewing_ssl.uncertainty.calibration import reliability_diagram
from chewing_ssl.uncertainty.decompose import infer_config_label
from chewing_ssl.uncertainty.mc_dropout import enable_mc_dropout, infer_recording_mc


def main() -> None:
    parser = argparse.ArgumentParser(description="MC-Dropout uncertainty evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--interim-root", type=Path, default=Path("data/interim_normalized_80_10_10")
    )
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed_80_10_10"))
    parser.add_argument("--dataset", default=None, help="Restrict to one dataset. Default: all.")
    parser.add_argument(
        "--mc-passes", type=int, default=30, help="Stochastic forward passes per window (T)."
    )
    parser.add_argument("--n-bins", type=int, default=15, help="Reliability-diagram bin count.")
    parser.add_argument("--device", default=None)
    add_postprocess_args(parser)
    # Match evaluate_bouts.py's SSL-specific override (see that module's
    # docstring): this pipeline's per-timestep trace has no equivalent to
    # classical ML's overlap-averaging smoothing.
    for action in parser._actions:
        if action.dest == "smooth_window_s":
            action.default = 2.0
    add_match_args(parser, default_match_mode="iou")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/uncertainty"))
    parser.add_argument(
        "--summary-csv", type=Path, default=Path("checkpoints/uncertainty_summary.csv")
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Identity for --summary-csv's row. Default: auto-generated from checkpoint metadata -- "
        "override when evaluating several checkpoints that share architecture/pretext_task/window "
        "(e.g. the nav-only/accel-only/all-channel variants), which would otherwise collide.",
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg = postprocess_config_from_args(args)

    model, meta = load_finetuned_model(args.checkpoint)
    model.to(device)
    enable_mc_dropout(model)
    label = infer_config_label(args.checkpoint)
    run_name = args.run_name or (
        f"{meta['pretext_task']}_{meta['architecture']}_w{meta['window_len']}_mc{args.mc_passes}"
    )
    window_len = meta["window_len"]

    print(f"\n{'=' * 60}")
    print(f"  MC-Dropout uncertainty evaluation  [{run_name}]")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Norm/channels : {label['norm']} / {label['channels']}")
    print(f"  Passes (T)    : {args.mc_passes}")
    print(f"  Device        : {device}")
    print(f"{'=' * 60}\n")

    idx = pd.read_parquet(args.processed_root / "index" / "finetune_test.parquet")
    if args.dataset:
        idx = idx[idx["dataset"] == args.dataset]
    recs = sorted(idx[["dataset", "tag"]].drop_duplicates().itertuples(index=False))

    results = []
    all_probs, all_labels, all_mi, all_entropy = [], [], [], []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ds, tag in recs:
        seg_dir = args.interim_root / ds / tag
        for seg_path in sorted(seg_dir.glob("segment_*.parquet")):
            df = pd.read_parquet(seg_path)
            if "label" not in df.columns:
                continue
            x = df[meta["channels"]].to_numpy(dtype=np.float32).T
            y = df["label"].to_numpy(dtype=np.float32)
            participant = f"{ds}/{tag}/{seg_path.stem}"
            print(f"  {participant} ...", end=" ", flush=True)

            mc = infer_recording_mc(
                model, x, window_len=window_len, n_passes=args.mc_passes, device=device
            )

            envelope = None
            if cfg.envelope_split:
                signal_mag = np.sqrt(
                    (df[ENVELOPE_CHANNELS].to_numpy(dtype=np.float64) ** 2).sum(axis=1)
                )
                envelope = compute_envelope(signal_mag, fs=FS, window_s=cfg.envelope_window_s)

            rec = {
                "participant": participant,
                "gt": y.astype(bool),
                "probs": mc["mean"],
                "envelope": envelope,
            }
            bout_result = evaluate_recording(
                rec, cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
            )
            bout_result["dataset"], bout_result["tag"], bout_result["segment"] = (
                ds,
                tag,
                seg_path.stem,
            )
            results.append(bout_result)

            covered = ~np.isnan(mc["mean"])
            all_probs.append(mc["mean"][covered])
            all_labels.append(y[covered])
            all_mi.append(mc["mutual_info"][covered])
            all_entropy.append(mc["entropy"][covered])

            np.savez(
                args.out_dir / f"{run_name}_{participant.replace('/', '_')}.npz",
                mean=mc["mean"],
                std=mc["std"],
                entropy=mc["entropy"],
                exp_entropy=mc["exp_entropy"],
                mutual_info=mc["mutual_info"],
                label=y,
            )
            print(
                f"GT={bout_result['n_gt_bouts']} bouts  Pred={bout_result['n_pred_bouts']} bouts  "
                f"F1={bout_result['f1']:.3f}  mean_std={np.nanmean(mc['std']):.4f}  "
                f"mean_MI={np.nanmean(mc['mutual_info']):.4f}"
            )

    # -- Bout-level sanity check (MC-mean vs. evaluate_bouts.py's single deterministic pass) --
    dual = aggregate_dual_mode(results)
    print(f"\n{'-' * 60}\n  Bout-level (from MC-mean probability trace, {len(results)} recordings)")
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
        "pipeline": "masked_ssl",
        "norm": label["norm"],
        "channels": label["channels"],
        "checkpoint": str(args.checkpoint),
        "mc_passes": args.mc_passes,
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
    print(f"  Per-recording arrays (mean/std/entropy/mutual_info/label) -> {args.out_dir}/*.npz")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
