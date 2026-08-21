"""Bout-by-bout evaluation of the classical-ML pipeline.

Also produces a per-bout ledger (--out-csv): every ground-truth bout tagged
detected/missed, every predicted bout tagged true-positive/false-alarm.

This script owns exactly ONE pipeline-specific step: computing each
recording's probability trace FRESH from `--model-path`
(`chewing_ssl.classical_ml.upscale_predictions.compute_upscaled_recordings`)
-- there is no cached prediction artifact read from disk (there used to be;
see that module's docstring for why that was a reproducibility risk). Every
run of this script reflects exactly whatever model `--model-path` currently
points at. Everything from postprocessing onward -- applying the shared
recipe, matching bouts under both IoU and tolerance modes, building the
per-bite ledger, and writing the results-summary row -- is shared code
(`chewing_ssl.common.evaluation.postprocess`/`pipeline`), identical to what
`chewing_ssl.ssl.masked.evaluate_bouts` and
`chewing_ssl.common.evaluation.compare_bites` use, so the two pipelines'
numbers are directly comparable and there is only one implementation of the
postprocessing/matching logic to maintain.

Matching (--match-mode, default "tolerance"):
  tolerance  A predicted bout matches a ground-truth bout when its onset AND
             offset are both within --tolerance-s seconds of the ground
             truth's -- "detected the same event at roughly the right time",
             not "detected pixel-perfect boundaries". Default --tolerance-s
             is 5.0, NOT the 2.0s a literal reading of "~100 rows" would
             suggest -- see docs/classical_ml.md §6a: because IoU's
             tolerance for a shift scales with bout length (median GT bout
             here is 10.9s) but this tolerance is a fixed absolute cap, 2.0s
             is actually stricter than IoU>=0.3 for bouts this long, not
             more lenient.
  iou        Strict-overlap matching: IoU(pred, gt) >= --iou-thresh. Combined
             with the duration-bounded postprocessing below, this is the
             best-measured combination so far (F1 0.458 -> 0.763, see §6a/§6c).
Both modes are always computed (see --summary-csv) regardless of which one
this flag selects for the console/ledger display above.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chewing_ssl.classical_ml.upscale_predictions import compute_upscaled_recordings
from chewing_ssl.common.data.windowing import TARGET_HZ, WINDOW_LEN
from chewing_ssl.common.evaluation.bouts import extract_bouts, match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.pipeline import (
    add_match_args,
    add_postprocess_args,
    aggregate_alignment,
    aggregate_dual_mode,
    append_summary_row,
    build_summary_row,
    evaluate_recording,
    ledger_rows_for_recording,
    postprocess_config_from_args,
)
from chewing_ssl.common.evaluation.postprocess import FS
from chewing_ssl.common.evaluation.reporting import (
    capture_console_report,
    print_alignment_summary,
    print_worst_offenders,
    save_console_report,
)
from chewing_ssl.common.evaluation.reporting import fmt as _fmt


def evaluate_raw(
    recordings: list[dict],
    threshold: float,
    min_bout_dur: float,
    match_mode: str,
    iou_thresh: float,
    tolerance_s: float,
) -> dict:
    """Raw (no postprocessing) threshold-only baseline, for the "does
    postprocessing actually help" comparison this script prints -- kept
    local rather than in the shared pipeline module since it's a one-off
    narrative specific to this script, not part of the shared recipe."""
    match_fn = (
        (lambda gt, pred: match_bouts_tolerant(gt, pred, tolerance_s=tolerance_s))
        if match_mode == "tolerance"
        else (lambda gt, pred: match_bouts(gt, pred, iou_thresh=iou_thresh))
    )
    tp = fp = fn = 0
    n_gt_bouts = n_pred_bouts = 0
    ious: list[float] = []
    for rec in recordings:
        binary = np.nan_to_num(rec["probs"], nan=0.0) >= threshold
        gt_bouts = extract_bouts(rec["gt"], fs=FS, min_dur=min_bout_dur)
        pred_bouts = extract_bouts(binary, fs=FS, min_dur=min_bout_dur)
        r = match_fn(gt_bouts, pred_bouts)
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
        ious += [m["iou"] for m in r["matches"]]
        n_gt_bouts += len(gt_bouts)
        n_pred_bouts += len(pred_bouts)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec_ = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec_,
        "f1": f1,
        "median_iou": float(np.median(ious)) if ious else float("nan"),
        "n_gt_bouts": n_gt_bouts,
        "n_pred_bouts": n_pred_bouts,
    }


def print_result(name: str, r: dict) -> None:
    print(
        f"  {name:<32s} prec={_fmt(r['precision'])}  rec={_fmt(r['recall'])}  f1={_fmt(r['f1'])}  "
        f"median_iou={_fmt(r['median_iou'])}  "
        f"GT={r['n_gt_bouts']} pred={r['n_pred_bouts']} (tp={r['tp']} fp={r['fp']} fn={r['fn']})"
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Raw (no postprocessing) binarization threshold.",
    )
    add_postprocess_args(parser)
    add_match_args(parser, default_match_mode="tolerance")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default="bout_ledger.csv",
        help="Write the full per-bout ledger (one row per GT/predicted bout, tagged "
        "detected/missed/false_alarm) here, e.g. checkpoints/ml_bout_ledger.csv. "
        "Uses the POSTPROCESSED predictions.",
    )
    parser.add_argument(
        "--top-n", type=int, default=15, help="Rows in the worst-offenders console table."
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Identity for --summary-csv's row, e.g. 'rf_4s' / 'rf_2s'. Defaults to --model-path's stem.",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        help="Recorded in --summary-csv for reference (e.g. 'random_forest').",
    )
    parser.add_argument(
        "--window-metrics-json",
        type=Path,
        default=None,
        help="Optional path to train.py's *_metrics.json, merged into --summary-csv's "
        "window_test_* columns -- best-effort cross-reference.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("checkpoints/classical_ml_summary.csv"),
        help="Append one row here every run -- the classical-ML analogue of "
        "checkpoints/masked_ablation_summary.csv. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Save the exact console report (per-recording table, glossary, alignment "
        "breakdown, worst offenders) to this markdown file. Default: "
        "checkpoints/<run-name>_eval_report.md. Pass an empty string to skip.",
    )
    args = parser.parse_args()

    run_name = args.run_name or args.model_path.stem
    report_path = args.report_path
    if report_path is None:
        report_path = Path("checkpoints") / f"{run_name}_eval_report.md"

    cfg = postprocess_config_from_args(args)
    window_len = int(args.window_seconds * TARGET_HZ) if args.window_seconds else WINDOW_LEN

    report_ctx = capture_console_report() if report_path else contextlib.nullcontext(None)
    with report_ctx as report_buffer:
        _run_evaluation(args, cfg, window_len, run_name)

    if report_path and report_buffer is not None:
        save_console_report(
            report_buffer, report_path, title=f"Bout-level Evaluation -- {run_name}"
        )
        print(f"  Full console report  →  {report_path}")


def _run_evaluation(args, cfg, window_len, run_name) -> None:
    print("\nBout-level evaluation: raw vs. postprocessed (classical-ML pipeline)")
    print("=" * 78)
    print(f"  Computing predictions fresh from {args.model_path} ...")
    recordings = compute_upscaled_recordings(
        args.model_path,
        args.features_path,
        args.interim_root,
        window_len,
        envelope_window_s=args.envelope_window_s,
    )
    print(
        f"  Recordings: {len(recordings)}   Match: {args.match_mode} "
        f"({'tolerance=' + str(args.tolerance_s) + 's' if args.match_mode == 'tolerance' else 'IoU>=' + str(args.iou_thresh)})   "
        f"min bout: {args.min_bout_dur}s   envelope-split: {cfg.envelope_split}   smooth: {cfg.smooth_window_s}s\n"
    )

    raw_result = evaluate_raw(
        recordings,
        args.threshold,
        args.min_bout_dur,
        args.match_mode,
        args.iou_thresh,
        args.tolerance_s,
    )
    post_results = [
        evaluate_recording(
            rec, cfg, args.min_bout_dur, args.match_mode, args.iou_thresh, args.tolerance_s
        )
        for rec in recordings
    ]
    post_dual = aggregate_dual_mode(post_results)
    post_display = post_dual[args.match_mode]
    post_display = {
        **post_display,
        "n_gt_bouts": post_dual["n_gt_bouts"],
        "n_pred_bouts": post_dual["n_pred_bouts"],
    }

    print_result("raw (threshold only)", raw_result)
    print_result("postprocessed (hysteresis+gapfill+splits)", post_display)

    delta_f1 = post_display["f1"] - raw_result["f1"]
    verdict = "HELPS" if delta_f1 > 0.01 else ("HURTS" if delta_f1 < -0.01 else "NO EFFECT")
    print(
        f"\n  Postprocessing (high={cfg.high}/low={cfg.low}, gap={cfg.max_gap_s}s, "
        f"dur=[{cfg.min_dur_s}, {cfg.max_dur_s}]s, valley={cfg.valley_thresh}, "
        f"envelope-split={cfg.envelope_split}, smooth={cfg.smooth_window_s}s) vs. raw: "
        f"F1 delta = {delta_f1:+.3f}  -> {verdict}"
    )

    ledger_rows = []
    for rec, result in zip(recordings, post_results):
        ident = {k: rec[k] for k in ("dataset", "participant", "tag", "segment")}
        ledger_rows += ledger_rows_for_recording(ident, result, args.match_mode)
    ledger = pd.DataFrame(ledger_rows)
    print_worst_offenders(ledger, top_n=args.top_n)
    print_alignment_summary(aggregate_alignment(post_results))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(args.out_csv, index=False)
        print(f"\n  Full per-bout ledger ({len(ledger)} rows) -> {args.out_csv}")

    if args.summary_csv:
        window_metrics = None
        if args.window_metrics_json and args.window_metrics_json.exists():
            raw = json.loads(args.window_metrics_json.read_text())
            window_metrics = raw.get("test", raw)
        row = build_summary_row(
            {"run_name": run_name, "model_type": args.model_type}, post_dual, window_metrics
        )
        append_summary_row(args.summary_csv, row)
        print(
            f"  Results summary row  ->  {args.summary_csv}  ({row['run_name']})  "
            f"bout_iou_f1={row['bout_iou_f1']:.3f}  bout_tolerance_f1={row['bout_tolerance_f1']:.3f}"
        )

    print("\n" + "=" * 78 + "\n")


if __name__ == "__main__":
    main()
