"""Recover the 11 S11 summary_row.csv files that were missing from
checkpoints/cross_sensor_evaluation/s11/<model_key>/ (their bout_ledger.csv
and plots were intact, only the aggregate summary row was gone) -- WITHOUT
re-running the expensive raw-CSV inference for all 11 configs.

Reconstructs each recording's exact gt_bouts/pred_bouts from its existing
bout_ledger.csv (ledger_rows_for_recording's own inverse: "gt" rows give
every ground-truth bout directly; "pred","false_alarm" rows give every
UNMATCHED predicted bout directly; each MATCHED predicted bout's bounds are
recovered from its paired "gt","detected" row's matched_start_s/
matched_end_s -- tolerance-mode matching is one-to-one, so this recovers
the complete, exact predicted-bout list), then re-runs match_bouts (IoU)
and match_bouts_tolerant on that reconstructed pair of lists per recording,
aggregated via aggregate_dual_mode -- byte-for-byte the same computation
build_summary_row would have produced from a fresh run, since it operates
on the same underlying bout lists.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chewing_ssl.common.evaluation.bouts import match_bouts, match_bouts_tolerant
from chewing_ssl.common.evaluation.pipeline import aggregate_dual_mode, append_summary_row, build_summary_row
from chewing_ssl.cross_sensor_evaluation.model_grid import MODEL_KEYS

ROOT = Path("checkpoints/cross_sensor_evaluation/s11")
IOU_THRESH = 0.3
TOLERANCE_S = 5.0


def _stats(result: dict) -> dict:
    tp, fp, fn = result["tp"], result["fp"], result["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    ious = [m["iou"] for m in result["matches"]]
    median_iou = float(pd.Series(ious).median()) if ious else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "median_iou": median_iou}


def reconstruct_recording(group: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    gt_bouts = []
    pred_bouts = []
    for _, row in group.iterrows():
        if row["bout_type"] == "gt":
            gt_bouts.append(
                {"start_s": row["start_s"], "end_s": row["end_s"], "duration_s": row["duration_s"]}
            )
            if row["status"] == "detected":
                pred_bouts.append(
                    {
                        "start_s": row["matched_start_s"],
                        "end_s": row["matched_end_s"],
                        "duration_s": row["matched_end_s"] - row["matched_start_s"],
                    }
                )
        else:  # bout_type == "pred", status == "false_alarm" (only unmatched preds get their own row)
            pred_bouts.append(
                {"start_s": row["start_s"], "end_s": row["end_s"], "duration_s": row["duration_s"]}
            )
    return gt_bouts, pred_bouts


def main() -> None:
    for key in MODEL_KEYS:
        out_path = ROOT / key / "summary_row.csv"
        if out_path.exists():
            print(f"  [skip] {key}: summary_row.csv already present")
            continue
        ledger_path = ROOT / key / "bout_ledger.csv"
        if not ledger_path.exists():
            print(f"  [MISSING LEDGER] {key}: {ledger_path} not found -- cannot reconstruct")
            continue

        ledger = pd.read_csv(ledger_path)
        results = []
        for (dataset, participant, tag, segment), group in ledger.groupby(
            ["dataset", "participant", "tag", "segment"]
        ):
            gt_bouts, pred_bouts = reconstruct_recording(group)
            iou_result = match_bouts(gt_bouts, pred_bouts, iou_thresh=IOU_THRESH)
            tol_result = match_bouts_tolerant(gt_bouts, pred_bouts, tolerance_s=TOLERANCE_S)
            results.append(
                {
                    "participant": participant,
                    "n_gt_bouts": len(gt_bouts),
                    "n_pred_bouts": len(pred_bouts),
                    "iou_stats": _stats(iou_result),
                    "tolerance_stats": _stats(tol_result),
                }
            )

        dual = aggregate_dual_mode(results)
        row = build_summary_row({"run_name": key, "model_key": key}, dual)
        append_summary_row(out_path, row)
        n_gt = sum(r["n_gt_bouts"] for r in results)
        n_pred = sum(r["n_pred_bouts"] for r in results)
        print(
            f"  [rebuilt] {key}: n_gt={n_gt} n_pred={n_pred} "
            f"iou_f1={dual['iou']['f1']:.3f} tolerance_f1={dual['tolerance']['f1']:.3f} -> {out_path}"
        )


if __name__ == "__main__":
    main()
