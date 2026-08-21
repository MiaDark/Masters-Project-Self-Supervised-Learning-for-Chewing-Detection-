"""Consolidate all 24 (12 models x 2 scopes -- native_test, s11) evaluation
results into one checkpoints/cross_sensor_evaluation/summary.csv. Reads each
model/scope's own summary_row.csv directly (both run_native_eval.py and
run_s11_eval.py write the same build_summary_row schema), so no console-log
parsing is needed.

Run from the project root:
  python -m chewing_ssl.cross_sensor_evaluation.build_summary
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chewing_ssl.cross_sensor_evaluation.model_grid import MODEL_KEYS

ROOT = Path("checkpoints/cross_sensor_evaluation")
SCOPES = ["native_test", "s11"]

SUMMARY_COLUMNS = [
    "n_gt_bouts",
    "n_pred_bouts",
    "bout_iou_precision",
    "bout_iou_recall",
    "bout_iou_f1",
    "bout_iou_median_iou",
    "bout_tolerance_precision",
    "bout_tolerance_recall",
    "bout_tolerance_f1",
    "bout_tolerance_median_iou",
]


def main() -> None:
    rows = []
    for scope in SCOPES:
        for key in MODEL_KEYS:
            path = ROOT / scope / key / "summary_row.csv"
            if not path.exists():
                print(f"  [MISSING] {scope}/{key}: {path} not found (run eval first)")
                continue
            r = pd.read_csv(path).iloc[0]
            rows.append({"scope": scope, "model_key": key, **{c: r[c] for c in SUMMARY_COLUMNS}})

    out = pd.DataFrame(rows)
    out_path = ROOT / "summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows -> {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
