from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chewing_ssl.cross_sensor_evaluation.model_grid import CLASSICAL_ML, MASKED_SSL, RAW_INTERIM

OUT_ROOT = Path("checkpoints/cross_sensor_evaluation/native_test")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def run_classical_ml(key: str, model_path: Path, features_path: Path) -> None:
    out_dir = OUT_ROOT / key
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nEVAL  {key}\n{'=' * 60}")
    _run([
        "chewing_ssl.classical_ml.evaluate_bouts",
        "--model-path", str(model_path),
        "--features-path", str(features_path),
        "--interim-root", str(RAW_INTERIM),
        "--out-csv", str(out_dir / "bout_ledger.csv"),
        "--report-path", str(out_dir / "eval_report.md"),
        "--summary-csv", str(out_dir / "summary_row.csv"),
        "--run-name", key,
    ])
    print(f"PLOT  {key}")
    _run([
        "chewing_ssl.classical_ml.plot_predictions",
        "--model-path", str(model_path),
        "--features-path", str(features_path),
        "--interim-root", str(RAW_INTERIM),
        "--all",
        "--out-dir", str(out_dir / "plots"),
    ])


def run_masked_ssl(key: str, checkpoint: Path, interim_root: Path, processed_root: Path) -> None:
    out_dir = OUT_ROOT / key
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nEVAL  {key}\n{'=' * 60}")
    _run([
        "chewing_ssl.ssl.masked.evaluate_bouts",
        "--checkpoint", str(checkpoint),
        "--interim-root", str(interim_root),
        "--processed-root", str(processed_root),
        "--pred-dir", str(out_dir / "preds"),
        "--out-csv", str(out_dir / "bout_ledger.csv"),
        "--report-path", str(out_dir / "eval_report.md"),
        "--summary-csv", str(out_dir / "summary_row.csv"),
    ])
    print(f"PLOT  {key}")
    _run([
        "chewing_ssl.ssl.masked.plot_predictions",
        "--checkpoint", str(checkpoint),
        "--interim-root", str(interim_root),
        "--raw-interim-root", str(RAW_INTERIM),
        "--processed-root", str(processed_root),
        "--all",
        "--out-dir", str(out_dir / "plots"),
    ])


def main() -> None:
    for key, (model_path, features_path, _medians, _norm) in CLASSICAL_ML.items():
        run_classical_ml(key, model_path, features_path)
    for key, (checkpoint, interim_root, processed_root, _norm) in MASKED_SSL.items():
        run_masked_ssl(key, checkpoint, interim_root, processed_root)
    print("\nALL NATIVE-TEST EVAL+PLOT RUNS DONE")


if __name__ == "__main__":
    main()
