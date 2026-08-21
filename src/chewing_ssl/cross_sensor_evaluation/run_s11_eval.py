from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chewing_ssl.cross_sensor_evaluation.model_grid import CLASSICAL_ML, MASKED_SSL

OUT_ROOT = Path("checkpoints/cross_sensor_evaluation/s11")
DATASET = "FEIT_eating_S11_RightTempleSubstitute_UNVALIDATED"
GLOBAL_STATS = Path("data/processed_80_10_10_clipped/global_stats.json")
FILES = sorted(Path("data/raw/s11_feit_eating").glob("*/*_eating.csv"))


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def _common_args(key: str, dir_: Path, normalization: str) -> list[str]:
    return [
        "--input",
        *[str(f) for f in FILES],
        "--dataset",
        DATASET,
        "--normalization",
        normalization,
        "--global-stats",
        str(GLOBAL_STATS),
        "--out-dir",
        str(dir_),
        "--out-csv",
        str(dir_ / "bout_ledger.csv"),
        "--summary-csv",
        str(dir_ / "summary_row.csv"),
        "--run-name",
        key,
    ]


def run_classical_ml(key: str, model_path: Path, medians_path: Path, normalization: str) -> None:
    dir_ = OUT_ROOT / key
    dir_.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nS11 EVAL  {key}  (normalization={normalization})\n{'=' * 60}")
    _run(
        [
            "chewing_ssl.predict_new_recording",
            "--model",
            "classical_ml",
            "--classical-ml-model-path",
            str(model_path),
            "--feature-medians",
            str(medians_path),
            *_common_args(key, dir_, normalization),
        ]
    )


def run_masked_ssl(key: str, checkpoint: Path, normalization: str) -> None:
    dir_ = OUT_ROOT / key
    dir_.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 60}\nS11 EVAL  {key}  (normalization={normalization})\n{'=' * 60}")
    _run(
        [
            "chewing_ssl.predict_new_recording",
            "--model",
            "masked_ssl",
            "--ssl-checkpoint",
            str(checkpoint),
            *_common_args(key, dir_, normalization),
        ]
    )


def main() -> None:
    if not FILES:
        raise SystemExit("No *_eating.csv files found under data/raw/s11_feit_eating")

    for key, (model_path, _test_features, medians_path, normalization) in CLASSICAL_ML.items():
        run_classical_ml(key, model_path, medians_path, normalization)
    for key, (checkpoint, _interim_root, _processed_root, normalization) in MASKED_SSL.items():
        run_masked_ssl(key, checkpoint, normalization)

    print("\nALL S11 EVAL RUNS DONE")


if __name__ == "__main__":
    main()
