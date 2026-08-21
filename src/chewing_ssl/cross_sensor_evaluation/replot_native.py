from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chewing_ssl.cross_sensor_evaluation.model_grid import CLASSICAL_ML, MASKED_SSL, RAW_INTERIM

OUT_ROOT = Path("checkpoints/cross_sensor_evaluation/native_test")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True)


def main() -> None:
    for key, (model_path, features_path, _medians, _norm) in CLASSICAL_ML.items():
        print(f"PLOT  {key}")
        _run([
            "chewing_ssl.classical_ml.plot_predictions",
            "--model-path", str(model_path),
            "--features-path", str(features_path),
            "--interim-root", str(RAW_INTERIM),
            "--all",
            "--out-dir", str(OUT_ROOT / key / "plots"),
        ])

    for key, (checkpoint, interim_root, processed_root, _norm) in MASKED_SSL.items():
        print(f"PLOT  {key}")
        _run([
            "chewing_ssl.ssl.masked.plot_predictions",
            "--checkpoint", str(checkpoint),
            "--interim-root", str(interim_root),
            "--raw-interim-root", str(RAW_INTERIM),
            "--processed-root", str(processed_root),
            "--all",
            "--out-dir", str(OUT_ROOT / key / "plots"),
        ])

    print("\nALL NATIVE-TEST REPLOTS DONE")


if __name__ == "__main__":
    main()
