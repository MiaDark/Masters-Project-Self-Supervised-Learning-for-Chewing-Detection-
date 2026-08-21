"""The 12-model grid shared by every script in this package: {classical_ml,
masked_ssl} x {global-norm, per-file-norm} x {all, nav-only, accel-only
channels}. Kept in one place so the model list and checkpoint/feature paths
are defined once rather than repeated per script."""
from __future__ import annotations

from pathlib import Path

GLOBAL_INTERIM = Path("data/interim_normalized_80_10_10_clipped")
GLOBAL_PROCESSED = Path("data/processed_80_10_10_clipped")
PERFILE_INTERIM = Path("data/interim_normalized_per_file_clipped")
PERFILE_PROCESSED = Path("data/processed_80_10_10_per_file_clipped")

RAW_INTERIM = Path("data/interim_clipped")

# key -> (model_path, test_features_path, feature_medians_path, normalization)
CLASSICAL_ML = {
    "classical_ml_global_all": (
        Path("checkpoints/train_ml_80_10_10_clipped/best_model.joblib"),
        GLOBAL_PROCESSED / "ml_features/test.parquet",
        GLOBAL_PROCESSED / "ml_features/feature_medians.json",
        "global",
    ),
    "classical_ml_global_nav": (
        Path("checkpoints/train_ml_80_10_10_clipped_nav_only/best_model.joblib"),
        GLOBAL_PROCESSED / "ml_features_nav_only/test.parquet",
        GLOBAL_PROCESSED / "ml_features_nav_only/feature_medians.json",
        "global",
    ),
    "classical_ml_global_accel": (
        Path("checkpoints/train_ml_80_10_10_clipped_accel_only/best_model.joblib"),
        GLOBAL_PROCESSED / "ml_features_accel_only/test.parquet",
        GLOBAL_PROCESSED / "ml_features_accel_only/feature_medians.json",
        "global",
    ),
    "classical_ml_perfile_all": (
        Path("checkpoints/train_ml_80_10_10_per_file_clipped/best_model.joblib"),
        PERFILE_PROCESSED / "ml_features_all/test.parquet",
        PERFILE_PROCESSED / "ml_features_all/feature_medians.json",
        "per_file",
    ),
    "classical_ml_perfile_nav": (
        Path("checkpoints/train_ml_80_10_10_per_file_clipped_nav_only/best_model.joblib"),
        PERFILE_PROCESSED / "ml_features_nav/test.parquet",
        PERFILE_PROCESSED / "ml_features_nav/feature_medians.json",
        "per_file",
    ),
    "classical_ml_perfile_accel": (
        Path("checkpoints/train_ml_80_10_10_per_file_clipped_accel_only/best_model.joblib"),
        PERFILE_PROCESSED / "ml_features_accel/test.parquet",
        PERFILE_PROCESSED / "ml_features_accel/feature_medians.json",
        "per_file",
    ),
}

# key -> (checkpoint, interim_root, processed_root, normalization)
MASKED_SSL = {
    "masked_ssl_global_all": (
        Path("checkpoints/finetune_masked_80_10_10_clipped/span_bigru_w200_best.pt"),
        GLOBAL_INTERIM, GLOBAL_PROCESSED, "global",
    ),
    "masked_ssl_global_nav": (
        Path("checkpoints/finetune_masked_80_10_10_clipped_nav_only/span_bigru_w200_best.pt"),
        GLOBAL_INTERIM, GLOBAL_PROCESSED, "global",
    ),
    "masked_ssl_global_accel": (
        Path("checkpoints/finetune_masked_80_10_10_clipped_accel_only/span_bigru_w200_best.pt"),
        GLOBAL_INTERIM, GLOBAL_PROCESSED, "global",
    ),
    "masked_ssl_perfile_all": (
        Path("checkpoints/finetune_masked_80_10_10_per_file_clipped/span_bigru_w200_best.pt"),
        PERFILE_INTERIM, PERFILE_PROCESSED, "per_file",
    ),
    "masked_ssl_perfile_nav": (
        Path("checkpoints/finetune_masked_80_10_10_per_file_clipped_nav_only/span_bigru_w200_best.pt"),
        PERFILE_INTERIM, PERFILE_PROCESSED, "per_file",
    ),
    "masked_ssl_perfile_accel": (
        Path("checkpoints/finetune_masked_80_10_10_per_file_clipped_accel_only/span_bigru_w200_best.pt"),
        PERFILE_INTERIM, PERFILE_PROCESSED, "per_file",
    ),
}

# Grid order used consistently across eval/plot/summary scripts.
MODEL_KEYS = list(CLASSICAL_ML) + list(MASKED_SSL)
