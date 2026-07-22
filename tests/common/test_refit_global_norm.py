from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS
from chewing_ssl.common.data.refit_global_norm import (
    apply_and_write_normalized,
    fit_train_only_global_stats,
    iter_segments,
)
from chewing_ssl.common.data.windowing import assign_splits, extract_participant


def test_iter_segments_filters_to_pretrain_eligible(synthetic_interim_tree):
    splits = assign_splits(synthetic_interim_tree, seed=42)
    all_segments = list(iter_segments(synthetic_interim_tree, splits, only_pretrain_eligible=False))
    eligible_only = list(iter_segments(synthetic_interim_tree, splits, only_pretrain_eligible=True))
    assert len(eligible_only) <= len(all_segments)
    assert len(eligible_only) > 0

    eligible_labels = {splits[ds][extract_participant(ds, tag)] for ds, tag, _, _ in eligible_only}
    assert eligible_labels <= {"train", "pretrain_only"}


def test_fit_train_only_global_stats_matches_manual_computation(synthetic_interim_tree):
    splits = assign_splits(synthetic_interim_tree, seed=42)
    stats = fit_train_only_global_stats(synthetic_interim_tree, splits, channels=["accel_x"])

    manual_values = []
    for _, _, path, _meta in iter_segments(
        synthetic_interim_tree, splits, only_pretrain_eligible=True
    ):
        df = pd.read_parquet(path, columns=["accel_x"])
        manual_values.append(df["accel_x"].to_numpy())
    expected_mean = np.concatenate(manual_values).mean()

    assert stats["accel_x"]["mean"] == pytest.approx(expected_mean)


def test_apply_and_write_normalized_writes_every_segment(synthetic_interim_tree, tmp_path):
    splits = assign_splits(synthetic_interim_tree, seed=42)
    stats = fit_train_only_global_stats(synthetic_interim_tree, splits)
    out_root = tmp_path / "interim_normalized"
    apply_and_write_normalized(synthetic_interim_tree, out_root, splits, stats)

    written = list(out_root.rglob("segment_00.parquet"))
    source = list(synthetic_interim_tree.rglob("segment_00.parquet"))
    assert len(written) == len(source)

    sample = pd.read_parquet(written[0])
    for c in SENSOR_CHANNELS:
        assert c in sample.columns


def test_apply_and_write_normalized_actually_normalizes(synthetic_interim_tree, tmp_path):
    splits = assign_splits(synthetic_interim_tree, seed=42)
    stats = fit_train_only_global_stats(synthetic_interim_tree, splits, channels=["accel_x"])
    out_root = tmp_path / "interim_normalized"
    apply_and_write_normalized(synthetic_interim_tree, out_root, splits, stats)

    written = next(out_root.rglob("segment_00.parquet"))
    source = synthetic_interim_tree / written.relative_to(out_root)

    df = pd.read_parquet(written, columns=["accel_x"])
    expected = (
        pd.read_parquet(source, columns=["accel_x"])["accel_x"] - stats["accel_x"]["mean"]
    ) / (stats["accel_x"]["std"] + 1e-8)
    assert np.allclose(df["accel_x"], expected)
