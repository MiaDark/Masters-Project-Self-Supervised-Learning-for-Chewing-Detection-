from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chewing_ssl.common.data.preprocess import (
    BUTTER_CUTOFF_HZ,
    DERIVATIVE_CHANNELS,
    RAW_CHANNELS,
    SENSOR_CHANNELS,
    add_derivative_channels,
    apply_global_stats,
    apply_lowpass_filter,
    fit_global_stats,
    resample_recording,
    resample_segment,
    segment_at_gaps,
    select_columns,
)


def test_sensor_channels_is_five_raw_plus_five_derivative():
    assert RAW_CHANNELS == [
        "accel_x",
        "accel_y",
        "accel_z",
        "nav_x_left_temple",
        "nav_y_left_temple",
    ]
    assert len(RAW_CHANNELS) == 5
    assert DERIVATIVE_CHANNELS == [f"{c}_d1" for c in RAW_CHANNELS]
    assert SENSOR_CHANNELS == RAW_CHANNELS + DERIVATIVE_CHANNELS
    assert len(SENSOR_CHANNELS) == 10
    assert not any("prox" in c for c in SENSOR_CHANNELS)
    assert "nav_x_right_temple" not in SENSOR_CHANNELS
    assert "nav_y_right_temple" not in SENSOR_CHANNELS


def test_select_columns_drops_proximity_right_temple_and_keeps_label():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="20ms"),
            "nav_x_left_temple": np.zeros(5),
            "nav_x_right_temple": np.zeros(5),  # present in source, must be dropped
            "prox_left_cheek": np.zeros(5),  # present in source, must be dropped -- out of scope
            "accel_x": np.zeros(5),
            "label": np.zeros(5),
            "food_type": ["x"] * 5,  # not a tracked channel, must be dropped
        }
    )
    out = select_columns(df)
    assert "nav_x_right_temple" not in out.columns
    assert "prox_left_cheek" not in out.columns
    assert "food_type" not in out.columns
    assert {"datetime", "nav_x_left_temple", "accel_x", "label"}.issubset(out.columns)


def test_apply_lowpass_filter_attenuates_high_frequency_more_than_low():
    fs = 50.0
    t = np.arange(0, 10, 1 / fs)
    low = np.sin(2 * np.pi * 1.5 * t)  # near the chewing fundamental -- should survive
    high = np.sin(2 * np.pi * 20.0 * t)  # well above cutoff -- should be attenuated
    df = pd.DataFrame({"accel_x": low + high})

    out = apply_lowpass_filter(df, channels=["accel_x"], cutoff_hz=BUTTER_CUTOFF_HZ, fs=fs)

    # Filtering can only remove energy, so the filtered low-frequency-only
    # signal's amplitude should be close to `low` alone, not `low + high`.
    assert out["accel_x"].std() < df["accel_x"].std()
    assert out["accel_x"].std() == pytest.approx(pd.Series(low).std(), rel=0.2)


def test_apply_lowpass_filter_skips_missing_channels():
    df = pd.DataFrame({"accel_x": np.zeros(200)})
    out = apply_lowpass_filter(df, channels=["accel_x", "nav_x_left_temple"], fs=50.0)
    assert "nav_x_left_temple" not in out.columns


def test_add_derivative_channels_matches_constant_slope():
    fs = 50.0
    n = 200
    slope = 3.0
    df = pd.DataFrame({"accel_x": slope * np.arange(n) / fs})  # linear ramp -> constant derivative

    out = add_derivative_channels(df, channels=["accel_x"], fs=fs)

    assert "accel_x_d1" in out.columns
    assert "accel_x" in out.columns  # original retained, not replaced
    assert np.allclose(out["accel_x_d1"], slope, atol=1e-6)


def test_add_derivative_channels_skips_missing_channels():
    df = pd.DataFrame({"accel_x": np.zeros(50)})
    out = add_derivative_channels(df, channels=["accel_x", "nav_x_left_temple"], fs=50.0)
    assert "nav_x_left_temple_d1" not in out.columns


def test_select_columns_without_label():
    df = pd.DataFrame(
        {"datetime": pd.date_range("2024-01-01", periods=3, freq="20ms"), "accel_x": [1, 2, 3]}
    )
    out = select_columns(df)
    assert "label" not in out.columns


def test_segment_at_gaps_splits_on_large_gap():
    t = pd.to_datetime(
        list(pd.date_range("2024-01-01", periods=10, freq="20ms"))
        + list(pd.date_range("2024-01-01 00:00:10", periods=10, freq="20ms"))  # ~10s gap
    )
    df = pd.DataFrame({"datetime": t, "accel_x": range(20)})
    segments = segment_at_gaps(df, gap_threshold_s=1.0)
    assert len(segments) == 2
    assert len(segments[0]) == 10
    assert len(segments[1]) == 10


def test_segment_at_gaps_no_gap_returns_single_segment():
    t = pd.date_range("2024-01-01", periods=20, freq="20ms")
    df = pd.DataFrame({"datetime": t, "accel_x": range(20)})
    segments = segment_at_gaps(df, gap_threshold_s=1.0)
    assert len(segments) == 1
    assert len(segments[0]) == 20


def test_resample_segment_label_is_nearest_not_interpolated():
    # Irregular timestamps (49-51ms spacing) -> resampling to 50Hz should still
    # produce a strictly 0/1 label (nearest-neighbour), never a fractional value.
    t0 = pd.Timestamp("2024-01-01")
    offsets_ms = np.cumsum([49, 51, 50, 49, 51, 50, 49, 51] * 5)
    t = t0 + pd.to_timedelta(offsets_ms, unit="ms")
    n = len(t)
    df = pd.DataFrame(
        {
            "datetime": t,
            "accel_x": np.sin(np.linspace(0, 10, n)),
            "label": (np.arange(n) % 6 < 3).astype(int),
        }
    )
    out = resample_segment(df, target_hz=50.0)
    assert set(out["label"].unique()) <= {0, 1}
    assert len(out) > 0
    # grid should be (approximately) evenly spaced
    dt = np.diff(out["datetime"].to_numpy()).astype("timedelta64[ms]").astype(float)
    assert np.allclose(dt, dt[0], atol=1.0)


def test_resample_segment_without_label():
    t = pd.date_range("2024-01-01", periods=50, freq="20ms")
    df = pd.DataFrame({"datetime": t, "accel_x": np.arange(50.0)})
    out = resample_segment(df, target_hz=50.0)
    assert "label" not in out.columns
    assert "accel_x" in out.columns


def test_fit_and_apply_global_stats_matches_manual_computation():
    seg1 = pd.DataFrame({"accel_x": [0.0, 2.0, 4.0]})
    seg2 = pd.DataFrame({"accel_x": [6.0, 8.0, 10.0]})
    stats = fit_global_stats([seg1, seg2], channels=["accel_x"])

    all_values = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    assert stats["accel_x"]["mean"] == all_values.mean()
    assert abs(stats["accel_x"]["std"] - all_values.std()) < 1e-9

    out = apply_global_stats(seg1.copy(), stats)
    expected = (seg1["accel_x"] - stats["accel_x"]["mean"]) / (stats["accel_x"]["std"] + 1e-8)
    assert np.allclose(out["accel_x"], expected)


def test_resample_recording_end_to_end(synthetic_sipm_csv):
    segments = resample_recording("SIPMEatingStudy", synthetic_sipm_csv)
    assert len(segments) == 1
    df, meta = segments[0]

    assert meta["has_label"] is True
    assert meta["filter"]["type"] == "butterworth_lowpass"
    assert meta["derivative"]["suffix"] == "_d1"
    assert "nav_x_right_temple" not in df.columns  # dropped even though present in source
    assert "prox_left_cheek" not in df.columns  # dropped -- out of scope for this model
    for c in ["nav_x_left_temple", "accel_x", "accel_x_d1", "label", "datetime"]:
        assert c in df.columns
    assert set(df["label"].unique()) <= {0, 1}
    # raw-scale (unnormalized) output -- no per-recording normalization anymore,
    # all 10 sensor channels present but not required to be ~N(0, 1)
    for c in SENSOR_CHANNELS:
        assert c in df.columns


def test_resample_recording_splits_on_gap(synthetic_sipm_csv_with_gap):
    segments = resample_recording("SIPMEatingStudy", synthetic_sipm_csv_with_gap)
    assert len(segments) == 2
    for _, meta in segments:
        assert meta["duration_s"] >= 0
