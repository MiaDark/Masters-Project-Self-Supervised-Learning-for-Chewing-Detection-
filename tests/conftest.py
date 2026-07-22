"""Shared synthetic-data fixtures.

CI has no access to the real (private, S3-hosted) datasets, so every test
that needs a recording CSV or an interim segment builds a small synthetic one
here instead -- same schema as the real data, fabricated values.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_sipm_csv(tmp_path):
    """A tiny SIPM-schema CSV: real column names, fabricated values, one dead channel
    (Nav RightTemple all-zero) just like real participants commonly show.

    700 samples @ 50Hz = 14s, comfortably above preprocess.py's default
    min_segment_s=10.0 so it survives the gap-segmentation length filter.
    """
    n = 700
    fs = 50.0
    t0 = 1_700_000_000.0
    rng = np.random.default_rng(0)
    label = (np.arange(n) % 100 < 30).astype(int)

    df = pd.DataFrame(
        {
            "Timestamp": t0 + np.arange(n) / fs,
            "Accelerometer/Raw.X": rng.normal(0, 1, n),
            "Accelerometer/Raw.Y": rng.normal(-9.8, 1, n),
            "Accelerometer/Raw.Z": rng.normal(0, 1, n),
            "Nav/Raw.X[LeftTemple]": rng.normal(0, 0.1, n),
            "Nav/Raw.Y[LeftTemple]": rng.normal(0, 0.1, n),
            "Nav/Raw.X[RightTemple]": np.zeros(n),
            "Nav/Raw.Y[RightTemple]": np.zeros(n),
            "Prox/Raw[LeftCheek]": rng.normal(24, 2, n),
            "Prox/Raw[RightCheek]": rng.normal(23, 2, n),
            "Chewing": label,
        }
    )
    path = tmp_path / "sensor_data_01.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_sipm_csv_with_gap(tmp_path):
    """Same schema, but with a large timestamp gap partway through (simulated dropout).

    700 samples (14s) on each side of the gap, so both resulting segments
    individually clear preprocess.py's default min_segment_s=10.0 filter.
    """
    half = 700
    n = 2 * half
    fs = 50.0
    t0 = 1_700_000_000.0
    rng = np.random.default_rng(1)
    timestamps = np.concatenate(
        [t0 + np.arange(half) / fs, t0 + half / fs + 30.0 + np.arange(half) / fs]  # 30s gap
    )
    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Accelerometer/Raw.X": rng.normal(0, 1, n),
            "Accelerometer/Raw.Y": rng.normal(-9.8, 1, n),
            "Accelerometer/Raw.Z": rng.normal(0, 1, n),
            "Nav/Raw.X[LeftTemple]": rng.normal(0, 0.1, n),
            "Nav/Raw.Y[LeftTemple]": rng.normal(0, 0.1, n),
            "Prox/Raw[LeftCheek]": rng.normal(24, 2, n),
            "Prox/Raw[RightCheek]": rng.normal(23, 2, n),
            "Chewing": (np.arange(n) % 80 < 20).astype(int),
        }
    )
    path = tmp_path / "sensor_data_02.csv"
    df.to_csv(path, index=False)
    return path


def make_interim_segment(
    root, dataset: str, tag: str, n_samples: int = 300, has_label: bool = True, seed: int = 0
):
    """Write one synthetic interim segment (parquet + json sidecar) for windowing tests."""
    from chewing_ssl.common.data.preprocess import SENSOR_CHANNELS

    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-01", periods=n_samples, freq="20ms")
    data = {"datetime": dt}
    for c in SENSOR_CHANNELS:
        data[c] = rng.normal(0, 1, n_samples)
    if has_label:
        data["label"] = (np.arange(n_samples) % 50 < 15).astype(int)
    df = pd.DataFrame(data)

    seg_dir = root / dataset / tag
    seg_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(seg_dir / "segment_00.parquet", index=False)

    meta = {
        "dataset": dataset,
        "source_path": "synthetic",
        "target_hz": 50.0,
        "duration_s": n_samples / 50.0,
        "n_samples": n_samples,
        "has_label": has_label,
    }
    (seg_dir / "segment_00.json").write_text(json.dumps(meta))
    return seg_dir


@pytest.fixture
def synthetic_interim_tree(tmp_path):
    """A minimal interim/<dataset>/<tag>/segment_00.{parquet,json} tree covering
    all five real datasets, enough participants to exercise the uniform 60/20/20
    split logic (small numbers, so splits won't hit exactly those ratios, but
    every item is still assigned to exactly one label). Spectacle_NonMeal uses
    real tag conventions ("p01_snippet001", participant + underscore + rest) so
    tests exercise the actual extract_participant split-on-first-underscore rule,
    not just the identity case the other four datasets happen to be."""
    root = tmp_path / "interim"
    for i in range(1, 6):
        make_interim_segment(root, "SIPMEatingStudy", f"p0{i}", n_samples=300, seed=i)
    for i in range(1, 6):
        make_interim_segment(root, "SussexEatingUK", f"EMTEQ{i}", n_samples=300, seed=10 + i)
    for i in range(1, 4):
        make_interim_segment(root, "FEIT_eating_S05", f"p0{i}", n_samples=300, seed=20 + i)
    for i in range(1, 4):
        make_interim_segment(root, "Office_eating_s05", f"p0{i}", n_samples=300, seed=30 + i)
    for i in range(1, 4):
        make_interim_segment(root, "Spectacle_NonMeal", f"p0{i}_snippet001", n_samples=300, seed=40 + i)
    return root
