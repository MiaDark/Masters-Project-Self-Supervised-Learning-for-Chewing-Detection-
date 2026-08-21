from __future__ import annotations

from pathlib import Path

import pandas as pd

# Per-dataset: raw column name -> canonical name.
SIPM_COLUMNS: dict[str, str] = {
    "Timestamp": "timestamp_raw",
    "Accelerometer/Raw.X": "accel_x",
    "Accelerometer/Raw.Y": "accel_y",
    "Accelerometer/Raw.Z": "accel_z",
    "Gyroscope/Raw.X": "gyro_x",
    "Gyroscope/Raw.Y": "gyro_y",
    "Gyroscope/Raw.Z": "gyro_z",
    "Nav/Raw.X[LeftTemple]": "nav_x_left_temple",
    "Nav/Raw.Y[LeftTemple]": "nav_y_left_temple",
    "Nav/Raw.X[RightTemple]": "nav_x_right_temple",
    "Nav/Raw.Y[RightTemple]": "nav_y_right_temple",
    "Nav/Raw.X[LeftBrow]": "nav_x_left_brow",
    "Nav/Raw.Y[LeftBrow]": "nav_y_left_brow",
    "Nav/Raw.X[RightBrow]": "nav_x_right_brow",
    "Nav/Raw.Y[RightBrow]": "nav_y_right_brow",
    "Prox/Raw[LeftCheek]": "prox_left_cheek",
    "Prox/Raw[RightCheek]": "prox_right_cheek",
    "Prox/Raw[LeftBrow]": "prox_left_brow",
    "Prox/Raw[RightBrow]": "prox_right_brow",
    "Prox/Raw[CentreBrow]": "prox_centre_brow",
    "Chewing": "label",
    "Bite Size": "bite_size",
    "Food Type": "food_type",
}

SUSSEX_COLUMNS: dict[str, str] = {
    "Timestamp": "timestamp_raw",
    "Accelerometer/Raw.X": "accel_x",
    "Accelerometer/Raw.Y": "accel_y",
    "Accelerometer/Raw.Z": "accel_z",
    "Gyroscope/Raw.X": "gyro_x",
    "Gyroscope/Raw.Y": "gyro_y",
    "Gyroscope/Raw.Z": "gyro_z",
    "Nav/Raw.X[LeftTemple]": "nav_x_left_temple",
    "Nav/Raw.Y[LeftTemple]": "nav_y_left_temple",
    "Nav/Raw.X[RightTemple]": "nav_x_right_temple",
    "Nav/Raw.Y[RightTemple]": "nav_y_right_temple",
    "Nav/Raw.X[LeftBrow]": "nav_x_left_brow",
    "Nav/Raw.Y[LeftBrow]": "nav_y_left_brow",
    "Nav/Raw.X[RightBrow]": "nav_x_right_brow",
    "Nav/Raw.Y[RightBrow]": "nav_y_right_brow",
    "Nav/Raw.X[LeftCheek]": "nav_x_left_cheek",
    "Nav/Raw.Y[LeftCheek]": "nav_y_left_cheek",
    "Nav/Raw.X[RightCheek]": "nav_x_right_cheek",
    "Nav/Raw.Y[RightCheek]": "nav_y_right_cheek",
    "Prox/Raw[LeftCheek]": "prox_left_cheek",
    "Prox/Raw[RightCheek]": "prox_right_cheek",
    "Prox/Raw[LeftBrow]": "prox_left_brow",
    "Prox/Raw[RightBrow]": "prox_right_brow",
    "Prox/Raw[CentreTopBrow]": "prox_centre_top_brow",
    "Prox/Raw[CentreBottomBrow]": "prox_centre_bottom_brow",
    "Chewing Event": "label",
    "Meal": "meal",
    "Food": "food_type",
    "Swallow Event": "swallow_event",
}

SPECTACLE_COLUMNS: dict[str, str] = {
    "Software Timestamp": "timestamp_raw",
    "Hardware Timestamp": "hw_timestamp_raw",
    # Spectacle CSVs come in at least 3 header layouts (see docs/data.md /
    # analyze_data.py schema report): some use space-separated names, others
    # the slash-style names shared with SIPM/Sussex. Map both onto the same
    # canonical column so no schema variant silently loses its sensor data.
    "Accelerometer X": "accel_x",
    "Accelerometer/Raw.X": "accel_x",
    "Accelerometer Y": "accel_y",
    "Accelerometer/Raw.Y": "accel_y",
    "Accelerometer Z": "accel_z",
    "Accelerometer/Raw.Z": "accel_z",
    "Gyroscope X": "gyro_x",
    "Gyroscope/Raw.X": "gyro_x",
    "Gyroscope Y": "gyro_y",
    "Gyroscope/Raw.Y": "gyro_y",
    "Gyroscope Z": "gyro_z",
    "Gyroscope/Raw.Z": "gyro_z",
    "Navigation Left Temple X": "nav_x_left_temple",
    "Nav/Raw.X[LeftTemple]": "nav_x_left_temple",
    "Navigation Left Temple Y": "nav_y_left_temple",
    "Nav/Raw.Y[LeftTemple]": "nav_y_left_temple",
    "Navigation Right Temple X": "nav_x_right_temple",
    "Nav/Raw.X[RightTemple]": "nav_x_right_temple",
    "Navigation Right Temple Y": "nav_y_right_temple",
    "Nav/Raw.Y[RightTemple]": "nav_y_right_temple",
    "Navigation Left Brow X": "nav_x_left_brow",
    "Nav/Raw.X[LeftBrow]": "nav_x_left_brow",
    "Navigation Left Brow Y": "nav_y_left_brow",
    "Nav/Raw.Y[LeftBrow]": "nav_y_left_brow",
    "Navigation Right Brow X": "nav_x_right_brow",
    "Nav/Raw.X[RightBrow]": "nav_x_right_brow",
    "Navigation Right Brow Y": "nav_y_right_brow",
    "Nav/Raw.Y[RightBrow]": "nav_y_right_brow",
    "Proximity Left Cheek": "prox_left_cheek",
    "Prox/Raw[LeftCheek]": "prox_left_cheek",
    "Proximity Right Cheek": "prox_right_cheek",
    "Prox/Raw[RightCheek]": "prox_right_cheek",
    "Proximity Left Brow": "prox_left_brow",
    "Prox/Raw[LeftBrow]": "prox_left_brow",
    "Proximity Right Brow": "prox_right_brow",
    "Prox/Raw[RightBrow]": "prox_right_brow",
    "Proximity Centre Brow": "prox_centre_brow",
    "Prox/Raw[CentreTopBrow]": "prox_centre_top_brow",
    "Label": "label",
    "Annotation": "annotation",
    "chewing_rate": "chewing_rate",
}

DIVERSE_NORMATIVE_COLUMNS: dict[str, str] = {
    "Timestamp": "timestamp_raw",
    "Accelerometer/Raw.X": "accel_x",
    "Accelerometer/Raw.Y": "accel_y",
    "Accelerometer/Raw.Z": "accel_z",
    "Gyroscope/Raw.X": "gyro_x",
    "Gyroscope/Raw.Y": "gyro_y",
    "Gyroscope/Raw.Z": "gyro_z",
    "Nav/Raw.X[LeftTemple]": "nav_x_left_temple",
    "Nav/Raw.Y[LeftTemple]": "nav_y_left_temple",
    "Prox/Raw[LeftTemple]": "prox_left_temple",
    "Annotations": "label",
}

# Shared by FEIT_eating_S05 and Office_eating_s05: same single-temple OCOsense
# device, space-separated header names, categorical "Label" column.
FEIT_S05_COLUMNS: dict[str, str] = {
    "Software Timestamp": "timestamp_raw",
    "Hardware Timestamp": "hw_timestamp_raw",
    "Accelerometer X": "accel_x",
    "Accelerometer Y": "accel_y",
    "Accelerometer Z": "accel_z",
    "Gyroscope X": "gyro_x",
    "Gyroscope Y": "gyro_y",
    "Gyroscope Z": "gyro_z",
    "Navigation Left Temple X": "nav_x_left_temple",
    "Navigation Left Temple Y": "nav_y_left_temple",
    "Proximity Left Temple": "prox_left_temple",
    "Label": "label",
}

# s11_feit_eating (new 2026 cohort): same device/software family as
# FEIT_eating_S05 (space-separated headers, ms timestamps), BUT recorded
# from the RIGHT temple sensor, not left -- this cohort has no
# nav_x/y_left_temple data at all. Every model in this project (classical
# ML's hand-crafted features, masked-SSL's raw input channels) was trained
# exclusively on LEFT-temple signal; right temple was previously audited
# and found unreliable/dead in a substantial fraction of the ORIGINAL
# datasets, which is why left temple was chosen as the standard channel to
# begin with (see docs/preprocessing.md). Mapping "Navigation Right Temple
# X/Y" onto the nav_x/y_left_temple column names below is therefore an
# UNVALIDATED SUBSTITUTION, not a claim the two sensors are equivalent --
# the dataset key's own name carries that caveat through every downstream
# output (file paths, ledger rows, console reports) so a result from this
# cohort can never be mistaken for a validated one. Also note the label
# column is "annotation_label" here, not "Label".
FEIT_S11_RIGHT_TEMPLE_COLUMNS: dict[str, str] = {
    "Software Timestamp": "timestamp_raw",
    "Hardware Timestamp": "hw_timestamp_raw",
    "Accelerometer X": "accel_x",
    "Accelerometer Y": "accel_y",
    "Accelerometer Z": "accel_z",
    "Gyroscope X": "gyro_x",
    "Gyroscope Y": "gyro_y",
    "Gyroscope Z": "gyro_z",
    "Navigation Right Temple X": "nav_x_left_temple",
    "Navigation Right Temple Y": "nav_y_left_temple",
    "Proximity Right Temple": "prox_left_temple",
    "annotation_label": "label",
}

# Categorical -> binary label maps. Values not present in the map (e.g.
# "Unknown") become NaN, same as the unlabeled gaps in the other datasets.
DIVERSE_NORMATIVE_LABEL_MAP = {"chewing": 1, "other": 0}
FEIT_S05_LABEL_MAP = {"Eating": 1, "Not Eating": 0}

DATASET_SPECS: dict[str, dict] = {
    "SIPMEatingStudy": {"columns": SIPM_COLUMNS, "timestamp_unit": "s"},
    "SussexEatingUK": {"columns": SUSSEX_COLUMNS, "timestamp_unit": "s"},
    "Spectacle": {"columns": SPECTACLE_COLUMNS, "timestamp_unit": "ms"},
    # Non-meal snippets sliced from Spectacle (see scripts/extract_spectacle_nonmeal.py);
    # same per-file schema/columns as Spectacle, just row-sliced, so reuse its column map.
    "Spectacle_NonMeal": {"columns": SPECTACLE_COLUMNS, "timestamp_unit": "ms"},
    "DiverseNormative": {
        "columns": DIVERSE_NORMATIVE_COLUMNS,
        "timestamp_unit": "s",
        "label_map": DIVERSE_NORMATIVE_LABEL_MAP,
    },
    "FEIT_eating_S05": {
        "columns": FEIT_S05_COLUMNS,
        "timestamp_unit": "ms",
        "label_map": FEIT_S05_LABEL_MAP,
    },
    "Office_eating_s05": {
        "columns": FEIT_S05_COLUMNS,
        "timestamp_unit": "ms",
        "label_map": FEIT_S05_LABEL_MAP,
    },
    "FEIT_eating_S11_RightTempleSubstitute_UNVALIDATED": {
        "columns": FEIT_S11_RIGHT_TEMPLE_COLUMNS,
        "timestamp_unit": "ms",
        "label_map": FEIT_S05_LABEL_MAP,
    },
}

# Canonical sensor channels used by the EDA time-series plots, grouped by
# the physical site the user asked to focus on. Not every group is present
# in every dataset; callers should check membership in the loaded frame.
NAV_TEMPLE_COLS = [
    "nav_x_left_temple",
    "nav_y_left_temple",
    "nav_x_right_temple",
    "nav_y_right_temple",
]
PROX_CHEEK_COLS = ["prox_left_cheek", "prox_right_cheek"]
PROX_BROW_COLS = [
    "prox_left_brow",
    "prox_right_brow",
    "prox_centre_brow",
    "prox_centre_top_brow",
    "prox_centre_bottom_brow",
]
PROX_TEMPLE_COLS = ["prox_left_temple"]
ACCEL_COLS = ["accel_x", "accel_y", "accel_z"]


def read_header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        import csv

        return [c.strip() for c in next(csv.reader(f))]


def load_recording(dataset: str, path: str | Path) -> pd.DataFrame:
    """Load one recording CSV, renamed onto the canonical schema.

    Returns a frame with a ``datetime`` column (parsed from the dataset's
    epoch timestamp) sorted in time order, plus whichever canonical sensor
    and label columns are present in this dataset's CSV.
    """
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {list(DATASET_SPECS)}")

    spec = DATASET_SPECS[dataset]
    column_map: dict[str, str] = spec["columns"]
    path = Path(path)

    header = read_header(path)
    usecols = [c for c in column_map if c in header]
    missing = set(column_map) - set(usecols)
    if missing:
        print(f"[load_recording] {dataset}: columns not in file (skipped): {sorted(missing)}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns=column_map)

    label_map = spec.get("label_map")
    if label_map and "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().map(label_map)

    if "label" in df.columns:
        # Unify label semantics across all datasets: 1 = chewing, everything
        # else (0, NaN, or an unmapped/unknown category) = not chewing.
        df["label"] = (df["label"] == 1).astype(int)

    df["datetime"] = pd.to_datetime(df["timestamp_raw"], unit=spec["timestamp_unit"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df
