# Data

A datasheet for the OCOsense datasets used in this project. It records
what is known about provenance, structure and sensors, and flags the fields that
must be confirmed against the actual CSV files. Treat the **To verify** items as
open tasks — fill them in once you have inspected the data, since downstream
preprocessing and labelling depend on them.

Participant counts confirmed directly against `data/processed_80_10_10/splits.json`'s per-dataset train+val+test totals (`docs/dissertation_classical_ml_sections.md` §3.1) — supersedes the earlier approximate counts below where they conflict.


## Sensor suite (OCOsense)
The OCOsense V2 glasses combine, per Emteq's published specifications:

- **9-axis IMU** — accelerometer, gyroscope, magnetometer; head motion and
  activity.
- **Altimeter (pressure sensor).**
- **Proximity sensors**
- **Navigation sensors**
For the purposes of this project, we will focus on the Accelerometer and Navigation data, later incorporating experiments with Proximity data as well.

### NAVIGATION: Left vs. Right Temple
Since all datasets include the Navigation Left Temple Signal, for further processing, we will exclude Navigation Right Temple, and stick with **Accelerometer** data and **Navigation Left Temple**.

## Scaling

All datasets are expected on disk in **raw ADC-count magnitude** (not physical
units), for a consistent training input across sources:

## Clipping

Physical sensor limits applied via `scripts/clip_raw_signals.py`, across all 5 raw-count datasets:

- Accelerometer: clipped to `[-5000, 5000]`
- Navigation: clipped to `[-50, 50]`


## Balancing: non-eating data from D5

D1-D4 total **~27.3h** of meal recordings across 133
files — all chewing/meal-context, no real-life non-eating background.
`Spectacle_NonMeal` (D5; `scripts/extract_spectacle_nonmeal.py` +
`topup_spectacle_nonmeal.py`) adds a matching amount of genuine non-eating data:

- Sourced from D5's 121 `Label`-carrying days (15 unlabelled days
  excluded — can't verify they're meal-free).
- "Non-meal" = outside every chewing bout (`find_bouts`), padded ±5 min to
  also exclude pre/post-meal handling that isn't itself labelled chewing.
- Snippets drawn round-robin across participants (capped at 20 min/day/pass)
  so the result isn't dominated by a handful of people.
- Result: **90 snippets, 22 participants, ~27.3h**, `label=0` throughout
  (verified). 11 of the initial snippets (dead/flat channels or
  NaN-corrupted rows, found via `eda.py` plots) were deleted and replaced
  with 13 freshly-selected, validated snippets to restore the target duration.
- Full provenance (source file + time range per snippet) is in
  `Spectacle_NonMeal/manifest.csv`.
