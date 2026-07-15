# Data

A datasheet for the OCOsense datasets used in this project. It records
what is known about provenance, structure and sensors, and flags the fields that
must be confirmed against the actual CSV files. Treat the **To verify** items as
open tasks — fill them in once you have inspected the data, since downstream
preprocessing and labelling depend on them.

## Structure

| Dataset             | Participants (approx.)   | Leaf file(s)                          |
|----------------------|--------------------------|----------------------------------------|
| SIPMEatingStudy      | ~35, folders `p01`–`p41` | `sensor_data.csv`                      |
| Spectacle   | `p0`–`p32` per week      | `daily_annotated.csv`                  |
| SussexEatingUk       | 41                               | `*filename.csv`                        |
| FEIT_eating_S05      | 36                               | `*_sensor_data.csv`                    |
| Office_eating_s05    | 26                                | `*_annotated_sensor_data.csv`          |
| Spectacle_NonMeal    | 22 (derived, see below)    | `*_snippetNNN.csv` (+ `manifest.csv`)  |

FEIT_eating_S05 and Office_eating_s05 are single-temple-device eating recordings.`Spectacle_NonMeal` is not a separately downloaded dataset — it's
extracted from `Spectacle`, see [Balancing](#balancing-non-eating-data-from-spectacle) below.

## Sensor suite (OCOsense)
The OCOsense V2 glasses combine, per Emteq's published specifications:

- **9-axis IMU** — accelerometer, gyroscope, magnetometer; head motion and
  activity.
- **Altimeter (pressure sensor).**
- **Proximity sensors**
- **Navigation sensors**
For the purposes of this project, we will focus on the Accelerometer and Navigation data, later incorporating experiments with Proximity data as well.

## Per-dataset structure

### SIPMEatingStudy
One folder per participant, each containing a single `sensor_data.csv`. The flat
structure makes this the natural dataset to prototype loaders and the SSL
pretext tasks against first.

- **Participants:** ~35 present in the range `p01`–`p41`. (*Missing participants had incosistent data*)

### Spectacle
One collection week (`week2`), each with participants `p0`–`p32`, and
within each participant one folder per recording day. 

- This is the largest and most free-living of the three (multiple days per
  participant across two weeks), so it is the main source of **unlabelled** data
  for SSL pre-training, and its annotations support fine-tuning.

### SussexEatingUk
41 participant objects, each with an `Eating/` subfolder of CSVs. The explicit
`Eating/` segmentation suggests data is already cut to eating episodes, which
makes this a strong candidate for **labelled fine-tuning and evaluation**.

### FEIT_eating_S05 / Office_eating_s05
Single-temple OCOsense devices (only `LeftTemple` Nav, one Proximity channel),
`Eating/`-subfolder layout like Sussex. Same role as SIPM/Sussex: **labelled
fine-tuning and evaluation** data, all meal-context.

### NAVIGATION: Left vs. Right Temple
Since all datasets include the Navigation Left Temple Signal, for further processing, we will exclude Navigation Right Temple, and stick with **Accelerometer** data and **Navigation Left Temple**.

## Scaling

All datasets are expected on disk in **raw ADC-count magnitude** (not physical
units), for a consistent training input across sources:

- **SIPMEatingStudy, SussexEatingUK, Spectacle** ship already scaled (divided)
  by emteqai's `scale_data()` — accelerometer ÷`417.53312946`, navigation
  ÷`240` (identical for `oco_v1`/`oco_v2` in `metadata_oco.yml`). Reverted by
  multiplying back (`scripts/revert_scaling.py` for SIPM/Sussex, into
  `.../raw_unscaled/`; same divisors applied manually to the Spectacle_NonMeal
  snippets below). Confirmed correct because reverted values land on clean
  integers (real ADC counts).
- **FEIT_eating_S05, Office_eating_s05** were never scaled — already
  raw-count magnitude natively.


## Clipping

Physical sensor limits applied via `scripts/clip_raw_signals.py`, across all 5 raw-count datasets
(SIPM, Sussex, FEIT_eating_S05, Office_eating_s05, Spectacle_NonMeal):

- Accelerometer: clipped to `[-5000, 5000]`
- Navigation: clipped to `[-50, 50]`


## Balancing: non-eating data from Spectacle

SIPM/Sussex/FEIT/Office total **~27.3h** of meal recordings across 133
files — all chewing/meal-context, no real-life non-eating background.
`Spectacle_NonMeal` (`scripts/extract_spectacle_nonmeal.py` +
`topup_spectacle_nonmeal.py`) adds a matching amount of genuine non-eating data:

- Sourced from Spectacle's 121 `Label`-carrying days (15 unlabelled days
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



## Important notes
- Label columns are the following:
  - SIPM: Chewing (1 when chewing; 0 when not chewing)
  - Spectacle: Label (1 when chewing, 0 when not chewing)
  - Sussex: Chewing Event (1.0 when chewing, 0 when not chewing)
  - FEIT_eating_S05 / Office_eating_s05: Label (`"Eating"`/`"Not Eating"`/`"Unknown"` categories, mapped to 1/0)
  - Spectacle_NonMeal: no chewing label column needed — `label=0` for every row by construction
- `loaders.load_recording` unifies all of the above to a single `label` column with
  `1` = chewing, everything else (`0`, `NaN`, unmapped category) = not chewing.


## Sensor activity across datasets
Nav RightTemple is the one unreliable channel:
- SIPM: 21 of 31 participants have it completely flat (p01, p04, p05, p07, p08, p12, p16, p17, p18, p21, p23, p25, p26, p27, p28, p30, p31, p32, p37, p38, p41)
- Sussex: perfect — 0 dead in 41 files
- Spectacle: worst — only 6/121 files where the column exists are actually live; 15/136 files don't even have the column

Everything else that is needed (LeftTemple Nav X/Y, all 3 accelerometer axes) is present and carrying real signal in every single one of the **208 files**.

*For cross-sensor prediction: Drop nav_x_right_temple/nav_y_right_temple from the channel set entirely.

## Splits and provenance

- **Splitting:** all train/validation/test splits must be **subject-disjoint**
  (see [decision 003](decisions/003-subject-disjoint-splits.md)). Hold out
  whole participants, never windows.
- **Ethics / consent:** record the study approvals and the data-sharing terms
  with Emteq Labs here. *To complete.*

## Next steps
- [ ] Normalization / standardization (per-channel, computed subject-disjoint per the split above).
