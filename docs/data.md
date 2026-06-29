# Data

A datasheet for the three OCOsense datasets used in this project. It records
what is known about provenance, structure and sensors, and flags the fields that
must be confirmed against the actual CSV files. Treat the **To verify** items as
open tasks — fill them in once you have inspected the data, since downstream
preprocessing and labelling depend on them.

## Structure

| Dataset           | Participants (approx.) | Extra nesting        | Leaf file(s)            |
|-------------------|------------------------|----------------------|-------------------------|
| SIPMEatingStudy   | ~35, folders `p01`–`p41` | none                 | `sensor_data.csv`       |
| Spectacle         | `p0`–`p32` per week      | `week2`, then per-day folders | `daily_annotated.csv` |
| SussexEatingUk    | 41                       | an `Eating/` subfolder | `*filename.csv`                 |



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

## Sensor suite (OCOsense)
The OCOsense V2 glasses combine, per Emteq's published specifications:

- **OCO™ optomyography (OMG) optical sensors** — contactless infrared sensors at
  the temples/cheeks that measure 3D (X, Y, Z) skin movement from facial muscle
  activity. These are the primary chewing signal (chewing drives temple/cheek
  muscles).
- **9-axis IMU** — accelerometer, gyroscope, magnetometer; head motion and
  activity.
- **Altimeter (pressure sensor).**
- **Proximity sensors.**
- **Navigation sensord**


## Known issues / to verify

- [ ] Exact column names, units and dtypes for each dataset's CSVs.
- [ ] Per-channel sampling rate(s) after processing; whether modalities are
      already time-aligned to a common clock.
- [ ] Label/annotation semantics: what the columns in `daily_annotated.csv` mean,
      and how the SussexEatingUk `Eating/` files are labelled.
- [ ] Exact participant counts and any gaps in each index range.
- [ ] Whether a participant index is consistent across Spectacle's two weeks.
- [ ] Missing-data / dropout patterns and any quality flags.

### Spectacle: 3 distinct `daily_annotated.csv` header schemas

Confirmed by inspecting all 136 downloaded `week2` files
(`src/chewing_ssl/data/eda.py` / `analyze_data.py`). The loader in
`src/chewing_ssl/data/loaders.py` maps both naming styles onto the same
canonical columns, but **15 of the 136 files have no `Label` column at all**
and so carry no chewing/meal annotation:

- `week2/p18/year_2024-month_11-day_20/daily_annotated.csv`
- `week2/p18/year_2024-month_11-day_21/daily_annotated.csv`
- `week2/p20/year_2024-month_12-day_11/daily_annotated.csv`
- `week2/p20/year_2024-month_12-day_9/daily_annotated.csv`
- `week2/p23/year_2024-month_11-day_21/daily_annotated.csv`
- `week2/p23/year_2024-month_11-day_23/daily_annotated.csv`
- `week2/p25/year_2024-month_12-day_13/daily_annotated.csv`
- `week2/p30/year_2025-month_1-day_3/daily_annotated.csv`
- `week2/p30/year_2025-month_1-day_4/daily_annotated.csv`
- `week2/p30/year_2025-month_1-day_5/daily_annotated.csv`
- `week2/p32/year_2024-month_12-day_10/daily_annotated.csv`
- `week2/p32/year_2024-month_12-day_11/daily_annotated.csv`
- `week2/p32/year_2024-month_12-day_12/daily_annotated.csv`
- `week2/p32/year_2024-month_12-day_13/daily_annotated.csv`
- `week2/p32/year_2024-month_12-day_9/daily_annotated.csv`

These files are unlabelled-only (fine for SSL pretraining input, unusable
for any labelled evaluation). They also carry a reduced sensor set: only
`LeftTemple` Nav (no `RightTemple`/Brow) and only cheek Proximity (no brow).

## Important notes
- Label columns are the following:
  - SIPM: Chewing (1 when chewing; 0 when not chewing)
  - Spectacle: Label (1 when chewing, 0 when not chewing)
  - Sussex: Chewing Event (1.0 when chewing, 0 when not chewing)


## Sensor activity across datasets
Nav RightTemple is the one unreliable channel:
- SIPM: 21 of 31 participants have it completely flat (p01, p04, p05, p07, p08, p12, p16, p17, p18, p21, p23, p25, p26, p27, p28, p30, p31, p32, p37, p38, p41)
- Sussex: perfect — 0 dead in 41 files
- Spectacle: worst — only 6/121 files where the column exists are actually live; 15/136 files don't even have the column

Everything else that is needed (LeftTemple Nav X/Y, both cheek proximity sensors, all 3 accelerometer axes) is present and carrying real signal in every single one of the **208 files**.

For cross-sensor prediction: Drop nav_x_right_temple/nav_y_right_temple from the channel set entirely.

## Splits and provenance

- **Splitting:** all train/validation/test splits must be **subject-disjoint**
  (see [decision 003](decisions/003-subject-disjoint-splits.md)). Hold out
  whole participants, never windows.
- **Ethics / consent:** record the study approvals and the data-sharing terms
  with Emteq Labs here. *To complete.*