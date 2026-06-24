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

## Splits and provenance

- **Splitting:** all train/validation/test splits must be **subject-disjoint**
  (see [decision 003](decisions/003-subject-disjoint-splits.md)). Hold out
  whole participants, never windows.
- **Ethics / consent:** record the study approvals and the data-sharing terms
  with Emteq Labs here. *To complete.*