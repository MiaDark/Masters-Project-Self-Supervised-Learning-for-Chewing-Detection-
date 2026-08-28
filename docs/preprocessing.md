
# Preprocessing

Raw CSV → windowed model input. Implemented in `src/chewing_ssl/common/data/preprocess.py`
(raw → interim) and `src/chewing_ssl/common/data/windowing.py` (interim → processed).
This covers **standard preprocessing only**. Augmentation (TF-C-style
time/frequency-domain augmentation) is a separate, later stage — see
[Relationship to augmentation](#relationship-to-augmentation) at the bottom.

## Channels

Exactly **5 raw channels**: `accel_x`, `accel_y`, `accel_z`, `nav_x_left_temple`,
`nav_y_left_temple`. Everything else (proximity, Nav RightTemple, gyroscope,
magnetometer, Euler, pressure) is dropped at the very first step and never
reappears anywhere downstream. Nav RightTemple is dropped because it's an
unreliable channel across every dataset (docs/data.md); proximity is simply
out of scope for this model.

Each raw channel also gets a first-derivative sibling (see below), so the
full channel set used everywhere downstream: normalization, windowing, the
model's input dimension is **10 channels**: `SENSOR_CHANNELS =
RAW_CHANNELS + DERIVATIVE_CHANNELS` in `preprocess.py`.

## Pipeline steps, in order

1. **Load + select columns** (`load_recording` + `select_columns`): rename
   to canonical column names (per dataset naming convention), unify the
   label to `1`=chewing/else=not-chewing, keep only `datetime` + the 5 raw
   channels + `label`. <br>
   ***CLIPPING RAW SIGNALS -50 50 and -5000 5000***

2. **Gap segmentation** (`segment_at_gaps`) — a timestamp gap much larger
   than the sampling period means the device was removed or dropped out,
   not a brief pause. Cut into separate segments rather than interpolating
   across a dropout, so no fake transition gets synthesized through a
   period with no real signal. Segments shorter than 10s are discarded.
3. **Resampling** (`resample_segment`) — linear interpolation onto a
   uniform 50 Hz grid, per segment. Every dataset's native rate varies
   slightly (45.5–50 Hz depending on source); resampling makes a window of
   N samples represent the same real-world duration regardless of source
   dataset. Label is carried over by nearest-neighbour, not interpolation
   (a fractional chewing label makes no sense).
4. **Butterworth low-pass filter** (`apply_lowpass_filter`) — zero-phase
   (`scipy.signal.filtfilt`), **8 Hz cutoff, order 4**, applied per-segment,
   per-channel, on the 5 raw channels only. Chosen from this project's own
   PSD analysis (chewing fundamental ~1.5–2 Hz, see `docs/data.md` /
   `windowing.py`), with headroom for the sharper jaw-impact transients
   within a chew cycle (harmonics above the fundamental), while cutting
   sensor/quantization noise that on this hardware dominates above
   ~10–15 Hz. Zero-phase specifically because chewing-cycle timing/phase
   alignment across channels matters — a causal IIR filter would introduce
   a rate-dependent lag that a plain low-pass doesn't need to pay.
5. **First derivative** (`add_derivative_channels`) — `np.gradient` of each
   *filtered* raw channel (per second, not per sample), appended as new
   `<channel>_d1` columns — **not** replacing the raw channel. Amplitude
   and rate-of-change carry complementary information about the chewing
   motion. Computed after filtering, not before: differentiating raw/noisy
   data amplifies high-frequency noise, so the signal has to be clean first.
6. **Normalization** is deliberately **not** part of `preprocess.py` at all
   — see the next section for why, and where it actually happens.

## Normalization: one global variant only, fit after splits exist

There is a single normalization strategy: a **global (population) z-score
per channel**, fit once on training-split participants only, saved, and
reapplied everywhere (val/test/inference). This is what satisfies "one
global model, constants saved and reapplied" — there is no per-recording
("personal") normalization variant; it was removed. Per-recording z-scoring
needs enough data *in that one recording* for its own mean/std to be a
stable estimate, which isn't reliably true here, and this project only ever
needs the one global model anyway — the extra complexity wasn't earning
its keep.

Fitting global stats correctly requires knowing the train/val/test split
*first* (decision 003: never let val/test participants leak into what the
model is normalized with) — but assigning splits requires knowing what
recordings/participants exist, which requires the raw-scale interim tree to
exist. That ordering dependency is why normalization isn't in `preprocess.py`
(which runs first and only knows about one recording at a time) — it's
`chewing_ssl.common.data.refit_global_norm`'s job, run after `windowing.py` has
produced `splits.json`. Concretely, three steps:

1. `preprocess.py` → raw-scale (filtered + derivative, unnormalized)
   segments at `data/interim/<dataset>/<tag>/`.
2. `windowing.py --splits-only --interim-root data/interim` → tag/participant
   discovery off the raw tree, writes `data/processed/splits.json`. (Tag
   discovery only needs directory structure, not normalized content, so the
   raw tree is sufficient here.)
3. `refit_global_norm.py` → fits mean/std per channel from **pretrain-eligible
   segments only** (D1/D2 train-split + D5 pretrain-only),
   writes `data/processed/global_stats.json`, and writes every segment —
   train, val, test, meal-test alike — normalized with those same fixed
   stats to `data/interim_normalized/<dataset>/<tag>/`.
4. `windowing.py --interim-root data/interim_normalized` (no `--splits-only`)
   → reuses the existing `splits.json`, builds the window index against the
   now-normalized segments.

## What `windowing.py` actually does

Two independent things, both subject-disjoint (decision 003):

- **Splits**: assigns each participant to train/val/test (D1/D2,
  60/20/20) or pretrain-only/meal-test-holdout (D5, 80/20),
  deterministically from a seed, and writes `splits.json`.
- **Window index**: NOT materialized window tensors — a lightweight index
  (one row per window: which segment file, which `[start, end)` sample
  range, which purpose e.g. `pretrain_train`/`finetune_val`). `WindowDataset`
  slices the actual window out of the (cached) segment parquet at access
  time. Windows are 4s (200 samples @ 50 Hz) with 50% overlap for train
  purposes, no overlap for eval purposes (overlapping eval windows are
  correlated and would inflate effective sample size).

It does **not** touch signal values at all — no filtering, no derivative, no
normalization, no augmentation. It only decides *which* stretches of
already-fully-preprocessed signal become training examples, and for whom.

## Where everything is saved

| Artifact | Path | Produced by |
|---|---|---|
| Raw-scale (filtered+derivative, unnormalized) segments | `data/interim/<dataset>/<tag>/segment_NN.parquet` + `.json` sidecar (filter/derivative provenance) | `preprocess.py` |
| Subject-disjoint participant → split assignment | `data/processed/splits.json` | `windowing.py --splits-only` |
| Global (train-only) normalization stats | `data/processed/global_stats.json` | `refit_global_norm.py` |
| Normalized, training-ready segments | `data/interim_normalized/<dataset>/<tag>/segment_NN.parquet` + `.json` | `refit_global_norm.py` |
| Window index (one row per window, not materialized tensors) | `data/processed/index/<purpose>.parquet` | `windowing.py` (2nd pass, against the normalized tree) |

## How this is applied at inference time

For a **new, unseen recording** (a real deployment scenario, not a held-out
test participant with pre-built segments):

1. Load the raw CSV, select the same 5 channels (`select_columns`).
2. Gap-segment, resample to 50 Hz — same functions, same parameters
   (`gap_threshold_s=1.0`, `target_hz=50.0`).
3. Apply the *same* Butterworth filter (cutoff=8 Hz, order=4, zero-phase) —
   these are fixed constants, not fit on data, so no state needs loading
   for this step.
4. Compute the derivative channels the same way.
5. Normalize using the **saved `global_stats.json`** —
   `apply_global_stats(df, json.load(open("data/processed/global_stats.json")))`.
   This is the only step that depends on saved, fitted state — deliberately:
   it's the one place cross-subject generalization risk actually lives, so
   it's the one place state needs to be versioned and reused carefully.
6. Window (same `WINDOW_LEN`/stride as training) and feed to the model.

## Visualizing the pipeline

`scripts/plot_preprocessing_steps.py` plots one channel through all four
stages (raw → filtered → derivative → normalized) on a real recording,
zoomed to the longest chewing bout so the plot actually shows chewing
dynamics:

```bash
python scripts/plot_preprocessing_steps.py --dataset SIPMEatingStudy --channel accel_z  # D1
```

Falls back to a clearly-labelled segment-level mean/std for the normalized
panel if `data/processed/global_stats.json` doesn't exist yet;
