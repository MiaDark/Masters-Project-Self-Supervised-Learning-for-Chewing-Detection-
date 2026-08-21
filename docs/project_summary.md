# Project Summary: Data, EDA, Preprocessing, and Augmentation

This document walks through the work completed so far, in the order it was
done and in the order each stage justifies the next: what data we have, what
exploratory analysis revealed about it, how that shaped the preprocessing
pipeline, and how the resulting windows feed into self-supervised contrastive
pretraining. It stops at the point where a pretrained encoder exists.

## 1. Datasets

The goal is a **global** (cross-subject) chewing-detection (bite-detection, where bite represents one bout of food inside the mouth) model trained via
self-supervised learning, so the underlying data needs to be plentiful,
diverse, and, critically for a self-supervised approach, not exclusively
about eating. Five datasets are used, all recorded on Emteq's OCOsense smart
glasses:

| Dataset | Role | Recordings |
|---|---|---|
| **SIPMEatingStudy** | Bite-level labelled meals | 31 |
| **SussexEatingUK** | Bite-level labelled meals | 40 |
| **FEIT_eating_S05** | Bite-level labelled meals | 36 |
| **Office_eating_s05** | Bite-level labelled meals | 26 |
| **Spectacle_NonMeal** | Non-eating background | 90 snippets / 22 participants |

The first four are lab/field meal recordings with per-bite chewing
annotations (~27.3 hours combined). All were self-supervised-training input,
but the model would have only ever seen *eating* contexts, which might be a real risk,
since a chewing detector deployed in the real world needs to also recognize
what *isn't* chewing (talking, walking, resting). **`Spectacle_NonMeal`**
addresses this: it was built by extracting non-eating time from the dataset, specifically
excluding every chewing bout (padded ±5 minutes to also exclude pre/post-meal
handling), drawn round-robin across participants so the result isn't
dominated by a handful of people, and sized to match the ~27.3 hours of meal
data.

**Data-integrity groundwork** (reverting the scaling), before any of the above could be trusted:
three of the five sources (SIPM, Sussex, and the Spectacle data
`Spectacle_NonMeal` is drawn from) ship pre-scaled by Emteq's own processing
pipeline (divided by fixed hardware calibration constants), while the other
two (FEIT, Office) are native raw sensor-count magnitude. Left uncorrected,
the *same* channel would carry wildly different numeric ranges depending on
which dataset a window came from — a model could trivially learn to
distinguish data source by magnitude rather than genuine sensor semantics.
This was reverted (multiplied back by the known calibration divisors,
verified by checking the results land on clean integers — true raw ADC
counts) and the results clipped to each channel's physical sensor range, so
every dataset now shares one consistent on-disk convention.

## 2. Exploratory Data Analysis

Two purpose-built tools drove every downstream design decision:

- **`eda.py`**: per-recording plots (signal overview), plus an
  aggregate summary across every recording in a dataset.
- **`analyze_data.py`**: cross-dataset schema comparison (which raw columns
  exist where, under which of the several naming conventions in use) and,
  critically, **per-sensor liveness checking**: does a channel actually carry
  live signal in every recording, or is it flat/disconnected in some?

Findings that directly shaped the pipeline:

- **Chewing has a consistent rhythm, ~1.5–2 Hz**, visible in the PSD
  comparison between chewing and non-chewing segments. This single number
  ends up grounding two independent later choices: the preprocessing
  low-pass filter cutoff, and the segment length used when disrupting
  temporal order during augmentation.
- **Navigation "RightTemple" is an unreliable channel.** In SIPM it's
  completely flat in 21 of 31 recordings; in the free-living Spectacle data
  it's live in only a handful of files (and absent from the header entirely
  in 15 of 136). By contrast, "LeftTemple" navigation is live in every
  single recording checked, across every dataset. This is why the channel
  set below keeps LeftTemple only.
- **The scaling inconsistency described in §1** was itself an EDA finding:
  it surfaced as an implausible ~300–400× magnitude gap between datasets for
  the same nominal channel, caught by comparing basic per-channel statistics
  across datasets before any modeling began.

## 3. Preprocessing

Implemented in `preprocess.py` (raw → interim) and `windowing.py` (interim →
processed), configured via `src/configs/preprocessing.yaml`. Every recording
goes through the same pipeline, in this order:

1. **Channel selection.** Keep exactly the channels that are (a) present and
   (b) genuinely live across *every* one of the five datasets, per the EDA
   findings above:

   | Raw channel | Reason |
   |---|---|
   | `accel_x`, `accel_y`, `accel_z` | Universal, always live |
   | `nav_x_left_temple`, `nav_y_left_temple` | Universal, always live |

   Proximity, RightTemple/Brow/Cheek navigation, gyroscope, magnetometer,
   Euler angles, and pressure are all dropped at this first step and never
   reappear downstream (add a reference to chewing detection papers from emteq labs). 
   Each of the 5 raw channels above also gets a **first-derivative sibling**,
   which is where the **10 output columns** come from: 5 raw +
   5 derivative.

2. **Gap segmentation.** A timestamp gap much larger than the sampling
   period means the device was removed or dropped out, not a brief pause.
   Such gaps cut the recording into separate segments rather than being
   interpolated across, so no fake transition gets synthesized through a
   period with no real signal. Segments shorter than 10 seconds are
   discarded as too short to be useful.

3. **Resampling.** Every dataset's native sampling rate varies slightly
   (45.5–50 Hz depending on source hardware/software); each segment is
   linearly resampled onto a uniform 50 Hz grid, so a window of *N* samples
   represents the same real-world duration regardless of which dataset it
   came from. The label (where present) is carried over by
   nearest-neighbour, not interpolation because a fractional chewing label makes
   no sense.

4. **Butterworth low-pass filter.** Zero-phase (`scipy.signal.filtfilt`),
   **8 Hz cutoff, order 4**, applied per-channel on the 5 raw channels.
   Chosen directly from the EDA's PSD finding: the ~1.5–2 Hz chewing
   fundamental needs headroom for its sharper harmonics (the jaw-impact
   transients within one chew cycle sit a few multiples above the
   fundamental), while sensor/quantization noise on this hardware dominates
   above roughly 10–15 Hz — 8 Hz sits between the two. Zero-phase
   specifically because a causal filter would introduce a rate-dependent
   timing lag across channels, which matters when chewing-cycle phase
   alignment between e.g. accelerometer and navigation is part of the
   signal.

5. **First derivative.** `np.gradient` of each *already-filtered* raw
   channel (per second, not per sample), appended as a new `<channel>_d1`
   column rather than replacing the raw channel. Because differentiating unfiltered
   data would amplify high-frequency noise, so filtering has to come first.
   Amplitude and rate-of-change carry complementary information about the
   chewing motion; for accelerometer specifically, the derivative also
   naturally suppresses the slow-varying gravity/head-tilt component that
   the raw channel still carries, so the two aren't as redundant as they
   might look.

This produces the **10-channel `SENSOR_CHANNELS`** set used everywhere
downstream: `accel_x`, `accel_y`, `accel_z`, `nav_x_left_temple`,
`nav_y_left_temple`, and each of those five again with a `_d1` suffix.

**Normalization** is deliberately *not* part of the steps above. It needs
the train/val/test participant split to exist first (so validation/test
statistics never leak into what the model is normalized with), but the
split needs to know what recordings exist, which needs the raw-scale interim
tree to exist. So the actual order is: preprocess → assign splits → fit a
single **global** (population) z-score per channel from *train-split
participants only* → apply that same fixed transform to every segment,
train and held-out alike. There is one normalization variant only.

**Windowing** turns each normalized segment into fixed-length training
examples: **4-second windows (200 samples @ 50 Hz)**, long enough to contain
a full chewing cycle at the ~1.5–2 Hz rhythm, short enough to generate many
windows per segment. 50% overlap for training-purpose windows (more signal,
no leakage risk since these aren't held out for evaluation), no overlap for
evaluation-purpose windows (overlapping eval windows are correlated and
would inflate the effective sample size). Labels are dense — one value per
timestep, carried straight through — not collapsed to one label per window
at this stage. Splits are **subject-disjoint** and computed independently
per dataset (no participant's data appears in more than one split), with
every dataset's train-split participants feeding pretraining and every
split's own purpose (`pretrain_train`/`val`, `finetune_train`/`val`/`test`)
recorded in a single window index — not materialized tensors, just which
segment file and sample range each window comes from, sliced at access time.

Current state of the processed index:

| Purpose | Windows | Participants |
|---|---|---|
| `pretrain_train` | 47,345 | 57 |
| `pretrain_val` | 4,701 | 11 |
| `finetune_train` | 56,735 | 64 |
| `finetune_val` | 9,154 | 24 |
| `finetune_test` | 10,270 | 28 |

## 4. Augmentations

Everything above is a **one-time, deterministic pass** that turns raw CSVs
into a canonical windowed dataset. Augmentation is different in kind: it's
**stochastic and applied live, per training batch**, not another offline
preprocessing step, the same window can be perturbed differently every time
it's drawn during training. This exists to support **self-supervised
contrastive pretraining**: since there are no chewing labels involved at
this stage, the model needs a proxy task, and that task is "recognize that a
perturbed copy of a window is still fundamentally the same signal, and tell
it apart from unrelated windows."

Implemented in `ssl/models/augmentation.py`, parameters in
`src/configs/augmentation.yaml`. Two families, each applied independently with
25% probability per window per training step (https://zitniklab.hms.harvard.edu/projects/TF-C/ and https://www.researchgate.net/publication/392838366_TFFC_time-frequency_fusion_consistency_for_semi-supervised_time_series_classification):

**Time-domain** (operates directly on the 10-channel `(C, T)` window):
- **Jitter**: additive Gaussian noise, per element.
- **Scaling**: one random multiplicative factor per (window, channel).
- **Permutation**: split the time axis into 8 chunks (~0.5s each, chosen to
  roughly match one chew-cycle length from the PSD finding above) and
  shuffle their order, disrupting between-cycle structure while mostly
  preserving within-cycle detail.
- **Masking**: zero out one contiguous span of timesteps.

**Frequency-domain** (transform to the window's rFFT spectrum first, apply,
then either inverse-transform back to time domain or use directly):
- **Low-pass**: brick-wall zero above a cutoff (4 Hz, deliberately
  different from preprocessing's 8 Hz filter, so it's a real additional
  perturbation rather than a no-op).
- **Phase-shift**: random phase offset per (window, channel), ±π.
- **Remove-frequency** / **Add-frequency**: randomly zero or perturb a
  fraction of frequency bins.

### How augmentation feeds into encoder training

Two different training recipes consume these augmentations, implemented so
they can be compared directly against each other:

**TF-C** (`ssl/tfc/pretrain_tfc.py`) uses *two separate encoders* — one
processing the raw time-domain window, one processing its frequency
spectrum (a new architecture, `FrequencyEncoder`, since none of the existing
encoders accept complex-valued input). Per batch: the original window and
its time-augmented view both go through the time encoder; the original
spectrum and its frequency-augmented view both go through the frequency
encoder; three contrastive losses are computed — time-view-vs-its-augmented-
self, frequency-view-vs-its-augmented-self, and (the paper's specific
contribution) the *original* window's time-embedding against its *own*
frequency-embedding, treating them as two views of the same signal that
should end up close together in a shared embedding space, with every other
window in the batch as a negative example. The three losses are summed
(equally weighted by default) and backpropagated through both encoders.

**TF-FC** (`ssl/tfc/pretrain_tffc.py`) uses one encoder (a 3-layer 1D
ResNet, per that paper's architecture) and a different fusion mechanism:
time-domain and frequency-domain augmented views are each flattened and
passed through Kernel PCA independently, then concatenated into one fused
vector. A single contrastive loss then pulls the *original* window's
encoder embedding toward this fused-augmented representation, again with
other windows in the batch as negatives.

In both cases, the contrastive loss (**NT-Xent**, the standard SimCLR-style
objective: maximize agreement between a positive pair's embeddings relative
to all other pairings in the batch) is the only training signal — no
chewing labels are used anywhere in this stage. What comes out is a
pretrained encoder that has learned to produce a meaningful numerical
summary of a sensor window, without ever having been told what "chewing"
means. Turning that into an actual chewing detector — fine-tuning a small
classifier on top using real labels, and evaluating it — is the next stage,
covered separately.
