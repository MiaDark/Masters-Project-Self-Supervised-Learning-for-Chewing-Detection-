# Dissertation Sections — Classical Machine Learning Pipeline

*Grounded in `docs/preprocessing.md`, `docs/classical_ml.md`, the actual
source in `src/chewing_ssl/classical_ml/` and `src/chewing_ssl/common/`,
and the current contents of `checkpoints/train_ml/`,
`checkpoints/train_ml_w2s0.25/`, `checkpoints/classical_ml_summary.csv`,
`checkpoints/ml_bout_ledger.csv` and `checkpoints/ml_bout_ledger_w2s0.25.csv`*

> **Status as of 2026-07-31 — 80/10/10 split rerun COMPLETE for the 4 s
> classical-ML configuration.** **The project's split ratio has moved from
> an (empirically) ~60/20/20 participant division to a genuine 80/10/10
> division**. **The
> classical-ML pipeline has been rerun end-to-end** — normalisation →
> windowing → segmentation (**now non-overlapping for every split,
> including train — a deliberate methodology change from the original
> pipeline's 50%-overlap-on-train convention, see §2.1**) → feature
> extraction → the same single `rf_06_regularized_leaf5` hyperparameter
> combination retrained from scratch → bout-level evaluation → a
> postprocessing re-sweep — against the new split, in a parallel
> `_80_10_10`-suffixed artefact tree (`data/processed_80_10_10/`,
> `checkpoints/train_ml_80_10_10/`) that does not overwrite the original
> 60/20/20 results. **Every number in §3.3, §4.6 (new), and §5.4–§5.8 below
> marked 80/10/10 is now real, verified data — nothing is pending for the
> 4 s configuration any longer.** The 2-second configuration was
> deliberately **not** rerun under 80/10/10 (scope decision: 4 s only for
> this pass), so its numbers throughout remain the original 60/20/20
> figures, clearly labelled as such.
>
> **All bold text in this document marks content added or changed in this
> latest (2026-07-31) update pass, specifically so it can be reviewed and
> revised later — nothing bold should be taken as fully polished prose
> yet.**
>
> **Newer work not reflected below (post-2026-07-31), read alongside this
> document for anything current**: a raw-signal clipping fix and a genuine
> per-sensor (nav-only/accel-only) retrained model family — this document's
> `rf_4s_80_10_10` model only ever exists as an all-channel model — feeding
> `checkpoints/cross_sensor_evaluation/CONCLUSION.md` (accuracy under sensor
> removal) and `docs/uncertainty.md` (predictive uncertainty, both
> pipelines, all 6 classical-ML channel/norm configs). Full provenance
> chain across every checkpoint generation, including this document's own:
> `checkpoints/uncertainty_models/MANIFEST.md`.

---

## 1. Preprocessing

### 1.1 Overview and rationale

All downstream pipelines in this project — the classical machine learning
baseline described here, and the self-supervised deep-learning pipelines —
consume the *same* preprocessed signal representation. Preprocessing is
implemented as a two-stage process (`src/chewing_ssl/common/data/preprocess.py`
for the raw-to-interim transformation and
`src/chewing_ssl/common/data/windowing.py` for split assignment and window
indexing), deliberately decoupled from any model-specific logic, so that a
difference in downstream performance between pipelines can be attributed to
the modelling approach rather than to inconsistent input treatment.

### 1.2 Channel selection

Each of the five source datasets is recorded via the OCOsense smart-glasses
platform, whose full sensor suite spans a 9-axis IMU (accelerometer,
gyroscope, magnetometer), proximity sensors, an altimeter, and navigation
(capacitive jaw-motion) sensors at left and right temple sites. This
project restricts its input to exactly **five raw channels**:
`accel_x`, `accel_y`, `accel_z`, `nav_x_left_temple`, `nav_y_left_temple`.

Two channels present in the raw data are deliberately excluded:

- **Navigation, right temple** — an audit of channel activity across all
  208 available recordings found this channel entirely flat (dead/
  disconnected) in a substantial fraction of files across every dataset
  (e.g. 21 of 31 SIPMEatingStudy participants), making it an unreliable
  input to rely on for a general model.
- **Proximity sensors** — considered out of scope for the present model,
  reserved as a candidate channel family for future extension rather than
  included provisionally.

Gyroscope and magnetometer channels are likewise not used: navigation and
accelerometer channels alone were judged sufficient to characterise jaw
motion and head-adjacent vibration during mastication, and restricting the
channel set keeps the feature space (see §2.2) and the model input
dimensionality tractable.

Each of the five raw channels is additionally paired with a **first-derivative
sibling** (see §1.3, step 5), giving a final channel set of 10 —
`SENSOR_CHANNELS = RAW_CHANNELS + DERIVATIVE_CHANNELS` — used identically by
every pipeline in the project, classical and deep-learning alike.

### 1.3 Pipeline steps, in order

1. **Load and column selection** (`load_recording`, `select_columns`).
   Each dataset ships with its own native column-naming convention (a
   consequence of differing acquisition software across the five source
   studies); a per-dataset column-mapping table renames every field onto a
   single canonical schema. The categorical/binary chewing annotation is
   unified to `1 = chewing`, everything else (`0`, `NaN`, or an unrecognised
   category, e.g. `"Unknown"`) treated as not-chewing. Only `datetime`, the
   five raw channels, and `label` survive this step; everything else
   (proximity, right-temple navigation, gyroscope, magnetometer, Euler
   angles, barometric pressure) is dropped and never reappears downstream.

2. **Gap segmentation** (`segment_at_gaps`). A timestamp gap substantially
   larger than the nominal sampling period indicates the device was removed
   or lost signal, not a brief real-world pause in activity. Rather than
   interpolating across such a gap — which would synthesise a fictitious
   transition through a period with no real sensor signal — each recording
   is cut into independent segments at any such gap. Segments shorter than
   10 seconds after this cut are discarded outright as too short to carry
   meaningful bout-level information.

3. **Resampling** (`resample_segment`). Native sampling rate varies across
   the five source datasets (45.5–50 Hz), a consequence of differing
   hardware/firmware revisions across acquisition studies. Every segment is
   linearly interpolated onto a uniform 50 Hz grid, so that a window of $N$
   samples represents an identical real-world duration regardless of which
   dataset it originated from — a precondition for combining data across
   datasets into one training population at all. The label channel is
   carried forward by **nearest-neighbour** assignment specifically, not
   interpolation, since a fractional chewing label has no physical meaning.

4. **Zero-phase Butterworth low-pass filtering** (`apply_lowpass_filter`).
   Applied per-segment, per-channel, to the five raw channels only (the
   derivative channels computed in the following step inherit the filtering
   through their construction). Implemented via `scipy.signal.filtfilt`
   (zero-phase, i.e. forward-backward filtering with no group delay),
   **8 Hz cutoff, 4th order**. The cutoff was chosen from this project's own
   power-spectral-density analysis of the raw signal: the fundamental
   mastication rhythm sits at approximately 1.5–2 Hz, and 8 Hz leaves
   headroom for the sharper jaw-impact transients within a single chew
   cycle (harmonic content above the fundamental) while removing
   sensor/quantisation noise that, on this hardware, dominates above
   roughly 10–15 Hz. The zero-phase property specifically matters because
   chewing-cycle timing and cross-channel phase alignment (e.g. between the
   two navigation axes) is informative signal in itself; a causal
   (forward-only) IIR filter would introduce a rate-dependent phase lag
   that a model reconstructing a bite's precise timing would then have to
   compensate for.

5. **First-derivative channels** (`add_derivative_channels`). Each
   already-filtered raw channel is differentiated (`np.gradient`, scaled per
   second rather than per sample) and appended as a new `<channel>_d1`
   column — additively, not replacing the raw channel. Amplitude and
   rate-of-change are treated as complementary descriptors of jaw motion.
   Differentiation is deliberately performed *after* filtering rather than
   before: differentiating unfiltered, noisy data would amplify
   high-frequency noise disproportionately, so the signal must already be
   clean before this step.

6. **Normalisation is deliberately not part of this stage** — see §1.4.

### 1.4 Normalisation

A single normalisation strategy is used throughout the project: a
**global (population) z-score per channel**, fitted once, saved, and
reapplied identically to every subsequent split (validation, test, and any
future unseen recording at inference time). No per-recording ("personal")
normalisation variant is used.

Two considerations motivate this choice. First, a per-recording z-score
requires enough data within that single recording for its own mean and
standard deviation to be a stable estimate — a condition not reliably met
here given how short some recordings/segments are — and the project has no
requirement for a personalised model that would justify the added
complexity. Second, and more importantly from a methodological standpoint,
fitting global statistics *correctly* requires already knowing the
train/validation/test split, since validation and test participants must
never contribute to the statistics a model is normalised against (this
follows the project's subject-disjoint-splitting principle, decision 003 —
see §3.1). This creates an ordering dependency: normalisation cannot happen
inside the raw-to-interim step (which processes one recording at a time,
before any split has been assigned), so it is deliberately deferred to a
dedicated step, `refit_global_norm.py`, run only after participant-to-split
assignment already exists.

Concretely, the pipeline runs in four ordered stages:

1. `preprocess.py` → raw-scale (filtered, derivative-augmented, but still
   **unnormalised**) segments, written to `data/interim/<dataset>/<tag>/`.
2. `windowing.py --splits-only`, run against the raw-scale tree → assigns
   every participant to a split (train/validation/test, or, for the
   pretraining-only dataset, pretrain-train/pretrain-validation) and writes
   `data/processed/splits.json`. This step only requires directory
   structure to discover which participants/recordings exist, not
   normalised signal content, so the raw tree suffices here.
3. `refit_global_norm.py` → fits per-channel mean and standard deviation
   from **training-split segments only**, writes these constants to
   `data/processed/global_stats.json`, then applies them to *every* segment
   — train, validation, and test alike — writing the result to
   `data/interim_normalized/<dataset>/<tag>/`. This is the only step in the
   entire pipeline that depends on fitted, saved state, and deliberately
   so: it is the one place where cross-subject generalisation risk
   genuinely lives, and is therefore the one place state needs to be
   versioned and reused with care.
4. `windowing.py`, re-run against the now-normalised tree (without
   `--splits-only`) → reuses the already-written `splits.json` and builds a
   window index against the normalised segments.

At inference time on a genuinely new, previously unseen recording, the same
sequence is followed exactly except step 5 (normalisation) reuses the
already-saved `global_stats.json` rather than refitting anything — this is
the only step whose behaviour depends on what the model saw during
training, everything else (filter cutoff, derivative computation, gap
threshold, resample rate) being fixed constants applied identically
regardless of the recording's origin.

### 1.5 Artefact summary

| Artefact | Path | Produced by |
|---|---|---|
| Raw-scale (filtered + derivative, unnormalised) segments | `data/interim/<dataset>/<tag>/segment_NN.parquet` | `preprocess.py` |
| Subject-disjoint participant → split assignment | `data/processed/splits.json` | `windowing.py --splits-only` |
| Global (training-split-only) normalisation constants | `data/processed/global_stats.json` | `refit_global_norm.py` |
| Normalised, training-ready segments | `data/interim_normalized/<dataset>/<tag>/segment_NN.parquet` | `refit_global_norm.py` |
| Window index (lightweight index, not materialised tensors) | `data/processed/index/<purpose>.parquet` | `windowing.py` (second pass) |

---

## 2. Methodology (Classical Machine Learning)

### 2.1 Windowing

Whereas the deep-learning pipelines consume a lightweight *index* into the
normalised segment files and slice out each window lazily at training time
(`WindowDataset`), the classical pipeline reads every window's data exactly
once, to compute a fixed-length feature vector from it, and therefore
**materialises** each window as a dense array up front
(`chewing_ssl.classical_ml.segment`).

- **Source**: `data/interim_normalized`, the same fully-preprocessed,
  filtered, derivative-augmented, globally-normalised 10-channel segments
  used by every other pipeline, read against the same `splits.json`, so
  that "which recordings are in which split" is identical across pipelines.
- **Window label**: a per-window binary target is required (a feature
  vector needs a single label, unlike the deep-learning path's dense
  per-timestep target), obtained by **majority vote** over the window's
  dense per-timestep labels — `mean(label) ≥ 0.5 → 1`. The idea to segment based on label
  was excluded, after thoroughly thinking and coming to a conclusion that during inference
  time, one doesn't have access to the labels. Windowing based on a label during training 
  can lead to inflated performance, which wouldn't be the case during inference. 
- **Window length and stride**: two configurations were evaluated across
  this work's history:
  - **4-second windows (200 samples at 50 Hz)** — the primary configuration
    reported throughout this work. **Originally run with 50% overlap on
    every split (see below); the 2026-07-31 80/10/10 rerun uses
    non-overlapping windows instead — see the callout immediately below.**
  - **2-second windows (100 samples), 25% overlap** — a secondary,
    finer-grained configuration investigated specifically to probe whether
    shorter windows improve bout-boundary precision at the cost of
    window-level classification noise (see §4.5). **Not rerun under
    80/10/10 (scope decision) — its numbers throughout this document are
    still the original 60/20/20, 25%-overlap results.**

  **Originally**, stride was controlled directly via a `--stride-seconds`
  argument (`overlap = 1 − stride / window`, this project's convention),
  and was applied **uniformly to every split, including validation and
  test** — a methodological simplification that was flagged as a
  limitation (§5.6 in earlier drafts of this document), since it meant
  validation/test windows were not independent samples in the way a
  non-overlapping evaluation convention would guarantee.

  > **Change for the 2026-07-31 80/10/10 rerun: `--stride-seconds` has
  > been removed from `segment.py` entirely, and stride is now
  > unconditionally `= window_len` (non-overlapping) for every split,
  > including train.** This was a deliberate methodological correction,
  > not an accident: it brings classical ML's own window convention in
  > line with the deep-learning pipelines' eval-window convention (§2.1
  > table in `docs/ssl_masked.md`: zero overlap for eval-purpose windows,
  > specifically to avoid inflating effective sample size via correlated
  > neighbouring windows) — applied here to *every* split, train included,
  > since the classical pipeline never lazily re-samples windows the way
  > the deep-learning pipelines' `WindowDataset` does; a materialized
  > training set benefits from the same non-redundancy. **One direct
  > consequence: the "caveat on comparing window-level numbers across
  > pipelines" in §3.3 (validation/test windows overlapping, therefore not
  > directly comparable to a non-overlapping pipeline) no longer applies
  > to the 80/10/10, 4 s classical-ML numbers — they are now computed on
  > genuinely independent windows, like the deep-learning pipelines'
  > always were.**

Output: `data/processed/ml_segments/{purpose}_X.npy` (dense float32 array,
shape `n_windows × 10 channels × 200 samples`), `_y.npy` (binary label per
window), and `_meta.parquet` (identity columns — dataset, participant, tag,
segment, window start index — aligned row-for-row with `X`/`y`).

### 2.2 Feature extraction

Each window's `(10, 200)` raw array is transformed into an **850-dimensional
feature vector** (`chewing_ssl.classical_ml.extract_features`), applying
four established families of hand-crafted, signal-processing features
independently to each of the 10 channels (via the `emteqai.utils.features` (**how do i reference this in the dissertation?ß**)
driver library, an existing project dependency, rather than reimplementing
standard human-activity-recognition feature sets):

| Family | Features per channel | Representative features |
|---|---|---|
| Statistical | 9 | mean, standard deviation, min, max, range, interquartile range, kurtosis, skew, RMS |
| Time-domain | 14 | integrated absolute value, average amplitude change, difference absolute standard deviation value, log-detector, mean absolute value (and two windowed variants), simple square integral, myopulse/Willison-style hemming/turning-point counts, temporal moment, variance, V-order, waveform length |
| Frequency-domain | 35 | top-5 power-spectral-density peak frequencies, 17 normalised PSD amplitudes, FFT spectral energy and entropy, PSD kurtosis/skew, a 9-bin PSD distribution over 0–20 Hz (Welch's method) |
| Wavelet | 27 | statistical features (mean/std/etc.) computed over each of 3 coarser levels of a Haar wavelet decomposition (decomposition level 3; the finest detail level is excluded) |

$10 \text{ channels} \times 85 \text{ features/channel} = 850$ features per
window.

Feature computation can introduce `NaN`/`Inf` values (e.g. a
frequency-domain feature undefined on a near-constant signal segment).
These are sanitised by replacing them with **per-column medians computed
from the training split only** — sanitisation statistics computed from
validation or test data would constitute label/distribution leakage — and
the medians themselves are persisted (`feature_medians.json`) for
reproducibility.

Output: `data/processed/ml_features/{purpose}.parquet`, one row per window,
850 feature columns plus 6 identity columns (`dataset`, `participant`,
`tag`, `segment`, `start`, `label`). Realised shapes:

| Configuration | Train | Validation | Test |
|---|---|---|---|
| 4 s window (`ml_features`) | (56,735, 856) | (18,272, 856) | (20,508, 856) |
| 2 s / 25 %-overlap window (`ml_features_w2s0.25`) | (75,810, 856) | (24,420, 856) | (27,398, 856) |

(856 = 850 features + 6 identity columns.)

### 2.3 Model family and training procedure

Every candidate classifier is wrapped identically in a two-stage
`sklearn.pipeline.Pipeline`: `StandardScaler` followed by the classifier.
Standardisation is a no-op for the tree-based models evaluated (scale-
invariant by construction) but required for the linear model, and using one
code path for every model choice avoids maintaining parallel
scaled/unscaled training logic. All classifiers are trained with
`class_weight="balanced"` by default, since the positive (chewing) class
constitutes only approximately 25–29% of windows — a real, moderate class
imbalance, not a 50/50 split.

Three classifier families are supported in the training script:
`RandomForestClassifier` (the family evaluated end-to-end in this work),
`LogisticRegression` (a linear baseline, included in the hyperparameter
search below but not carried through bout-level evaluation).

### 2.4 Hyperparameter search

Rather than training a single hand-picked configuration, model selection is
performed via an explicit hyperparameter sweep
(`chewing_ssl.classical_ml.train --sweep-config src/configs/train_ml.yaml`)
over a **20-combination grid**: 10 Random Forest variants (varying
`n_estimators` ∈ {100, 300, 500}, `max_depth` ∈ {unlimited, 10, 20},
`min_samples_leaf` ∈ {1, 5, 10}, `max_features` ∈ {`sqrt`, `log2`, 0.3}, and
`class_weight` ∈ {`balanced`, `balanced_subsample`}, in the specific
combinations listed in the sweep configuration) and 10 Logistic Regression
variants (varying inverse-regularisation strength `C` across
{0.01, 0.1, 1, 10, 100}, the L1/L2/elastic-net mixing parameter, solver
choice, and the presence/absence of class weighting). Every combination is
trained once, evaluated on both the validation and test splits, and its
fitted pipeline persisted independently
(`checkpoints/train_ml/sweep_models/{name}.joblib`), alongside every
combination's window-level validation and test predictions
(`checkpoints/train_ml/sweep_predictions/{name}_{val,test}.parquet`) and one
combined results table (`checkpoints/train_ml/sweep_results.csv`).

**Model selection criterion**: the combination achieving the highest
**validation-split** F1 score is selected — deliberately the validation
split, not test, so that the held-out test split remains untouched by any
model-selection decision (mirroring the same no-leakage discipline applied
to feature-median sanitisation in §2.2). The selected pipeline is copied to
`checkpoints/train_ml/best_model.joblib`, the artefact all downstream
bout-level evaluation is run against.

**Selected model (4-second configuration)**: `rf_06_regularized_leaf5` — a
`RandomForestClassifier` with `n_estimators=300`, `max_depth=None`
(unrestricted), `min_samples_leaf=5`, `max_features='sqrt'`,
`class_weight='balanced'` — validation F1 = 0.8232 (verified directly from
`checkpoints/train_ml/sweep_results.csv`), narrowly ahead of the runner-up
combinations (see §3.3).

**On what basis Random Forest specifically, over Logistic Regression, stated
precisely — the win is real but narrower and less repeatedly re-verified
than it might look.** The 20-combination grid above is a genuine head-to-head:
against the original 60/20/20 split, `rf_06_regularized_leaf5` (val F1
0.8232) beats the best Logistic Regression combination,
`lr_10_C10_liblinear` (val F1 0.8202), by 0.0030 — a real but narrow margin,
consistent with §3.3's own "17 of 20 combinations land within a 0.81-0.82
band" observation. **This 20-combination sweep was run exactly once, on
this original split.** Every later retrain of the classical-ML pipeline —
the 80/10/10 rerun in this document (§3.1), and a further raw-signal-clipping
data revision produced after this document was written (`checkpoints/
uncertainty_models/MANIFEST.md`'s "Gen 3", covering the nav-only/accel-only
sensor-ablation variants) — reused `rf_06_regularized_leaf5`'s exact
hyperparameters directly via a single-combination sweep config
(`src/configs/train_ml_rf06_only.yaml`), rather than re-running the full
grid each time. This was a deliberate efficiency decision, not an
oversight — that config file's own comment states it exists to reproduce
the known winner "without re-running the other 19 combos" — but it does
mean the RF-over-LR choice was genuinely re-contested only once, not
re-verified at every later data revision. **The one time a full grid *was*
rerun on later (80/10/10, clipping-fix-applied) data** — for the
sensor-ablation study — **Logistic Regression actually won narrowly**
(val F1 0.845 vs. Random Forest's 0.842), inside the same sub-0.01 plateau
described above. That later study substituted a dedicated Random-Forest-only
rerun back in for consistency with every other channel-subset model in its
own 6-way grid (all of which are Random Forest, since only RF has a natural
tree-ensemble uncertainty mechanism — see `docs/uncertainty.md`), not
because Logistic Regression was disqualified on accuracy grounds. The
honest summary: Random Forest is the project's consistent classical-ML
model family across every configuration by deliberate choice for continuity
and downstream uncertainty analysis, grounded in one genuine sweep win at
the outset — not a claim that it re-wins the sweep every single time it has
been checked.

**2-second configuration**: rather than re-running the full 20-combination
sweep at this window size, the 4-second sweep's winning hyperparameter
combination (`rf_06_regularized_leaf5`, identical hyperparameters) was
transferred and retrained on the 2-second feature set unchanged
(`checkpoints/train_ml_w2s0.25/`, confirmed by hash comparison to be the
sole entry in that configuration's own `sweep_results.csv`). A dedicated
sweep at the 2-second window size — which given the sensitivity observed in
§3.3, might recover some of the window-level performance drop this
configuration exhibits — is explicitly left as future work.

Validation F1: window-level F1 (not bout-level) — computed over all 18,272 validation windows in the 4s config, pooled into one number, not averaged per-recording. Concretely, from train.py's compute_metrics():


`preds = (probs >= 0.5).astype(int)
f1 = f1_score(y_true, preds, zero_division=0.0)`

where in run_sweep(): <br>

`val_probs = pipeline.predict_proba(X_val)[:, 1]` <br>
`val_metrics = compute_metrics(y_val.to_numpy(), val_probs, threshold=0.5)`<br>
X_val is the validation split's 850-feature matrix, transformed by the StandardScaler fit on the training split only (never refit on validation data).
y_val is each validation window's majority-vote ground-truth label (§2.1 in the doc — mean(per-timestep label) ≥ 0.5).
probs are the fitted pipeline's continuous predict_proba outputs on those validation windows, binarised at 0.5.
f1_score (scikit-learn) is computed once over the flat array of all 18,272 validation predictions/labels together — a single pooled F1, not a per-recording average.
This is the exact number pick_best(results, metric="val_f1") compares across all 20 sweep combinations to choose the winner — so 0.8232 specifically means: of rf_06_regularized_leaf5's predictions on those 18,272 held-out validation windows, F1 (precision/recall harmonic mean at threshold 0.5) was 0.8232.

---

## 3. Results — Train / Validation / Test

### 3.1 Split construction

Splitting follows a strict **subject-disjoint** discipline throughout the
project (decision 003): a participant assigned to one split contributes
*all* of their data to that split and none to any other, and this
assignment is made *before* any windowing, feature extraction, or model
fitting occurs. Windows, not participants, are never used as the unit of
split assignment — this prevents a model from ever being evaluated on
windows drawn from a recording it (or its normalisation/feature-median
statistics) was partly trained on.

Splitting is computed **independently per dataset** (participant
identifiers are not comparable across datasets — the same code, e.g.
`"p01"`, denotes different physical individuals in different studies), with
a fixed seed (42) for reproducibility. `src/configs/preprocessing.yaml` has
specified an **80% / 10% / 10%** train/validation/test division since
before this project's original 60/20/20 results were produced — but
`windowing.py`'s split-assignment step unconditionally reuses an existing
`splits.json` if one is already present on disk rather than regenerating
it from the current config, so every result up to and including §3.4/§5
below was actually computed against a `splits.json` frozen at an earlier,
~60/20/20 division (verified by direct comparison: re-running split
assignment fresh against the current config produces different per-dataset
participant counts than what was actually on disk). This was identified,
and a fresh `splits.json` regenerated from the config's real 80/10/10
ratio, on 2026-07-31.

**Original (60/20/20, frozen `splits.json`) division** — the split every
result below §3.4 was computed against, until the 2026-07-31 rerun:

| Dataset | Train | Validation | Test | Total participants |
|---|---|---|---|---|
| SIPMEatingStudy | 18 | 6 | 7 | 31 |
| SussexEatingUK | 24 | 8 | 8 | 40 |
| FEIT_eating_S05 | 21 | 7 | 8 | 36 |
| Office_eating_s05 | 15 | 5 | 6 | 26 |
| Spectacle_NonMeal | 13 | 4 | 5 | 22 |

**New (80/10/10, regenerated `splits.json`) division**, used for the
2026-07-31 rerun (`data/processed_80_10_10/splits.json`, verified directly
against the file on disk):

| Dataset | Train | Validation | Test | Total participants |
|---|---|---|---|---|
| SIPMEatingStudy | 24 | 3 | 4 | 31 |
| SussexEatingUK | 32 | 4 | 4 | 40 |
| FEIT_eating_S05 | 28 | 3 | 5 | 36 |
| Office_eating_s05 | 20 | 2 | 4 | 26 |
| Spectacle_NonMeal | 17 | 2 | 3 | 22 |

(`Spectacle_NonMeal` is a derived, fully-labelled — `label=0` throughout —
non-eating subset extracted from the larger `Spectacle` free-living
dataset, included to balance the otherwise all-chewing-context training
population with genuine non-eating background activity; see
`docs/data.md` for its construction.)

Both divisions are subject-disjoint by construction (decision 003); moving
from 60/20/20 to 80/10/10 trades some validation/test population for a
larger training population, at the cost of noticeably smaller absolute
validation/test participant counts per dataset (e.g. Office_eating_s05's
validation split shrinks from 5 to 2 participants) — a real, acknowledged
precision/data-efficiency tradeoff, not a free improvement.

The held-out test population under the original 60/20/20 split comprises
**944 ground-truth chewing bouts** — the fixed denominator against which
every *60/20/20* bout-level result in §5 is reported. **Under the new
80/10/10 split, the held-out (4 s) test population comprises **524
ground-truth chewing bouts** instead — smaller in absolute terms, per the
tradeoff just described, and verified directly from
`checkpoints/classical_ml_summary_80_10_10.csv`. Both denominators are
used in §5 below, clearly labelled by which split they belong to.**

### 3.2 What each split is used for

- **Training split**: fits (a) the `StandardScaler`'s per-feature mean/
  variance, (b) each candidate classifier's parameters, and (c) the
  feature-sanitisation medians (§2.2). Also the source of the global
  per-channel normalisation constants applied at the raw-signal level
  (§1.4) — note this is fit on a broader "pretrain-eligible" training
  population than classical ML's own train split alone, since those
  constants are shared across all of the project's pipelines.
- **Validation split**: used exclusively for **model selection** — the
  hyperparameter combination with the highest validation-split F1 is what
  becomes `best_model.joblib` (§2.4). Never used to fit any parameter or
  statistic directly.
- **Test split**: touched only once model selection is already finalised.
  Used for (a) the window-level classification metrics reported below, and
  (b) — after full postprocessing — the bout-level detection metrics
  reported in §5, which are the primary metrics of interest for this work's
  actual task (detecting individual chewing bouts, not classifying
  independent windows).

### 3.3 Window-level classification results

Window-level metrics are computed via
`compute_metrics(y_true, probs, threshold=0.5)`:

1. The fitted pipeline's `predict_proba` yields a continuous probability of
   the positive (chewing) class per window (the `StandardScaler`, fit on
   training data only, transforms the input features first).
2. Probabilities are binarised at a fixed 0.5 threshold.
3. Accuracy, precision, recall, and F1 (`scikit-learn`, with
   `zero_division=0.0`) are computed on the binarised predictions.
4. Area under the ROC curve is computed on the **continuous probabilities**
   directly, making it the one reported metric independent of the 0.5
   threshold choice.

**Selected model, `rf_06_regularized_leaf5` (4-second configuration, original
60/20/20 split)**:

| Split | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Validation | 0.896 | 0.814 | 0.833 | 0.823 | 0.951 |
| Test | 0.905 | 0.797 | 0.897 | **0.844** | **0.961** |

**Same model, retrained from scratch against the new 80/10/10 split**
(`checkpoints/train_ml_80_10_10/sweep_results.csv`, identical
hyperparameters, feature set, and non-overlapping-window segmentation —
verified directly from the file on disk, 2026-07-31):

| Split | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Validation | 0.904 | 0.793 | 0.895 | 0.841 | 0.962 |
| Test | 0.912 | 0.829 | 0.873 | **0.851** | **0.964** |

Both accuracy and AUC improve slightly under 80/10/10 (more training data),
and test F1 improves marginally (0.844 → 0.851); the balance between
precision and recall also shifts (test recall 0.897 → 0.873, test precision
0.797 → 0.829) — consistent with a larger, more representative training
population producing a model that is marginally more conservative rather
than a straightforward across-the-board improvement. Given the smaller
absolute test-participant counts under 80/10/10 (§3.1), some of this
movement should be attributed to a different, smaller sample of held-out
recordings rather than the ratio change alone — a caveat equally applicable
to the bout-level numbers once those are recomputed against this split.

**Same model family, 2-second configuration** (transferred hyperparameters,
§2.4, original 60/20/20 split — **not yet rerun under 80/10/10**, see the
status note at the top of this document):

| Split | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Validation | 0.879 | 0.775 | 0.806 | 0.790 | 0.937 |
| Test | 0.888 | 0.760 | 0.876 | 0.814 | 0.948 |

Window-level classification is noticeably noisier at 2 seconds
(validation F1 0.823 → 0.790) — an expected consequence of a shorter window
spanning fewer complete chewing cycles at the roughly 1.5–2 Hz mastication
rhythm (approximately 3–4 cycles at 2 s, versus 6–8 at 4 s), leaving less
signal to average over per feature vector.

**Sweep-wide observations** (full 20-row table in
`checkpoints/train_ml/sweep_results.csv`): 17 of 20 hyperparameter
combinations land within a narrow validation-F1 band of 0.81–0.82,
indicating that, within this hand-crafted feature representation,
hyperparameter tuning of either model family plateaus quickly and does not
offer large further gains. The one substantial outlier is a Logistic
Regression variant with `class_weight` explicitly disabled
(`lr_09_C1_l2_noclassweight`), which trades recall for precision on this
imbalanced label distribution and drops validation F1 to 0.788 — a clear
demonstration that class weighting materially matters for this task, not
an incidental modelling choice.

**A caveat on comparing window-level numbers across pipelines — applies to
the 2-second configuration and the original 60/20/20, 4-second numbers
only.** Because the same stride was applied uniformly to every split
(§2.1), the validation and test windows evaluated here **overlap** (50% at
the original 4 s configuration, 25% at 2 s) — unlike the deep-learning
pipelines' evaluation convention, in which validation/test windows are
deliberately non-overlapping specifically to avoid inflating effective
sample size through correlated neighbouring windows. Those numbers are
internally valid (correctly computed, with no cross-split data leakage)
but should be read as "this classifier does at least this well" at the
window level, rather than a clean apples-to-apples comparison against
another pipeline's own window-level metric computed under a non-overlapping
convention. **This caveat does NOT apply to the 80/10/10, 4-second
results reported just above (§2.1's 2026-07-31 change): those validation/
test windows are non-overlapping, so their window-level numbers ARE
directly comparable to the deep-learning pipelines' own window-level
convention** — though see `docs/ssl_masked.md` §6's discussion
for why window-level F1 still isn't a clean apples-to-apples number across
pipelines for a *different* reason (majority-vote window label vs.
per-timestep scoring). Bout-level metrics (§5) never carried this caveat
either way, since bout extraction operates on the dense, non-windowed
ground-truth and postprocessed-prediction arrays directly.

---

## 4. Postprocessing

### 4.1 Motivation: from window predictions to a continuous decision

The trained classifier emits one probability per **window** (a 4-second, or
2-second, span), but the actual task — detecting individual chewing
**bouts** with a start and end time — requires a decision at every
individual **timestep**. `chewing_ssl.classical_ml.upscale_predictions`
bridges this gap: for every distinct recording, it re-predicts across all
of that recording's windows and "paints" each window's probability across
every original-timestep index it covers. Where windows overlap (as they do
here, by design — §2.1), a given timestep's final probability is the
**mean of every window that covers it** — an implicit smoothing effect that
turns out to materially shape which further postprocessing is and is not
useful (§4.3). Ground truth requires no such reconstruction, since it is
already stored as a dense per-timestep array.

This computation is always performed **fresh, in memory**, from whichever
model checkpoint is currently pointed at — there is no cached prediction
artefact persisted to disk. An earlier version of this pipeline did cache
upscaled predictions to disk; this was removed as a reproducibility
hazard, since a cached artefact can silently go stale if the underlying
model is retrained without also regenerating it. Every evaluation reported
in this work reflects predictions computed directly from the exact model
checkpoint cited.

### 4.2 The postprocessing recipe

Once a continuous per-timestep probability trace exists, discrete bout
candidates are extracted via a shared postprocessing pipeline
(`chewing_ssl.common.evaluation.postprocess`), applied identically by every
pipeline in the project (a deliberate design choice, so that any
performance difference between classical ML and the deep-learning
pipelines reflects the underlying model, not different postprocessing
code). Applied in order:

1. **Two-threshold (hysteresis) binarisation**: a candidate bout begins
   when probability rises above a `high` threshold and does not end until
   probability falls back below a lower `low` threshold — a hysteresis band
   is more robust to brief probability dips near a single threshold than a
   single cutoff would be.
2. **Gap-filling**: bridges any non-chewing gap shorter than `max_gap_s`
   between two hysteresis-detected bouts, on the premise that a very brief
   dip likely reflects a momentary lapse in confidence rather than a
   genuine pause.
3. **Duration-bounded splitting**: any bout outside a plausible
   `[min_dur_s, max_dur_s]` duration range is recursively split at its
   lowest-probability interior point(s) until every resulting piece falls
   within bounds, recovering separate candidate bouts from an implausibly
   long merged span rather than leaving it intact.
4. **Interior-dip (valley) splitting**: a generalisation of (3) — any bout
   with an interior probability dip below a `valley_thresh`, even if not
   long enough to trigger the duration bound, is also split at that dip.
5. **Raw-signal envelope splitting** (§4.4): an additional split pass
   operating on the raw sensor amplitude envelope rather than the
   classifier's probability trace.
6. **Minimum-duration filtering**: any remaining bout shorter than
   `min_dur_s` is discarded outright as implausibly brief.

An optional smoothing step (a short moving average applied to the
probability trace before hysteresis) also exists in the shared pipeline,
but is **not used** for the classical-ML configuration
(`smooth_window_s=0.0`): the window-overlap averaging in §4.1 already acts
as an implicit smoother, so an additional explicit smoothing pass was found
to be unnecessary here (unlike the deep-learning pipelines, whose
per-timestep probability has no equivalent built-in smoothing and does
benefit from an explicit 2-second moving average).

### 4.3 Why the postprocessing recipe needed re-derivation for this pipeline

The postprocessing pipeline above did not arrive at its current form by
assumption — it was derived through a sequence of empirical investigations
specific to how this classifier's predictions actually behave, summarised
here because the reasoning is itself methodologically significant.

**Finding 1 — the classifier's raw output is already smooth, and naively
applying deep-learning-tuned postprocessing actively hurts it.**
Ground-truth bouts in the 57-recording test population (4-second
configuration) have a median duration of 10.9 s and a median gap of 2.5 s
between consecutive bouts within the same recording — but 40.6% of those
gaps are shorter than 2 seconds, reflecting genuinely closely-spaced real
eating episodes (e.g. successive bites within one meal), not isolated
events. **The raw, threshold-only prediction (no postprocessing at all)**
**already produces coherent bouts with a median duration of 10.0 seconds** and
0% under 2 seconds — there is no fragmentation problem of the kind
per-timestep deep-learning predictions typically exhibit, because the
window-overlap averaging in §4.1 already smooths the signal considerably.
Applying a postprocessing recipe tuned for a noisier, per-timestep deep-
learning probability trace (hysteresis + gap-filling + a flat minimum-
duration filter) to this already-smooth signal **reduced** bout-level F1
from 0.458 to 0.369. The mechanism: gap-filling (with a 1-second default)
bridges exactly the genuine gaps between distinct-but-close ground-truth
bouts (recall 40.6% of real gaps are under 2 s), merging two or more
genuinely separate chewing episodes into one over-long predicted blob.
Greedy overlap-based matching (§5.2) can then only credit that merged blob
against a single ground-truth bout; every other real bout it absorbed
becomes an uncredited false negative — precision improves (fewer, more
confident predicted bouts) but recall collapses, and F1 nets negative
overall.

**Finding 2 — duration-bounded splitting recovers the merged-blob problem,
directly confirming the diagnosis.** Generalising the postprocessing
recipe to include duration bounds and recursive splitting at low-
probability interior points (§4.2, step 3) — rather than leaving an
over-long merged blob intact — recovered bout-level F1 from 0.458 to
**0.679** under unchanged matching criteria, and specifically fixed the
regression the naive recipe introduced (0.369 → 0.679). This is a clean,
validated confirmation that the merged-blob mechanism identified in Finding
1 was the actual failure mode, not a coincidental correlation.

**Finding 3 — strict overlap-based matching over-penalises short,
slightly-shifted bouts, motivating an additional, complementary matching
criterion.** Independent of the postprocessing recipe itself, a short bout
whose predicted onset/offset are shifted by only a few hundred milliseconds
from ground truth can fall well below a 0.3 intersection-over-union (IoU)
threshold even though a human inspecting the corresponding plot would
readily call it correctly detected — visible as an unusually low median IoU
(≈0.58–0.61) among *already-matched* pairs. A second matching criterion,
onset/offset **tolerance matching** (§5.2), was introduced specifically to
target this failure mode independently of the postprocessing pipeline
itself; both matching criteria are computed for every evaluation run
regardless of which is used for the headline number reported (see §5.2 for
why a naive reading of "5-second tolerance" is, counter-intuitively,
*stricter* than IoU≥0.3 for bouts of this length, not more lenient).

**Finding 4 — gap-fill tuning has zero measured effect, and the reason
reveals a hard ceiling on probability-based fixes.** A natural follow-up
hypothesis — reduce `max_gap_s` so that gap-filling stops bridging distinct
bouts — was tested by sweeping this parameter from 0.2 to 1.0 seconds, and
produced **byte-identical results at every value tested**. Inspecting the
raw probability trace for a representative problem recording
(`SussexEatingUK/EMTEQRAP13`) explained why: a 170-second span that
hysteresis alone (i.e. before gap-filling ever runs) already treats as one
continuous bout, because probability never drops below the hysteresis
`low` threshold (0.35 at the time) anywhere in that span — it plateaus
around 0.6–0.7 across the entire multi-bite eating episode. Gap-filling was
never the actual merge mechanism for this failure mode: the window-averaged
probability trace itself never falls low enough to trigger a genuine gap in
the first place, since 50%-overlapping-window averaging smooths straight
through the brief real pauses between individual bites. This established
that any fix operating purely on the classifier's *probability* output has
a hard ceiling for this class of failure — motivating Finding 5.

**Finding 5 — the raw sensor signal, unlike the smoothed probability
trace, still shows bite-to-bite structure, and can be used to split
directly.** Although the classifier's probability plateaus through a
multi-bite episode, the underlying raw navigation-channel amplitude still
visibly modulates bite-by-bite. A rolling-RMS amplitude envelope of the
combined `nav_x_left_temple`/`nav_y_left_temple` magnitude (a 1.5-second
window — deliberately coarser than the approximately 0.5–0.7 s single-chew
ripple at the mastication rhythm, so as to detect inter-bite pauses rather
than individual chews) is computed directly from the raw signal, and
`scipy.signal.find_peaks` applied to its negation locates candidate
inter-bite pauses (valleys), with a peak-prominence threshold relative to
that specific bout's own envelope standard deviation and a minimum
resulting piece-duration floor. This needed genuine empirical tuning to get
right: a naive first attempt (prominence fraction 0.5, minimum piece
duration 2.0 s) over-split badly — recall improved (0.584 → 0.706) but
predicted-bout count exploded 4.4× (679 → 2,970) and precision collapsed
(0.811 → 0.224, net F1 0.679 → 0.340), because it was splitting at the
single-chew ripple itself rather than genuine bite-to-bite pauses. A
subsequent sweep over prominence (2.0–5.0) and minimum-piece-duration
(2.0–7.0 s), isolating the postprocessing effect under IoU-mode matching,
found a broad F1-maximising plateau around prominence ≈ 2.5–3.0 and
minimum piece duration ≈ 5–7 s.

**Finding 6 — a finer window/stride granularity motivates re-tuning
hysteresis itself, and the re-tuned values generalise back to the original
configuration.** Testing a 2-second/25%-overlap window configuration as a
means of improving boundary precision directly (rather than only refining
postprocessing around a coarser signal) produced noisier window-level
classification, as already noted in §3.3, which initially manifested
downstream as more false-alarm bouts under postprocessing settings tuned
for the smoother 4-second signal. These false alarms were not attributable
to brief noise blips (median false-alarm duration was 8.7 s, a legitimate
bout-scale length, not an instant spurious detection a stricter minimum-
duration filter would catch) but to a genuine hysteresis-threshold
mismatch: re-sweeping the hysteresis `high`/`low` thresholds against the
noisier 2-second predictions found a clear peak at `high=0.75, low=0.52`
(up from `0.65/0.35`) — a wider, higher-confidence band better suited to a
noisier probability trace, which needs more sustained evidence before
committing to a bout boundary. Re-sweeping the envelope-splitting
parameters (Finding 5) on top of this found a further peak at
`envelope_prominence=2.5, envelope_min_piece_dur=8.0` (from 3.0/6.0).
Critically, this re-tuned postprocessing configuration was **not**
2-second-specific: applying it to the original 4-second configuration
*also* improved its results, confirming the earlier hysteresis/envelope
values had simply not been optimal to begin with, rather than the two
window sizes genuinely requiring different postprocessing regimes. These
re-tuned values are the current shared defaults, reported in §4.4.

### 4.4 Final postprocessing configuration

The configuration below is shared verbatim by both window-size
configurations, and by every pipeline in the project (with the sole,
already-noted exception of the smoothing step, §4.2):

| Parameter | Value | Role |
|---|---|---|
| Hysteresis `high` | 0.75 | Bout-start threshold |
| Hysteresis `low` | 0.52 | Bout-end threshold |
| Gap-fill `max_gap_s` | 1.0 s | Maximum bridged gap between nearby bouts |
| Minimum duration `min_dur_s` | 2.0 s | Discard bouts shorter than this |
| Maximum duration `max_dur_s` | 35.0 s | Split bouts longer than this at their lowest-probability point(s) |
| Interior-dip threshold `valley_thresh` | 0.35 | Split at any interior probability dip below this (contribution on this dataset measured as negligible, F1 0.759→0.760 — there is usually no such dip to find, per Finding 4) |
| Envelope window | 1.5 s | Rolling-RMS window for the raw-signal amplitude envelope |
| Envelope minimum piece duration | 8.0 s | Minimum resulting bout length after envelope-based splitting |
| Envelope peak prominence | 2.5 | Relative-to-bout-std prominence threshold for envelope-valley detection |
| Probability smoothing window | 0.0 s (classical ML) | Not used for this pipeline — see §4.2 |

### 4.5 Effect of window-size configuration on postprocessed results

| | 4-second window | 2-second / 25%-overlap window |
|---|---|---|
| Bout F1, tolerance matching | 0.512 | **0.551** |
| Bout F1, IoU ≥ 0.3 | **0.763** | 0.743 |

The 2-second configuration is ahead on tolerance-mode F1 but trails
narrowly on IoU-mode F1, a difference within the range one would expect
given neither configuration has yet had a window-size-specific
hyperparameter sweep (§2.4) — i.e. this comparison is not yet a clean,
fully-optimised comparison of the two window sizes, only of the same
hyperparameters transferred across them.

### 4.6 Postprocessing re-tuning under the 80/10/10 split

**Motivation.** §4.3's postprocessing recipe (§4.4's final values) was
tuned against the original 60/20/20 split's test population. Since the
80/10/10 rerun changes both the model (retrained from scratch, §2.4) and
the test population it's evaluated against (§3.1), there's no guarantee
the old recipe is still optimal — worth re-checking rather than assuming.

**Method.** A dedicated coordinate-descent sweep tool was built for this
(`chewing_ssl.classical_ml.sweep_postprocess`, backed by
`chewing_ssl.common.evaluation.sweep`): the expensive step (computing each
test recording's probability trace from the model) runs once, then each
of the 8 postprocessing parameters is tuned one at a time — holding the
other 7 at their current running-best value — rather than a full
cross-product grid (which, at ~4-6 values per parameter across 8
parameters, would be tens of thousands of combinations; coordinate descent
costs `sum(choices per parameter)` instead, 37 combinations total here).
This mirrors the same coordinate-descent approach already used for this
project's SSL hyperparameter tuning (see the unified SSL document, §6).
Each step optimizes for `bout_tolerance_f1` specifically (configurable),
while recording `bout_iou_f1` alongside it for every combination tried, so
the two criteria's agreement/disagreement is always visible, not just the
one being optimized.

```bash
python -m chewing_ssl.classical_ml.sweep_postprocess \
  --model-path checkpoints/train_ml_80_10_10/best_model.joblib \
  --interim-root data/interim_normalized_80_10_10 \
  --features-path data/processed_80_10_10/ml_features/test.parquet \
  --out-csv checkpoints/postprocess_sweep_rf_4s_80_10_10.csv
```

**Result** (`checkpoints/postprocess_sweep_rf_4s_80_10_10.csv`, 37
combinations tried, real run against the 80/10/10 model and test split):

| Parameter | §4.4 default | Re-tuned value |
|---|---|---|
| Hysteresis `high` | 0.75 | **0.70** |
| Hysteresis `low` | 0.52 | 0.52 (unchanged) |
| Gap-fill `max_gap_s` | 1.0 s | **0.5 s** |
| Maximum duration `max_dur_s` | 35.0 s | **40.0 s** |
| Interior-dip `valley_thresh` | 0.35 | **0.25** |
| Envelope prominence | 2.5 | **2.0** |
| Envelope minimum piece duration | 8.0 s | 8.0 s (unchanged) |
| Smoothing window | 0.0 s | 0.0 s (unchanged) |

| Metric | §4.4 default recipe | Re-tuned recipe |
|---|---|---|
| Bout F1, tolerance = 5.0 s | 0.509 | **0.533** (+0.024, +4.7% relative) |
| Bout F1, IoU ≥ 0.3 | 0.744 | **0.752** (+0.008, +1.1% relative) |

**Both metrics improve together here** — unlike some individual steps
within the sweep (e.g. re-tuning `max_dur_s` alone moved IoU-F1 to 0.760,
higher than the final combo's 0.752, before the subsequent
`envelope_prominence` step traded a little of that back for a larger
tolerance-F1 gain) — confirming the two criteria don't always pull in the
same direction, consistent with §5.2's discussion of what each one
actually measures. **This is a genuine, verified improvement, not yet
adopted as the new default**: it is reported here as a finding, and
§5.4's headline numbers above still use the §4.4 defaults, pending a
decision on whether to promote the re-tuned recipe project-wide (it would
need re-validating against the masked-SSL pipeline too, since §4.2's
design principle is one shared recipe across both pipelines).

**Methodological caveat carried over unchanged from §4.7**: this re-tuning
was performed against the same 31 test recordings whose bout-level metrics
are reported in §5.4 — the same pre-existing lack of a dedicated
postprocessing-validation split applies here too, now compounded across
two separate tuning passes (the original §4.3 derivation, and this
80/10/10 re-sweep) against what is structurally the same kind of held-out
population each time.

### 4.7 Explicit methodological limitation

Every postprocessing parameter reported above (`high`/`low`,
`envelope_prominence`/`envelope_min_piece_dur`, the tolerance-matching
window in §5.2) was tuned by directly sweeping against the same 57
held-out test recordings whose bout-level metrics are reported as this
work's final results — there is no separate postprocessing-validation
split analogous to the train/validation/test separation the classifier's
own hyperparameters (§2.4) receive. This is an explicit, acknowledged gap
in methodological rigour: these values are demonstrated to work well on
this specific test population, but have not been verified to generalise to
an independent, unseen population held out specifically for postprocessing
tuning. Introducing such a split is identified as a priority item for
future work.

---

## 5. Evaluation

### 5.1 Window-level metrics — definitions

As already introduced in §3.3: accuracy, precision, recall, and F1
(scikit-learn implementations, `zero_division=0.0`) computed on
threshold-binarised predictions (threshold 0.5), plus AUC computed on the
continuous probability output. These answer the classification-style
question "is this independent window correctly labelled chewing / not
chewing" — informative for model selection (§2.4), but, as demonstrated
throughout §4.3, not a reliable proxy for the bout-level detection quality
that is the actual object of interest in this work.

### 5.2 Bout-level metrics — definitions

**Bout extraction.** A "bout" is any contiguous run of positive
(chewing) samples in a binary array sampled at 50 Hz — applied identically
to the dense ground-truth label array and to the postprocessed prediction
array (§4). Bouts shorter than 0.5 s are discarded as noise before any
matching is attempted.

**Matching.** Predicted bouts are matched against ground-truth bouts under
two independent criteria, both **always computed** for every evaluation
run regardless of which is used for the headline console/report number:

- **Intersection-over-union (IoU) matching**: a predicted bout matches a
  ground-truth bout if their temporal IoU is at least 0.3. Candidate pairs
  are matched greedily, highest-IoU pair first, each ground-truth and each
  predicted bout consumed at most once (a one-to-one assignment).
- **Onset/offset tolerance matching**: a predicted bout matches a
  ground-truth bout when **both** its onset and its offset fall within a
  fixed `tolerance_s` (5.0 seconds, the project default) of the ground
  truth's onset and offset respectively — independent of bout duration or
  overlap fraction. Candidate pairs are again matched greedily, tightest
  total (onset + offset) deviation first, one-to-one.

  Counter-intuitively, a fixed absolute tolerance of this kind is *stricter*
  than IoU ≥ 0.3 for bouts of the length typical in this dataset (median
  ground-truth bout 10.9 s), not more lenient as a literal reading of
  "5-second tolerance" might suggest: IoU's effective tolerance for a
  boundary shift scales with bout length (a 10.9 s bout can absorb a
  multi-second shift and still clear IoU ≥ 0.3), whereas a fixed absolute
  cap does not. Reproducing IoU ≥ 0.3's own effective leniency would
  require a tolerance closer to 15 seconds — at which point the criterion
  becomes closer to "detected somewhere nearby" than a meaningful
  boundary-precision check. 5.0 seconds (roughly half the median
  ground-truth bout duration) was chosen as a deliberate middle ground
  between the two extremes, not a value intended to reproduce IoU's own
  numbers.

**Precision, recall, F1**: for either matching criterion, true positives
are matched pairs, false positives are predicted bouts left unmatched, and
false negatives are ground-truth bouts left unmatched:
$\text{Precision} = TP/(TP+FP)$, $\text{Recall} = TP/(TP+FN)$, F1 their
harmonic mean. True/false positive/negative counts are **summed across all
57 test recordings before** precision, recall, and F1 are computed once —
a pooled aggregate, not a mean of per-recording F1 scores — so that
recordings with more ground-truth bouts are weighted proportionally to
their actual contribution to the 944-bout test population.

**Median IoU** is reported as a secondary diagnostic: the median IoU among
*already-matched* pairs only, a measure of boundary-alignment quality
independent of whether a pair was matched at all.

### 5.3 Final window-level results (test split)

| Configuration | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 4 s (`rf_4s`) | 0.905 | 0.797 | 0.897 | **0.844** | **0.961** |
| 2 s / 25% overlap (`rf_2s`) | 0.888 | 0.760 | 0.876 | 0.814 | 0.948 |

### 5.4 Final bout-level results (test split)

**`rf_4s`, 80/10/10 split (524 ground-truth bouts) — real, verified results
as of 2026-07-31**, from `checkpoints/rf_4s_80_10_10_eval_report.md` /
`checkpoints/classical_ml_summary_80_10_10.csv`, produced by:

```bash
python -m chewing_ssl.classical_ml.evaluate_bouts \
  --model-path checkpoints/train_ml_80_10_10/best_model.joblib \
  --features-path data/processed_80_10_10/ml_features/test.parquet \
  --interim-root data/interim_normalized_80_10_10 \
  --run-name rf_4s_80_10_10 --model-type random_forest \
  --summary-csv checkpoints/classical_ml_summary_80_10_10.csv \
  --out-csv checkpoints/ml_bout_ledger_80_10_10.csv
```

**This script's one pipeline-specific responsibility is computing the
per-timestep probability trace fresh from `--model-path` — everything
from postprocessing onward (`apply_full_postprocessing`, both matching
modes, the per-bite ledger) is shared code, identical to what the masked
SSL pipeline calls (see `docs/bite_evaluation.md`).**

| Configuration | Match mode | Precision | Recall | F1 | Median IoU |
|---|---|---|---|---|---|
| **`rf_4s_80_10_10`** | **IoU ≥ 0.3** | **0.723** | **0.767** | **0.744** | **0.622** |
| **`rf_4s_80_10_10`** | **Tolerance = 5.0 s** | **0.495** | **0.525** | **0.509** | **0.698** |

**(524 ground-truth bouts, 556 predicted bouts — smaller than the original
60/20/20 split's 944/970 simply because the 80/10/10 test population has
fewer held-out participants, §3.1; not a change in per-bout detection
difficulty.)**

**Raw (threshold-only, no postprocessing) vs. postprocessed, same model**
— confirms postprocessing still helps substantially under the new split,
consistent with §4.3's original findings: raw precision=0.351 recall=0.231
f1=0.278; postprocessed precision=0.495 recall=0.525 **f1=0.509**, a
**+0.231 F1** improvement (tolerance mode, GT=524, pred: 345 raw → 556
postprocessed).

**Original 60/20/20 split results (superseded above for `rf_4s`; `rf_2s`
was never rerun under 80/10/10, so this remains its only result)**:

| Configuration | Match mode | Precision | Recall | F1 | Median IoU |
|---|---|---|---|---|---|
| `rf_4s` (60/20/20) | IoU ≥ 0.3 | 0.753 | 0.773 | 0.763 | 0.676 |
| `rf_4s` (60/20/20) | Tolerance = 5.0 s | 0.505 | 0.519 | 0.512 | 0.746 |
| `rf_2s` (60/20/20, not rerun) | IoU ≥ 0.3 | 0.694 | 0.801 | 0.743 | 0.705 |
| `rf_2s` (60/20/20, not rerun) | Tolerance = 5.0 s | 0.514 | 0.593 | **0.551** | 0.747 |

(944 predicted-versus-ground-truth bout counts under the original split:
`rf_4s` predicts 970 bouts against 944 ground truth; `rf_2s` predicts
1,090 — verified directly from `checkpoints/classical_ml_summary.csv`.)

**Reading the movement**: IoU-mode F1 fell slightly under 80/10/10
(0.763→0.744) while tolerance-mode F1 fell more (0.512→0.509, roughly
flat) — both within the range plausibly explained by a smaller,
differently-composed test population (§3.1) rather than a genuine
regression in detection quality, especially since the underlying
window-level classifier's own test F1 slightly *improved* under 80/10/10
(§3.3: 0.844→0.851). Disentangling "smaller test set" from "non-overlapping
windows changed the classifier's probability characteristics" from "genuine
model quality change" would need a dedicated ablation (e.g. re-running the
80/10/10 model's bout-level eval against the *original* 60/20/20 test
participants) — not done here, flagged as a limitation in §5.8.

### 5.5 Bout-outcome taxonomy — a finer-grained diagnostic

Precision/recall/F1 answer "how many bouts were detected", but not "what
did detection failures look like". A complementary, match-mode-independent
taxonomy is computed by grouping temporally-overlapping ground-truth and
predicted bouts into maximal connected clusters, and labelling each
cluster by how many ground-truth (`n_gt`) and predicted (`n_pred`) bouts it
contains:

- **match** — a clean one-to-one correspondence between one ground-truth
  bite and one predicted bout;
- **merge** — one predicted bout spans two or more real bites (they have
  been bridged into a single detection);
- **split** — one real bite is instead covered by two or more separate
  predicted bouts;
- **tangle** — a chain of two or more ground-truth bites and two or more
  predicted bouts, all mutually overlapping — typically a dense run of
  closely-spaced real bites where boundary imprecision has decoupled bite
  count from predicted-bout count on both sides at once;
- **missed** — a ground-truth bite with no overlapping prediction at all;
- **false alarm** — a predicted bout with no overlapping ground-truth bite
  at all.

**80/10/10 split, `rf_4s_80_10_10` — real, verified results as of
2026-07-31**, computed directly from `checkpoints/ml_bout_ledger_80_10_10.csv`
(the `--out-csv` output of the same `evaluate_bouts.py` invocation as §5.4),
against the full 524-bite ground-truth population:

| Cluster label | 4 s, 80/10/10 (real) | 4 s, 60/20/20 (original) | 2 s, 60/20/20 (not rerun) |
|---|---|---|---|
| match | **172 bites (32.8%)** | 291 bites (30.8%) | 382 bites (40.5%) |
| merge | **38 bites (7.3%)** | 61 bites (6.5%) | 58 bites (6.1%) |
| split | **29 bites (5.5%)** | 45 bites (4.8%) | 66 bites (7.0%) |
| tangle | **243 bites (46.4%)** | 453 bites (48.0%) | 355 bites (37.6%) |
| missed | **42 bites (8.0%)** | 94 bites (10.0%) | 83 bites (8.8%) |

**The proportions are strikingly stable across the split change** — every
category's percentage under 80/10/10 is within ~2 points of the original
60/20/20, 4 s figures (match 32.8% vs. 30.8%, tangle 46.4% vs. 48.0%,
missed 8.0% vs. 10.0%). **This is a meaningful robustness check**: the
dominant-failure-mode finding below isn't an artefact of one particular
60/20/20 test population — it reproduces under a materially different
split (different participants, non-overlapping windows, retrained model).

The dominant failure mode at both window sizes is **tangle**, not a clean
merge or split — nearly half of all ground-truth bites at the 4-second
configuration (and over a third at 2 seconds) fall into a cluster where
boundary imprecision has decoupled bite count from predicted-bout count on
both sides simultaneously, rather than a single, individually-diagnosable
merge or split event. The 2-second configuration converts a meaningfully
larger share of bites into clean one-to-one matches (40.5% versus 30.8%
under the original 60/20/20 4 s split) at the cost of somewhat more split
events (7.0% versus 4.8%) — consistent with finer temporal resolution
improving boundary disambiguation within dense chewing bursts, at a modest
cost of occasionally over-segmenting a single bite. **Whether this still
holds at 2 s under 80/10/10 is untested (2 s was not rerun, §3.1).**

### 5.6 Boundary-alignment quality among matched pairs

**80/10/10, `rf_4s_80_10_10` — real, verified results**, computed from the
275 bite pairs matched under tolerance-mode criteria in
`checkpoints/ml_bout_ledger_80_10_10.csv`, alongside the original figures:

| Statistic | 4 s, 80/10/10 (real, n=275) | 4 s, 60/20/20 (original, n=490) | 2 s, 60/20/20 (not rerun, n=560) |
|---|---|---|---|
| Onset lag, mean (median) | **−0.51 s (−0.66 s)** | −0.43 s (−0.42 s) | −0.27 s (−0.43 s) |
| Offset lag, mean (median) | **−0.74 s (−0.88 s)** | −0.37 s (−0.44 s) | −0.62 s (−0.64 s) |
| Duration difference, mean (median) | **−0.23 s (−0.44 s)** | +0.06 s (+0.19 s) | −0.35 s (−0.20 s) |
| IoU among matched pairs, mean (median) | **0.687 (0.709)** | 0.730 (0.755) | 0.739 (0.767) |

**The systematic early-prediction tendency (negative lag) is somewhat more
pronounced under 80/10/10** than the original 60/20/20 4 s result — both
onset and offset lag roughly doubled in magnitude (onset −0.43→−0.51,
offset −0.37→−0.74), and matched-pair IoU is modestly lower (0.755→0.709
median). This is consistent with, though not conclusively caused by, the
non-overlapping-window change (§2.1): the original 60/20/20 numbers came
from a 50%-overlap-averaged probability trace (an implicit smoother, §4.1),
whereas the 80/10/10 predictions are painted from non-overlapping windows
only, which is less smoothed and could plausibly shift boundary timing
somewhat — a hypothesis worth checking directly (e.g. by comparing the raw,
pre-postprocessing probability traces) rather than asserted here as fact.
The direction and rough scale of the effect (a second or so, not tens of
seconds) is still consistent with §5.5's tangle-driven diagnosis: it's the
discrete match/no-match decision, not gross boundary error, that accounts
for most of the gap between window-level and bout-level performance.

### 5.7 Reference point: comparison against the self-supervised pipeline

**80/10/10, real, verified head-to-head — both pipelines now retrained and
re-evaluated against the identical 524-ground-truth-bite test population**
(`checkpoints/classical_ml_summary_80_10_10.csv`,
`checkpoints/masked_ablation_summary_80_10_10.csv`; full analysis in
`docs/bite_evaluation.md`, which owns this comparison in more depth):

| Pipeline | Window test F1 | Bout F1, IoU ≥ 0.3 | Bout F1, tolerance = 5 s |
|---|---|---|---|
| Classical ML, 4 s (`rf_4s_80_10_10`) | 0.851 | 0.744 | 0.509 |
| Masked-reconstruction SSL (`span_bigru_w200`, 80/10/10) | **0.826*** | **0.766** | **0.548** |

**(\*window-level F1 is not a like-for-like number between the two
pipelines even here — see the caveat below — included for completeness,
not as a fair ranking.)**

**Under the new 80/10/10 split, the self-supervised pipeline now leads
classical ML on BOTH bout-level metrics** (IoU-mode 0.766 vs. 0.744;
tolerance-mode 0.548 vs. 0.509) — a reversal from the original 60/20/20
comparison, where classical ML's 4 s configuration led on IoU-mode
(0.763 vs. 0.731) and only trailed on tolerance-mode (0.512 vs. 0.527).
**This should not be over-read as "SSL is now definitively better"**: both
models were retrained on a different, smaller test population (§3.1), the
classical-ML segmentation convention changed (non-overlapping, §2.1) in
the same rerun, and classical ML's own postprocessing re-tuning (§4.6)
found a further +0.024 tolerance-F1 improvement (to 0.533) not yet folded
into the number above — an equivalent re-tuning pass has not been run for
the SSL pipeline, so this comparison is not yet controlling for "has each
pipeline had its postprocessing optimized under this split." See
`docs/bite_evaluation.md` for the fuller discussion, including a
bite-by-bite ledger comparing where each pipeline succeeds/fails on the
exact same ground-truth bites.

**Original 60/20/20 comparison (superseded above, kept for reference)**:

| Pipeline | Window test F1 | Bout F1, IoU ≥ 0.3 | Bout F1, tolerance = 5 s |
|---|---|---|---|
| Classical ML, 4 s (`rf_4s`) | 0.844 | **0.763** | 0.512 |
| Classical ML, 2 s (`rf_2s`) | 0.814 | 0.743 | **0.551** |
| Masked-reconstruction SSL, best configuration | 0.799 | 0.731 | 0.527 |

Under that split, the classical machine-learning pipeline led on
window-level F1 and on IoU-mode bout-level F1 at both window sizes, while
the self-supervised pipeline edged ahead of the 4-second classical
configuration specifically on tolerance-mode F1, and the 2-second
classical configuration (never rerun under 80/10/10) remained ahead of
both on that metric.

### 5.8 Summary of limitations

1. **Window-level validation/test metrics use overlapping windows under the
   original 60/20/20 run (§3.3) — this is fixed for the 80/10/10, 4 s
   configuration (§2.1's non-overlapping change), but the 2-second
   configuration (never rerun) still carries this limitation.**
2. Postprocessing hyperparameters have no dedicated validation split
   (§4.7) — tuned directly against the same test population whose
   bout-level metrics are reported as final results. **This gap is now
   compounded, not resolved, by the 80/10/10 re-tuning pass (§4.6): two
   separate tuning passes have now been run against what is structurally
   the same kind of held-out population each time.**
3. Only the Random Forest classifier family has been carried through full
   bout-level evaluation; Logistic Regression and Gradient Boosting remain
   evaluated at the window level only.
4. The 2-second configuration has not received its own dedicated
   hyperparameter sweep (§2.4), **nor was it rerun under the 80/10/10 split
   at all (§3.1) — every 2 s number in this document is now from a
   different, older split than every 4 s number**, an asymmetry worth
   resolving before treating any 4 s-vs-2 s comparison in this document as
   conclusive.
5. The dominant bout-level failure mode (§5.5, "tangle") indicates that
   further gains are more likely to come from improved boundary
   resolution within dense chewing bursts — plausibly a finer window/
   stride grid, or a detection improvement rather than a postprocessing
   change — than from further postprocessing tuning alone, given how
   thoroughly the postprocessing parameter space has already been explored
   (§4.3). **This conclusion is reinforced, not undermined, by the
   80/10/10 rerun — §5.5's tangle-share is stable (46.4% vs. 48.0%) across
   the split change.**
6. **The 80/10/10 rerun is complete for the 4-second classical-ML
   configuration** (§3.1 split regeneration, §2.1 non-overlapping
   segmentation, §2.4 retrained model, §3.3 window-level results, §4.6
   postprocessing re-tuning, §5.4–§5.7 bout-level results) — **nothing
   about the 4 s configuration remains pending**. What remains open:
   (a) the 2-second configuration has not been rerun at all under 80/10/10
   (item 4 above); (b) the postprocessing re-tuning found in §4.6 has not
   been adopted as the new default (§5.4 still reports the §4.4 defaults);
   (c) §5.7's head-to-head against the SSL pipeline compares two models
   whose postprocessing has been tuned to different degrees (classical ML
   re-tuned under 80/10/10, SSL not yet) — see `docs/bite_evaluation.md`
   for the fuller cross-pipeline discussion.
7. **§5.6's finding that boundary-alignment quality (matched-pair IoU,
   onset/offset lag magnitude) is somewhat worse under 80/10/10's
   non-overlapping windows than the original 60/20/20's overlap-averaged
   ones is plausible but not proven** — disentangling "non-overlapping
   windows changed the probability trace's smoothness" from "a smaller,
   differently-composed test population" from "the retrained model itself
   behaves differently" was not attempted here (it would need e.g.
   re-running the 80/10/10 model's evaluation against the *original*
   60/20/20 test participants specifically) and is flagged as a natural
   follow-up.
