# Classical ML baseline

A non-deep-learning comparison point to the deep-learning pipelines in this
project. Same underlying data (`data/interim_normalized`, the same
subject-disjoint `data/processed/splits.json`), different pipeline: instead
of learning a representation, this path extracts ~850 hand-crafted
signal-processing features per window and trains a standard classifier
directly.

**The actual, completed comparison target is masked-reconstruction SSL**
(`docs/ssl_masked.md`) — its 8-way ablation (2 pretext tasks × 2
architectures × 2 window sizes) is fully pretrained, fine-tuned, and
evaluated, with results in `checkpoints/masked_ablation_summary.csv`. The
contrastive TF-C path (implemented per decision
[004](decisions/004_contrastive_pretext_task.md), `ssl/tfc/`) is
checkpoint-compatible with fine-tuning but has not yet been pretrained at
scale, and — unlike the other pipelines — has no dedicated writeup doc yet
(`docs/tfc.md` doesn't exist; if you're looking for it, decision 004 and the
module docstrings in `ssl/tfc/` are the only source right now). §7 below
gives the current side-by-side numbers against masked-reconstruction's best
combination.

**Superseded by later work, not reflected in this document**: everything
below still reports the original 60/20/20-split numbers only. The current,
higher-provenance numbers (80/10/10 split, retrained from scratch) are in
`docs/dissertation_classical_ml_sections.md` — read that document for
anything you'd cite in the dissertation itself; this one is kept as the
detailed investigation log its polished counterpart condenses from (the
step-by-step postprocessing derivation in §6 particularly). Newer still,
and not reflected in either document: a raw-signal clipping fix and a
genuine per-sensor (nav-only/accel-only) retrained model family
(`checkpoints/uncertainty_models/MANIFEST.md`'s "Gen 3"), feeding
`checkpoints/cross_sensor_evaluation/` and `docs/uncertainty.md`.

## Commands — the full pipeline, both window-size configurations

Five scripts, run in this order. `checkpoints/` holds every model-derived
artifact (models, plots) — `data/processed/` holds only preprocessed *data*
(segments, features), never predictions. There is no separate "upscale
predictions" script/artifact anymore — `evaluate_bouts.py` and
`plot_predictions.py` each compute predictions FRESH from `--model-path`
every run (`chewing_ssl.classical_ml.upscale_predictions.
compute_upscaled_recordings`, a library function, not a CLI) — see
`docs/bite_evaluation.md` for why a cached prediction artifact was a
reproducibility risk (silently stale if the model changed without
regenerating it) and was removed.

**4s window (50% overlap)**:
```bash
python -m chewing_ssl.classical_ml.segment --out data/processed/ml_segments --stride-seconds 2.0
python -m chewing_ssl.classical_ml.extract_features --segments-dir data/processed/ml_segments --out data/processed/ml_features
python -m chewing_ssl.classical_ml.train --sweep-config src/configs/train_ml.yaml --features-dir data/processed/ml_features --out-dir checkpoints/train_ml
python -m chewing_ssl.classical_ml.evaluate_bouts \
  --model-path checkpoints/train_ml/best_model.joblib --features-path data/processed/ml_features/test.parquet \
  --run-name rf_4s --model-type random_forest --summary-csv checkpoints/classical_ml_summary.csv
python -m chewing_ssl.classical_ml.plot_predictions --all \
  --model-path checkpoints/train_ml/best_model.joblib --features-path data/processed/ml_features/test.parquet \
  --out-dir checkpoints/plots/ml_predictions_4s/all
```

**2s window (25% overlap)** — same five steps, `_w2s0.25`-suffixed paths throughout, plus `--window-seconds 2.0`:
```bash
python -m chewing_ssl.classical_ml.segment --out data/processed/ml_segments_w2s0.25 --window-seconds 2.0 --stride-seconds 1.5
python -m chewing_ssl.classical_ml.extract_features --segments-dir data/processed/ml_segments_w2s0.25 --out data/processed/ml_features_w2s0.25
python -m chewing_ssl.classical_ml.train --sweep-config src/configs/train_ml.yaml --features-dir data/processed/ml_features_w2s0.25 --out-dir checkpoints/train_ml_w2s0.25
python -m chewing_ssl.classical_ml.evaluate_bouts \
  --model-path checkpoints/train_ml_w2s0.25/best_model.joblib --features-path data/processed/ml_features_w2s0.25/test.parquet \
  --window-seconds 2.0 --run-name rf_2s --model-type random_forest --summary-csv checkpoints/classical_ml_summary.csv
python -m chewing_ssl.classical_ml.plot_predictions --all \
  --model-path checkpoints/train_ml_w2s0.25/best_model.joblib --features-path data/processed/ml_features_w2s0.25/test.parquet \
  --window-seconds 2.0 --out-dir checkpoints/plots/ml_predictions_w2s0.25/all
```

`evaluate_bouts.py` and `plot_predictions.py` both call the shared
postprocessing/evaluation code (`common/evaluation/postprocess.py`/
`pipeline.py`) internally — see `docs/bite_evaluation.md` for that
architecture and for `compare_bites.py`, a later step that compares this
pipeline's output against masked-reconstruction SSL's directly.

## 1. Segmentation (`chewing_ssl.classical_ml.segment`)

Windowing.py's `WindowDataset` stores a lightweight *index* (segment file +
sample range) and slices the real window out of a cached parquet at access
time. Classical ML reads each window's data exactly once (to compute its
feature vector), so `segment_ml.py` instead **materializes** every window as
a dense array up front, with its label attached.

- **Source**: `data/interim_normalized` (same fully-preprocessed, filtered,
  derivative-computed, globally-normalized 10-channel segments as the
  deep-learning path) + the same `splits.json`, so that results stay directly
  comparable in terms of *which* recordings are in train/val/test.
- **Label**: majority vote over the window's dense per-timestep labels
  (`mean >= 0.5 → 1`), meaning one label per window (not per-timestep;
  classical ML needs a single target per feature vector).
- **Overlap**: window = 4.0s/2.0s (200 samples/100 samples @ 50Hz from `src/configs/preprocessing.yaml`).
  `--stride-seconds` controls overlap; the 4s configuration uses **50%** overlap, whereas the 2s configuration uses **25%** overlap, **for every split, including val/test**.

Output: `data/processed/ml_segments/{purpose}_X.npy` (float32,
`n_windows × 10 channels × 200 samples`), `_y.npy` (0/1 per window),
`_meta.parquet` (identity columns, aligned row-for-row with X/y).

## 2. Feature extraction (`chewing_ssl.classical_ml.extract_features`)

Each window's `(10, 200)` array becomes an 850-dimensional feature vector
using `emteqai.utils.features` drivers, applied per-channel:

| Family | Driver | Features/channel | What it captures |
|---|---|---|---|
| Statistical | `stat_driver` | 9 | mean, std, skew, kurtosis, percentiles, ... |
| Time-domain | `time_driver` | 14 | zero-crossing rate, RMS, peak count, slope, ... |
| Frequency-domain | `freq_driver` | 35 | Welch PSD band energies/ratios |
| Wavelet | `wavelet_driver` | 27 | Haar wavelet decomposition (level 3) energy per sub-band |

`10 channels × 85 features/channel = 850 features/window`.

Output: `data/processed/ml_features/{purpose}.parquet`, one row per window,
850 feature columns + 6 meta columns (`dataset, participant, tag, segment,
start, label`), plus `feature_medians.json` (the train-split medians used to
fill NaN/Inf in val/test — computed from train only, never val/test, to avoid
leakage). Shapes:

| | train | val | test |
|---|---|---|---|
| 4s window (`ml_features`) | `(56735, 856)` | `(18272, 856)` | `(20508, 856)` |
| 2s/25%-overlap (`ml_features_w2s0.25`) | `(75810, 856)` | `(24420, 856)` | `(27398, 856)` |

## 3. Model training (`chewing_ssl.classical_ml.train`)

`Pipeline(StandardScaler, classifier)` — scaling is a no-op for tree models
but required for `logistic_regression`, and one code path handles every
`--model` choice uniformly. Three choices, all with `class_weight="balanced"`
(the positive class is ~25-29% of windows, not 50/50):

| `--model` | Classifier | Notes |
|---|---|---|
| `random_forest` (default) | `RandomForestClassifier(n_estimators=300)` | reported below |
| `logistic_regression` | `LogisticRegression(max_iter=2000)` | linear baseline; **to be explored**|
| `gradient_boosting` | `HistGradientBoostingClassifier` | handles NaN natively; **to be explored**|

One advantage classical ML has that the deep-learning path lacks: feature
importance is directly inspectable (`plot_feature_importance` —
`model.feature_importances_` for tree models, `|coef_|` for logistic
regression), saved to `{model}_feature_importance.png` (single-run mode
only). A learned embedding has no equivalent — you can't ask a contrastive
encoder "which input statistic mattered."

### 3a. Hyperparameter sweep (`--sweep-config src/configs/train_ml.yaml`)

`train.py --sweep-config <path>` trains every combination listed in a YAML
config (10 RF, 10 LR), each varying the parameters most likely to move performance for
that model family, instead of the single `--model` run:

- Every combo's fitted `Pipeline` -> `checkpoints/train_ml/sweep_models/{name}.joblib`
- Every combo's val AND test window-level predictions -> `checkpoints/train_ml/sweep_predictions/{name}_{val,test}.parquet`
- One combined row-per-combo results table -> `checkpoints/train_ml/sweep_results.csv`
- The combo with the best `--sweep-metric` (default `val_f1` — **val**) is copied
  to `checkpoints/train_ml/best_model.joblib`, this is what
  `evaluate_bouts.py`/`plot_predictions.py --model-path` points at.

**Result of the first sweep run** (same train/val/test split as §4 below):

| name | model | val F1 | test F1 | train time |
|---|---|---|---|---|
| **rf_06_regularized_leaf5** (winner) | random_forest, `min_samples_leaf=5` | **0.823** | 0.844 | 50s |
| rf_04_shallow_depth10 | random_forest, `max_depth=10` | 0.823 | 0.832 | 27s |
| rf_02_baseline_300 (original single-run default) | random_forest | 0.819 | 0.846 | 42s |
| lr_10_C10_liblinear | logistic_regression, `C=10, solver=liblinear` | 0.820 | 0.843 | 322s |
| lr_09_C1_l2_noclassweight | logistic_regression, no `class_weight` | 0.788 | 0.844 | 4s |

Full 20-row table in `checkpoints/train_ml/sweep_results.csv`.

**Two honest findings, not just a leaderboard:**

1. **Hyperparameter tuning within this feature set plateaus fast.** 17 of
   20 combos land within val F1 0.81-0.82 — a ~1.5 point band. The single
   real outlier is `lr_09` (`class_weight=None` on data that's ~25-29%
   positive), which trades recall for precision and drops val F1 to 0.788.
2. **The window-level winner is not a clear bout-level winner.** Carrying
   `rf_06_regularized_leaf5` through `upscale_predictions.py` ->
   `evaluate_bouts.py` (§6b's then-current defaults) gave bout-level
   F1=0.755 (IoU) / 0.478 (tolerance) — statistically indistinguishable
   from the previous default model's 0.759 / 0.489. (Both numbers predate
   §6c's postprocessing re-tune; see there for current figures.) This is
   the same window-level-vs-
   bout-level gap §6 already documents, one level up: optimizing window
   classification (what the sweep selects on) doesn't reliably move
   bout-level detection (what actually matters for the end task), because
   bout-level scoring depends on the *temporal* structure of errors, not
   just how many windows are correctly labelled in aggregate. If bout-level
   performance is the real target, a sweep should probably select on a
   bout-level metric directly rather than `val_f1`

    **TODO**:
    - selecting best model on bout-level performance, rather than the current window level performance.

`rf_06_regularized_leaf5` is nonetheless what's currently at
`checkpoints/train_ml/best_model.joblib` and what §5-§6's numbers below
were regenerated against, since it's at least as good as the alternative
and the sweep infrastructure is there to re-select against a different
`--sweep-metric` if bout-level tuning becomes the priority.

## 4. Testing

`compute_metrics(y_true, probs, threshold=0.5)`:

1. `pipeline.predict_proba(X)[:, 1]`: the fitted `StandardScaler` (fit on
   train only) transforms `X`, then the classifier outputs a continuous
   probability of the positive (chewing) class per window.
2. `preds = (probs >= threshold).astype(int)`: binarize at `threshold=0.5`.
3. `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
   (scikit-learn, `zero_division=0.0`) computed on `(y_true, preds)`.
4. `roc_auc_score(y_true, probs)`: computed on the **continuous
   probabilities**, not `preds`; this is the one metric independent of the
   0.5 threshold choice.


**`rf_06_regularized_leaf5`, current best (§3a sweep winner, `checkpoints/train_ml/best_model.joblib`):**

| Split | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| val | 0.896 | 0.814 | 0.833 | 0.823 | 0.951 |
| test | 0.905 | 0.797 | 0.897 | **0.844** | **0.961** |

(Original single-run default, `random_forest` with library defaults +
`n_estimators=300`, for reference: val F1=0.819, test F1=0.846 — see §3a
for why the sweep's gain over this is real but small.)

**Caveat — not strictly comparable to the masked-reconstruction SSL
numbers.** The deep-learning path's `finetune_test`/`finetune_val` windows
are non-overlapping by design (`docs/architecture.md`: "overlapping eval
windows are highly correlated and would inflate the effective sample
size"). This classical-ML run's val/test windows **do** overlap (50% for the
4s config, 25% for 2s — `--stride-seconds` was applied uniformly to every
split), which correlates neighbouring evaluation windows and likely inflates
these window-level metrics somewhat versus a non-overlapping-eval comparison.
The test AUC=0.961/F1=0.844 numbers are internally valid (correctly computed,
no data leakage) but should be read as "classical ML does at least this
well" at the window level, not as a clean apples-to-apples number against
masked-reconstruction SSL's own window-level F1 (0.799 best combo,
`docs/ssl_masked.md`). See §7 below for the level at which the two
pipelines *are* directly comparable — bout-level detection, under shared
postprocessing code and parameters.

## 5. From window predictions to a continuous timeline

`train.py`'s model predicts one probability per **window** (4s / 200
timesteps), but a real recording needs a decision at every **timestep** for
visualization and bout-level evaluation.
`chewing_ssl.classical_ml.upscale_predictions.compute_upscaled_recordings`
(a library function, not a script — see `docs/bite_evaluation.md`) bridges
this: it re-predicts on `--features-path` and, for every `(dataset,
participant, tag, segment)` recording, "paints" each window's probability
across every original-row index it covers. Where windows overlap (they do
here — 50% stride), a timestep's final probability is the **mean of every
window covering it**. Ground truth needs no upscaling — it's already dense
per-timestep in `data/interim_normalized`.

This runs FRESH, in-memory, every time `evaluate_bouts.py`/
`plot_predictions.py`/`compare_bites.py` is invoked — there is no cached
prediction artifact on disk (there used to be, `checkpoints/
ml_predictions_upscaled/`; removed as a reproducibility risk, since a cached
artifact can silently go stale if the model is retrained without
regenerating it — see `docs/bite_evaluation.md`).

`plot_predictions.py` (`--all` for every test recording, or
`--dataset/--participant/--tag/--segment` for one) overlays
`nav_x_left_temple`/`nav_y_left_temple` against ground-truth shading (top
panel) and predicted shading (bottom panel) (`checkpoints/plots/ml_predictions/all/`).
By default the bottom panel now shows **postprocessed** predictions with
**tolerance-based** bout matching — the same defaults as `evaluate_bouts.py`
(see §6) — not the raw thresholded prediction it used previously. Pass
`--raw` to get the old raw-threshold/strict-IoU behavior back for comparison.

## 6. Does this pipeline need postprocessing? (bout-level investigation)

**Ground-truth bout shape** (across all 57 test recordings, no minimum
duration filter): median duration **10.9s**, median gap between consecutive
bouts in the same recording **2.5s** — but **40.6%** of those gaps are under
2s. These are real eating episodes close together in time (e.g. successive
bites in a meal), not isolated events.

**Raw predictions (threshold=0.5, no smoothing) are already smooth**: 596
predicted bouts vs. 944 ground-truth bouts, median predicted-bout duration
**10.0s**, **0%** under 2 seconds. There is no fragmentation problem to
solve — the upscaling step's probability-averaging over 50%-overlapping
windows already acts as an implicit smoother, so raw thresholded output
doesn't look like noisy per-timestep deep-learning predictions would.

**Applying the existing postprocessing defaults makes it worse:**

```
python -m chewing_ssl.classical_ml.evaluate_bouts

  raw (threshold only)                      prec=0.592  rec=0.374  f1=0.458  median_iou=0.577  GT=944 pred=596
  postprocessed (hysteresis+gapfill+mindur) prec=0.674  rec=0.254  f1=0.369  median_iou=0.612  GT=944 pred=356
  Postprocessing vs. raw: F1 delta = -0.089  -> HURTS
```

A parameter sweep (lighter hysteresis, shorter gap-fill, shorter minimum
duration) found **no configuration that beat raw** — every variant either
had zero effect (predictions are already too coarse at this timescale for
the sub-2s thresholds to trigger) or actively hurt recall. The mechanism:
`postprocess.py`'s gap-fill (default `max_gap_s=1.0`) bridges the very gaps
between distinct-but-close ground-truth bouts (40.6% of real gaps are <2s),
merging two-or-more genuine chewing episodes into one long predicted blob.
Greedy IoU matching can only assign that merged blob to *one* ground-truth
bout, so every other real bout it swallowed becomes a false negative —
precision goes up (fewer, more confident predicted bouts) but recall drops
sharply, and F1 nets negative.

**Conclusion**: the deep-learning path's postprocessing defaults were tuned
for per-timestep predictions from a noisier model and **do not transfer**
to this pipeline — do not apply them here as-is. This isn't evidence that
postprocessing is categorically useless for classical ML, only that this
particular hysteresis/gap-fill/min-duration recipe, at these parameter
values, was tuned for the wrong temporal resolution. If bout-level
detection needs to improve further, the more promising direction is
**boundary precision** (why do matched bouts have median IoU only
~0.58–0.61, and why are 591/944 ground-truth bouts unmatched at all) —
likely addressed by a finer window/stride grid (test at
`--stride-seconds` below 2.0, or a shorter window than 4s) rather than
post-hoc smoothing of an already-coarse prediction.

**Window-level vs. bout-level metrics are answering different questions,
and the large gap between them (test F1=0.846 vs. bout-level F1=0.458) is
expected, not a contradiction.** Window-level F1 asks "is each independent
4s window correctly labelled chewing/not-chewing" — a classification
question. Bout-level F1 asks "does a temporally-precise predicted span
(start, end) overlap a true span by ≥30% IoU", which is a much stricter,
segmentation-style question where a single correctly-classified window
contributes to many different candidate bout boundaries. A model can score
very well on the former while still merging or slightly mis-timing bouts
under the latter.

### 6a. Update: tolerance matching + duration-bounded postprocessing

The analysis above pinned two concrete, separately-fixable failure modes:

1. **Strict IoU over-penalizes short, slightly-shifted bouts.** A 1-2s bout
   that's onset/offset-shifted by a few hundred milliseconds can drop well
   below a 0.3 IoU threshold even though a human looking at the plot would
   call it correctly detected — this is the "median IoU only ~0.58-0.61"
   symptom.
2. **Gap-filling merges close-together real bouts into one blob**, and
   greedy IoU matching can only credit that blob against a single
   ground-truth bout, every other real bout it swallowed becomes a false
   negative, which is why applying the deep-learning path's postprocessing
   defaults *hurt* recall here.

Both are now addressed directly, in `chewing_ssl.common.evaluation`:

- **`postprocess.split_long_bouts`** (`--post-max-dur`, default 35.0s, paired
  with `--post-min-dur`, default 2.0s — real bite bouts run ~3-17s per the
  EDA in `docs/architecture.md`): any postprocessed bout outside `[min, max]`
  duration is implausible. Instead of leaving an over-long merged blob
  intact (failure mode 2) it's recursively split at its lowest-predicted-
  probability point(s) until every piece is within the cap, recovering
  separate candidate bouts that can each be matched to their own
  ground-truth bout.
- **`bouts.match_bouts_tolerant`** (`--match-mode tolerance`, the new
  default in both `evaluate_bouts.py` and `plot_predictions.py`): a
  predicted bout matches ground truth when its onset AND offset are both
  within `--tolerance-s` seconds of the ground truth's — independent of
  bout length or IoU. Targets failure mode 1 without touching the
  postprocessing pipeline at all.

Both are opt-in at the function level (`match_bouts`/plain `postprocess()`
without `max_dur_s` keep their old behavior — the SSL masked-reconstruction
pipeline's `evaluate_bouts.py`/`visualize_inference.py` calls are
unaffected) but are the new **defaults** for the classical-ML scripts.

**Measured results** (57 test recordings, GT=944 bouts, `random_forest`,
same run as §6):

| Config | precision | recall | F1 | median IoU |
|---|---|---|---|---|
| raw, IoU≥0.3 (§6 baseline) | 0.592 | 0.374 | 0.458 | 0.577 |
| old postprocessing, IoU≥0.3 (§6 baseline) | 0.674 | 0.254 | 0.369 | 0.612 |
| **new postprocessing (duration bounds), IoU≥0.3** | **0.811** | **0.584** | **0.679** | 0.587 |
| new postprocessing, tolerance=2.0s | 0.161 | 0.115 | 0.134 | 0.861 |
| new postprocessing, tolerance=5.0s (current default) | 0.359 | 0.258 | 0.301 | 0.782 |
| new postprocessing, tolerance=10.0s | 0.595 | 0.428 | 0.498 | 0.684 |
| new postprocessing, tolerance=15.0s | 0.810 | 0.583 | 0.678 | 0.586 |

**`split_long_bouts` is a clean, validated win**: F1 0.458 → 0.679 under
unchanged IoU≥0.3 matching, and it's what fixes the old postprocessing's
regression (0.369 → 0.679), which confirms that the merged-blob mechanism §6
diagnosed was the real problem, and splitting recovers it.

**`match_bouts_tolerant` at a literal "~2s / 100 rows" reading is
*stricter* than IoU≥0.3 here, not more lenient**, the opposite of the
original intuition. IoU's effective tolerance for a boundary shift scales
with bout length (median GT bout is 10.9s, so IoU≥0.3 already tolerates
shifts of several seconds); a fixed absolute onset/offset cap does not,
so a bout can be shifted just enough to clear IoU≥0.3 while failing a 2s
two-sided cap. The table above only reaches IoU-equivalent numbers around
tolerance=15s, by which point the criterion is closer to "detected
somewhere nearby" than a meaningful precision check. **5.0s (roughly half
the median GT bout duration) is the current default** — a deliberate
middle ground, not a value that reproduces IoU's numbers. If boundary
precision matters more than recall for your use case, prefer
`--match-mode iou` (paired with the new postprocessing) instead — it's the
best-measured combination so far.

### 6b. Update: why gap-fill tuning doesn't help, and what does

A natural next guess after §6a was "tune `--post-max-gap` down, so
gap-filling stops bridging distinct bouts." **Measured effect: zero.**
Sweeping `--post-max-gap` from 0.2s to 1.0s produces byte-identical
results at every value. Digging into why (`chewing_ssl.classical_ml.
evaluate_bouts.load_recordings`'s raw probability trace for the biggest
merged bout in a representative recording, `SussexEatingUK/EMTEQRAP13`):
a 170-second span that hysteresis alone (`fill_gaps` never even runs)
already treats as one continuous bout, because the probability **never
drops below 0.37** anywhere in that span — it plateaus around 0.6-0.7 for
the entire multi-bite eating episode. Gap-filling was never the merge
mechanism here: hysteresis's own `low` threshold (0.35) is never crossed
in the first place, because `train_ml.py`'s window-averaged predictions
(50%-overlapping windows, mean-pooled by `upscale_predictions.py`) smooth
straight through the brief pauses between individual bites.

That means a *probability*-based fix has a hard ceiling: `split_bouts`'s
`valley_thresh` (generalizing the old duration-only `split_long_bouts` to
also split at any interior probability dip, not just when a bout exceeds
`--post-max-dur`) is still a legitimate generalization and stays on by
default (`--post-valley 0.35`), but measured contribution on this dataset
is negligible (F1 0.759 → 0.760) — there's usually no dip to find.

**The raw sensor signal, unlike the classifier's probability, still shows
the bite-to-bite structure clearly** (visible as amplitude modulation in
`plot_predictions.py`'s output, e.g. `EMTEQRAP13`). `compute_envelope`
(rolling-RMS magnitude of `nav_x_left_temple`/`nav_y_left_temple`, 1.5s
window — deliberately coarser than the ~0.5-0.7s single-chew ripple at the
1.5-2Hz chewing rhythm, docs/architecture.md) plus `split_bouts_by_envelope`
(`scipy.signal.find_peaks` on the negated envelope, prominence relative to
that bout's own envelope std, minimum spacing between valleys) recovers
this structure directly from the signal instead of the smoothed
probability.

**This needed real tuning** — the naive first attempt (prominence_frac=0.5,
min_piece_dur_s=2.0) over-split badly: recall jumped (0.584→0.706) but
predicted-bout count exploded 4.4x (679→2970) and precision collapsed
(0.811→0.224, F1 0.679→0.340), because it was splitting at the ~0.5-0.7s
single-chew ripple, not just bite-to-bite pauses. A sweep over
`--envelope-prominence` (2.0-5.0) × `--envelope-min-piece-dur` (2.0-7.0),
all under `--match-mode iou` to isolate the postprocessing effect:

| prominence | min_piece_dur | precision | recall | F1 | pred bouts |
|---|---|---|---|---|---|
| (no envelope split, §6a) | — | 0.811 | 0.584 | 0.679 | 679 |
| 0.5 | 2.0 | 0.224 | 0.706 | 0.340 | 2970 |
| 2.0 | 4.0 | 0.602 | 0.844 | 0.703 | 1325 |
| 2.5 | 7.0 | 0.717 | 0.815 | 0.763 | 1072 |
| **3.0** | **6.0** (§6b's choice, superseded by §6c) | **0.747** | **0.771** | **0.759** | 975 |
| 4.0 | 4.0 | 0.790 | 0.632 | 0.702 | 756 |

F1 peaks in a broad plateau around prominence=2.5-3.0 / min_piece_dur=5-7;
`--envelope-prominence 3.0 --envelope-min-piece-dur 6.0` were the defaults
set in this round (round numbers inside that plateau -- superseded by §6c's
re-tune below). **Under the actual default match mode** (`tolerance`, 5.0s),
these defaults took postprocessed F1 from 0.300 to **0.489** and recall
from 0.258 to **0.497** — nearly double, on top of §6a's already-large jump
from raw. `EMTEQRAP13` alone (the plot above) goes from 1/22 GT bouts
detected under the original raw predictions to 18/22 under these defaults.

**Remaining gap**: even at these defaults, ~23-50% of GT bouts (matching
mode dependent) were still missed. The natural next lever, per §6's
original framing, is finer window/stride granularity in `segment_ml.py`'s
feature extraction — tried next, in §6c.

### 6c. Update: 2s windows + re-tuned postprocessing

Tried `segment.py --window-seconds 2.0 --stride-seconds 1.5` (25% overlap,
by this project's `overlap = 1 - stride/window` convention) against the
§6b defaults, transferring the §3a sweep-winning hyperparameters
(`rf_06_regularized_leaf5`) unchanged (no fresh hyperparameter sweep at
this window size):

| | 4s window (§6b) | 2s window, §6b's postprocessing |
|---|---|---|
| val F1 (window-level) | 0.823 | 0.790 |
| bout F1, tolerance | 0.478 | 0.495 |
| bout F1, IoU≥0.3 | 0.755 | 0.715 |

Window-level classification got slightly noisier (expected: a 2s window
only spans ~3-4 chew cycles at the 1.5-2Hz chewing rhythm vs. ~6-8 for 4s,
so there's less to average per window), and that noise showed up
downstream as more false-alarm bouts after postprocessing (1146 vs. 979)
— a mixed result under §6b's postprocessing, which was tuned for the
smoother 4s signal specifically, not this noisier one.

**Diagnosing the false alarms**: their duration wasn't the tell (median
8.7s vs. the 4s config's 12.0s — legitimate-length false detections, not
instant noise blips that a larger `--post-min-dur` would filter). What
actually helped was tightening hysteresis: sweeping `--post-high`/
`--post-low` against the 2s upscaled predictions found a clear peak at
**`--post-high 0.75 --post-low 0.52`** (up from 0.65/0.35) — a wider,
higher-confidence band that a noisier probability trace needs to avoid
spurious bout starts/restarts. Re-sweeping `--envelope-prominence`/
`--envelope-min-piece-dur` on top of that found a further peak at
**`--envelope-prominence 2.5 --envelope-min-piece-dur 8.0`** (from 3.0/6.0).

**This re-tuned postprocessing also improves the original 4s pipeline** —
it wasn't 2s-specific, §6b's hysteresis/envelope values were just not
optimal to begin with. Both configs improved, and **these are now the
package's global `--post-high`/`--post-low`/`--envelope-prominence`/
`--envelope-min-piece-dur` defaults**:

| | 4s window | 2s window |
|---|---|---|
| bout F1, tolerance (was / now) | 0.478 / **0.512** | 0.495 / **0.551** |
| bout F1, IoU≥0.3 (was / now) | 0.755 / **0.763** | 0.715 / **0.743** |

2s windows are now ahead on tolerance-mode F1 (0.551 vs. 0.512) and close
behind on IoU-mode F1 (0.743 vs. 0.763, within noise of each other given
neither config has had a window-size-specific hyperparameter sweep). On
`SIPMEatingStudy/p01` specifically (a visual reference point, already a
good recording): 33/40 GT bouts detected under the original defaults →
38/40 (1 false alarm) under 2s windows + re-tuned postprocessing.

**Caveat on methodology**: `--post-high`/`--post-low`/`--envelope-*` were
tuned by sweeping directly against these 57 test recordings' bout-level
metrics, the same numbers reported above — there's no separate
postprocessing-validation split the way `train.py`'s model hyperparameters
have val/test. This is a pre-existing gap in the whole postprocessing-tuning
approach (§6a's tolerance-s and §6b's envelope parameters were tuned the
same way), not new to this round, but worth being explicit about: these
values are picked to work well on this specific test set, not verified to
generalize beyond it.

**Not yet done**: a full 20-combo hyperparameter sweep (§3a) at the 2s/25%
window -- only one hyperparameter combination (last session's 4s winner)
has been tried here. Given window-level F1 dropped (0.823→0.790), a sweep
tuned for the noisier 2s features (more trees, different regularization)
might recover some of that gap and improve the bout-level numbers further.
2s windows also cost meaningfully more compute (75,810 train windows vs.
56,735 -- roughly 1.33x, from the smaller stride) for a currently-marginal
net gain, which is worth weighing before switching the documented default
away from 4s.

## 7. Comparing against masked-reconstruction SSL

`evaluate_bouts.py` supports a `--summary-csv` (default
`checkpoints/classical_ml_summary.csv`), the classical-ML analogue of
`docs/ssl_masked.md`'s own `checkpoints/masked_ablation_summary.csv`
— same column schema, and the same both-match-modes-always-computed
convention. As of this round, the postprocessing/evaluation code itself is
also shared, not just the schema — see `docs/bite_evaluation.md` for the
unified architecture (`common/evaluation/postprocess.py`/`pipeline.py`) and
`common.evaluation.compare_bites` for a bite-by-bite ledger comparing both
pipelines directly.

```bash
python -m chewing_ssl.classical_ml.evaluate_bouts \
  --model-path checkpoints/train_ml/best_model.joblib \
  --features-path data/processed/ml_features/test.parquet \
  --run-name rf_4s --model-type random_forest \
  --summary-csv checkpoints/classical_ml_summary.csv

python -m chewing_ssl.classical_ml.evaluate_bouts \
  --model-path checkpoints/train_ml_w2s0.25/best_model.joblib \
  --features-path data/processed/ml_features_w2s0.25/test.parquet --window-seconds 2.0 \
  --run-name rf_2s --model-type random_forest \
  --summary-csv checkpoints/classical_ml_summary.csv
```

**Current numbers** (both summary CSVs, verified against a real run of each
pipeline — GT=944 bouts in both, confirming the same held-out test
population). SSL's numbers below reflect `docs/bite_evaluation.md`'s
validated probability-smoothing addition (`smooth_window_s=2.0`, SSL-only —
its bout F1 was 0.674 IoU / 0.489 tolerance before that change):

| | window-level test F1 | bout F1, IoU≥0.3 | bout F1, tolerance=5s |
|---|---|---|---|
| **Classical ML, 4s (`rf_4s`)** | **0.844** | **0.763** | 0.512 |
| Classical ML, 2s (`rf_2s`) | 0.814 | 0.743 | **0.551** |
| Masked SSL, best combo (`span_bigru_w200`) | 0.799 | 0.731 | 0.527 |

Classical ML still leads on window-level F1 and IoU-mode bout F1 at both
window sizes, but the gap narrowed substantially on IoU-mode after SSL's
smoothing fix (0.674→0.731, now within 3 points of `rf_4s`'s 0.763). On
tolerance-mode specifically, `span_bigru_w200` (0.527) now beats `rf_4s`
(0.512), though `rf_2s` (0.551) is still ahead of both. Full masked-SSL
per-combination numbers (2 pretext tasks × 2 architectures × 2 window sizes)
are in `checkpoints/masked_ablation_summary.csv` (`docs/ssl_masked.md`
§8) — note only `span_bigru_w200` has been re-evaluated with the smoothing
fix so far; the other 7 rows still reflect the pre-fix default.

## Open items / not yet done

- **Full 20-combo hyperparameter sweep at the 2s/25% window** (§6c) — only
  one hyperparameter combination has been tried at this window size (the
  4s sweep's winner, transferred unchanged). Window-level F1 dropped there
  (0.823→0.790); a sweep tuned for the noisier 2s features specifically
  might recover it and push the bout-level numbers further.
- **Overlapping eval windows** (§4 caveat) — a clean non-overlapping-eval
  variant (`--stride-seconds` equal to window length for val/test only,
  overlap for train only, mirroring the deep-learning path's convention)
  would make the window-level comparison to masked-reconstruction SSL direct;
  not yet run. Applies to both the 4s and 2s configurations. (Bout-level
  comparison is already direct — see §7 — since it doesn't depend on
  window-level evaluation overlap at all.)
- **Bout-level boundary precision** (§6) — tolerance matching (§6a), the
  envelope split (§6b/§6c), and the 2s window (§6c) all paper over or
  recover around imprecise boundaries and merged predictions at
  evaluation/postprocessing time, but only the window-size change (§6c)
  touches the underlying classifier's temporal precision itself, and even
  that's unproven without a proper sweep. ~40-45% of GT bouts still go
  undetected even at §6c's best measured config (tolerance mode).
- **Postprocessing parameters have no held-out validation split** (§6c
  caveat) — `--tolerance-s`, `--envelope-prominence`/`--envelope-min-piece-dur`,
  and `--post-high`/`--post-low` have all been tuned directly against the
  57 test recordings' bout-level metrics reported throughout §6. A proper
  val/test split for postprocessing hyperparameters (mirroring what
  `train.py`'s model hyperparameters already have) would make these
  numbers more trustworthy.
- **`gradient_boosting`/`logistic_regression`** have not been run end-to-end
  through `evaluate_bouts.py` — only `random_forest` (specifically
  `rf_06_regularized_leaf5`, §3a's sweep winner) has full bout-level metrics
  at every stage.



For a supervisor update, I would avoid the implementation details and focus on the **main findings, why post-processing was needed, what was changed, and the impact**. Something like this:

---

### Classical ML chewing detection – update

I evaluated the Random Forest classifier using 4s windows and converted the window-level probabilities into continuous chewing timelines through probability upscaling (overlapping windows were averaged at each timestep). The classifier achieved strong window-level performance (**test F1 = 0.844, AUC = 0.961**), although these results are not directly comparable with the SSL models due to different evaluation window overlap strategies.

The main challenge was the gap between window-level classification and bout-level detection. Although the classifier correctly identified chewing windows, the resulting temporal bouts were often merged or had imprecise boundaries. Initial deep-learning post-processing (hysteresis + gap filling + minimum duration filtering) reduced performance because gap filling merged neighbouring chewing events, decreasing recall (**bout F1 dropped from 0.458 to 0.369**).

I therefore investigated alternative post-processing strategies. The main improvements were:

* **Duration-constrained splitting:** long predicted bouts were recursively split at probability minima to recover separate chewing episodes instead of treating multiple bites as one event.
* **Tolerance-based matching:** introduced to reduce the penalty for small onset/offset shifts during evaluation.
* **Signal-based bout splitting:** since probability averaging smooths over individual bite pauses, I used the original sensor signal envelope to identify internal valleys and separate merged bouts.

These changes substantially improved bout-level performance. With the updated post-processing pipeline, bout F1 increased from **0.458 → 0.679 (IoU matching)**, mainly by recovering previously merged events. Further tuning of hysteresis thresholds and envelope splitting improved performance again, reaching approximately:

* **4s windows:** bout F1 = **0.512** (tolerance matching), **0.763** (IoU ≥ 0.3)
* **2s windows:** bout F1 = **0.551** (tolerance matching), **0.743** (IoU ≥ 0.3)

I also tested reducing the window size from 4s to 2s to improve temporal resolution. This slightly reduced window-level classification performance (F1: 0.823 → 0.790), but improved temporal detection after post-processing because shorter windows preserved more boundary information.

Current conclusions:

* The classifier itself performs well at the window level.
* The main limitation is temporal resolution rather than classification accuracy.
* Post-processing is essential, but generic smoothing/gap-filling approaches are unsuitable because they merge naturally occurring consecutive bites.
* Shorter windows combined with signal-informed splitting provide better bout detection, although further validation is needed because post-processing parameters were tuned on the current test set.

Remaining work:

1. Run a dedicated hyperparameter sweep for the 2s window configuration.
2. Evaluate with non-overlapping validation/test windows for fair comparison with SSL models.
3. Tune post-processing parameters on a separate validation set rather than directly on test recordings.
