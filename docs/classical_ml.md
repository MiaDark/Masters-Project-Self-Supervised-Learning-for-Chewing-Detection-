# Classical ML baseline

A non-deep-learning comparison point to the TF-C/TF-FC contrastive-pretraining path.
Same underlying data (`data/interim_normalized`, the same subject-disjoint
`data/processed/splits.json`), different pipeline: instead of learning a
representation, this path extracts ~850 hand-crafted signal-processing
features per window and trains a standard classifier directly.

Five scripts, run in this order:

```bash
python -m chewing_ssl.classical_ml.segment          --stride-seconds 2.0
python -m chewing_ssl.classical_ml.extract_features
python -m chewing_ssl.classical_ml.train         --model random_forest
python -m chewing_ssl.classical_ml.upscale_predictions --model-path checkpoints/train_ml/random_forest.joblib --window-seconds 4.0
python -m chewing_ssl.classical_ml.evaluate_bouts
```

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
  (`mean >= 0.5 → 1`)m neaning one label per window (not per-timestep;
  classical ML needs a single target per feature vector).
- **Overlap**: window = 4.0s (200 samples @ 50Hz, from `configs/preprocessing.yaml`).
  `--stride-seconds` controls overlap; the current run uses **50% overlap**
  (`--stride-seconds 2.0`) **for every split, including val/test** — a
  deliberate deviation from the deep-learning path (which only overlaps
  `train`/`pretrain` windows, never eval windows), made to get more windows
  out of a modest amount of labelled data. See the caveat in §4.

Output: `data/processed/ml_segments/{purpose}_X.npy` (float32,
`n_windows × 10 channels × 200 samples`), `_y.npy` (0/1 per window),
`_meta.parquet` (identity columns, aligned row-for-row with X/y).

**Current split sizes** (50% overlap):

| Split | Windows | Participants | Positive rate |
|---|---|---|---|
| train | 56,735 | 91 | 25.4% |
| val | 18,272 | 30 | 29.1% |
| test | 20,508 | 34 | 28.7% |

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

**Sanitization.** Some windows are genuinely flat or near-constant (e.g. a
recording with a dead channel, or a quiet non-eating stretch) — this drives
degenerate `0/0` divisions inside the frequency-domain driver and
catastrophic-cancellation warnings in the statistical driver's
skew/kurtosis. `sanitize()` replaces `inf`/`-inf` with `NaN`, then fills all
`NaN` with **train-split-only medians** (`feature_medians.json`) — the same
no-leakage principle as normalization: val/test never influence a fitted
statistic. This isn't a theoretical edge case — it fires on real data in
every split.

Output: `data/processed/ml_features/{purpose}.parquet`, one row per window,
850 feature columns + 6 meta columns (`dataset, participant, tag, segment,
start, label`). Shapes: train `(56735, 856)`, val `(18272, 856)`, test
`(20508, 856)`.

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
regression), saved to `{model}_feature_importance.png`. A learned embedding
has no equivalent — you can't ask a contrastive encoder "which input
statistic mattered."

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

Both val and test are run through this identically; val is reported for
reference (it's what you'd watch if you were doing a hyperparameter sweep),
test is the final, held-out number.

**Random Forest, current run:**

| Split | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| val | 0.895 | 0.824 | 0.814 | 0.819 | 0.949 |
| test | 0.908 | 0.814 | 0.881 | **0.846** | **0.959** |

**Caveat — not strictly comparable to the TF-C/TF-FC numbers.** The
deep-learning path's `finetune_test`/`finetune_val` windows are
non-overlapping by design (`docs/architecture.md`: "overlapping eval
windows are highly correlated and would inflate the effective sample
size"). This classical-ML run's val/test windows **do** overlap (50%,
`--stride-seconds 2.0` was applied uniformly to every split), which
correlates neighbouring evaluation windows and likely inflates these
window-level metrics somewhat versus a non-overlapping-eval comparison. The
test AUC=0.959/F1=0.846 numbers are internally valid (correctly computed,
no data leakage — see the race-condition fix note below) but should be read
as "classical ML does at least this well," not as a clean apples-to-apples
number against TF-C (F1=0.759) / TF-FC (F1=0.808).

*(Historical note: an earlier run reported a similar-looking F1=0.842 from a
race condition — `segment_ml.py` was re-run with overlap while a background
`extract_ml_features.py` job was still reading the stale pre-overlap files,
producing train/val features computed on different data than test. That
result was discarded; the numbers above are from a clean, fully re-run
pipeline with `segment` added to the row-identity key and verified 0
duplicate rows in every split's meta table.)*

## 5. From window predictions to a continuous timeline

`train_ml.py`'s model predicts one probability per **window** (4s / 200
timesteps), but a real recording needs a decision at every **timestep** for
visualization and bout-level evaluation. `upscale_predictions.py` bridges
this: it re-predicts on `data/processed/ml_features/test.parquet` and, for
every `(dataset, participant, tag, segment)` recording, "paints" each
window's probability across every original-row index it covers. Where
windows overlap (they do here — 50% stride), a timestep's final probability
is the **mean of every window covering it** (`n_windows_covering` records
how many). Ground truth needs no upscaling — it's already dense
per-timestep in `data/interim_normalized`.

Output: `data/processed/ml_predictions_upscaled/manifest.csv` (the identity
source of truth — filenames are opaque `recording_NNNN.parquet` indices,
since `tag`/`segment` can themselves contain underscores) + one parquet per
recording (`row_idx, pred_prob, pred_label, n_windows_covering`).

`plot_predictions.py` (`--all` for every test recording, or
`--dataset/--participant/--tag/--segment` for one) overlays
`nav_x_left_temple`/`nav_y_left_temple` against ground-truth shading (top
panel) and predicted shading (bottom panel) — visual confirmation that
predicted bouts track ground truth closely across every test recording
checked (`checkpoints/plots/ml_predictions/all/`).

## 6. Does this pipeline need postprocessing? (bout-level investigation)

The deep-learning path already has a postprocessing module
(`chewing_ssl.common.evaluation.postprocess`: hysteresis thresholding → gap-filling →
minimum-duration filtering) and a bout-level evaluator
(`evaluate_bouts.py`, IoU-matching between predicted and ground-truth
chewing bouts). `evaluate_ml_bouts.py` reuses both — `extract_bouts`/
`match_bouts` unchanged, applied to this pipeline's upscaled predictions —
to check whether the same smoothing helps here too, rather than assuming it
does because it exists.

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
(start, end) overlap a true span by ≥30% IoU" — a much stricter,
segmentation-style question where a single correctly-classified window
contributes to many different candidate bout boundaries. A model can score
very well on the former while still merging or slightly mis-timing bouts
under the latter.

## Open items / not yet done

- **No hyperparameter search** for `train_ml.py` — current numbers use
  library defaults + `class_weight="balanced"`, not a tuned config (see §3).
- **Overlapping eval windows** (§4 caveat) — a clean non-overlapping-eval
  variant (`--stride-seconds` equal to window length for val/test only,
  overlap for train only, mirroring the deep-learning path's convention)
  would make the comparison to TF-C/TF-FC direct; not yet run.
- **Bout-level boundary precision** (§6) is the concrete next lever if
  bout-level scores need to improve — not postprocessing tuning.
- **`gradient_boosting`/`logistic_regression`** have not been run end-to-end
  through `upscale_predictions.py`/`evaluate_ml_bouts.py` — only
  `random_forest` has full metrics at every stage.
