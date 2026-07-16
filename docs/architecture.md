# Architecture

## Package layout

All importable logic lives in `src/chewing_ssl/`. Notebooks and scripts are
entry points only — anything reusable belongs in the package so it is testable
and covered by CI.

```
src/chewing_ssl/
├── data/          # S3 download, preprocessing, dataset/windowing
├── models/        # multimodal encoder, SSL pretext heads, detection head
├── training/      # pretrain.py (SSL), finetune.py (chewing detection)
├── uncertainty/   # MC-Dropout inference
├── energy/        # per-sensor energy model + accuracy/energy trade-off
├── tracking/      # MLflow helpers (consistent run logging)
└── utils/         # config loading, seeding
```

## SSL Pretraining Plan Layout
Sussex and SIPM datasets are bite level annotations, whereas Spectacle is meal level annotations. 
Bite-level (chewing-event) detection is harder than meal-level detection, and that's actually informative: from the EDA, bite bouts in SIPM/Sussex are short (~3–17s, mean 7–11s) with the chewing-specific 1.5–2 Hz rhythm only visible at that resolution, whereas Spectacle's "bouts" are whole meals (mean 147s, up to 8 min), separable by gross signal envelope alone. So meal-level would score higher on paper, but it's a weaker demonstration of why the sensor suite and the masked-reconstruction pretext task matter — accelerometer-only HAR could mostly solve meal detection.

We define two independent problems:

- Primary task: bite-level chewing detection. Fine-tune and evaluate on SIPM + Sussex only (both have per-bite ground truth), subject-disjoint split (e.g. 60/20/20 of participants, per decision 003).
- Secondary task: free-living meal-detection transfer test. Take the same pretrained encoder (possibly with a small new head, or by aggregating bite-level outputs into time windows) and evaluate against held-out Spectacle participants' Label column. This tests whether what the model learned in the lab generalizes to real-world, multi-day, sensor-dropout conditions, which directly feeds the uncertainty/energy trade-off story, since that's precisely where confidence should degrade.

### Data allocation:
- SSL pretraining (unlabelled, label-blind): all of Spectacle, including the 15 files with no Label at all, plus the training-split participants of SIPM/Sussex. Per decision 001, Spectacle is the natural backbone here since it's largest and most free-living.
- Bite-level fine-tune/test: SIPM + Sussex, subject-disjoint — and critically, exclude these test participants from pretraining too, not just fine-tuning, so the encoder never sees them in any form before final evaluation.
Meal-level transfer test: a held-out slice of Spectacle participants, never used for pretraining or threshold-tuning, evaluated only at the end.

## Data flow

The pipeline moves data through three on-disk stages, each a separate directory,
so the raw download is never mutated and any step can be re-run from a clean
input.

```
S3 (emteq-data)
   │  chewing_ssl.data.download
   ▼
data/raw/        # exactly as downloaded, per-dataset structure preserved
   │  preprocessing: sync modalities, filter, normalise
   ▼
data/interim/    # cleaned, time-aligned per-recording data
   │  windowing: fixed-length segments, label assignment, split tiers
   ▼
data/processed/  # model-ready tensors + subject-disjoint splits
```

## Model pipeline

```
                 ┌─────────────────────────┐
data/processed → │  Multimodal encoder      │
                 └─────────────────────────┘
                       │            │
        ┌──────────────┘            └──────────────┐
        ▼                                          ▼
  SSL pre-training                          Fine-tuning (few labels)
  • masked reconstruction                   • chewing-detection head
  • cross-sensor prediction                        │
  (large unlabelled data)                          ▼
        │                                  ┌─────────────────┐
        └── pretrained weights ──────────► │ Predictions     │
                                           └─────────────────┘
                                              │            │
                                ┌─────────────┘            └────────────┐
                                ▼                                       ▼
                    MC-Dropout uncertainty                   Energy model
                    (confidence per config)                  (per-sensor power)
                                └──────────────┬───────────────────────┘
                                               ▼
                              Accuracy / confidence / energy trade-off
```

The rationale for each stage is recorded in the
[decision records](decisions/001_ssl_pretext_tasks.md).

### Normalization (raw -> interim)

Implemented in `chewing_ssl.data.preprocess`. Two reasons normalization is
required at all, both grounded in the EDA:

1. **Channels live on incomparable scales.** Nav ≈ -0.5 to 0.5, Proximity ≈
   5–45, Accelerometer ≈ -10 to 10 (m/s²). Fed in raw, a model's loss is
   dominated by whichever channel has the largest numbers (Proximity), not
   whichever is most informative for chewing.
2. **Per-participant baseline offset is a confound.** The Nav/Proximity
   baseline level differs by participant (fit, skin tone, calibration) — a
   nuisance variable, not signal. Left uncorrected, a model can partly learn
   "this baseline level correlates with this participant's chewing rate"
   instead of the actual temporal pattern, which leaks participant identity
   and hurts generalization to held-out test participants (decision 003).

Both reasons are addressed by z-score normalization (`(x - mean) / std`) per
channel, but *which* mean/std to use is a genuine design fork, so both
variants are produced side by side rather than picked upfront:

- **`personal`** — each segment is normalized using its own mean/std.
  Directly removes the per-participant baseline-offset confound. At
  deployment this needs a short calibration window or a running estimate;
  not a blocker, just an extra step.
- **`global`** — one mean/std per channel, pooled across the whole corpus
  (`data/interim/global_stats.json`), applied uniformly. Simpler at
  deployment (no calibration step), but risks the model keying on absolute
  baseline level as a per-participant shortcut rather than the dynamic
  chewing pattern.

Both are written to `data/interim/{personal,global}/<dataset>/<tag>/`, sharing
the same gap-segmentation and resampling (done once, cached, then normalized
two ways) so the only difference between them is the normalization itself —
making it a clean ablation once both a personalized and a global model are
trained and compared.

**Resolved:** `data/interim/global_stats.json` was originally pooled over
*all* recordings, computed before the subject-disjoint train/val/test split
existed -- flagged as a caveat during windowing. `chewing_ssl.data.refit_global_norm`
has since refit it on pretrain-eligible participants only (the same
participants whose windows feed the `pretrain` purpose: 1,212 segments) and
re-applied the corrected stats to every segment, including held-out
val/test/meal-test ones (so they're still normalized consistently, just
never having contributed to fitting the stats). This worked without
re-resampling anything: each `personal` segment's own `channel_stats` (in
its `.json` sidecar) make its per-recording normalization exactly invertible
(`raw = personal_value * std + mean`), validated by round-tripping the old
stats through the same path and confirming it reproduced the original
`global` variant to float32 precision (~1e-7). The pre-split stats are kept
at `data/interim/global_stats_presplit_backup.json` for reference.

### Windowing (interim -> processed)

Implemented in `chewing_ssl.data.windowing`. A preprocessed segment ranges
from 10 seconds to ~10.5 hours (Spectacle's free-living days) -- far too
variable to feed a model directly, and the wrong shape for either pretext
task in any case. Windowing turns segments into fixed-length, fixed-shape
training examples, which is required for four reasons:

1. **Pretext tasks need a bounded unit to operate on.** Masked reconstruction
   needs a defined span to hide and reconstruct; cross-sensor prediction
   needs a defined span over which one channel is predicted from the others.
   Neither parses against an open-ended stream.
2. **Batched training needs a fixed tensor shape.** Equal-length windows
   give every example in a batch the same shape with no padding/masking
   machinery.
3. **It's the source of view diversity for the model to learn from**, and
   the practical reason a modest number of raw recordings (208 files) turns
   into a large pretraining corpus (~1.8M windows) -- one long Spectacle
   segment alone yields thousands of overlapping windows.
4. **Window length is shared by pretraining and fine-tuning**, since both
   stages feed the same encoder -- it's an architectural choice now, not a
   per-stage hyperparameter.

**Window index, not materialized tensors.** With 4s windows and 50% overlap,
naively materializing every pretraining window as a dense array would mean
tens of GB of duplicated floats for data that already exists on disk as
interim segments. Instead `windowing.py` writes a lightweight index --one row
per window recording which segment file it comes from and its
`[start, end)` sample range -- to `data/processed/index/<variant>/<purpose>.parquet`,
and `WindowDataset` slices the real window out of a (small, LRU-cached)
segment array at access time.

**Window length and stride:**

- **4 seconds** (200 samples @ 50 Hz) -- long enough to contain several
  chewing cycles at the ~1.5–2 Hz rhythm found in the PSD analysis, short
  enough to generate many windows per segment.
- **50% overlap (stride 100)** for `pretrain` and `finetune_train` windows --
  more training signal, no leakage concern since these aren't held out for
  evaluation.
- **No overlap (stride 200)** for `finetune_val`, `finetune_test` and
  `meal_test` windows -- overlapping eval windows are highly correlated and
  would inflate the effective sample size, making metrics look more stable
  than they are.

**Labels: dense, per-timestep, not majority-vote.** Each fine-tuning/eval
window carries a label *array* the same length as the window (carried
straight from the interim segment), not one label per window. Bite bouts
(3–17s) are comparable in length to a 4s window, so a window can straddle a
bout boundary -- empirically, ~20% of sampled fine-tuning windows do (mixed
0/1 labels within the window). Dense labeling avoids collapsing that
boundary information into a single ambiguous label, at the cost of requiring
a sequence-output (not single-vector) fine-tuning head later. Pretraining
windows never carry a label even when the source segment has one --
masked reconstruction and cross-sensor prediction are self-supervised by
construction (decision 001).

**Subject-disjoint splits and window purposes** (`data/processed/splits.json`):
computed independently per dataset (SIPM and Spectacle happen to share a
`p01`–`p32`-style participant-naming convention, but they are different
studies with different physical people -- the split logic and the window
index both keep `dataset` and `participant` as a joint key throughout, so
this naming overlap is cosmetic, not a leakage risk).

- **SIPM, Sussex** (60/20/20 by participant): `train` participants feed both
  `pretrain` and `finetune_train`; `val`/`test` participants feed only
  `finetune_val`/`finetune_test` respectively, and are excluded from
  pretraining entirely -- the conservative reading of decision 003 ("all
  splits subject-disjoint... hold out whole participants") applied to
  pretraining too, not just fine-tuning.
- **Spectacle** (80/20 by participant): `pretrain_only` participants feed
  `pretrain`; the `meal_test_holdout` slice feeds the `meal_test` purpose
  (the secondary, free-living meal-detection transfer task) and is excluded
  from pretraining.

## Reproducibility

- Every run seeds all RNGs via `chewing_ssl.utils.seed.set_seed` and logs the
  seed and config to MLflow.
- Configs live in `configs/` and are passed as arguments — hyper-parameters are
  never hard-coded.
- Splits are subject-disjoint; the raw data is read-only.