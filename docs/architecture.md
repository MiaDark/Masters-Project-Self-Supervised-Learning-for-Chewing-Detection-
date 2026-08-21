# Architecture

## Package layout

All importable logic lives in `src/chewing_ssl/`.

```
src/chewing_ssl/
├── common/            # shared by both pipelines
│   ├── data/          # S3 download, preprocessing, dataset/windowing
│   ├── evaluation/    # bout extraction/matching, postprocessing
│   ├── tracking/      # MLflow helpers (consistent run logging)
│   └── utils/         # config loading, seeding
├── classical_ml/      # feature-based (sklearn) pipeline
├── ssl/               # self-supervised pipeline
│   ├── models/        # multimodal encoder, SSL pretext heads, detection head
│   ├── masked/        # variant 1: masked-reconstruction pretraining + finetune.py
│   └── tfc/           # variants 2 & 3: TF-C / TF-FC contrastive pretraining
├── uncertainty/       # MC-Dropout (SSL) + tree-ensemble (classical ML) predictive
│                      # uncertainty, calibration -- see docs/uncertainty.md
└── energy/            # per-sensor energy model + accuracy/energy trade-off --
                       # still an empty stub (__init__.py only), not yet built
```

## SSL Pretraining Plan Layout

**This section originally described the pre-implementation plan (3 datasets:
SIPM, Sussex, Spectacle) — corrected below to match what was actually
built (5 datasets), verified directly against the live window index
(`data/processed_80_10_10/index/*.parquet`).** The historical plan is kept
below in the "Original plan" note for anyone tracing how the scope grew.

Chewing bite bouts are short (empirically, per `docs/ssl_masked.md` §2.1's
EDA: mean 11.9s, median 10.1s, p5-p95 3.0-27.1s) with the chewing-specific
1.5-2 Hz rhythm only visible at that resolution — this is why bite-level
detection is the harder, primary task, not a meal/session-level one.

**Data allocation, as actually implemented** (`docs/data.md`,
`docs/project_summary.md` §1): all five datasets — SIPMEatingStudy,
SussexEatingUK, FEIT_eating_S05, and Office_eating_s05 (bite-level labelled
meals) plus Spectacle_NonMeal (a derived non-eating background set, `label=0`
throughout, built specifically to balance the otherwise all-eating-context
training population) — contribute to **both** pretraining and bite-level
fine-tune/test, subject-disjointly:

- **SSL pretraining** (unlabelled, label-blind): the `train`-split
  participants of all five datasets.
- **Bite-level fine-tune/val/test**: the `val`/`test`-split participants of
  all five datasets (confirmed: `finetune_test.parquet` genuinely spans all
  5 dataset names, not just two) — and, critically, these participants are
  excluded from pretraining too, not just fine-tuning, so the encoder never
  sees them in any form before final evaluation.

There is currently no separate meal-level transfer-test purpose in the
implemented pipeline (the original plan's third population) — every
dataset, including Spectacle_NonMeal, is evaluated at the same bite/window
level as the others.

*(Original plan, superseded: pretrain on Spectacle + SIPM/Sussex
train-split only; bite-level fine-tune/test on SIPM+Sussex only; a
separate Spectacle meal-level transfer test. FEIT_eating_S05 and
Office_eating_s05 were added, and Spectacle's role changed from
"meal-level transfer test" to "Spectacle_NonMeal, a bite-level-comparable
non-eating contributor," before real training began.)*

## Data flow

The pipeline moves data through three on-disk stages, each a separate directory,
so the raw download is never mutated and any step can be re-run from a clean
input.

```
S3 (emteq-data)
   │  chewing_ssl.common.data.download
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

1. **Channels live on incomparable scales.** Nav ≈ -0.5 to 0.5, Proximity ≈
   5–45, Accelerometer ≈ -10 to 10 (m/s²). Fed in raw, a model's loss is
   dominated by whichever channel has the largest numbers (Proximity), not
   whichever is most informative for chewing.
- **`global`** — one mean/std per channel, pooled across the whole corpus
  (`data/interim/global_stats.json`), applied uniformly. Simpler at
  deployment (no calibration step), but risks the model keying on absolute
  baseline level as a per-participant shortcut rather than the dynamic
  chewing pattern.

Written to `data/interim/{personal,global}/<dataset>/<tag>/`, sharing
the same gap-segmentation and resampling (done once, cached, then normalized).

### Windowing (interim -> processed)

Implemented in `chewing_ssl.common.data.windowing`.

1. **Pretext tasks need a bounded unit to operate on.** Masked reconstruction
   needs a defined span to hide and reconstruct; cross-sensor prediction
   needs a defined span over which one channel is predicted from the others.
   Neither parses against an open-ended stream.
2. **Batched training needs a fixed tensor shape.** Equal-length windows
   give every example in a batch the same shape with no padding/masking
   machinery.
3. **It's the source of view diversity for the model to learn from**, and
   the practical reason a modest number of raw recordings turns into a
   pretraining corpus much larger than the recording count alone would
   suggest -- one long Spectacle segment alone yields thousands of
   overlapping windows. **Corrected figure**: the original (60/20/20)
   window index totals 128,205 windows across all five purposes
   (`pretrain_train` 47,345 + `pretrain_val` 4,701 + `finetune_train`
   56,735 + `finetune_val` 9,154 + `finetune_test` 10,270, verified
   directly against `data/processed/index/*.parquet` — matches
   `docs/project_summary.md`'s own table); the regenerated 80/10/10 index
   totals ~155,681. Neither is close to the "~1.8M" previously stated here
   — that figure doesn't match either split generation and was likely an
   early, pre-windowing back-of-envelope estimate that was never corrected.
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

**Corrected below** — this originally described the pre-implementation
plan's binary SIPM/Sussex-vs-Spectacle split scheme; the actually-built
pipeline treats all five datasets uniformly (verified directly against
`data/processed_80_10_10/splits.json`):

- **Every dataset** (SIPMEatingStudy, SussexEatingUK, FEIT_eating_S05,
  Office_eating_s05, Spectacle_NonMeal), **80/10/10 by participant**,
  computed independently per dataset: `train` participants feed both
  `pretrain` and `finetune_train`; `val`/`test` participants feed only
  `finetune_val`/`finetune_test` respectively, and are excluded from
  pretraining entirely -- the conservative reading of decision 003 ("all
  splits subject-disjoint... hold out whole participants") applied to
  pretraining too, not just fine-tuning. Confirmed per-dataset counts
  (`docs/dissertation_classical_ml_sections.md` §3.1): SIPM 24/3/4, Sussex
  32/4/4, FEIT 28/3/5, Office 20/2/4, Spectacle_NonMeal 17/2/3.
- There is no separate meal-level transfer-test purpose in the current
  pipeline (see the "SSL Pretraining Plan Layout" section above) — the
  original plan's `pretrain_only`/`meal_test_holdout` Spectacle-specific
  purposes and the 60/20/20 SIPM/Sussex ratio are both superseded.

## Reproducibility

- Every run seeds all RNGs via `chewing_ssl.common.utils.seed.set_seed` and logs the
  seed and config to MLflow.
- Configs live in `src/configs/` and are passed as arguments — hyper-parameters are
  never hard-coded.
- Splits are subject-disjoint; the raw data is read-only.