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

## Reproducibility

- Every run seeds all RNGs via `chewing_ssl.utils.seed.set_seed` and logs the
  seed and config to MLflow.
- Configs live in `configs/` and are passed as arguments — hyper-parameters are
  never hard-coded.
- Splits are subject-disjoint; the raw data is read-only.