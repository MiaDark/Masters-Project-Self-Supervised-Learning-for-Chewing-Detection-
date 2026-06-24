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