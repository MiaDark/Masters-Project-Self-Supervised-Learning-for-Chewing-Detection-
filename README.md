# Chewing-SSL

Self-supervised multimodal representation learning for chewing detection from
Emteq Labs smart-glasses sensor data (MSc project, COMP66060).

## Project structure

The package is split into two independent pipelines -- **classical ML** and
**SSL** -- plus a `common/` package holding everything both depend on.

```
chewing-ssl/
├── configs/                        # YAML configs (data, preprocessing, augmentation)
├── src/chewing_ssl/
│   ├── data/                       # gitignored data lake (raw/interim/processed + reports)
│   ├── common/                     # shared by both pipelines
│   │   ├── data/                   # download, preprocess, windowing/splits, EDA
│   │   ├── evaluation/             # bout extraction/matching, postprocessing
│   │   ├── tracking/               # MLflow helpers
│   │   └── utils/                  # config loading, seeding
│   ├── classical_ml/               # feature-based (sklearn) pipeline: segment,
│   │                                # extract_features, train, upscale_predictions,
│   │                                # evaluate_bouts, plot_predictions
│   ├── ssl/                        # self-supervised pipeline
│   │   ├── models/                 # encoders, heads, augmentation -- shared across variants
│   │   ├── masked/                 # variant 1: masked-reconstruction pretraining
│   │   └── tfc/                    # variants 2 & 3: TF-C and TF-FC contrastive pretraining
│   ├── uncertainty/                # MC-Dropout inference
│   └── energy/                     # per-sensor energy model + trade-off analysis
└── tests/
    ├── common/
    └── ssl/
```

See [docs/architecture.md](docs/architecture.md) for what each module does and
[docs/classical_ml.md](docs/classical_ml.md) for the classical-ML pipeline in detail.
