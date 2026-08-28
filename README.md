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
│   │   └── masked/                 # variant 1: masked-reconstruction pretraining
│   ├── uncertainty/                # MC-Dropout inference
│   └── energy/                     # per-sensor energy model + trade-off analysis
└── tests/
    ├── common/
    └── ssl/
```

See [docs/architecture.md](docs/architecture.md) for what each module does and
[docs/classical_ml.md](docs/classical_ml.md) for the classical-ML pipeline in detail.

## Quick start

```bash
git clone https://github.com/MiaDark/Masters-Project-Self-Supervised-Learning-for-Chewing-Detection-
cd Masters-Project-Self-Supervised-Learning-for-Chewing-Detection-
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[classical-ml]"   # optional
```

The raw sensor recordings are not included in this repository (proprietary,
Emteq Labs data-sharing agreement) -- cloning gets you the full pipeline code,
not the data. See [docs/data.md](docs/data.md) for the expected data layout if
you have access to the recordings separately.

### Tests

```bash
pytest
```

### Running a pipeline

Each stage is a runnable module, e.g.:

```bash
# Classical ML: segment -> extract features -> train -> evaluate
python -m chewing_ssl.classical_ml.segment --interim-root data/interim_normalized_80_10_10
python -m chewing_ssl.classical_ml.extract_features --segments-dir data/processed_80_10_10/ml_segments
python -m chewing_ssl.classical_ml.train --features-dir data/processed_80_10_10/ml_features
python -m chewing_ssl.classical_ml.evaluate_bouts --model-path checkpoints/train_ml/best_model.joblib

# Self-supervised: pretrain -> fine-tune -> evaluate
python -m chewing_ssl.ssl.masked.pretrain --architecture bigru --pretext-task span
python -m chewing_ssl.ssl.masked.finetune --checkpoint <pretrain-checkpoint>
python -m chewing_ssl.ssl.masked.evaluate_bouts --checkpoint <finetune-checkpoint>
```

See [docs/classical_ml.md](docs/classical_ml.md) and [docs/ssl_masked.md](docs/ssl_masked.md)
for the full command sequences, flags, and expected outputs behind this dissertation's results.
