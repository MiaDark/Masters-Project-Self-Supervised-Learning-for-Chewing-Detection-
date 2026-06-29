# Chewing-SSL

Self-supervised multimodal representation learning for chewing detection from
Emteq Labs smart-glasses sensor data (MSc project, COMP66060).

## Project structure

```
chewing-ssl/
├── configs/                 # YAML configs (data, pretrain, finetune)
├── data/                    # gitignored
│   ├── raw/                 # exactly as downloaded from S3
│   ├── interim/             # cleaned / synchronised intermediate data
│   └── processed/           # windowed, model-ready tensors
├── notebooks/               # exploration only; refactor logic into src/
├── scripts/                 # one-off shell/python entry points
├── src/chewing_ssl/         # the importable package (all real logic lives here)
│   ├── data/                # download.py, preprocessing, datasets
│   ├── models/              # encoder, pretext heads, detection head
│   ├── training/            # pretrain.py, finetune.py
│   ├── uncertainty/         # MC-Dropout inference
│   ├── energy/              # per-sensor energy model + trade-off analysis
│   ├── tracking/            # MLflow helpers
│   └── utils/               # config loading, seeding
└── tests/
```
