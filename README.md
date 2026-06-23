# Chewing-SSL

Self-supervised multimodal representation learning for chewing detection from
Emteq Labs smart-glasses sensor data (MSc project, COMP66060).

## Project structure

```
chewing-ssl/
├── configs/                 # YAML configs (data, pretrain, finetune)
├── data/                    # gitignored — never committed
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

The guiding rule: **notebooks are for looking, `src/` is for logic.** Anything
you'd run twice belongs in the package so it's testable and reproducible.

## 1. Setup

```bash
git clone <your-repo-url> chewing-ssl && cd chewing-ssl
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                 # makes `chewing_ssl` importable everywhere
cp .env.example .env             # then edit .env with your real values
```

Install the PyTorch build that matches your hardware (CPU vs CUDA) by following
the selector at https://pytorch.org/get-started/locally/ — don't rely on the
bare `torch` in requirements for GPU work.

## 2. AWS credentials

Use the standard AWS credential chain rather than hard-coding keys:

```bash
aws configure        # stores keys in ~/.aws/credentials (gitignored location)
```

Set your bucket and prefix in `.env` (`S3_BUCKET`, `S3_PREFIX`) or in
`configs/data.yaml`.

## 3. Download data

```bash
python -m chewing_ssl.data.download --dry-run     # preview what will be pulled
python -m chewing_ssl.data.download               # download into data/raw/
```

Re-runs are idempotent: files already present with a matching size are skipped.

## 4. MLflow tracking

```bash
make mlflow-ui      # opens http://127.0.0.1:5000
```

Every training run logs params, metrics, and the model artifact through
`chewing_ssl.tracking.mlflow_utils`. Promote your best checkpoints into the
MLflow Model Registry by passing `registered_name=` to `log_pytorch_model`.

For a shareable history, point `MLFLOW_TRACKING_URI` at a server backed by a
SQLite/Postgres store and an S3 artifact location instead of the local `file:`
store.

## 5. Suggested workflow

1. Branch per piece of work (`git checkout -b feat/masked-reconstruction`).
2. Prototype in a notebook, then move stable code into `src/`.
3. Run experiments via the package entry points; every run is logged to MLflow.
4. Keep configs in `configs/` and pass the config path as an argument — never
   bury hyper-parameters in code.
5. Commit small and often; open a PR to `main` when a phase is complete.

## Reproducibility checklist

- [ ] Seed set via `chewing_ssl.utils.seed.set_seed` and logged to MLflow.
- [ ] Train/val/test split is **subject-disjoint**.
- [ ] Config file logged as an MLflow artifact for every run.
- [ ] Raw data untouched; all transforms produce new files in `interim`/`processed`.
