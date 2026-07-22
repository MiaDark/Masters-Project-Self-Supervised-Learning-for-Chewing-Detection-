# Chewing-SSL

Self-supervised multimodal representation learning for chewing detection from
Emteq Labs OCOsense smart-glasses sensor data (MSc project, COMP66060).

This site documents the *why* and the *how* of the project. The public API is
documented from the code's docstrings (see **Reference**); these pages cover
everything that spans more than a single module.

## Where to look

- **[Getting started](getting_started.md)** — install, configure credentials, run a download.
- **[Data](data.md)** — the three OCOsense datasets: structure, sensors, known quirks.
- **[Architecture](architecture.md)** — package layout and how data flows through the pipeline.
- **[Project summary](project_summary.md)** — datasets → EDA → preprocessing → augmentation, up to pretraining.
- **[Classical ML baseline](classical_ml.md)** — segmentation, feature extraction, training, testing, and the bout-level postprocessing investigation for the non-deep-learning comparison model.
- **[Experiments](experiments.md)** — how runs are tracked and reproduced (work in progress).
- **[Decision records](decisions/0001-ssl-pretext-tasks.md)** — short notes on *why* key choices were made; these feed directly into the dissertation methodology.
- **[Reference](api/data.md)** — API documentation generated from docstrings.

## Documentation principle

Docstrings are the single source of truth for the API. These narrative pages
never paraphrase function behaviour — they link to the generated reference
instead, so there is only one place to update.