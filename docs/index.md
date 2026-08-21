# Chewing-SSL

Self-supervised multimodal representation learning for chewing detection from
Emteq Labs OCOsense smart-glasses sensor data (MSc project, COMP66060).

This site documents the *why* and the *how* of the project.

## Where to look

- **[Getting started](getting_started.md)** — install, configure credentials, run a download.
- **[Data](data.md)** — the five OCOsense datasets: structure, sensors, known quirks.
- **[Architecture](architecture.md)** — package layout and how data flows through the pipeline.
- **[Preprocessing](preprocessing.md)** — raw CSV → windowed model input, and the normalization ordering.
- **[Project summary](project_summary.md)** — datasets → EDA → preprocessing → augmentation, up to pretraining.
- **[Classical ML baseline](classical_ml.md)** — segmentation, feature extraction, training, testing (superseded numbers-wise by the dissertation sections below; kept as the detailed investigation log).
- **[Dissertation sections: classical ML](dissertation_classical_ml_sections.md)** — the polished, current-numbers (80/10/10 split) writeup of the classical-ML pipeline, methods through results.
- **[SSL masked reconstruction](ssl_masked.md)** — the primary pipeline: pretext task, architecture ablation, hyperparameter tuning, the 80/10/10 rerun, and the cross-sensor ablation.
- **[Bite-level evaluation](bite_evaluation.md)** — the shared postprocessing/bout-matching architecture both pipelines use, and the head-to-head bite-by-bite comparison.
- **[Uncertainty quantification](uncertainty.md)** — MC-Dropout (SSL) and tree-ensemble (classical ML) predictive uncertainty, calibration, and what it says about sensor selection.
- **[Experiments](experiments.md)** — how runs are tracked (MLflow) and where evaluation results actually live.
- **[Decision records](decisions/001_ssl_pretext_tasks.md)** — short notes on *why* key choices were made, each now annotated with what actually got implemented; these feed directly into the dissertation methodology.

## On the "Reference" API docs

There is no generated API-reference site in this repository (no `mkdocs`/
`sphinx`/`pdoc` config exists) — an earlier version of this page linked to
one (`api/data.md`) that was never built. Docstrings in the source are
still the right place to check exact function signatures/behaviour; there
just isn't a rendered, browsable version of them yet. Worth setting up
before submission if the dissertation or its appendices reference "see the
API docs" anywhere — otherwise, don't promise a reference site that
doesn't exist.