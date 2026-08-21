# Experiments

How runs are tracked, and where to actually find results — as opposed to
what's reported in the narrative docs (`docs/ssl_masked.md`,
`docs/classical_ml.md`/`docs/dissertation_classical_ml_sections.md`,
`docs/uncertainty.md`), which is the *curated, written-up* subset.

## Live tracking: MLflow

Every training script (`pretrain.py`, `finetune.py`, `train.py`, and the TF-C
variants) logs through `chewing_ssl.common.tracking.mlflow_utils` — one
`start_run()` context manager per run, `log_params`/`log_metrics` throughout,
so every run's full hyperparameters and per-epoch metrics are captured, not
just whatever ended up quoted in a doc afterward.

- **Backend**: local file store, `mlruns/` at the project root (gitignored —
  this is a local record, not a committed artifact). Configurable via
  `.env`'s `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` if a remote backend
  is ever needed; defaults to zero-setup local file storage.
- **View it**: `mlflow ui` from the project root (or any parent of `mlruns/`),
  then open the printed local URL — every run, every metric curve, every
  logged hyperparameter, filterable/sortable in one place. This is the right
  place to check "did I actually try X" or "what was the exact config for
  run Y", rather than re-deriving it from a checkpoint filename.
- **Nested runs**: hyperparameter sweeps (`train.py --sweep-config`, the SSL
  tuning passes) log one child run per combination under a parent run for
  the whole sweep (`start_run(..., nested=True)`), so a sweep's results are
  browsable as a group, not a flat list of unrelated runs.

## What's *not* in MLflow

Anything computed after training finishes — bout-level postprocessing/
evaluation sweeps (`common/evaluation/sweep.py`), the cross-sensor ablation
study, uncertainty quantification — writes directly to CSV/JSON under
`checkpoints/` instead (`checkpoints/*_summary.csv`, `checkpoints/
uncertainty_summary.csv`, `checkpoints/cross_sensor_evaluation/summary.csv`,
...), not to MLflow. These are the files the narrative docs actually cite
numbers from — MLflow has the *training* run history; `checkpoints/` has the
*evaluation* results.

## Checkpoint provenance

With this many retrains (5 checkpoint generations, at last count — original
60/20/20, the 80/10/10 split regeneration, an early per-file-normalization
attempt, a raw-signal clipping fix, and the genuinely-per-sensor-retrained
family that fed the cross-sensor/uncertainty work), it stopped being obvious
from a directory name alone which checkpoint is current for a given claim.
`checkpoints/uncertainty_models/MANIFEST.md` documents the full generation
timeline (dates, what changed each round, which is superseded by what) — the
first place to check before citing a number from `checkpoints/` directly
rather than from one of the narrative docs above.
