"""
Two modes:

  Single run (default): train one --model with library-default hyperparameters
  (or whatever hardcoded in MODELS below), wrapped in Pipeline(StandardScaler,
  classifier) -- scaling is a no-op for tree models but required for logistic
  regression, and applying it uniformly keeps one code path for every choice.

  Sweep (--sweep-config configs/train_ml.yaml): train every hyperparameter
  combination listed in the YAML config, evaluate each on val AND test, save
  every combo's fitted pipeline + val/test predictions, write one combined
  results CSV (--out-dir/sweep_results.csv), and copy the combo with the best
  --sweep-metric (default val_f1) to --out-dir/best_model.joblib -- that's
  the file upscale_predictions.py should point at next.

Reports the same metric set as the rest of this project (accuracy,
precision, recall, F1, AUC) on val (for reference / model selection) and
test (final, held-out), plus a feature-importance plot for tree-based
models in single-run mode -- one of classical ML's real advantages over the
deep-learning path is that *which* features mattered is directly
inspectable, unlike a learned embedding.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chewing_ssl.common.tracking.mlflow_utils import log_artifact, log_metrics, log_params, start_run
from chewing_ssl.common.utils.config import load_train_ml_config

META_COLUMNS = ["dataset", "participant", "tag", "segment", "start", "label"]

MODELS = {
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced", random_state=seed, n_jobs=-1
    ),
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=seed
    ),
    "gradient_boosting": lambda seed: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=seed
    ),
}

MODEL_BUILDERS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "gradient_boosting": HistGradientBoostingClassifier,
}


def load_split(features_dir: Path, purpose: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_parquet(features_dir / f"{purpose}.parquet")
    meta = df[[c for c in META_COLUMNS if c in df.columns and c != "label"]]
    y = df["label"]
    X = df.drop(columns=[c for c in META_COLUMNS if c in df.columns])
    return X, y, meta


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    n_unique = len(np.unique(y_true))
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0.0)),
        "recall": float(recall_score(y_true, preds, zero_division=0.0)),
        "f1": float(f1_score(y_true, preds, zero_division=0.0)),
        "auc": float(roc_auc_score(y_true, probs)) if n_unique > 1 else float("nan"),
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    print(
        f"{prefix}f1={metrics['f1']:.3f}  prec={metrics['precision']:.3f}  "
        f"rec={metrics['recall']:.3f}  auc={metrics['auc']:.3f}  acc={metrics['accuracy']:.3f}"
    )


def plot_feature_importance(
    pipeline: Pipeline, feature_names: list[str], model_name: str, out_path: Path, top_n: int = 30
) -> None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        ylabel = "importance"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
        ylabel = "|coefficient|"
    else:
        print(f"  {model_name} has no inspectable feature importance -- skipping plot")
        return

    order = np.argsort(importance)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(order) + 1))
    ax.barh(range(len(order)), importance[order][::-1], color="#1565C0")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=7)
    ax.set_xlabel(ylabel, fontsize=9)
    ax.set_title(f"Top {len(order)} features -- {model_name}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  ->  {out_path}")


# ── Sweep mode ───────────────────────────────────────────────────────────────


def build_model(model_type: str, params: dict, seed: int):
    """Instantiate a classifier for `model_type` with `params`, seeded."""
    if model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model type {model_type!r} (expected one of {list(MODEL_BUILDERS)})"
        )
    kwargs = dict(params)
    if model_type == "random_forest":
        kwargs.setdefault("n_jobs", -1)
    return MODEL_BUILDERS[model_type](random_state=seed, **kwargs)


def save_predictions(
    meta: pd.DataFrame, y_true: pd.Series, probs: np.ndarray, out_path: Path
) -> None:
    df = meta.copy()
    df["y_true"] = y_true.to_numpy()
    df["pred_prob"] = probs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def run_sweep(
    config: dict, features_dir: Path, out_dir: Path, seed: int, threshold: float
) -> pd.DataFrame:
    """Train every combo in `config` ({model_type: [{name, **params}, ...]}),
    evaluate each on val+test, save every combo's fitted pipeline + val/test
    predictions, and return one results row per combo (does not pick a
    winner -- see pick_best)."""
    X_train, y_train, _ = load_split(features_dir, "train")
    X_val, y_val, meta_val = load_split(features_dir, "val")
    X_test, y_test, meta_test = load_split(features_dir, "test")
    print(
        f"  Train : {len(y_train):,} windows x {X_train.shape[1]} features  ({y_train.mean():.1%} positive)"
    )
    print(f"  Val   : {len(y_val):,} windows  ({y_val.mean():.1%} positive)")
    print(f"  Test  : {len(y_test):,} windows  ({y_test.mean():.1%} positive)")

    models_dir = out_dir / "sweep_models"
    preds_dir = out_dir / "sweep_predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        (model_type, raw_params)
        for model_type, param_list in config.items()
        for raw_params in param_list
    ]
    print(f"\n  Sweeping {len(combos)} combinations ({', '.join(config)}) ...\n")

    rows = []
    for i, (model_type, raw_params) in enumerate(combos, start=1):
        params = {k: v for k, v in raw_params.items() if k != "name"}
        name = raw_params.get("name", f"{model_type}_{i:02d}")
        print(f"  [{i}/{len(combos)}] {name}  ({model_type}: {params})")

        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", build_model(model_type, params, seed))]
        )
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0

        val_probs = pipeline.predict_proba(X_val)[:, 1]
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        val_metrics = compute_metrics(y_val.to_numpy(), val_probs, threshold)
        test_metrics = compute_metrics(y_test.to_numpy(), test_probs, threshold)
        print_metrics(val_metrics, prefix=f"      ({elapsed:5.1f}s) val : ")
        print_metrics(test_metrics, prefix="               test: ")

        model_path = models_dir / f"{name}.joblib"
        joblib.dump(pipeline, model_path)
        save_predictions(meta_val, y_val, val_probs, preds_dir / f"{name}_val.parquet")
        save_predictions(meta_test, y_test, test_probs, preds_dir / f"{name}_test.parquet")

        rows.append(
            {
                "name": name,
                "model_type": model_type,
                **{f"param_{k}": v for k, v in params.items()},
                "train_seconds": round(elapsed, 1),
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["auc"],
                "model_path": str(model_path),
            }
        )

    return pd.DataFrame(rows)


def load_input_meta(features_dir: Path) -> dict:
    """Best-effort read of {features_dir}/feature_meta.json, written by
    extract_features.py -- documents how the input was actually processed
    (feature families, channels, and, if segment.py wrote its own
    segment_meta.json upstream, window length / per-purpose stride /
    train-overlap too). Returns {} rather than raising for older
    features_dir trees that predate this file -- it's provenance for
    MLflow logging, not a required input to training."""
    meta_path = features_dir / "feature_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def flatten_input_params(meta: dict) -> dict:
    """feature_meta.json (+ its nested "segmentation" key) -> a flat dict of
    MLflow-loggable params (MLflow params must be flat scalars, not nested
    dicts/lists). Prefixed input_* so these are visually grouped apart from
    the model's own hyperparameters in the MLflow UI. Drops any key whose
    value is None (segmentation info is entirely absent for a features_dir
    built before segment.py wrote segment_meta.json)."""
    if not meta:
        return {}
    seg = meta.get("segmentation") or {}
    purpose_stride = seg.get("purpose_stride") or {}
    flat = {
        "input_families": ",".join(meta.get("families", [])),
        "input_fs": meta.get("fs"),
        "input_n_channels": len(meta.get("channels", [])),
        "input_n_features_total": meta.get("n_features_total"),
        "input_segments_dir": meta.get("segments_dir"),
        "input_window_len": seg.get("window_len"),
        "input_window_seconds": seg.get("window_seconds"),
        "input_train_overlap": seg.get("train_overlap"),
        "input_train_stride": purpose_stride.get("train"),
        "input_eval_stride": purpose_stride.get("val"),
        "input_interim_root": seg.get("interim_root"),
        "input_splits_path": seg.get("splits_path"),
    }
    return {k: v for k, v in flat.items() if v is not None}


def pick_best(results: pd.DataFrame, metric: str = "val_f1") -> pd.Series:
    """Select the sweep row with the best `metric` (higher is better).

    Selection uses a val metric by default, deliberately -- test stays
    held-out for the final reported number, not used to choose the winner
    (same no-leakage principle extract_ml_features.py's sanitize() follows
    for feature medians, see docs/classical_ml.md §2).
    """
    return results.loc[results[metric].idxmax()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a classical ML baseline for window-level chewing detection."
    )
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed/ml_features"))
    parser.add_argument("--model", choices=list(MODELS), default="random_forest")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/train_ml"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep-config",
        type=Path,
        default=None,
        help="Path to a YAML hyperparameter sweep (see configs/train_ml.yaml). When given, "
        "trains every combination instead of the single --model run, saves each combo's "
        "model + val/test predictions, writes one combined results CSV, and copies the "
        "best --sweep-metric combo to --out-dir/best_model.joblib.",
    )
    parser.add_argument(
        "--sweep-metric",
        default="val_f1",
        help="Column in the results CSV used to pick the best combo (default: val_f1).",
    )
    args = parser.parse_args()

    if args.sweep_config:
        print(f"\n{'=' * 60}")
        print("  Classical ML hyperparameter sweep")
        print(f"{'=' * 60}")
        config = load_train_ml_config(args.sweep_config)
        input_meta = load_input_meta(args.features_dir)
        input_params = flatten_input_params(input_meta)
        if input_params:
            print(f"  Input provenance found ({args.features_dir}/feature_meta.json) -- logging to MLflow too")

        with start_run(run_name=f"sweep_{args.sweep_config.stem}"):
            log_params(
                {
                    "mode": "sweep",
                    "sweep_config": str(args.sweep_config),
                    "sweep_metric": args.sweep_metric,
                    "seed": args.seed,
                    "threshold": args.threshold,
                    "features_dir": str(args.features_dir),
                    "n_combos": sum(len(v) for v in config.values()),
                    **input_params,
                }
            )

            results = run_sweep(config, args.features_dir, args.out_dir, args.seed, args.threshold)

            metric_cols = [
                "val_accuracy", "val_precision", "val_recall", "val_f1", "val_auc",
                "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc",
                "train_seconds",
            ]
            for _, row in results.iterrows():
                with start_run(run_name=row["name"], nested=True):
                    param_cols = [c for c in row.index if c.startswith("param_") and pd.notna(row[c])]
                    log_params({"model_type": row["model_type"], **{c: row[c] for c in param_cols}})
                    log_metrics({c: float(row[c]) for c in metric_cols if pd.notna(row[c])})

            args.out_dir.mkdir(parents=True, exist_ok=True)
            results_path = args.out_dir / "sweep_results.csv"
            results.to_csv(results_path, index=False)
            print(f"\n  Saved combined results ({len(results)} combos) -> {results_path}")
            log_artifact(str(results_path))

            best = pick_best(results, args.sweep_metric)
            best_path = args.out_dir / "best_model.joblib"
            shutil.copy(best["model_path"], best_path)
            print(f"\n  Best combo by {args.sweep_metric}: {best['name']} ({best['model_type']})")
            print(
                f"    val : f1={best['val_f1']:.3f}  prec={best['val_precision']:.3f}  "
                f"rec={best['val_recall']:.3f}  auc={best['val_auc']:.3f}  acc={best['val_accuracy']:.3f}"
            )
            print(
                f"    test: f1={best['test_f1']:.3f}  prec={best['test_precision']:.3f}  "
                f"rec={best['test_recall']:.3f}  auc={best['test_auc']:.3f}  acc={best['test_accuracy']:.3f}"
            )
            print(f"  Copied -> {best_path}")

            log_params({"best_combo_name": best["name"], "best_combo_model_type": best["model_type"]})
            log_metrics({f"best_{c}": float(best[c]) for c in metric_cols if pd.notna(best[c])})

        print(f"\n{'=' * 60}\n")
        return

    print(f"  Classical ML baseline: {args.model}")

    input_meta = load_input_meta(args.features_dir)
    input_params = flatten_input_params(input_meta)
    if input_params:
        print(f"  Input provenance found ({args.features_dir}/feature_meta.json) -- logging to MLflow too")

    X_train, y_train, _ = load_split(args.features_dir, "train")
    X_val, y_val, _ = load_split(args.features_dir, "val")
    X_test, y_test, _ = load_split(args.features_dir, "test")
    print(
        f"  Train : {len(y_train):,} windows x {X_train.shape[1]} features  ({y_train.mean():.1%} positive)"
    )
    print(f"  Val   : {len(y_val):,} windows  ({y_val.mean():.1%} positive)")
    print(f"  Test  : {len(y_test):,} windows  ({y_test.mean():.1%} positive)")

    with start_run(run_name=f"{args.model}_single"):
        log_params(
            {
                "mode": "single",
                "model": args.model,
                "seed": args.seed,
                "threshold": args.threshold,
                "features_dir": str(args.features_dir),
                "n_train_windows": len(y_train),
                "n_features": X_train.shape[1],
                **input_params,
            }
        )

        pipeline = Pipeline([("scaler", StandardScaler()), ("model", MODELS[args.model](args.seed))])

        print(f"\n  Fitting {args.model} ...")
        pipeline.fit(X_train, y_train)

        val_probs = pipeline.predict_proba(X_val)[:, 1]
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        val_metrics = compute_metrics(y_val.to_numpy(), val_probs, args.threshold)
        test_metrics = compute_metrics(y_test.to_numpy(), test_probs, args.threshold)

        print()
        print_metrics(val_metrics, prefix="  val : ")
        print_metrics(test_metrics, prefix="  test: ")
        log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        args.out_dir.mkdir(parents=True, exist_ok=True)
        model_path = args.out_dir / f"{args.model}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"\n  Saved model -> {model_path}")

        metrics_path = args.out_dir / f"{args.model}_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {"val": val_metrics, "test": test_metrics, "threshold": args.threshold}, indent=2
            )
        )
        print(f"  Saved metrics -> {metrics_path}")
        log_artifact(str(metrics_path))

        importance_path = args.out_dir / f"{args.model}_feature_importance.png"
        plot_feature_importance(pipeline, list(X_train.columns), args.model, importance_path)
        if importance_path.exists():
            log_artifact(str(importance_path))

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
