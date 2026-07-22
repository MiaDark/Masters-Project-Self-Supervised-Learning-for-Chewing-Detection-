"""Classical ML baseline for window-level chewing-bout detection, using the
tabular features from extract_ml_features.py -- a non-deep-learning
comparison point to the TF-C/TF-FC contrastive-pretraining path
(chewing_ssl.ssl.tfc.finetune_window).

Model choices (--model), all wrapped in Pipeline(StandardScaler, classifier)
-- scaling is a no-op for tree models but required for logistic regression,
and applying it uniformly keeps one code path for every choice:
  - random_forest      RandomForestClassifier (default)
  - logistic_regression LogisticRegression
  - gradient_boosting   HistGradientBoostingClassifier (handles NaN natively,
                        but features are already sanitized by extract_ml_features.py)

Reports the same metric set as the rest of this project (accuracy,
precision, recall, F1, AUC) on val (for reference) and test (final,
held-out), plus a feature-importance plot for tree-based models -- one of
classical ML's real advantages over the deep-learning path is that *which*
features mattered is directly inspectable, unlike a learned embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a classical ML baseline for window-level chewing detection."
    )
    parser.add_argument("--features-dir", type=Path, default=Path("data/processed/ml_features"))
    parser.add_argument("--model", choices=list(MODELS), default="random_forest")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/train_ml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  Classical ML baseline: {args.model}")
    print(f"{'=' * 60}")

    X_train, y_train, _ = load_split(args.features_dir, "train")
    X_val, y_val, _ = load_split(args.features_dir, "val")
    X_test, y_test, _ = load_split(args.features_dir, "test")
    print(
        f"  Train : {len(y_train):,} windows x {X_train.shape[1]} features  ({y_train.mean():.1%} positive)"
    )
    print(f"  Val   : {len(y_val):,} windows  ({y_val.mean():.1%} positive)")
    print(f"  Test  : {len(y_test):,} windows  ({y_test.mean():.1%} positive)")

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

    plot_feature_importance(
        pipeline,
        list(X_train.columns),
        args.model,
        args.out_dir / f"{args.model}_feature_importance.png",
    )

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
