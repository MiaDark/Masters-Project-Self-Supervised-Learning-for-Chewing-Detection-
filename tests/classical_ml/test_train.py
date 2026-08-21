from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from chewing_ssl.classical_ml.train import (
    build_model,
    load_split,
    pick_best,
    run_sweep,
    save_predictions,
)


def _make_features_dir(tmp_path: Path) -> Path:
    """A tiny synthetic ml_features/ directory: train/val/test.parquet, each
    with a handful of numeric feature columns plus META_COLUMNS."""
    rng = np.random.default_rng(0)
    features_dir = tmp_path / "ml_features"
    features_dir.mkdir()

    for purpose, n in [("train", 60), ("val", 20), ("test", 20)]:
        label = (rng.random(n) > 0.5).astype(int)
        # make features weakly informative so AUC isn't degenerate
        f0 = label + rng.normal(0, 0.5, size=n)
        f1 = rng.normal(0, 1, size=n)
        df = pd.DataFrame(
            {
                "dataset": "synthetic",
                "participant": [f"p{i % 3}" for i in range(n)],
                "tag": "tag",
                "segment": "segment_00",
                "start": np.arange(n) * 200,
                "label": label,
                "feat_0": f0,
                "feat_1": f1,
            }
        )
        df.to_parquet(features_dir / f"{purpose}.parquet", index=False)

    return features_dir


def test_build_model_random_forest():
    model = build_model("random_forest", {"n_estimators": 10, "max_depth": 3}, seed=42)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10
    assert model.max_depth == 3
    assert model.random_state == 42
    assert model.n_jobs == -1  # default injected when not specified


def test_build_model_random_forest_respects_explicit_n_jobs():
    model = build_model("random_forest", {"n_estimators": 10, "n_jobs": 1}, seed=42)
    assert model.n_jobs == 1


def test_build_model_logistic_regression():
    model = build_model(
        "logistic_regression", {"C": 0.5, "l1_ratio": 1.0, "solver": "saga"}, seed=7
    )
    assert isinstance(model, LogisticRegression)
    assert model.C == 0.5
    assert model.solver == "saga"
    assert model.random_state == 7


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model("not_a_model", {}, seed=0)


def test_save_predictions_writes_expected_columns(tmp_path):
    meta = pd.DataFrame({"dataset": ["a", "b"], "participant": ["p1", "p2"]})
    y_true = pd.Series([0, 1])
    probs = np.array([0.2, 0.8])
    out_path = tmp_path / "preds.parquet"

    save_predictions(meta, y_true, probs, out_path)

    result = pd.read_parquet(out_path)
    assert list(result.columns) == ["dataset", "participant", "y_true", "pred_prob"]
    assert result["y_true"].tolist() == [0, 1]
    assert result["pred_prob"].tolist() == [0.2, 0.8]


def test_pick_best_selects_max_of_given_metric():
    results = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "val_f1": [0.5, 0.9, 0.7],
            "test_f1": [0.9, 0.4, 0.6],
        }
    )
    assert pick_best(results, "val_f1")["name"] == "b"
    assert pick_best(results, "test_f1")["name"] == "a"


def test_run_sweep_trains_every_combo_and_saves_artifacts(tmp_path):
    features_dir = _make_features_dir(tmp_path)
    out_dir = tmp_path / "out"
    config = {
        "random_forest": [
            {"name": "rf_a", "n_estimators": 5, "max_depth": 2},
            {"name": "rf_b", "n_estimators": 5, "max_depth": None},
        ],
        "logistic_regression": [
            {"name": "lr_a", "C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "max_iter": 200},
        ],
    }

    results = run_sweep(config, features_dir, out_dir, seed=42, threshold=0.5)

    assert len(results) == 3
    assert set(results["name"]) == {"rf_a", "rf_b", "lr_a"}
    assert set(results["model_type"]) == {"random_forest", "logistic_regression"}
    for col in ["val_f1", "val_precision", "val_recall", "val_accuracy", "val_auc"]:
        assert results[col].between(0, 1).all() or results[col].isna().all()

    # every combo's model + predictions actually landed on disk
    for _, row in results.iterrows():
        assert Path(row["model_path"]).exists()
        assert (out_dir / "sweep_predictions" / f"{row['name']}_val.parquet").exists()
        assert (out_dir / "sweep_predictions" / f"{row['name']}_test.parquet").exists()

    # param columns are prefixed and combo-specific (NaN where not applicable)
    assert "param_n_estimators" in results.columns
    assert "param_C" in results.columns
    rf_row = results[results["name"] == "rf_a"].iloc[0]
    lr_row = results[results["name"] == "lr_a"].iloc[0]
    assert rf_row["param_n_estimators"] == 5
    assert pd.isna(rf_row["param_C"])
    assert lr_row["param_C"] == 1.0
    assert pd.isna(lr_row["param_n_estimators"])


def test_run_sweep_predictions_match_val_split_size(tmp_path):
    features_dir = _make_features_dir(tmp_path)
    out_dir = tmp_path / "out"
    config = {"logistic_regression": [{"name": "lr_only", "C": 1.0, "max_iter": 200}]}

    run_sweep(config, features_dir, out_dir, seed=0, threshold=0.5)

    _, y_val, _ = load_split(features_dir, "val")
    val_preds = pd.read_parquet(out_dir / "sweep_predictions" / "lr_only_val.parquet")
    assert len(val_preds) == len(y_val)
