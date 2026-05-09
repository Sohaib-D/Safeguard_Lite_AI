from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import ml.training as training_module
from ml.optimization import JAX_AVAILABLE
from ml.training import TrainingConfig, pick_best_model, train_and_select_best_model


def build_training_csv(tmp_path: Path) -> Path:
    X, y = make_classification(
        n_samples=180,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    labels = {0: "Normal", 1: "DDoS", 2: "PortScan"}
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["service"] = [
        "http" if idx % 3 == 0 else "dns" if idx % 3 == 1 else "ssh"
        for idx in range(len(df))
    ]
    df["attack_class"] = [labels[val] for val in y]
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_train_and_select_best_model_creates_artifacts(tmp_path, monkeypatch):
    csv_path = build_training_csv(tmp_path)

    original_build_model_registry = training_module.build_model_registry

    def build_single_thread_registry(random_state: int):
        models = original_build_model_registry(random_state)
        if "random_forest" in models:
            models["random_forest"].set_params(n_jobs=1)
        if "xgboost" in models:
            models["xgboost"].set_params(n_jobs=1)
        if "lightgbm" in models:
            models["lightgbm"].set_params(n_jobs=1)
        return models

    monkeypatch.setattr(
        training_module, "build_model_registry", build_single_thread_registry
    )

    config = TrainingConfig(
        model_output_dir=str(tmp_path / "trained"),
        preprocessing_artifacts_dir=str(tmp_path / "prep"),
        selection_metric="f1",
        reduction_method="none",
    )

    result = train_and_select_best_model(
        csv_path=csv_path, target_column="attack_class", config=config
    )

    assert result["best_model_name"] in result["available_models"]
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(
        result["results_table"].columns
    )
    assert (tmp_path / "trained" / "best_model.pkl").exists()
    assert (tmp_path / "trained" / "model_results.csv").exists()
    assert (tmp_path / "trained" / "confusion_matrix.csv").exists()
    assert (tmp_path / "trained" / "per_class_report.csv").exists()


def test_pick_best_model_rejects_unknown_metric():
    df = pd.DataFrame(
        [
            {
                "model": "a",
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.9,
                "f1": 0.9,
                "roc_auc": 0.9,
            },
            {
                "model": "b",
                "accuracy": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "roc_auc": 0.8,
            },
        ]
    )

    with pytest.raises(ValueError, match="selection_metric must be one of"):
        pick_best_model(df, "not_a_metric")


def test_training_optimization_metadata_is_saved(tmp_path, monkeypatch):
    csv_path = build_training_csv(tmp_path)

    original_build_model_registry = training_module.build_model_registry

    def build_single_model_registry(random_state: int):
        models = original_build_model_registry(random_state)
        return {"logistic_regression": models["logistic_regression"]}

    monkeypatch.setattr(
        training_module, "build_model_registry", build_single_model_registry
    )

    config = TrainingConfig(
        model_output_dir=str(tmp_path / "trained_opt"),
        preprocessing_artifacts_dir=str(tmp_path / "prep_opt"),
        selection_metric="f1",
        enable_feature_pruning=True,
        feature_prune_max_features=5,
        enable_jax_conversion=True,
    )

    result = train_and_select_best_model(
        csv_path=csv_path, target_column="attack_class", config=config
    )
    bundle = joblib.load(tmp_path / "trained_opt" / "best_model.pkl")

    assert "optimization" in result
    assert "optimization" in bundle
    assert bundle["optimization"]["quantization"]["enabled"] is True
    if JAX_AVAILABLE:
        assert bundle["optimization"]["jax"]["enabled"] is True
    else:
        assert bundle["optimization"]["jax"]["enabled"] is False
        assert "reason" in bundle["optimization"]["jax"]
    assert bundle["selected_feature_indices"] is not None
    if bundle["optimization"]["feature_pruning"]["applied"]:
        assert len(bundle["feature_names"]) <= 5
