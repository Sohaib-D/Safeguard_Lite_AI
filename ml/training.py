"""
Model training and evaluation utilities for intrusion detection.

Supported models:
- Logistic Regression
- Random Forest
- XGBoost (optional, if installed)
- LightGBM (optional, if installed)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.optimization import (
    OptimizationConfig,
    build_jax_metadata,
    export_onnx_artifact,
    measure_serialized_size_bytes,
    optimize_with_feature_pruning,
    quantize_bundle,
)
from ml.preprocessing import PreprocessingConfig, preprocess_csv, set_global_seed

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None


@dataclass
class TrainingConfig:
    model_output_dir: str = "models/trained"
    preprocessing_artifacts_dir: str = "models/preprocessing"
    test_size: float = 0.2
    random_state: int = 42
    missing_strategy: str = "impute"
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    scaler_type: str = "standard"
    drop_columns: list[str] | None = None
    stratify: bool = True
    enable_correlation_filter: bool = False
    correlation_threshold: float = 0.95
    reduction_method: str = "none"
    pca_n_components: int | float | None = None
    tree_selection_threshold: str | float = "median"
    tree_selection_max_features: int | None = None
    selection_metric: str = "f1"
    enable_feature_pruning: bool = False
    feature_prune_max_features: int | None = None
    feature_prune_importance_threshold: float | None = None
    optimization_max_metric_drop: float = 0.02
    quantize_dtype: str = "float32"
    export_onnx: bool = False
    enable_jax_conversion: bool = False


def build_preprocessing_config(
    target_column: str, config: TrainingConfig
) -> PreprocessingConfig:
    return PreprocessingConfig(
        target_column=target_column,
        test_size=config.test_size,
        random_state=config.random_state,
        missing_strategy=config.missing_strategy,
        numeric_impute_strategy=config.numeric_impute_strategy,
        categorical_impute_strategy=config.categorical_impute_strategy,
        scaler_type=config.scaler_type,
        drop_columns=config.drop_columns,
        stratify=config.stratify,
        enable_correlation_filter=config.enable_correlation_filter,
        correlation_threshold=config.correlation_threshold,
        reduction_method=config.reduction_method,
        pca_n_components=config.pca_n_components,
        tree_selection_threshold=config.tree_selection_threshold,
        tree_selection_max_features=config.tree_selection_max_features,
    )


def build_model_registry(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )

    return models


def compute_roc_auc(y_true: pd.Series, y_score: np.ndarray, n_classes: int) -> float:
    """Compute ROC-AUC for binary or multiclass settings."""
    try:
        if n_classes <= 2:
            if y_score.ndim == 2:
                return roc_auc_score(y_true, y_score[:, 1])
            return roc_auc_score(y_true, y_score)
        return roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
    except ValueError:
        return float("nan")


def evaluate_model(
    model: Any, X_test: np.ndarray, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate a fitted model on the test split."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    class_count = len(np.unique(y_test))

    average = "binary" if class_count == 2 else "weighted"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "roc_auc": (
            compute_roc_auc(y_test, y_proba, class_count)
            if y_proba is not None
            else float("nan")
        ),
    }
    return metrics


def decode_labels(
    encoded_values: pd.Series | np.ndarray, label_classes: list[str]
) -> np.ndarray:
    """Decode integer labels back to their original class names."""
    encoded_array = np.asarray(encoded_values, dtype=int)
    classes = np.asarray(label_classes, dtype=object)
    return classes[encoded_array]


def build_multiclass_reports(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    label_classes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate confusion matrix and per-class metrics for the selected model.

    Per-class accuracy here is the class hit rate over actual samples of that class,
    which is numerically equivalent to class recall.
    """
    y_pred = model.predict(X_test)
    y_true_decoded = decode_labels(y_test, label_classes)
    y_pred_decoded = decode_labels(y_pred, label_classes)

    class_names = sorted(set(y_true_decoded) | set(y_pred_decoded))
    cm = confusion_matrix(y_true_decoded, y_pred_decoded, labels=class_names)
    confusion_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    confusion_df.index.name = "actual"
    confusion_df.columns.name = "predicted"

    report_dict = classification_report(
        y_true_decoded,
        y_pred_decoded,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for class_name in class_names:
        support = int(report_dict[class_name]["support"])
        true_positive = int(confusion_df.loc[class_name, class_name])
        class_accuracy = true_positive / support if support else 0.0
        per_class_rows.append(
            {
                "class_name": class_name,
                "support": support,
                "per_class_accuracy": class_accuracy,
                "precision": report_dict[class_name]["precision"],
                "recall": report_dict[class_name]["recall"],
                "f1": report_dict[class_name]["f1-score"],
            }
        )

    per_class_df = (
        pd.DataFrame(per_class_rows).sort_values(by="class_name").reset_index(drop=True)
    )
    return confusion_df, per_class_df


def fit_and_compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, float]]]:
    """Train all available models and return ranking table plus fitted models."""
    models = build_model_registry(random_state)
    fitted_models: dict[str, Any] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        metrics_by_model[model_name] = evaluate_model(model, X_test, y_test)

    results_df = pd.DataFrame.from_dict(metrics_by_model, orient="index").reset_index()
    results_df = results_df.rename(columns={"index": "model"})
    results_df = results_df.sort_values(
        by=["f1", "roc_auc", "accuracy"], ascending=False
    ).reset_index(drop=True)
    return results_df, fitted_models, metrics_by_model


def pick_best_model(results_df: pd.DataFrame, selection_metric: str) -> str:
    """Select the best model name using the configured metric."""
    allowed_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    if selection_metric not in allowed_metrics:
        raise ValueError(f"selection_metric must be one of {sorted(allowed_metrics)}")

    ranked = results_df.sort_values(
        by=[selection_metric, "f1", "roc_auc", "accuracy"],
        ascending=False,
        na_position="last",
    )
    return str(ranked.iloc[0]["model"])


def save_best_model(
    output_dir: str | Path,
    best_model_name: str,
    best_model: Any,
    results_df: pd.DataFrame,
    preprocessing_results: dict[str, Any],
    training_config: TrainingConfig,
    confusion_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    optimization_metadata: dict[str, Any] | None = None,
    bundle_overrides: dict[str, Any] | None = None,
) -> None:
    """Persist the chosen model bundle and metric table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": preprocessing_results["feature_names"],
        "label_classes": preprocessing_results["label_encoder"].classes_.tolist(),
        "preprocessor": preprocessing_results["preprocessor"],
        "correlation_filter": preprocessing_results["correlation_filter"],
        "feature_engineer": preprocessing_results["feature_engineer"],
        "training_config": training_config,
        "selected_feature_indices": preprocessing_results.get(
            "selected_feature_indices"
        ),
        "optimization": optimization_metadata or {},
    }
    if bundle_overrides:
        model_bundle.update(bundle_overrides)

    joblib.dump(model_bundle, output_path / "best_model.pkl", compress=3)
    results_df.to_csv(output_path / "model_results.csv", index=False)
    confusion_df.to_csv(output_path / "confusion_matrix.csv")
    per_class_df.to_csv(output_path / "per_class_report.csv", index=False)


def train_and_select_best_model(
    csv_path: str | Path,
    target_column: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Run preprocessing, train candidate models, and save the best one."""
    set_global_seed(config.random_state)

    preprocessing_config = build_preprocessing_config(target_column, config)
    preprocessing_results = preprocess_csv(
        csv_path=csv_path,
        artifacts_dir=config.preprocessing_artifacts_dir,
        config=preprocessing_config,
    )

    results_df, fitted_models, metrics_by_model = fit_and_compare_models(
        preprocessing_results["X_train_processed"],
        preprocessing_results["X_test_processed"],
        preprocessing_results["y_train"],
        preprocessing_results["y_test"],
        config.random_state,
    )

    best_model_name = pick_best_model(results_df, config.selection_metric)
    label_classes = preprocessing_results["label_encoder"].classes_.tolist()
    best_model = fitted_models[best_model_name]

    optimization_config = OptimizationConfig(
        enable_feature_pruning=config.enable_feature_pruning,
        feature_prune_max_features=config.feature_prune_max_features,
        feature_prune_importance_threshold=config.feature_prune_importance_threshold,
        max_metric_drop=config.optimization_max_metric_drop,
        quantize_dtype=config.quantize_dtype,
        export_onnx=config.export_onnx,
        enable_jax_conversion=config.enable_jax_conversion,
    )
    optimization_result = optimize_with_feature_pruning(
        model_name=best_model_name,
        model=best_model,
        X_train=preprocessing_results["X_train_processed"],
        X_test=preprocessing_results["X_test_processed"],
        y_train=preprocessing_results["y_train"],
        y_test=preprocessing_results["y_test"],
        feature_names=preprocessing_results["feature_names"],
        selection_metric=config.selection_metric,
        config=optimization_config,
    )
    original_engineered_feature_count = len(preprocessing_results["feature_names"])
    best_model = optimization_result["model"]
    preprocessing_results["selected_feature_indices"] = optimization_result[
        "selected_feature_indices"
    ]
    preprocessing_results["feature_names"] = optimization_result[
        "selected_feature_names"
    ]

    confusion_df, per_class_df = build_multiclass_reports(
        best_model,
        preprocessing_results["X_test_processed"][
            :, preprocessing_results["selected_feature_indices"]
        ],
        preprocessing_results["y_test"],
        label_classes,
    )

    pre_quantization_bundle = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": preprocessing_results["feature_names"],
        "label_classes": label_classes,
        "preprocessor": preprocessing_results["preprocessor"],
        "correlation_filter": preprocessing_results["correlation_filter"],
        "feature_engineer": preprocessing_results["feature_engineer"],
        "training_config": config,
        "selected_feature_indices": preprocessing_results["selected_feature_indices"],
    }
    size_before_bytes = measure_serialized_size_bytes(pre_quantization_bundle)
    quantized_bundle, quantization_metadata = quantize_bundle(
        pre_quantization_bundle, config.quantize_dtype
    )
    size_after_bytes = measure_serialized_size_bytes(quantized_bundle)

    onnx_metadata = (
        export_onnx_artifact(
            best_model,
            config.model_output_dir,
            len(preprocessing_results["feature_names"]),
        )
        if config.export_onnx
        else {"requested": False, "exported": False}
    )
    jax_metadata = (
        build_jax_metadata(best_model, label_classes)
        if config.enable_jax_conversion
        else {"enabled": False, "reason": "JAX conversion was not requested."}
    )
    optimization_metadata = {
        "feature_pruning": {
            "enabled": config.enable_feature_pruning,
            "applied": optimization_result["pruning_applied"],
            "selected_feature_count": len(
                preprocessing_results["selected_feature_indices"]
            ),
            "original_feature_count": original_engineered_feature_count,
            "feature_importance": (
                optimization_result["feature_importance"].to_dict(orient="records")
                if optimization_result["feature_importance"] is not None
                else []
            ),
            "metrics_after_pruning": optimization_result["metrics"],
        },
        "quantization": {
            **quantization_metadata,
            "size_before_bytes": size_before_bytes,
            "size_after_bytes": size_after_bytes,
        },
        "onnx_export": onnx_metadata,
        "jax": jax_metadata,
    }

    save_best_model(
        output_dir=config.model_output_dir,
        best_model_name=best_model_name,
        best_model=quantized_bundle["model"],
        results_df=results_df,
        preprocessing_results=preprocessing_results,
        training_config=config,
        confusion_df=confusion_df,
        per_class_df=per_class_df,
        optimization_metadata=optimization_metadata,
        bundle_overrides={
            "preprocessor": quantized_bundle["preprocessor"],
            "correlation_filter": quantized_bundle["correlation_filter"],
            "feature_engineer": quantized_bundle["feature_engineer"],
            "selected_feature_indices": quantized_bundle["selected_feature_indices"],
        },
    )

    return {
        "best_model_name": best_model_name,
        "best_model": quantized_bundle["model"],
        "results_table": results_df,
        "metrics_by_model": metrics_by_model,
        "preprocessing_results": preprocessing_results,
        "available_models": list(fitted_models.keys()),
        "confusion_matrix": confusion_df,
        "per_class_report": per_class_df,
        "optimization": optimization_metadata,
    }
