"""
Deployment optimization utilities for trained intrusion-detection models.

Features:
- optional feature pruning with retraining on the selected model family
- lightweight float32 quantization for sklearn-compatible artifacts
- optional ONNX export when skl2onnx is installed
- optional JAX metadata generation for LogisticRegression inference
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

try:
    import jax.numpy as _jnp_module

    jnp_module: Any = _jnp_module
    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    jnp_module = None
    JAX_AVAILABLE = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    SKL2ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    convert_sklearn = None
    FloatTensorType = None
    SKL2ONNX_AVAILABLE = False


@dataclass
class OptimizationConfig:
    enable_feature_pruning: bool = False
    feature_prune_max_features: int | None = None
    feature_prune_importance_threshold: float | None = None
    max_metric_drop: float = 0.02
    quantize_dtype: str = "float32"
    export_onnx: bool = False
    enable_jax_conversion: bool = False


def _compute_roc_auc(y_true: pd.Series, y_score: np.ndarray, n_classes: int) -> float:
    try:
        if n_classes <= 2:
            if y_score.ndim == 2:
                return roc_auc_score(y_true, y_score[:, 1])
            return roc_auc_score(y_true, y_score)
        return roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
    except ValueError:
        return float("nan")


def evaluate_model_metrics(
    model: Any, X_test: np.ndarray, y_test: pd.Series
) -> dict[str, float]:
    """Lightweight copy of the training evaluator to avoid circular imports."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    class_count = len(np.unique(y_test))
    average = "binary" if class_count == 2 else "weighted"

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        "roc_auc": (
            _compute_roc_auc(y_test, y_proba, class_count)
            if y_proba is not None
            else float("nan")
        ),
    }


def extract_feature_importance(
    model: Any, feature_names: list[str]
) -> pd.DataFrame | None:
    """Return a ranked feature-importance DataFrame when the model exposes one."""
    values: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None:
        return None

    importance_df = pd.DataFrame(
        {
            "feature_index": list(range(len(feature_names))),
            "feature": feature_names,
            "importance": values,
        }
    ).sort_values(by="importance", ascending=False)
    return importance_df.reset_index(drop=True)


def select_pruned_feature_indices(
    importance_df: pd.DataFrame,
    config: OptimizationConfig,
) -> list[int]:
    """Pick the most useful features by threshold and/or top-k."""
    working_df = importance_df.copy()

    if config.feature_prune_importance_threshold is not None:
        working_df = working_df[
            working_df["importance"] >= config.feature_prune_importance_threshold
        ]

    if config.feature_prune_max_features is not None:
        working_df = working_df.head(config.feature_prune_max_features)

    if working_df.empty:
        return list(range(len(importance_df)))

    return sorted(int(idx) for idx in working_df["feature_index"].tolist())


def optimize_with_feature_pruning(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    selection_metric: str,
    config: OptimizationConfig,
) -> dict[str, Any]:
    """Optionally retrain the selected model on a pruned feature set."""
    baseline_metrics = evaluate_model_metrics(model, X_test, y_test)
    result = {
        "model": model,
        "selected_feature_indices": list(range(len(feature_names))),
        "selected_feature_names": list(feature_names),
        "metrics": baseline_metrics,
        "feature_importance": extract_feature_importance(model, feature_names),
        "pruning_applied": False,
    }

    if not config.enable_feature_pruning:
        return result

    importance_df = result["feature_importance"]
    if importance_df is None:
        return result

    selected_indices = select_pruned_feature_indices(importance_df, config)
    if len(selected_indices) >= len(feature_names):
        return result

    pruned_feature_names = [feature_names[idx] for idx in selected_indices]
    X_train_pruned = np.asarray(X_train)[:, selected_indices]
    X_test_pruned = np.asarray(X_test)[:, selected_indices]

    pruned_model = clone(model)
    pruned_model.fit(X_train_pruned, y_train)
    pruned_metrics = evaluate_model_metrics(pruned_model, X_test_pruned, y_test)

    baseline_value = float(result["metrics"].get(selection_metric, float("-inf")))
    pruned_value = float(pruned_metrics.get(selection_metric, float("-inf")))

    if pruned_value + config.max_metric_drop < baseline_value:
        return result

    result.update(
        {
            "model": pruned_model,
            "selected_feature_indices": selected_indices,
            "selected_feature_names": pruned_feature_names,
            "metrics": pruned_metrics,
            "pruning_applied": True,
        }
    )
    return result


def _cast_ndarray_dtype(values: np.ndarray, dtype: str) -> np.ndarray:
    if values.dtype.kind != "f":
        return values
    return values.astype(dtype)


def _quantize_known_attributes(obj: Any, dtype: str) -> int:
    converted = 0
    for attr_name in (
        "coef_",
        "intercept_",
        "mean_",
        "scale_",
        "var_",
        "statistics_",
        "components_",
        "explained_variance_",
        "singular_values_",
    ):
        if hasattr(obj, attr_name):
            values = getattr(obj, attr_name)
            if isinstance(values, np.ndarray):
                setattr(obj, attr_name, _cast_ndarray_dtype(values, dtype))
                converted += 1
    return converted


def _walk_quantize(obj: Any, dtype: str) -> int:
    converted = _quantize_known_attributes(obj, dtype)

    if hasattr(obj, "steps"):
        for _name, step in obj.steps:
            converted += _walk_quantize(step, dtype)

    if hasattr(obj, "transformers_"):
        for _name, transformer, _columns in obj.transformers_:
            converted += _walk_quantize(transformer, dtype)

    if hasattr(obj, "named_transformers_"):
        for transformer in obj.named_transformers_.values():
            converted += _walk_quantize(transformer, dtype)

    if hasattr(obj, "estimator_"):
        converted += _walk_quantize(obj.estimator_, dtype)

    if hasattr(obj, "estimator"):
        converted += _walk_quantize(obj.estimator, dtype)

    return converted


def quantize_bundle(
    bundle: dict[str, Any], dtype: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a float32-leaning copy of the bundle when supported."""
    quantized_bundle = copy.deepcopy(bundle)
    converted_attributes = 0

    converted_attributes += _walk_quantize(quantized_bundle["model"], dtype)
    converted_attributes += _walk_quantize(quantized_bundle["preprocessor"], dtype)

    feature_engineer = quantized_bundle.get("feature_engineer")
    if feature_engineer is not None:
        converted_attributes += _walk_quantize(feature_engineer, dtype)

    return quantized_bundle, {
        "enabled": True,
        "dtype": dtype,
        "converted_attributes": converted_attributes,
    }


def measure_serialized_size_bytes(bundle: dict[str, Any], compress: int = 3) -> int:
    """Measure serialized bundle size on disk."""
    fd, temp_path = mkstemp(suffix=".joblib")
    os.close(fd)
    Path(temp_path).unlink(missing_ok=True)
    try:
        joblib.dump(bundle, temp_path, compress=compress)
        return Path(temp_path).stat().st_size
    finally:
        Path(temp_path).unlink(missing_ok=True)


def export_onnx_artifact(
    model: Any,
    output_dir: str | Path,
    input_width: int,
) -> dict[str, Any]:
    """Export a supported sklearn model to ONNX when skl2onnx is installed."""
    if not SKL2ONNX_AVAILABLE:
        return {
            "requested": True,
            "exported": False,
            "reason": "skl2onnx is not installed.",
        }

    if convert_sklearn is None or FloatTensorType is None:  # pragma: no cover
        return {
            "requested": True,
            "exported": False,
            "reason": "ONNX exporter is unavailable.",
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / "best_model.onnx"

    initial_types = [("float_input", FloatTensorType([None, input_width]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types)
    onnx_path.write_bytes(onnx_model.SerializeToString())

    return {
        "requested": True,
        "exported": True,
        "path": str(onnx_path),
        "size_bytes": int(onnx_path.stat().st_size),
    }


def build_jax_metadata(model: Any, label_classes: list[str]) -> dict[str, Any]:
    """Return JAX inference metadata for LogisticRegression models when possible."""
    if not JAX_AVAILABLE:
        return {
            "enabled": False,
            "reason": "jax is not installed.",
        }

    if not isinstance(model, LogisticRegression):
        return {
            "enabled": False,
            "reason": (
                "JAX conversion is currently implemented for "
                "LogisticRegression only."
            ),
        }

    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.asarray(model.intercept_, dtype=np.float32)

    _ = jnp_module.asarray(coef)
    _ = jnp_module.asarray(intercept)

    return {
        "enabled": True,
        "model_type": "logistic_regression",
        "coef": coef,
        "intercept": intercept,
        "classes": list(label_classes),
    }


def predict_proba_with_jax(jax_metadata: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Run LogisticRegression inference with JAX-backed array ops."""
    if not JAX_AVAILABLE:
        raise RuntimeError("jax is not installed.")
    if not jax_metadata.get("enabled"):
        raise RuntimeError("JAX metadata is not enabled for this model bundle.")

    coef = jnp_module.asarray(jax_metadata["coef"], dtype=jnp_module.float32)
    intercept = jnp_module.asarray(jax_metadata["intercept"], dtype=jnp_module.float32)
    features = jnp_module.asarray(np.asarray(X), dtype=jnp_module.float32)
    logits = features @ coef.T + intercept

    if logits.ndim == 1 or logits.shape[1] == 1:
        logits = logits.reshape(-1, 1)
        probs_pos = 1.0 / (1.0 + jnp_module.exp(-logits))
        probs = jnp_module.concatenate([1.0 - probs_pos, probs_pos], axis=1)
    else:
        shifted = logits - jnp_module.max(logits, axis=1, keepdims=True)
        exp_scores = jnp_module.exp(shifted)
        probs = exp_scores / jnp_module.sum(exp_scores, axis=1, keepdims=True)

    return np.asarray(probs)


def predict_with_jax(jax_metadata: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Return predicted class indices using the JAX inference path."""
    probabilities = predict_proba_with_jax(jax_metadata, X)
    return np.asarray(np.argmax(probabilities, axis=1), dtype=int)
