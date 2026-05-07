from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

from joblib import Memory
import numpy as np
import pandas as pd

from backend.core.config import settings
from backend.core.logging_config import configure_logger
from ml.explainability import (
    compute_shap_values,
    load_model_bundle,
    transform_input_for_model,
)
from ml.optimization import predict_proba_with_jax, predict_with_jax
from backend.services.recommendation_service import RecommendationService
from backend.services.validation_service import InputSchema, ValidationService

PREDICTION_MEMORY = Memory(settings.prediction_cache_dir, verbose=0)


def _build_local_explanation_rows(
    explanation: Any,
    feature_frame: pd.DataFrame,
    row_index: int,
    predicted_index: int,
    top_k: int,
) -> list[dict[str, Any]]:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        row_values = values[row_index, :, predicted_index]
    else:
        row_values = values[row_index]

    row_features = feature_frame.iloc[row_index]
    local_df = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "feature_value": row_features.values,
            "shap_value": row_values,
            "abs_shap_value": np.abs(row_values),
        }
    ).sort_values(by="abs_shap_value", ascending=False)

    return [
        {
            "feature": str(row["feature"]),
            "feature_value": (
                row["feature_value"].item()
                if hasattr(row["feature_value"], "item")
                else row["feature_value"]
            ),
            "shap_value": round(float(row["shap_value"]), 6),
            "abs_shap_value": round(float(row["abs_shap_value"]), 6),
        }
        for _, row in local_df.head(top_k).iterrows()
    ]


def _build_global_importance_rows(
    explanation: Any, feature_frame: pd.DataFrame
) -> list[dict[str, Any]]:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        importance = np.abs(values).mean(axis=(0, 2))
    else:
        importance = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "mean_abs_shap": importance,
        }
    ).sort_values(by="mean_abs_shap", ascending=False)

    return [
        {
            "feature": str(row["feature"]),
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
        }
        for _, row in importance_df.head(10).iterrows()
    ]


def _predict_with_bundle(
    bundle: dict[str, Any],
    transformed: pd.DataFrame,
    use_jax: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    jax_metadata = bundle.get("optimization", {}).get("jax", {})
    probabilities_out: np.ndarray | None
    if use_jax and jax_metadata.get("enabled"):
        probabilities_out = predict_proba_with_jax(jax_metadata, transformed.to_numpy())
        indices = predict_with_jax(jax_metadata, transformed.to_numpy())
        return indices, probabilities_out

    model = bundle["model"]
    indices = np.asarray(model.predict(transformed.to_numpy()))
    probabilities_out = (
        np.asarray(model.predict_proba(transformed.to_numpy()))
        if hasattr(model, "predict_proba")
        else None
    )
    return indices, probabilities_out


@PREDICTION_MEMORY.cache
def run_cached_prediction(
    bundle_path: str,
    raw_json: str,
    include_explanations: bool,
    explanation_top_k: int,
    max_explanations: int,
    use_jax: bool,
) -> dict[str, Any]:
    bundle = load_model_bundle(bundle_path)
    raw_df = pd.read_json(io.StringIO(raw_json), orient="split")
    transformed = transform_input_for_model(bundle, raw_df)
    label_classes = list(bundle.get("label_classes", []))

    predicted_indices, class_probabilities = _predict_with_bundle(
        bundle, transformed, use_jax=use_jax and not include_explanations
    )
    predicted_labels = [
        label_classes[int(idx)] if label_classes else str(idx)
        for idx in predicted_indices
    ]

    explanation = None
    if include_explanations:
        explain_result = compute_shap_values(
            bundle_path=bundle_path,
            raw_sample_df=raw_df,
            background_df=raw_df.head(min(len(raw_df), 100)),
        )
        explanation = explain_result["explanation"]
        transformed = explain_result["sample_transformed"]

    predictions: list[dict[str, Any]] = []
    for idx, label in enumerate(predicted_labels):
        confidence = None
        prob_map = None
        if class_probabilities is not None:
            row_probs = class_probabilities[idx]
            prob_map = {
                class_name: round(float(prob), 6)
                for class_name, prob in zip(label_classes, row_probs)
            }
            confidence = round(float(np.max(row_probs)), 6)

        top_contributions: list[dict[str, Any]] = []
        if explanation is not None and idx < max_explanations:
            top_contributions = _build_local_explanation_rows(
                explanation=explanation,
                feature_frame=transformed,
                row_index=idx,
                predicted_index=int(predicted_indices[idx]),
                top_k=explanation_top_k,
            )

        predictions.append(
            {
                "row_index": idx,
                "predicted_label": label,
                "predicted_index": int(predicted_indices[idx]),
                "confidence": confidence,
                "class_probabilities": prob_map,
                "top_contributions": top_contributions,
            }
        )

    summary = {
        "prediction_count": len(predictions),
        "labels": {},
        "global_feature_importance": (
            _build_global_importance_rows(explanation, transformed)
            if explanation is not None
            else []
        ),
    }
    return {"predictions": predictions, "summary": summary}


class ModelService:
    """Inference and explainability wrapper around the saved best-model bundle."""

    def __init__(self, bundle_path: str, max_explanations: int = 10):
        self.bundle_path = bundle_path
        self.max_explanations = max_explanations
        self.bundle = load_model_bundle(bundle_path)
        self.validation_service = ValidationService(self._build_input_schema())
        self.recommendation_service = RecommendationService()
        self.logger = configure_logger(
            "safeguard.backend.model", settings.log_file_path
        )

    def ping(self) -> bool:
        return Path(self.bundle_path).exists()

    @property
    def model_name(self) -> str:
        return str(self.bundle["model_name"])

    @property
    def label_classes(self) -> list[str]:
        return list(self.bundle.get("label_classes", []))

    def get_model_info(self) -> dict[str, Any]:
        optimization = self.bundle.get("optimization", {})
        return {
            "model_name": self.model_name,
            "label_classes": self.label_classes,
            "feature_count": len(self.bundle.get("feature_names", [])),
            "feature_names": list(self.bundle.get("feature_names", [])),
            "raw_input_schema": {
                "required_columns": self.validation_service.schema.required_columns,
                "numeric_columns": self.validation_service.schema.numeric_columns,
                "categorical_columns": (
                    self.validation_service.schema.categorical_columns
                ),
            },
            "training_config": (
                vars(self.bundle.get("training_config"))
                if self.bundle.get("training_config")
                else {}
            ),
            "artifacts": {
                "preprocessor": self.bundle.get("preprocessor") is not None,
                "correlation_filter": self.bundle.get("correlation_filter") is not None,
                "feature_engineer": self.bundle.get("feature_engineer") is not None,
            },
            "optimization": {
                "feature_pruning": optimization.get("feature_pruning", {}),
                "quantization": optimization.get("quantization", {}),
                "onnx_export": optimization.get("onnx_export", {}),
                "jax": optimization.get("jax", {}),
                "prediction_cache_dir": settings.prediction_cache_dir,
            },
        }

    def predict(
        self,
        raw_df: pd.DataFrame,
        include_explanations: bool = True,
        explanation_top_k: int = 5,
    ) -> dict[str, Any]:
        raw_df = self.validation_service.sanitize_and_validate(raw_df)
        raw_json = raw_df.to_json(orient="split")
        self.logger.info(
            "Prediction request accepted.",
            extra={
                "event_type": "prediction_started",
                "details": {
                    "row_count": len(raw_df),
                    "include_explanations": include_explanations,
                },
            },
        )
        cached_result = run_cached_prediction(
            bundle_path=self.bundle_path,
            raw_json=raw_json,
            include_explanations=include_explanations,
            explanation_top_k=explanation_top_k,
            max_explanations=self.max_explanations,
            use_jax=settings.use_jax_inference,
        )

        predictions: list[dict[str, Any]] = []
        aggregated_actions: list[str] = []
        for item in cached_result["predictions"]:
            label = str(item["predicted_label"])
            recommendation = self.recommendation_service.get_recommendation(label)
            recommendation_severity = str(recommendation["severity"])
            recommendation_suggestions = cast(list[str], recommendation["suggestions"])
            if recommendation_severity.lower() not in {"info", "normal"}:
                self.logger.warning(
                    "Attack detected during prediction.",
                    extra={
                        "event_type": "attack_detected",
                        "details": {
                            "predicted_label": label,
                            "severity": recommendation_severity,
                            "row_index": int(item["row_index"]),
                        },
                    },
                )
            for suggestion in recommendation_suggestions:
                if suggestion not in aggregated_actions:
                    aggregated_actions.append(suggestion)

            predictions.append(
                {
                    "row_index": int(item["row_index"]),
                    "predicted_label": label,
                    "predicted_index": int(item["predicted_index"]),
                    "confidence": item.get("confidence"),
                    "class_probabilities": item.get("class_probabilities"),
                    "top_contributions": item.get("top_contributions", []),
                    "recommendation_severity": recommendation_severity,
                    "recommendations": recommendation_suggestions,
                }
            )

        summary_labels = self._label_counts(
            [str(prediction["predicted_label"]) for prediction in predictions]
        )
        result = {
            "model_name": self.model_name,
            "predictions": predictions,
            "summary": {
                "prediction_count": len(predictions),
                "labels": summary_labels,
                "global_feature_importance": cached_result["summary"][
                    "global_feature_importance"
                ],
                "recommended_actions": aggregated_actions,
            },
        }
        self.logger.info(
            "Prediction completed.",
            extra={
                "event_type": "prediction_completed",
                "details": {
                    "row_count": len(predictions),
                    "labels": summary_labels,
                },
            },
        )
        return result

    def _label_counts(self, labels: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def sanitize_and_validate(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        return self.validation_service.sanitize_and_validate(raw_df)

    def _build_input_schema(self) -> InputSchema:
        preprocessor = self.bundle["preprocessor"]
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []

        for name, _transformer, columns in preprocessor.transformers_:
            column_list = cast(list[str], list(columns))
            if name == "num":
                numeric_columns.extend(column_list)
            elif name == "cat":
                categorical_columns.extend(column_list)

        required_columns = numeric_columns + categorical_columns
        return InputSchema(
            required_columns=required_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
