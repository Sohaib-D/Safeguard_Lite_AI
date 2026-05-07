from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class FeatureRecord(BaseModel):
    features: dict[str, Any]


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    include_explanations: bool = True
    explanation_top_k: int = Field(default=5, ge=1, le=25)


class LocalExplanation(BaseModel):
    feature: str
    feature_value: Any
    shap_value: float
    abs_shap_value: float


class PredictionItem(BaseModel):
    row_index: int
    predicted_label: str
    predicted_index: int
    confidence: float | None
    class_probabilities: dict[str, float] | None = None
    top_contributions: list[LocalExplanation] = Field(default_factory=list)
    recommendation_severity: str = "info"
    recommendations: list[str] = Field(default_factory=list)


class PredictionSummary(BaseModel):
    prediction_count: int
    labels: dict[str, int]
    global_feature_importance: list[dict[str, float | str]] = Field(
        default_factory=list
    )
    recommended_actions: list[str] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    model_name: str
    predictions: list[PredictionItem]
    summary: PredictionSummary
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UploadRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)
    source_name: str = "api_json"

    @model_validator(mode="after")
    def validate_records(self) -> "UploadRequest":
        if not self.records:
            raise ValueError("records cannot be empty.")
        return self
