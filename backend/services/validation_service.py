from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backend.core.config import settings
from backend.core.logging_config import configure_logger
from backend.utils.sanitization import sanitize_dataframe


class InputValidationError(ValueError):
    """Raised when uploaded or submitted records fail schema validation."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []


@dataclass
class InputSchema:
    required_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]


class ValidationService:
    """Validate and sanitize incoming inference data against the saved model schema."""

    def __init__(self, schema: InputSchema):
        self.schema = schema
        self.logger = configure_logger(
            "safeguard.backend.validation", settings.log_file_path
        )

    def sanitize_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise InputValidationError(
                "Input data is empty.", ["No rows were provided."]
            )

        sanitized = sanitize_dataframe(df)
        self._validate_columns(sanitized)
        sanitized = self._coerce_numeric_columns(sanitized)
        sanitized = self._coerce_categorical_columns(sanitized)
        return sanitized[self.schema.required_columns].copy()

    def _validate_columns(self, df: pd.DataFrame) -> None:
        incoming_columns = df.columns.tolist()
        missing = [
            col for col in self.schema.required_columns if col not in incoming_columns
        ]
        unexpected = [
            col for col in incoming_columns if col not in self.schema.required_columns
        ]
        errors: list[str] = []

        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}.")
        if unexpected:
            errors.append(f"Unexpected columns: {', '.join(unexpected)}.")

        if errors:
            self.logger.warning(
                "Column validation failed.",
                extra={
                    "event_type": "column_validation_failed",
                    "details": {"errors": errors, "incoming_columns": incoming_columns},
                },
            )
            raise InputValidationError("Column validation failed.", errors)

    def _coerce_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        coerced = df.copy()
        errors: list[str] = []

        for col in self.schema.numeric_columns:
            series = pd.to_numeric(coerced[col], errors="coerce")
            invalid_mask = series.isna() & coerced[col].notna()
            if invalid_mask.any():
                bad_examples = (
                    coerced.loc[invalid_mask, col].astype(str).head(3).tolist()
                )
                errors.append(
                    f"Column '{col}' must be numeric. Invalid values: {bad_examples}."
                )
            coerced[col] = series

        if errors:
            self.logger.warning(
                "Data type validation failed.",
                extra={
                    "event_type": "datatype_validation_failed",
                    "details": {"errors": errors},
                },
            )
            raise InputValidationError("Data type validation failed.", errors)
        return coerced

    def _coerce_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        coerced = df.copy()
        for col in self.schema.categorical_columns:
            coerced[col] = coerced[col].astype("string").replace({"<NA>": pd.NA})
        return coerced
