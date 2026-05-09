"""
Reusable preprocessing utilities for intrusion-detection datasets.

Features:
- load raw CSV data
- handle missing values by dropping or imputing
- encode categorical feature columns
- encode target labels
- scale numeric features
- detect and remove highly correlated features
- optionally apply PCA or tree-based feature selection
- split into train/test sets
- save fitted transformers with joblib
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


@dataclass
class PreprocessingConfig:
    target_column: str
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


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop highly correlated columns using the training split only."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.columns_to_drop_: list[str] = []
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CorrelationFilter":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationFilter expects a pandas DataFrame as input.")

        self.feature_names_in_ = X.columns.tolist()
        numeric_df = X.select_dtypes(include=["number", "bool"])
        self.columns_to_drop_ = []

        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            self.columns_to_drop_ = [
                column
                for column in upper.columns
                if (upper[column] > self.threshold).any()
            ]

        self.feature_names_out_ = [
            col for col in self.feature_names_in_ if col not in self.columns_to_drop_
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationFilter expects a pandas DataFrame as input.")
        return X.drop(columns=self.columns_to_drop_, errors="ignore")

    def get_support(self) -> list[str]:
        return self.feature_names_out_


def set_global_seed(seed: int) -> None:
    """Set Python and NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_raw_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a raw CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and whitespace."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    return cleaned


def separate_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV.")

    feature_df = df.copy()
    y = feature_df.pop(target_column)

    if drop_columns:
        existing = [col for col in drop_columns if col in feature_df.columns]
        feature_df = feature_df.drop(columns=existing)

    return feature_df, y


def handle_missing_rows(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Handle missing values before transformation.

    - 'drop': remove rows with any missing value in features or target
    - 'impute': keep rows and let sklearn imputers handle feature nulls
    """
    combined = X.copy()
    combined["__target__"] = y

    if strategy == "drop":
        combined = combined.dropna(axis=0)
    elif strategy == "impute":
        combined = combined.dropna(subset=["__target__"])
    else:
        raise ValueError("missing_strategy must be either 'drop' or 'impute'.")

    y_clean = combined.pop("__target__")
    return combined, y_clean


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature column names."""
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def build_scaler(scaler_type: str) -> Any:
    """Build the requested scaler."""
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler_type must be either 'standard' or 'minmax'.")


def build_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder instance."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_feature_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: PreprocessingConfig,
) -> ColumnTransformer:
    """Create a ColumnTransformer for numeric and categorical features."""
    numeric_steps: list[tuple[str, Any]] = []
    categorical_steps: list[tuple[str, Any]] = []

    if config.missing_strategy == "impute":
        numeric_steps.append(
            ("imputer", SimpleImputer(strategy=config.numeric_impute_strategy))
        )
        categorical_steps.append(
            ("imputer", SimpleImputer(strategy=config.categorical_impute_strategy))
        )

    numeric_steps.append(("scaler", build_scaler(config.scaler_type)))
    categorical_steps.append(("encoder", build_one_hot_encoder()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=categorical_steps)

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No usable feature columns were found after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def encode_labels(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Encode string or categorical target labels."""
    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(y.astype(str))
    return pd.Series(encoded, index=y.index, name=y.name), label_encoder


def apply_correlation_filter(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, CorrelationFilter | None]:
    """Optionally remove highly correlated raw feature columns."""
    if not config.enable_correlation_filter:
        return X_train, X_test, None

    correlation_filter = CorrelationFilter(threshold=config.correlation_threshold)
    X_train_filtered = correlation_filter.fit_transform(X_train)
    X_test_filtered = correlation_filter.transform(X_test)
    return X_train_filtered, X_test_filtered, correlation_filter


def build_feature_engineer(
    config: PreprocessingConfig,
) -> Pipeline | None:
    """Build optional post-preprocessing feature engineering pipeline."""
    steps: list[tuple[str, Any]] = []

    if config.reduction_method == "none":
        return None
    if config.reduction_method == "pca":
        pca_kwargs: dict[str, Any] = {"random_state": config.random_state}
        if config.pca_n_components is not None:
            pca_kwargs["n_components"] = config.pca_n_components
        steps.append(("pca", PCA(**pca_kwargs)))
        return Pipeline(steps=steps)
    if config.reduction_method == "tree":
        estimator = ExtraTreesClassifier(
            n_estimators=200,
            random_state=config.random_state,
            n_jobs=-1,
        )
        selector = SelectFromModel(
            estimator=estimator,
            threshold=config.tree_selection_threshold,
            max_features=config.tree_selection_max_features,
        )
        steps.append(("tree_selector", selector))
        return Pipeline(steps=steps)

    raise ValueError("reduction_method must be one of: 'none', 'pca', 'tree'.")


def generate_engineered_feature_names(
    feature_engineer: Pipeline | None,
    input_feature_names: list[str],
) -> list[str]:
    """Return output feature names after optional dimensionality reduction."""
    if feature_engineer is None:
        return input_feature_names

    step_name = feature_engineer.steps[-1][0]
    transformer = feature_engineer.steps[-1][1]

    if step_name == "pca":
        component_count = getattr(
            transformer, "n_components_", len(input_feature_names)
        )
        return [f"pca_component_{idx + 1}" for idx in range(component_count)]

    if step_name == "tree_selector":
        support = transformer.get_support()
        return [name for name, keep in zip(input_feature_names, support) if keep]

    return input_feature_names


def apply_feature_engineering(
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    y_train: pd.Series,
    config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray, Pipeline | None]:
    """Optionally apply PCA or tree-based feature selection."""
    feature_engineer = build_feature_engineer(config)
    if feature_engineer is None:
        return X_train_processed, X_test_processed, None

    X_train_engineered = feature_engineer.fit_transform(X_train_processed, y_train)
    X_test_engineered = feature_engineer.transform(X_test_processed)
    return X_train_engineered, X_test_engineered, feature_engineer


def fit_transform_dataset(
    df: pd.DataFrame,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """Fit preprocessing objects and create train/test splits."""
    set_global_seed(config.random_state)

    df = clean_dataframe(df)
    X, y = separate_features_and_target(df, config.target_column, config.drop_columns)
    X, y = handle_missing_rows(X, y, config.missing_strategy)
    y_encoded, label_encoder = encode_labels(y)

    stratify_labels = y_encoded if config.stratify and y_encoded.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_labels,
    )

    X_train, X_test, correlation_filter = apply_correlation_filter(
        X_train, X_test, config
    )

    numeric_columns, categorical_columns = get_feature_types(X_train)
    preprocessor = build_feature_preprocessor(
        numeric_columns, categorical_columns, config
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    preprocessed_feature_names = preprocessor.get_feature_names_out().tolist()
    X_train_engineered, X_test_engineered, feature_engineer = apply_feature_engineering(
        X_train_processed,
        X_test_processed,
        y_train,
        config,
    )
    engineered_feature_names = generate_engineered_feature_names(
        feature_engineer, preprocessed_feature_names
    )

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_engineered,
        "X_test_processed": X_test_engineered,
        "X_train_preprocessed": X_train_processed,
        "X_test_preprocessed": X_test_processed,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "correlation_filter": correlation_filter,
        "feature_engineer": feature_engineer,
        "feature_names": engineered_feature_names,
        "preprocessed_feature_names": preprocessed_feature_names,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


def save_artifacts(
    artifacts_dir: str | Path,
    preprocessor: ColumnTransformer,
    label_encoder: LabelEncoder,
    metadata: dict[str, Any],
    correlation_filter: CorrelationFilter | None = None,
    feature_engineer: Pipeline | None = None,
) -> None:
    """Persist fitted transformers and metadata for later reuse."""
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")
    joblib.dump(label_encoder, artifacts_path / "label_encoder.joblib")
    joblib.dump(metadata, artifacts_path / "preprocessing_metadata.joblib")

    if correlation_filter is not None:
        joblib.dump(correlation_filter, artifacts_path / "correlation_filter.joblib")
    if feature_engineer is not None:
        joblib.dump(feature_engineer, artifacts_path / "feature_engineer.joblib")


def preprocess_csv(
    csv_path: str | Path,
    artifacts_dir: str | Path,
    config: PreprocessingConfig,
) -> dict[str, Any]:
    """End-to-end CSV preprocessing workflow."""
    df = load_raw_csv(csv_path)
    results = fit_transform_dataset(df, config)

    metadata = {
        "target_column": config.target_column,
        "test_size": config.test_size,
        "random_state": config.random_state,
        "missing_strategy": config.missing_strategy,
        "numeric_impute_strategy": config.numeric_impute_strategy,
        "categorical_impute_strategy": config.categorical_impute_strategy,
        "scaler_type": config.scaler_type,
        "drop_columns": config.drop_columns or [],
        "enable_correlation_filter": config.enable_correlation_filter,
        "correlation_threshold": config.correlation_threshold,
        "reduction_method": config.reduction_method,
        "pca_n_components": config.pca_n_components,
        "tree_selection_threshold": config.tree_selection_threshold,
        "tree_selection_max_features": config.tree_selection_max_features,
        "feature_names": results["feature_names"],
        "preprocessed_feature_names": results["preprocessed_feature_names"],
        "numeric_columns": results["numeric_columns"],
        "categorical_columns": results["categorical_columns"],
        "label_classes": results["label_encoder"].classes_.tolist(),
        "correlation_dropped_columns": (
            results["correlation_filter"].columns_to_drop_
            if results["correlation_filter"] is not None
            else []
        ),
    }
    save_artifacts(
        artifacts_dir,
        results["preprocessor"],
        results["label_encoder"],
        metadata,
        correlation_filter=results["correlation_filter"],
        feature_engineer=results["feature_engineer"],
    )
    return results
