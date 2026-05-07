"""
SHAP-based explainability utilities for trained intrusion-detection models.

Designed to work with the model bundle saved as ``best_model.pkl`` and to return
matplotlib figures / pandas DataFrames that can be rendered directly in Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def load_model_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load the saved best-model bundle."""
    return joblib.load(bundle_path)


def _ensure_dataframe(
    X: pd.DataFrame | np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Convert transformed feature arrays into a named DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=feature_names)


def transform_input_for_model(
    bundle: dict[str, Any],
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the saved preprocessing stack to raw input rows.

    The returned DataFrame is the exact feature matrix used by the model and SHAP.
    """
    features = raw_df.copy()

    correlation_filter = bundle.get("correlation_filter")
    if correlation_filter is not None:
        features = correlation_filter.transform(features)

    transformed = bundle["preprocessor"].transform(features)

    feature_engineer = bundle.get("feature_engineer")
    if feature_engineer is not None:
        transformed = feature_engineer.transform(transformed)

    selected_feature_indices = bundle.get("selected_feature_indices")
    if selected_feature_indices is not None:
        transformed = np.asarray(transformed)[:, selected_feature_indices]

    feature_names = bundle["feature_names"]
    return _ensure_dataframe(transformed, feature_names)


def build_explainer(
    bundle: dict[str, Any],
    background_data: pd.DataFrame,
) -> shap.Explainer:
    """Create a SHAP explainer using transformed background data."""
    model = bundle["model"]
    return shap.Explainer(model, background_data)


def compute_shap_values(
    bundle_path: str | Path,
    raw_sample_df: pd.DataFrame,
    background_df: pd.DataFrame | None = None,
    max_background_rows: int = 100,
) -> dict[str, Any]:
    """
    Load the saved model bundle and compute SHAP values for a sample dataset.

    Returns transformed features, predictions, the explainer, and SHAP Explanation.
    """
    bundle = load_model_bundle(bundle_path)
    sample_transformed = transform_input_for_model(bundle, raw_sample_df)

    if background_df is None:
        background_df = raw_sample_df

    if len(background_df) > max_background_rows:
        background_df = background_df.sample(n=max_background_rows, random_state=42)

    background_transformed = transform_input_for_model(bundle, background_df)
    explainer = build_explainer(bundle, background_transformed)
    explanation = explainer(sample_transformed)

    predicted_indices = bundle["model"].predict(sample_transformed.to_numpy())
    label_classes = bundle.get("label_classes", [])
    predicted_labels = [
        label_classes[int(idx)] if label_classes else str(idx)
        for idx in np.asarray(predicted_indices)
    ]

    return {
        "bundle": bundle,
        "sample_transformed": sample_transformed,
        "background_transformed": background_transformed,
        "explainer": explainer,
        "explanation": explanation,
        "predicted_indices": np.asarray(predicted_indices),
        "predicted_labels": predicted_labels,
    }


def _select_class_index(
    explanation: shap.Explanation,
    class_names: list[str] | None,
    class_name: str | None,
) -> int | None:
    """Map a user-specified class name to its SHAP output index."""
    values = np.asarray(explanation.values)
    if values.ndim < 3:
        return None

    if class_name is None:
        return 0

    if not class_names:
        raise ValueError("class_name was provided, but no class names are available.")

    if class_name not in class_names:
        raise ValueError(
            f"Unknown class_name '{class_name}'. Expected one of: {class_names}"
        )

    return class_names.index(class_name)


def _extract_shap_matrix(
    explanation: shap.Explanation,
    class_names: list[str] | None = None,
    class_name: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """Extract a 2D SHAP matrix from binary or multiclass explanations."""
    values = np.asarray(explanation.values)
    class_index = _select_class_index(explanation, class_names, class_name)

    if values.ndim == 2:
        return values, None
    if values.ndim == 3:
        if class_index is None:
            class_index = 0
        selected_name = class_names[class_index] if class_names else str(class_index)
        return values[:, :, class_index], selected_name

    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def plot_global_feature_importance(
    explanation: shap.Explanation,
    feature_frame: pd.DataFrame,
    class_names: list[str] | None = None,
    class_name: str | None = None,
    max_display: int = 20,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Plot global SHAP feature importance and return the figure plus summary table.

    The figure is matplotlib-based and can be rendered via ``st.pyplot(fig)``.
    """
    shap_matrix, selected_class = _extract_shap_matrix(
        explanation, class_names, class_name
    )
    importance = np.abs(shap_matrix).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_frame.columns,
            "mean_abs_shap": importance,
        }
    ).sort_values(by="mean_abs_shap", ascending=False)

    top_df = importance_df.head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top_df))))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color="#2f6db2")
    title = "SHAP Global Feature Importance"
    if selected_class is not None:
        title += f" ({selected_class})"
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig, importance_df.reset_index(drop=True)


def compute_local_explanation(
    explanation: shap.Explanation,
    feature_frame: pd.DataFrame,
    row_index: int,
    class_names: list[str] | None = None,
    class_name: str | None = None,
    max_display: int = 15,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Compute a local explanation for one instance and return table + waterfall plot.

    The DataFrame is ideal for ``st.dataframe`` and the figure for ``st.pyplot``.
    """
    shap_matrix, selected_class = _extract_shap_matrix(
        explanation, class_names, class_name
    )
    if row_index < 0 or row_index >= len(feature_frame):
        raise IndexError(
            f"row_index {row_index} is out of bounds for {len(feature_frame)} rows."
        )

    row_values = shap_matrix[row_index]
    row_features = feature_frame.iloc[row_index]
    local_df = (
        pd.DataFrame(
            {
                "feature": feature_frame.columns,
                "feature_value": row_features.values,
                "shap_value": row_values,
                "abs_shap_value": np.abs(row_values),
            }
        )
        .sort_values(by="abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )

    base_values = np.asarray(explanation.base_values)
    if base_values.ndim == 0:
        row_base_value = float(base_values)
    elif base_values.ndim == 1:
        row_base_value = float(base_values[row_index])
    else:
        class_index = _select_class_index(explanation, class_names, class_name) or 0
        row_base_value = float(base_values[row_index, class_index])

    exp = shap.Explanation(
        values=row_values,
        base_values=row_base_value,
        data=row_features.values,
        feature_names=feature_frame.columns.tolist(),
    )

    plt.close("all")
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    if selected_class is not None:
        fig.axes[0].set_title(f"SHAP Local Explanation ({selected_class})")
    fig.tight_layout()
    return local_df, fig
