from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import PreprocessingConfig, fit_transform_dataset


def build_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "service": ["http", "dns", "http", "ssh", "dns", None],
            "label": ["Normal", "DDoS", "Normal", "PortScan", "DDoS", "Normal"],
        }
    )


def test_fit_transform_dataset_with_imputation_and_scaling():
    df = build_sample_frame()
    config = PreprocessingConfig(
        target_column="label",
        missing_strategy="impute",
        scaler_type="standard",
        stratify=False,
    )

    result = fit_transform_dataset(df, config)

    assert result["X_train_processed"].shape[0] > 0
    assert result["X_test_processed"].shape[0] > 0
    assert len(result["feature_names"]) > 0
    assert set(result["label_encoder"].classes_) == {"DDoS", "Normal", "PortScan"}


def test_fit_transform_dataset_with_drop_missing_strategy_removes_null_rows():
    df = build_sample_frame()
    config = PreprocessingConfig(
        target_column="label", missing_strategy="drop", stratify=False
    )

    result = fit_transform_dataset(df, config)

    total_rows_after_drop = len(result["X_train_raw"]) + len(result["X_test_raw"])
    assert total_rows_after_drop == 4


def test_correlation_filter_removes_highly_correlated_columns():
    df = pd.DataFrame(
        {
            "f0": [1, 2, 3, 4, 5, 6],
            "f1": [2, 4, 6, 8, 10, 12],
            "service": ["http", "dns", "http", "ssh", "dns", "ssh"],
            "label": ["Normal", "DDoS", "Normal", "DDoS", "Normal", "DDoS"],
        }
    )
    config = PreprocessingConfig(
        target_column="label",
        enable_correlation_filter=True,
        correlation_threshold=0.90,
        stratify=False,
    )

    result = fit_transform_dataset(df, config)

    assert result["correlation_filter"] is not None
    assert (
        "f1" in result["correlation_filter"].columns_to_drop_
        or "f0" in result["correlation_filter"].columns_to_drop_
    )


def test_fit_transform_dataset_raises_for_missing_target():
    df = build_sample_frame().drop(columns=["label"])
    config = PreprocessingConfig(target_column="label")

    with pytest.raises(ValueError, match="Target column 'label' not found"):
        fit_transform_dataset(df, config)
