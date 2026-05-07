"""
CLI entrypoint for preprocessing an intrusion-detection CSV dataset.

Example:
    python scripts/preprocess_dataset.py ^
        --input data/raw/my_dataset.csv ^
        --target label ^
        --artifacts-dir models/preprocessing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing import PreprocessingConfig, preprocess_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a raw CSV dataset for ML training."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--target", required=True, help="Target/label column name.")
    parser.add_argument(
        "--artifacts-dir",
        default="models/preprocessing",
        help="Directory to save joblib artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to save split datasets.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Train/test split ratio."
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--missing-strategy",
        choices=["drop", "impute"],
        default="impute",
        help="How to handle missing values.",
    )
    parser.add_argument(
        "--numeric-impute-strategy",
        choices=["mean", "median", "most_frequent"],
        default="median",
        help="Imputation strategy for numeric columns.",
    )
    parser.add_argument(
        "--categorical-impute-strategy",
        choices=["most_frequent", "constant"],
        default="most_frequent",
        help="Imputation strategy for categorical columns.",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax"],
        default="standard",
        help="Feature scaling method for numeric columns.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help="Optional feature columns to exclude before training.",
    )
    parser.add_argument(
        "--enable-correlation-filter",
        action="store_true",
        help="Remove highly correlated raw numeric features before encoding/scaling.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold for dropping raw numeric features.",
    )
    parser.add_argument(
        "--reduction-method",
        choices=["none", "pca", "tree"],
        default="none",
        help="Optional dimensionality reduction or feature selection method.",
    )
    parser.add_argument(
        "--pca-n-components",
        default=None,
        help="PCA components as int or float variance ratio, e.g. 20 or 0.95.",
    )
    parser.add_argument(
        "--tree-selection-threshold",
        default="median",
        help="Threshold for tree-based feature selection, e.g. median, mean, or numeric value.",
    )
    parser.add_argument(
        "--tree-selection-max-features",
        type=int,
        default=None,
        help="Optional cap on number of features kept by tree-based selection.",
    )
    return parser.parse_args()


def parse_optional_numeric(value: str | None) -> int | float | None:
    """Parse CLI numeric values that may be int or float or omitted."""
    if value is None:
        return None
    if "." in value:
        return float(value)
    return int(value)


def parse_threshold(value: str) -> str | float:
    """Keep symbolic thresholds like 'median' or convert numeric strings."""
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    args = parse_args()

    config = PreprocessingConfig(
        target_column=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        missing_strategy=args.missing_strategy,
        numeric_impute_strategy=args.numeric_impute_strategy,
        categorical_impute_strategy=args.categorical_impute_strategy,
        scaler_type=args.scaler,
        drop_columns=args.drop_columns,
        enable_correlation_filter=args.enable_correlation_filter,
        correlation_threshold=args.correlation_threshold,
        reduction_method=args.reduction_method,
        pca_n_components=parse_optional_numeric(args.pca_n_components),
        tree_selection_threshold=parse_threshold(args.tree_selection_threshold),
        tree_selection_max_features=args.tree_selection_max_features,
    )

    results = preprocess_csv(
        csv_path=args.input,
        artifacts_dir=args.artifacts_dir,
        config=config,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["X_train_processed"], output_dir / "X_train_processed.joblib")
    joblib.dump(results["X_test_processed"], output_dir / "X_test_processed.joblib")
    joblib.dump(results["y_train"], output_dir / "y_train.joblib")
    joblib.dump(results["y_test"], output_dir / "y_test.joblib")

    pd.DataFrame(results["X_train_raw"]).to_csv(
        output_dir / "X_train_raw.csv", index=False
    )
    pd.DataFrame(results["X_test_raw"]).to_csv(
        output_dir / "X_test_raw.csv", index=False
    )
    pd.DataFrame({"target": results["y_train"]}).to_csv(
        output_dir / "y_train.csv", index=False
    )
    pd.DataFrame({"target": results["y_test"]}).to_csv(
        output_dir / "y_test.csv", index=False
    )

    print(f"Saved preprocessing artifacts to: {args.artifacts_dir}")
    print(f"Saved processed train/test outputs to: {output_dir}")
    print(f"Train rows: {len(results['X_train_raw'])}")
    print(f"Test rows: {len(results['X_test_raw'])}")
    print(f"Processed feature count: {len(results['feature_names'])}")
    print(f"Encoded classes: {results['label_encoder'].classes_.tolist()}")


if __name__ == "__main__":
    main()
