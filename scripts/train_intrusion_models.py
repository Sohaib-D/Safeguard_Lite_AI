"""
CLI entrypoint for training multiple intrusion-detection classifiers.

Example:
    python scripts/train_intrusion_models.py ^
        --input data/processed/intrusion_datasets_merged.csv ^
        --target label_text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import TrainingConfig, train_and_select_best_model


def parse_optional_numeric(value: str | None) -> int | float | None:
    if value is None:
        return None
    if "." in value:
        return float(value)
    return int(value)


def parse_threshold(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare intrusion-detection classifiers."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--target", required=True, help="Target/label column name.")
    parser.add_argument(
        "--model-output-dir",
        default="models/trained",
        help="Directory for trained models and metrics.",
    )
    parser.add_argument(
        "--preprocessing-artifacts-dir",
        default="models/preprocessing",
        help="Directory for preprocessing joblib artifacts.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Train/test split ratio."
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        default="f1",
    )
    parser.add_argument(
        "--missing-strategy", choices=["drop", "impute"], default="impute"
    )
    parser.add_argument(
        "--numeric-impute-strategy",
        choices=["mean", "median", "most_frequent"],
        default="median",
    )
    parser.add_argument(
        "--categorical-impute-strategy",
        choices=["most_frequent", "constant"],
        default="most_frequent",
    )
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--drop-columns", nargs="*", default=[])
    parser.add_argument("--enable-correlation-filter", action="store_true")
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument(
        "--reduction-method", choices=["none", "pca", "tree"], default="none"
    )
    parser.add_argument("--pca-n-components", default=None)
    parser.add_argument("--tree-selection-threshold", default="median")
    parser.add_argument("--tree-selection-max-features", type=int, default=None)
    parser.add_argument("--enable-feature-pruning", action="store_true")
    parser.add_argument("--feature-prune-max-features", type=int, default=None)
    parser.add_argument(
        "--feature-prune-importance-threshold", type=float, default=None
    )
    parser.add_argument("--optimization-max-metric-drop", type=float, default=0.02)
    parser.add_argument("--quantize-dtype", choices=["float32"], default="float32")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--enable-jax-conversion", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = TrainingConfig(
        model_output_dir=args.model_output_dir,
        preprocessing_artifacts_dir=args.preprocessing_artifacts_dir,
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
        selection_metric=args.selection_metric,
        enable_feature_pruning=args.enable_feature_pruning,
        feature_prune_max_features=args.feature_prune_max_features,
        feature_prune_importance_threshold=args.feature_prune_importance_threshold,
        optimization_max_metric_drop=args.optimization_max_metric_drop,
        quantize_dtype=args.quantize_dtype,
        export_onnx=args.export_onnx,
        enable_jax_conversion=args.enable_jax_conversion,
    )

    results = train_and_select_best_model(
        csv_path=args.input,
        target_column=args.target,
        config=config,
    )

    print("\nModel Evaluation Results")
    print(results["results_table"].to_string(index=False))
    print("\nConfusion Matrix")
    print(results["confusion_matrix"].to_string())
    print("\nPer-Class Summary")
    print(results["per_class_report"].to_string(index=False))
    print("\nOptimization Summary")
    print(results["optimization"])
    print(f"\nAvailable models trained: {', '.join(results['available_models'])}")
    print(f"Best model: {results['best_model_name']}")
    print(f"Saved best model bundle to: {config.model_output_dir}/best_model.pkl")


if __name__ == "__main__":
    main()
