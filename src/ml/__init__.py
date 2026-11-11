"""
Machine Learning layer for the Financial Intelligence & Stock Analytics Platform.

Provides regression, classification, clustering, evaluation, and persistence
for historical stock market features with strict chronological validation and
zero look-ahead bias.
"""

from src.ml.evaluation import (
    calculate_regression_metrics,
    evaluate_zero_return_baseline,
    calculate_classification_metrics,
    evaluate_majority_baseline,
    calculate_clustering_metrics,
)
from src.ml.regression import (
    prepare_regression_dataset,
    chronological_train_val_test_split,
    train_linear_regression,
    train_random_forest_regressor,
    get_feature_importance,
)
from src.ml.classification import (
    prepare_classification_dataset,
    train_logistic_regression,
    train_random_forest_classifier,
)
from src.ml.clustering import (
    extract_stock_cluster_features,
    find_optimal_clusters,
    fit_stock_clusters,
)

__all__ = [
    "calculate_regression_metrics",
    "evaluate_zero_return_baseline",
    "calculate_classification_metrics",
    "evaluate_majority_baseline",
    "calculate_clustering_metrics",
    "prepare_regression_dataset",
    "chronological_train_val_test_split",
    "train_linear_regression",
    "train_random_forest_regressor",
    "get_feature_importance",
    "prepare_classification_dataset",
    "train_logistic_regression",
    "train_random_forest_classifier",
    "extract_stock_cluster_features",
    "find_optimal_clusters",
    "fit_stock_clusters",
]
