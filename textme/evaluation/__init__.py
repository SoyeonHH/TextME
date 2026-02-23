"""
Evaluation utilities for TextME.

Extracted from: EfficientBind/evaluation/
"""

from .config import parse_args
from .eval_utils import (
    calculate_similarity,
    get_recall_metrics,
    get_recall_metrics_multi_sentence,
    get_accuracy_metrics,
    get_ranking_metrics,
    zeroshot_binary,
    get_mAP_from_similarity_matrix,
    save_predictions,
)

__all__ = [
    "parse_args",
    "calculate_similarity",
    "get_recall_metrics",
    "get_recall_metrics_multi_sentence",
    "get_accuracy_metrics",
    "get_ranking_metrics",
    "zeroshot_binary",
    "get_mAP_from_similarity_matrix",
    "save_predictions",
]
