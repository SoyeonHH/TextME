"""
TextME: Bridging Unseen Modalities Through Text Descriptions

A text-only modality expansion framework that projects diverse modalities
into LLM embedding space without requiring paired cross-modal data.

This codebase is extracted from the actual EfficientBind implementation.
"""

__version__ = "1.0.0"
__author__ = "Soyeon Hong, Jinchan Kim, Jaegook You, Seungtaek Choi, Suha Kwak, Hyunsouk Cho"

from .models.projector import ProjectionHead, get_activation
from .models.encoders import (
    build_encoder,
    ENCODER_DIM,
    MODEL_OFFSET_DATA,
    ANCHOR_DIM,
    process_embeddings,
)
from .losses import (
    ContrastiveLoss,
    HardNegativeContrastiveLoss,
    TripletLoss,
    TextMSELoss,
    build_loss,
)
from .data import CaptionDataset, build_dataloader, DATASET_CONFIG
from .evaluation import (
    calculate_similarity,
    get_recall_metrics,
    get_accuracy_metrics,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Models
    "ProjectionHead",
    "get_activation",
    "build_encoder",
    "process_embeddings",
    # Losses
    "ContrastiveLoss",
    "HardNegativeContrastiveLoss",
    "TripletLoss",
    "TextMSELoss",
    "build_loss",
    # Data
    "CaptionDataset",
    "build_dataloader",
    # Constants
    "ENCODER_DIM",
    "MODEL_OFFSET_DATA",
    "ANCHOR_DIM",
    "DATASET_CONFIG",
    # Evaluation
    "calculate_similarity",
    "get_recall_metrics",
    "get_accuracy_metrics",
]
