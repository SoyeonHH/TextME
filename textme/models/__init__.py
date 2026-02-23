"""
Model components for TextME.

Extracted from: EfficientBind/src/models.py and EfficientBind/src/projector.py
"""

from .projector import ProjectionHead, get_activation
from .encoders import (
    build_encoder,
    BaseEncoder,
    CLIPEncoder,
    CLAPEncoder,
    LanguageBindEncoder,
    Uni3DEncoder,
    CXRCLIPEncoder,
    MoleculeSTMEncoder,
    RemoteCLIPEncoder,
    ENCODER_DIM,
    MODEL_OFFSET_DATA,
    ANCHOR_DIM,
    process_embeddings,
    generate_offset_config,
)

__all__ = [
    # Projector
    "ProjectionHead",
    "get_activation",
    # Encoders
    "build_encoder",
    "BaseEncoder",
    "CLIPEncoder",
    "CLAPEncoder",
    "LanguageBindEncoder",
    "Uni3DEncoder",
    "CXRCLIPEncoder",
    "MoleculeSTMEncoder",
    "RemoteCLIPEncoder",
    # Constants
    "ENCODER_DIM",
    "MODEL_OFFSET_DATA",
    "ANCHOR_DIM",
    # Utilities
    "process_embeddings",
    "generate_offset_config",
]
