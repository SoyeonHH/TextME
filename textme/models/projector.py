"""
Projection Head for TextME.

Two-layer MLP that maps encoder embeddings to the LLM anchor space.
Extracted from actual EfficientBind implementation (src/projector.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def get_activation(activation: str) -> nn.Module:
    """Get activation function by name.

    Reference: EfficientBind/src/projector.py
    """
    activation = activation.lower()
    if activation == 'gelu':
        return nn.GELU()
    elif activation == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'silu':
        return nn.SiLU(inplace=True)
    elif activation == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)


class ProjectionHead(nn.Module):
    """
    Projection head for modality alignment.

    Two layer MLP with GELU activation.
    Reference: https://github.com/zehanwang01/OmniBind/blob/main/omni_model/projector.py
    Extracted from: EfficientBind/src/projector.py

    Args:
        in_dim: Input dimension (encoder embedding dim)
        proj_dim: Hidden/projection layer dimension (typically 2 * in_dim)
        out_dim: Output dimension (LLM embedding dim, e.g., 2560 for Qwen3-Embedding-4B)
        init_mode: Weight initialization mode ('xav' for xavier, 'eye' for identity)
        dim_act: Activation function name (default: 'gelu')
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        out_dim: int,
        init_mode: str = 'xav',
        dim_act: str = 'gelu',
    ):
        super(ProjectionHead, self).__init__()

        self.mlps = nn.Sequential(
            nn.Linear(in_dim, proj_dim, bias=True),
            nn.BatchNorm1d(proj_dim),
            get_activation(dim_act),
            nn.Linear(proj_dim, out_dim, bias=True),
            nn.BatchNorm1d(out_dim),
        )

        self.init_weights(init_mode)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with L2 normalization.

        Args:
            x: Input embeddings [batch_size, in_dim]

        Returns:
            Normalized projected embeddings [batch_size, out_dim]
        """
        embs = self.mlps(x)
        return F.normalize(embs, dim=-1)

    def init_weights(self, mode: str):
        """Initialize weights."""
        if mode == 'eye':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.eye_(m)
        elif mode == 'xav':
            for m in self.parameters():
                if m.dim() > 1:
                    nn.init.xavier_uniform_(m)

    def get_device(self):
        """Get the device of the model parameters."""
        return next(self.parameters()).device
