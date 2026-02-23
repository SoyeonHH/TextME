"""
Loss functions for TextME training.

Extracted from actual EfficientBind implementation (src/model_utils/criterion.py)
Reference: LanguageBind (https://github.com/PKU-YuanGroup/LanguageBind)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss with temperature scaling.

    Reference: LanguageBind (https://github.com/PKU-YuanGroup/LanguageBind)
    Extracted from: EfficientBind/src/model_utils/criterion.py

    Args:
        temperature: Softmax temperature for similarity scaling (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embedding_A: Tensor, embedding_B: Tensor) -> Tensor:
        """
        Compute contrastive loss between two embeddings.

        Args:
            embedding_A: Tensor of shape [batch_size, embedding_dim]
            embedding_B: Tensor of shape [batch_size, embedding_dim]

        Returns:
            loss: Scalar loss value
        """
        # Normalize embeddings
        embedding_A = F.normalize(embedding_A, dim=-1)
        embedding_B = F.normalize(embedding_B, dim=-1)

        # Compute similarity matrix
        similarity = torch.matmul(embedding_A, embedding_B.t()) / self.temperature

        # Create labels for positive pairs (diagonal)
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # Compute loss
        loss = self.criterion(similarity, labels)

        return loss


class HardNegativeContrastiveLoss(nn.Module):
    """
    Hard Negative Contrastive Loss for robust training.

    Filters out negative samples that are too easy or too hard,
    focusing on informative hard negatives within a similarity threshold range.

    Extracted from: EfficientBind/src/model_utils/criterion.py

    Args:
        temperature: Temperature parameter (tau) for similarity scaling (default: 0.07)
        top_perc_margin: Upper percentile threshold relative to positive score (default: 0.95)
        bottom_perc_margin: Lower percentile threshold relative to positive score (default: 0.05)
        max_neg: Maximum number of negatives to use per query (default: None = use all)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        top_perc_margin: float = 0.95,
        bottom_perc_margin: float = 0.05,
        max_neg: int = None
    ):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.top_perc_margin = top_perc_margin
        self.bottom_perc_margin = bottom_perc_margin
        self.max_neg = max_neg

    def forward(self, embedding_A: Tensor, embedding_B: Tensor) -> Tensor:
        """
        Compute hard negative contrastive loss.

        Args:
            embedding_A: Source embeddings [batch_size, dim]
            embedding_B: Target embeddings [batch_size, dim]

        Returns:
            loss: Scalar loss value
        """
        embedding_A = F.normalize(embedding_A, dim=-1)
        embedding_B = F.normalize(embedding_B, dim=-1)

        batch_size = embedding_A.size(0)
        device = embedding_A.device

        # Compute similarity matrix
        similarity = torch.matmul(embedding_A, embedding_B.t())
        pos_scores = torch.diag(similarity)

        # Create threshold matrix for each query based on its positive pair
        thresholds = pos_scores.unsqueeze(1) * self.top_perc_margin

        # Create lower threshold matrix for filtering too high similarity scores
        lower_thresholds = pos_scores.unsqueeze(1) * self.bottom_perc_margin

        # Create mask for valid negatives (similarity between lower and upper thresholds, and not self)
        negative_mask = (
            (similarity <= thresholds) &
            (similarity >= lower_thresholds) &
            (~torch.eye(batch_size, dtype=torch.bool, device=device))
        )

        # Optionally limit number of negatives
        if self.max_neg is not None and self.max_neg < batch_size - 1:
            # Find scores of valid negatives (use large negative number for invalid ones)
            valid_scores = similarity.clone()
            valid_scores[~negative_mask] = -float('inf')

            _, topk_indices = torch.topk(valid_scores, min(self.max_neg + 1, batch_size), dim=1)

            new_mask = torch.zeros_like(negative_mask)
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, min(self.max_neg + 1, batch_size))
            new_mask[batch_indices, topk_indices] = True

            new_mask = new_mask & (~torch.eye(batch_size, dtype=torch.bool, device=device))
            negative_mask = new_mask

        total_loss = None
        valid_count = 0

        for i in range(batch_size):
            # Skip if no negatives for this query
            if not torch.any(negative_mask[i]):
                continue

            # Get indices of negatives for this query
            neg_indices = torch.where(negative_mask[i])[0]
            candidates = torch.cat([embedding_B[i:i+1], embedding_B[neg_indices]], dim=0)

            query_sim = torch.matmul(embedding_A[i:i+1], candidates.t()) / self.temperature

            # First item (index 0) is positive
            target = torch.zeros(1, dtype=torch.long, device=device)

            loss = self.criterion(query_sim, target)

            if total_loss is None:
                total_loss = loss.sum()
            else:
                total_loss = total_loss + loss.sum()
            valid_count += 1

        if total_loss is None:
            # Return zero loss with proper gradient tracking
            return torch.zeros(1, device=device, requires_grad=True)
        return total_loss / valid_count


class TripletLoss(nn.Module):
    """
    Triplet Loss for NLI-style training.

    For NLI data:
        - anchor: premise embeddings
        - positive: entailment hypothesis embeddings
        - negative: contradiction hypothesis embeddings

    The loss encourages:
        sim(anchor, positive) > sim(anchor, negative) + margin

    Formula:
        loss = max(0, margin - sim(anchor, positive) + sim(anchor, negative))

    Extracted from: EfficientBind/src/model_utils/criterion.py

    Args:
        margin: Margin for triplet loss (default: 0.3)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor
    ) -> Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Tensor of shape [batch_size, embedding_dim] - premise embeddings
            positive: Tensor of shape [batch_size, embedding_dim] - entailment embeddings
            negative: Tensor of shape [batch_size, embedding_dim] - contradiction embeddings

        Returns:
            loss: Scalar loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        # Compute cosine similarities for each sample in the batch
        pos_similarity = torch.sum(anchor * positive, dim=-1)  # [batch_size]
        neg_similarity = torch.sum(anchor * negative, dim=-1)  # [batch_size]

        # Triplet loss: we want pos_similarity > neg_similarity + margin
        losses = F.relu(self.margin - pos_similarity + neg_similarity)

        return losses.mean()


class TextMSELoss(nn.Module):
    """
    MSE loss for text embedding alignment.

    Extracted from: EfficientBind/src/model_utils/criterion.py
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, embedding_A: Tensor, embedding_B: Tensor) -> Tensor:
        """
        Compute MSE loss between normalized embeddings.

        Args:
            embedding_A: Source embeddings [batch_size, dim]
            embedding_B: Target embeddings [batch_size, dim]

        Returns:
            loss: Scalar MSE loss
        """
        embedding_A = F.normalize(embedding_A, dim=-1)
        embedding_B = F.normalize(embedding_B, dim=-1)
        return self.criterion(embedding_A, embedding_B)


def calculate_feature_similarity(features_a: Tensor, features_b: Tensor) -> Tensor:
    """
    Calculate cosine similarity between two sets of features.

    Extracted from: EfficientBind/src/model_utils/criterion.py

    Args:
        features_a: First set of features [batch_size, dim]
        features_b: Second set of features [batch_size, dim]

    Returns:
        similarity: Cosine similarity scores [batch_size]
    """
    features_a = F.normalize(features_a, dim=-1)
    features_b = F.normalize(features_b, dim=-1)

    # Calculate pairwise similarity
    similarity = torch.sum(features_a * features_b, dim=-1)
    return similarity


def build_loss(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to build loss by name.

    Args:
        loss_name: One of 'contrastive', 'hard_negative', 'triplet', 'mse'
        **kwargs: Additional arguments for specific losses

    Returns:
        Loss module instance
    """
    loss_name = loss_name.lower()

    if loss_name == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif loss_name in ['hard_negative', 'hnc']:
        return HardNegativeContrastiveLoss(**kwargs)
    elif loss_name == 'triplet':
        return TripletLoss(**kwargs)
    elif loss_name == 'mse':
        return TextMSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}. "
                        f"Available: contrastive, hard_negative, triplet, mse")
