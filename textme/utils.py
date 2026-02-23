"""
Utilities for TextME.

Training utilities, metrics computation, and helper functions.
"""

import os
import random
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import autocast


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking training metrics like loss.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@contextmanager
def get_autocast(precision: str = 'fp32'):
    """
    Context manager for automatic mixed precision.

    Args:
        precision: One of 'fp32', 'fp16', 'bf16', 'amp'
    """
    if precision in ['fp16', 'amp']:
        with autocast(dtype=torch.float16):
            yield
    elif precision == 'bf16':
        with autocast(dtype=torch.bfloat16):
            yield
    else:
        yield


def get_input_dtype(precision: str = 'fp32') -> torch.dtype:
    """Get input dtype based on precision setting."""
    if precision == 'fp16':
        return torch.float16
    elif precision == 'bf16':
        return torch.bfloat16
    else:
        return torch.float32


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count trainable and total parameters in a model.

    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_trainable_parameters(model: nn.Module, model_name: str = "Model"):
    """Log trainable vs total parameters."""
    trainable, total = count_parameters(model)
    logging.info(
        f"{model_name}: {trainable:,} trainable params, "
        f"{total:,} total params ({100 * trainable / total:.2f}% trainable)"
    )


# =============================================================================
# Offset I/O
# =============================================================================

def save_offset(
    offset: Tensor,
    save_path: str,
    offset_type: str = 'text'
):
    """
    Save computed offset vector.

    Args:
        offset: Offset tensor [dim]
        save_path: Directory to save offset
        offset_type: 'text' or 'modal'
    """
    os.makedirs(save_path, exist_ok=True)
    filename = f"{offset_type}_offset.pkl"
    filepath = os.path.join(save_path, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(offset.cpu().numpy(), f)

    logging.info(f"Saved {offset_type} offset to {filepath}")


def load_offset(
    load_path: str,
    offset_type: str = 'text',
    device: str = 'cuda'
) -> Tensor:
    """
    Load precomputed offset vector.

    Args:
        load_path: Directory containing offset files
        offset_type: 'text' or 'modal'
        device: Device to load tensor on

    Returns:
        Offset tensor [dim]
    """
    filename = f"{offset_type}_offset.pkl"
    filepath = os.path.join(load_path, filename)

    # Also try legacy naming convention
    if not os.path.exists(filepath):
        legacy_names = {
            'text': 'text_embed_mean.pkl',
            'modal': 'img_embed_mean.pkl'
        }
        filepath = os.path.join(load_path, legacy_names.get(offset_type, filename))

    with open(filepath, 'rb') as f:
        offset = pickle.load(f)

    return torch.tensor(offset, device=device, dtype=torch.float32)


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    save_path: str,
    best_metric: float = None,
    config: Dict = None,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state (optional)
        epoch: Current epoch
        save_path: Path to save checkpoint
        best_metric: Best validation metric (optional)
        config: Training configuration (optional)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logging.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    strict: bool = True,
) -> Dict:
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer to restore state (optional)
        scheduler: Scheduler to restore state (optional)
        strict: Whether to strictly match state dict keys

    Returns:
        Checkpoint dict with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")

    return checkpoint


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_recall_at_k(
    similarity_matrix: Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute Recall@K metrics.

    Args:
        similarity_matrix: Similarity scores [num_queries, num_candidates]
        k_values: List of K values to compute

    Returns:
        Dict with R@K metrics
    """
    num_queries = similarity_matrix.size(0)

    # Get rankings
    _, rankings = similarity_matrix.sort(dim=1, descending=True)

    # Find where ground truth (diagonal) ranks
    gt_indices = torch.arange(num_queries, device=similarity_matrix.device)

    metrics = {}
    for k in k_values:
        # Check if ground truth is in top-k
        top_k = rankings[:, :k]
        correct = (top_k == gt_indices.unsqueeze(1)).any(dim=1)
        recall = correct.float().mean().item() * 100
        metrics[f'R@{k}'] = recall

    return metrics


def compute_mrr(similarity_matrix: Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        similarity_matrix: Similarity scores [num_queries, num_candidates]

    Returns:
        MRR value
    """
    num_queries = similarity_matrix.size(0)

    # Get rankings
    _, rankings = similarity_matrix.sort(dim=1, descending=True)

    # Find where ground truth (diagonal) ranks
    gt_indices = torch.arange(num_queries, device=similarity_matrix.device)

    # Find rank of ground truth for each query
    ranks = (rankings == gt_indices.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1

    mrr = (1.0 / ranks.float()).mean().item() * 100
    return mrr


def compute_retrieval_metrics(
    query_embeddings: Tensor,
    candidate_embeddings: Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute full retrieval metrics.

    Args:
        query_embeddings: Query embeddings [N, dim]
        candidate_embeddings: Candidate embeddings [N, dim]
        k_values: K values for Recall@K

    Returns:
        Dict with all retrieval metrics
    """
    # Compute similarity matrix
    similarity = torch.matmul(query_embeddings, candidate_embeddings.t())

    metrics = compute_recall_at_k(similarity, k_values)
    metrics['MRR'] = compute_mrr(similarity)
    metrics['MedianR'] = compute_median_rank(similarity)
    metrics['MeanR'] = compute_mean_rank(similarity)

    return metrics


def compute_median_rank(similarity_matrix: Tensor) -> float:
    """Compute median rank of ground truth items."""
    num_queries = similarity_matrix.size(0)
    _, rankings = similarity_matrix.sort(dim=1, descending=True)
    gt_indices = torch.arange(num_queries, device=similarity_matrix.device)
    ranks = (rankings == gt_indices.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    return ranks.float().median().item()


def compute_mean_rank(similarity_matrix: Tensor) -> float:
    """Compute mean rank of ground truth items."""
    num_queries = similarity_matrix.size(0)
    _, rankings = similarity_matrix.sort(dim=1, descending=True)
    gt_indices = torch.arange(num_queries, device=similarity_matrix.device)
    ranks = (rankings == gt_indices.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    return ranks.float().mean().item()


def compute_classification_accuracy(
    embeddings: Tensor,
    class_embeddings: Tensor,
    labels: Tensor,
    top_k: List[int] = [1, 5]
) -> Dict[str, float]:
    """
    Compute zero-shot classification accuracy.

    Args:
        embeddings: Sample embeddings [N, dim]
        class_embeddings: Class prototype embeddings [C, dim]
        labels: Ground truth labels [N]
        top_k: K values for Top-K accuracy

    Returns:
        Dict with accuracy metrics
    """
    # Compute similarity to all classes
    similarity = torch.matmul(embeddings, class_embeddings.t())

    metrics = {}
    for k in top_k:
        _, top_k_preds = similarity.topk(k, dim=1)
        correct = (top_k_preds == labels.unsqueeze(1)).any(dim=1)
        accuracy = correct.float().mean().item() * 100
        metrics[f'Top-{k}'] = accuracy

    return metrics
