"""
Evaluation utilities for TextME.

Extracted from: EfficientBind/evaluation/eval_utils.py
"""

import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import average_precision_score
from scipy.special import softmax


def save_predictions(metrics_dict, filename="predictions.json"):
    """Save prediction metrics to JSON file."""
    with open(filename, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def _run_on_single_gpu(model, batch_sequence_output_list, batch_visual_output_list):
    """
    Compute similarity matrix on single GPU.

    Extracted from: EfficientBind/evaluation/eval_utils.py
    """
    sim_matrix = []
    for idx1 in tqdm(range(len(batch_sequence_output_list)), desc="Computing similarities"):
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2 in range(len(batch_visual_output_list)):
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits = sequence_output @ visual_output.T
            b1b2_logits = b1b2_logits.cpu().detach().numpy().item()
            each_row.append(b1b2_logits)
        each_row = np.array(each_row)
        sim_matrix.append(each_row)
    return sim_matrix


def calculate_similarity(model, batch_anchor_list, batch_target_list):
    """
    Calculate similarity matrix between anchor and target embeddings.

    Extracted from: EfficientBind/evaluation/eval_utils.py

    Args:
        model: Model (used for device placement, can be None)
        batch_anchor_list: List of anchor embeddings
        batch_target_list: List of target embeddings

    Returns:
        Similarity matrix as numpy array
    """
    print(f"Calculating similarity for {len(batch_anchor_list)} anchor and {len(batch_target_list)} target")

    sim_matrix = _run_on_single_gpu(model, batch_anchor_list, batch_target_list)
    sim_matrix = np.array(sim_matrix)

    return sim_matrix


def get_ranking_metrics(sim_matrix, top_k=[1, 5, 10], device=None):
    """
    Calculate ranking metrics from similarity matrix.

    Extracted from: EfficientBind/evaluation/eval_utils.py
    """
    if not torch.is_tensor(sim_matrix):
        sim_matrix = torch.tensor(sim_matrix)

    sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)

    ranks = torch.zeros(sim_matrix.shape[0], dtype=torch.long, device=device)
    for i in range(sim_matrix.shape[0]):
        matches = torch.where(sorted_indices[i] == i)[0]
        if len(matches) > 0:
            ranks[i] = matches[0].item()
        else:
            ranks[i] = sim_matrix.shape[0] - 1

    print(f"Rank statistics - Min: {ranks.min().item()}, Max: {ranks.max().item()}, Mean: {ranks.float().mean().item()}")

    results = {}
    for k in top_k:
        results[f"R@{k}"] = float((ranks < k).sum().item() * 100 / len(ranks))

    results["MedianR"] = float(torch.median(ranks.float() + 1).item())
    results["MeanR"] = float(torch.mean(ranks.float() + 1).item())
    results["Std_Rank"] = float(torch.std(ranks.float() + 1).item())
    results['MR'] = results["MedianR"]

    return results


def get_recall_metrics(sim_tensor, top_k=[1, 5, 10]):
    """
    Calculate recall metrics from similarity matrix.

    Extracted from: EfficientBind/evaluation/eval_utils.py

    Args:
        sim_tensor: Similarity matrix [N, N]
        top_k: List of k values for R@k computation

    Returns:
        Dictionary with R@1, R@5, R@10, MedianR, MeanR, MR
    """
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    sim_indices = torch.argsort(sim_tensor, dim=-1, descending=True)

    ranks = torch.zeros(sim_tensor.shape[0], dtype=torch.long, device=sim_tensor.device)
    for i in range(sim_tensor.shape[0]):
        matches = torch.where(sim_indices[i] == i)[0]
        if len(matches) > 0:
            ranks[i] = matches[0].item()
        else:
            ranks[i] = sim_tensor.shape[0] - 1

    results = {}
    for k in top_k:
        results[f"R@{k}"] = float((ranks < k).sum().item() * 100 / len(ranks))

    results["MedianR"] = float(torch.median(ranks.float() + 1).item())
    results["MeanR"] = float(torch.mean(ranks.float() + 1).item())
    results["Std_Rank"] = float(torch.std(ranks.float() + 1).item())
    results['MR'] = results["MedianR"]

    return results


def get_recall_metrics_multi_sentence(sim_tensor, top_k=[1, 5, 10]):
    """
    Calculate recall metrics for multi-sentence evaluation (e.g., Flickr, Clotho).

    Directly from: https://github.com/Deferf/Experiments
    Extracted from: EfficientBind/evaluation/eval_utils.py

    Args:
        sim_tensor: Similarity matrix [sentences_per_sample, N, N]
        top_k: List of k values for R@k computation

    Returns:
        Dictionary with R@1, R@5, R@10, MedianR, MeanR, MR
    """
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor for similarity ranking
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)

    # Extract ranks
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))

    # Validate ranks
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
    mask = ~torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]

    if valid_ranks.numel() == 0:
        raise ValueError("No valid ranks computed. Check similarity matrix.")

    if valid_ranks.max() >= sim_tensor.shape[0]:
        print(f"Warning: valid_ranks contains values >= {sim_tensor.shape[0]} (max: {valid_ranks.max()})")
        valid_ranks = valid_ranks.clamp(0, sim_tensor.shape[0] - 1)

    results = {f"R@{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]

    return results


def get_accuracy_metrics(args, sim_tensor, source_label_list, target_label_list, top_k=[1, 5, 10]):
    """
    Calculate Top-K accuracy for classification tasks.

    Extracted from: EfficientBind/evaluation/eval_utils.py

    Args:
        args: Arguments (requires dataset_name)
        sim_tensor: Similarity matrix [N_samples, N_classes]
        source_label_list: List of ground truth labels
        target_label_list: List of class labels/names
        top_k: List of k values for Top@k computation

    Returns:
        Dictionary with Top@1, Top@5, Top@10 accuracies
    """
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)

    results = {}
    dataset_name = getattr(args, 'dataset_name', '').lower()

    if dataset_name == 'imagenet':
        for k in top_k:
            if k == 1:
                correct_count = 0
                for i in range(sim_tensor.shape[0]):
                    max_index = torch.argmax(sim_tensor[i])
                    if int(source_label_list[i]) == int(max_index):
                        correct_count += 1
                results[f"Top@{k}"] = float(correct_count * 100 / len(source_label_list))
            else:
                correct_count = 0
                for i in range(sim_tensor.shape[0]):
                    top_k_indices = torch.topk(sim_tensor[i], k).indices
                    if int(source_label_list[i]) in top_k_indices.tolist():
                        correct_count += 1
                results[f"Top@{k}"] = float(correct_count * 100 / len(source_label_list))
    else:
        for k in top_k:
            if k == 1:
                correct_count = 0
                for i in range(sim_tensor.shape[0]):
                    max_index = torch.argmax(sim_tensor[i])
                    if source_label_list[i] == target_label_list[max_index]:
                        correct_count += 1
                results[f"Top@{k}"] = float(correct_count * 100 / len(source_label_list))
            else:
                correct_count = 0
                for i in range(sim_tensor.shape[0]):
                    top_k_indices = np.array(torch.topk(sim_tensor[i], k).indices)
                    target_label_max_idx_list = [target_label_list[idx] for idx in top_k_indices]
                    if source_label_list[i] in target_label_max_idx_list:
                        correct_count += 1
                results[f"Top@{k}"] = float(correct_count * 100 / len(source_label_list))

    return results


def zeroshot_binary(image_embeddings, text_model, labels, label_dict):
    """
    Zero-shot binary classification evaluation.

    Extracted from: EfficientBind/evaluation/eval_utils.py

    Args:
        image_embeddings: Image embeddings tensor
        text_model: Text encoder model with encode_text method
        labels: Ground truth binary labels
        label_dict: {0: "negative_class_name", 1: "positive_class_name"}

    Returns:
        Dictionary with AUROC, Accuracy, F1 scores
    """
    result = {}
    with torch.no_grad():
        class_names = list(label_dict.values())
        text_embeddings = text_model.encode_text(class_names)

    image_embeddings = image_embeddings.cpu().numpy()
    text_embeddings = text_embeddings.cpu().numpy()
    labels = np.array(labels)

    similarities = metrics.pairwise.cosine_similarity(image_embeddings, text_embeddings)
    similarities = softmax(similarities, axis=1)

    try:
        fpr, tpr, thresholds = metrics.roc_curve(labels, similarities[:, 1])
        result["AUROC"] = metrics.auc(fpr, tpr)
    except Exception as e:
        result["AUROC"] = 0.0

    result["Accuracy"] = metrics.accuracy_score(labels, np.argmax(similarities, axis=1))
    result["F1"] = metrics.f1_score(labels, np.argmax(similarities, axis=1))

    result["AUROC(Avg)"] = result["AUROC"]
    result["F1(Avg)"] = result["F1"]
    result["Accuracy(Avg)"] = result["Accuracy"]

    s = "\n".join(f"{k}: {v}" for k, v in result.items())
    print(s)

    return result


def get_mAP_from_similarity_matrix(sim_matrix, ground_truth_labels, class_to_idx_map):
    """
    Calculate mAP from similarity matrix and ground truth labels.

    Extracted from: EfficientBind/evaluation/eval_utils.py

    Args:
        sim_matrix: [num_samples, num_classes] similarity scores
        ground_truth_labels: List of lists, each sample's labels
        class_to_idx_map: class name -> index mapping

    Returns:
        Dictionary with mAP score
    """
    num_samples, num_classes = sim_matrix.shape

    # Create ground truth matrix
    gt_matrix = np.zeros((num_samples, num_classes))
    for i, labels in enumerate(ground_truth_labels):
        for label in labels:
            if label in class_to_idx_map:
                gt_matrix[i, class_to_idx_map[label]] = 1.0

    # Calculate mAP
    ap_scores = []
    for i in range(num_samples):
        y_true = gt_matrix[i]
        y_score = sim_matrix[i].cpu().numpy() if torch.is_tensor(sim_matrix) else sim_matrix[i]

        # Calculate only if there is at least one positive label
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_score)
            ap_scores.append(ap)

    mAP = np.mean(ap_scores) if ap_scores else 0.0
    return {'mAP': mAP * 100}


def do_CL_eval(X, Y, neg_Y):
    """
    Contrastive learning evaluation for molecule-text retrieval.

    Extracted from: EfficientBind/evaluation/retrieval.py

    Args:
        X: Query embeddings [B, d]
        Y: Positive target embeddings [B, d]
        neg_Y: Negative target embeddings [T, B, d]

    Returns:
        Accuracy score
    """
    import torch.nn.functional as F

    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1)  # B, 1, d

    Y = Y.unsqueeze(0)
    Y = torch.cat([Y, neg_Y], dim=0)  # T, B, d
    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]
    labels = torch.zeros(B).long().to(logits.device)  # B*1

    pred = logits.argmax(dim=1, keepdim=False)

    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    return CL_acc
