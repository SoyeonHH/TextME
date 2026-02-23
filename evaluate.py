#!/usr/bin/env python3
"""
Stage 3: Evaluation for TextME.

Evaluates trained TextME projections on cross-modal retrieval and zero-shot classification.

Extracted from: EfficientBind/evaluation/retrieval.py and classification.py

Usage:
    # Text-to-Audio Retrieval
    python evaluate.py \
        --val_al_ret_data AudioCaps \
        --source_model_name languagebind \
        --target_model_name clap \
        --pivot_model_name qwen3_embed_4b \
        --use_projection \
        --use_offset \
        --load_projection_checkpoint \
        --languagebind_proj_checkpoint /path/to/checkpoint.pt \
        --clap_proj_checkpoint /path/to/checkpoint.pt

    # Zero-shot 3D Classification
    python evaluate.py \
        --val_p_cls_data modelnet40 \
        --source_model_name languagebind \
        --target_model_name uni3d \
        --pivot_model_name qwen3_embed_4b \
        --use_projection \
        --use_offset \
        --load_projection_checkpoint \
        --uni3d_proj_checkpoint /path/to/checkpoint.pt
"""

import os
import warnings
warnings.filterwarnings('ignore')

import logging
import torch
import numpy as np
import random
import datetime
import json
import copy
from tqdm import tqdm
import torch.nn.functional as F

from textme.evaluation.config import parse_args
from textme.evaluation.eval_utils import (
    calculate_similarity,
    get_recall_metrics,
    get_recall_metrics_multi_sentence,
    get_accuracy_metrics,
    save_predictions,
)


# --- Logging Setup ---
def random_seed(seed=42, rank=0):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def setup_logging(log_file, level, include_host=False):
    """Setup logging configuration."""
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S'
        )

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def build_encoder_by_name(name, args, device):
    """
    Build encoder by name.

    Extracted from: EfficientBind/evaluation/retrieval.py

    Note: This is a placeholder. Users should implement their own encoder
    loading based on the actual EfficientBind model wrappers.
    """
    name = (name or "").lower()

    # Import actual model classes from textme
    from textme.models import build_encoder, ProjectionHead, ENCODER_DIM

    # Build base encoder
    encoder = build_encoder(name, device=device)

    # Add projection if needed
    if getattr(args, 'use_projection', False):
        encoder_dim = ENCODER_DIM.get(name, 768)
        out_dim = getattr(args, 'out_dim', 2560)

        encoder.projector = ProjectionHead(
            in_dim=encoder_dim,
            proj_dim=2 * encoder_dim,
            out_dim=out_dim,
            init_mode=getattr(args, 'init_mode', 'xav'),
            dim_act=getattr(args, 'dim_act', 'gelu'),
        ).to(device)

        # Load projection checkpoint if available
        if getattr(args, 'load_projection_checkpoint', False):
            checkpoint_path = getattr(args, f'{name}_proj_checkpoint', None)
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'projector_state_dict' in checkpoint:
                    encoder.projector.load_state_dict(checkpoint['projector_state_dict'])
                elif 'model_state_dict' in checkpoint:
                    encoder.projector.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"Loaded projection checkpoint from {checkpoint_path}")

    encoder.eval()
    return encoder


def get_modality_embeddings(input_batch, modality_encoder, is_multi_sentence,
                            cut_off_points_, total_samples_processed, modality="image"):
    """
    Get modality embeddings from input batch.

    Extracted from: EfficientBind/evaluation/retrieval.py
    """
    if modality == "image":
        modality_input = input_batch.get('image', input_batch.get('images', None))
        if modality_input is None:
            return None, 0

        batch_size = len(modality_input) if isinstance(modality_input, list) else modality_input.size(0)

        if is_multi_sentence:
            s_idx, e_idx = total_samples_processed, total_samples_processed + batch_size
            filter_indices = [idx - s_idx for idx in cut_off_points_ if s_idx <= idx < e_idx]
            if len(filter_indices) > 0:
                if isinstance(modality_input, torch.Tensor):
                    modality_input = modality_input[filter_indices, ...]
                elif isinstance(modality_input, list):
                    modality_input = [modality_input[i] for i in filter_indices]

        if not modality_input or (isinstance(modality_input, torch.Tensor) and modality_input.size(0) == 0):
            return None, batch_size

        modality_embeddings = modality_encoder.encode_image(modality_input)

    elif modality == "audio":
        modality_input = input_batch.get('audio', input_batch.get('audios', None))
        if modality_input is None:
            return None, 0

        batch_size = len(modality_input)

        if is_multi_sentence:
            s_idx, e_idx = total_samples_processed, total_samples_processed + batch_size
            filter_indices = [idx - s_idx for idx in cut_off_points_ if s_idx <= idx < e_idx]
            if len(filter_indices) > 0:
                modality_input = [modality_input[i] for i in filter_indices]

        if not modality_input:
            return None, batch_size

        modality_embeddings = modality_encoder.encode_audio(modality_input)

    elif modality == "point":
        modality_input = input_batch.get('pc', input_batch.get('point_cloud', None))
        colors = input_batch.get('rgb', None)

        if modality_input is None:
            return None, 0

        batch_size = len(modality_input) if isinstance(modality_input, list) else modality_input.size(0)

        modality_embeddings = modality_encoder.encode_point(modality_input, colors)

    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return modality_embeddings, batch_size


@torch.no_grad()
def evaluate_text2modality_retrieval(args, dataset_name, modality_encoder, dataloader,
                                      device, text_encoder=None, modality="image"):
    """
    Evaluate text-to-modality retrieval performance.

    Extracted from: EfficientBind/evaluation/retrieval.py

    Args:
        args: Evaluation arguments
        dataset_name: Name of the dataset
        modality_encoder: Encoder for the target modality
        dataloader: DataLoader for evaluation data
        device: Device to use
        text_encoder: Text encoder (if different from modality encoder)
        modality: Target modality type ("image", "audio", "point")

    Returns:
        Dictionary with retrieval metrics (R@1, R@5, R@10, MR, MeanR)
    """
    modality_encoder.eval()
    if text_encoder is not None:
        text_encoder.eval()

    # Multi-sentence setup
    is_multi_sentence = getattr(args, 'multi_sentence', False) and hasattr(dataloader.dataset, 'cut_off_points')
    cut_off_points_ = []
    sentence_num_ = -1
    modality_num_ = -1

    if is_multi_sentence:
        cut_off_points_ = dataloader.dataset.cut_off_points
        sentence_num_ = dataloader.dataset.sample_len
        modality_num_ = getattr(dataloader.dataset, f'{modality}_num', len(dataloader.dataset))
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]
        logging.info(f"Eval under multi-sentence setting for {dataset_name.upper()}.")
        logging.info(f"Sentence num: {sentence_num_}, Modality num: {modality_num_}")

    transformed_modality_embeddings_list = []
    text_embeddings_list = []
    total_samples_processed = 0

    logging.info(f"Starting text-to-modality retrieval evaluation for {dataset_name.upper()}...")

    for batch in tqdm(dataloader, desc=f"Retrieval Eval ({dataset_name.upper()})"):
        # Get text embeddings
        text_input = batch.get('raw_text', batch.get('text', batch.get('caption', [])))

        if text_encoder is not None:
            text_embeddings = text_encoder.encode_text(text_input)
        else:
            text_embeddings = modality_encoder.encode_text(text_input)

        text_embeddings_list.extend(text_embeddings.cpu())

        # Get modality embeddings
        modality_embeddings, batch_size = get_modality_embeddings(
            batch, modality_encoder, is_multi_sentence,
            cut_off_points_, total_samples_processed, modality
        )

        if modality_embeddings is None:
            total_samples_processed += batch_size
            continue

        if not isinstance(modality_embeddings, torch.Tensor):
            modality_embeddings = torch.tensor(modality_embeddings).to(device)
        elif modality_embeddings.device != device:
            modality_embeddings = modality_embeddings.to(device)

        transformed_modality_embeddings_list.extend(modality_embeddings.cpu())
        total_samples_processed += batch_size

    # Calculate similarity
    logging.info("Calculating similarity matrix...")
    sim_matrix = calculate_similarity(modality_encoder, text_embeddings_list, transformed_modality_embeddings_list)

    # Calculate metrics
    if is_multi_sentence:
        print(f"{dataset_name.upper()} before reshape, sim matrix shape: {sim_matrix.shape}")
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate(
                (sim_matrix[s_:e_], np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                axis=0
            ))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        metrics = get_recall_metrics_multi_sentence(sim_matrix)
    else:
        metrics = get_recall_metrics(sim_matrix)

    logging.info(f"{dataset_name.upper()} Text-to-Modality Retrieval Results:")
    logging.info('\t>>> R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.format(
        metrics['R@1'], metrics['R@5'], metrics['R@10'], metrics['MR'], metrics['MeanR']
    ))

    # Save results
    if getattr(args, 'save_logs', True):
        output_dir = os.path.join(args.log_base_path, f'{args.mode}_text2modality/{dataset_name}')
        os.makedirs(output_dir, exist_ok=True)

        pred_file = os.path.join(output_dir, "predictions.json")
        save_predictions(metrics, pred_file)
        logging.info(f"Saved prediction metrics to {pred_file}")

        results_file = os.path.join(output_dir, "results.jsonl")
        with open(results_file, "a+") as f:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "dataset": dataset_name,
                "mode": "text2modality",
                "modality": modality,
                "metrics": metrics,
            }
            f.write(json.dumps(log_entry, indent=4))
            f.write("\n")
        logging.info(f"Saved results summary to {results_file}")

    return metrics


@torch.no_grad()
def evaluate_classification(args, dataset_name, text_encoder, modality_encoder,
                            dataloader, class_templates, class_names):
    """
    Evaluate zero-shot classification performance.

    Extracted from: EfficientBind/evaluation/classification.py

    Args:
        args: Evaluation arguments
        dataset_name: Name of the dataset
        text_encoder: Text encoder
        modality_encoder: Modality encoder
        dataloader: DataLoader for evaluation data
        class_templates: List of prompt templates for class names
        class_names: List of class names

    Returns:
        Dictionary with classification metrics (Top@1, Top@5, etc.)
    """
    if text_encoder is not None:
        text_encoder.eval()
    modality_encoder.eval()

    batch_source_embeddings_list = []
    batch_target_embeddings_list = []
    batch_source_label_list = []

    # Create class embeddings with templates
    batch_target_set = []
    for class_name in class_names:
        batch_target_set.append([template(class_name) for template in class_templates])

    logging.info(f"Encoding {len(class_names)} classes with {len(class_templates)} templates each...")

    with torch.no_grad():
        # Encode classes
        for text in tqdm(batch_target_set, desc="Class encoding"):
            text_embeddings = []
            for text_with_prompt in text:
                if text_encoder is not None:
                    text_embeddings.append(text_encoder.encode_text([text_with_prompt]))
                else:
                    text_embeddings.append(modality_encoder.encode_text([text_with_prompt]))

            text_embeddings = torch.stack(text_embeddings)
            text_embeddings = text_embeddings.reshape(1, len(class_templates), -1).mean(dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

            batch_target_embeddings_list.append(text_embeddings)

        # Encode samples
        for batch in tqdm(dataloader, desc="Sample encoding"):
            labels = batch.get('label', batch.get('labels', []))
            inputs = batch.get('pc', batch.get('image', batch.get('audio', None)))
            colors = batch.get('rgb', None)

            # Get modality embeddings
            if 'pc' in batch:
                embeddings = modality_encoder.encode_point(inputs, colors)
            elif 'image' in batch:
                embeddings = modality_encoder.encode_image(inputs)
            elif 'audio' in batch:
                embeddings = modality_encoder.encode_audio(inputs)
            else:
                raise ValueError("No valid input found in batch")

            batch_source_embeddings_list.append(embeddings)

            if len(batch_source_label_list) == 0:
                batch_source_label_list.append(list(labels) if not isinstance(labels, list) else labels)
            else:
                batch_source_label_list[0].extend(labels if isinstance(labels, list) else list(labels))

    source_tensor = torch.cat(batch_source_embeddings_list, dim=0)
    target_tensor = torch.stack(batch_target_embeddings_list, dim=0).squeeze(1)

    # Calculate similarity
    logging.info("Calculating similarity matrix...")
    if text_encoder is not None:
        sim_matrix = calculate_similarity(text_encoder, source_tensor, target_tensor)
    else:
        sim_matrix = calculate_similarity(modality_encoder, source_tensor, target_tensor)

    # Calculate accuracy metrics
    logging.info("Calculating accuracy metrics...")
    metrics = get_accuracy_metrics(args, sim_matrix, batch_source_label_list[0], list(class_names))

    if 'Top@5' in metrics:
        logging.info(f"{dataset_name} Classification Results:")
        logging.info('\t>>> Top-1 Accuracy: {:.1f}'.format(metrics['Top@1']))
        logging.info('\t>>> Top-5 Accuracy: {:.1f}'.format(metrics['Top@5']))
    else:
        logging.info(f"{dataset_name} Classification Results:")
        logging.info('\t>>> Top-1 Accuracy: {:.1f}'.format(metrics['Top@1']))

    # Save results
    if getattr(args, 'save_logs', True):
        output_dir = os.path.join(args.log_base_path, f'{dataset_name}_classification')
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, "results.jsonl")
        with open(results_file, "a+") as f:
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "dataset": dataset_name,
                "mode": "classification",
                "metrics": metrics,
            }
            f.write(json.dumps(log_entry, indent=4))
            f.write("\n")
        logging.info(f"Saved results summary to {results_file}")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.distributed = False

    # Setup dataset name
    if not hasattr(args, 'dataset_name') or args.dataset_name is None:
        task_to_dataset = {
            'coco': 'coco',
            'flickr': 'coco',
            'audiocaps': 'audiocaps',
            'clotho': 'audiocaps',
            'modelnet40': 'objaverse',
            'scanobjnn': 'objaverse',
        }

        if hasattr(args, 'val_il_ret_data') and args.val_il_ret_data:
            args.dataset_name = task_to_dataset.get(args.val_il_ret_data[0], 'coco')
        elif hasattr(args, 'val_al_ret_data') and args.val_al_ret_data:
            args.dataset_name = task_to_dataset.get(args.val_al_ret_data[0], 'audiocaps')
        elif hasattr(args, 'val_p_cls_data') and args.val_p_cls_data:
            args.dataset_name = task_to_dataset.get(args.val_p_cls_data[0], 'objaverse')
        elif hasattr(args, 'val_x_cls_data') and args.val_x_cls_data:
            args.dataset_name = 'chestxray'
        else:
            args.dataset_name = 'default'

    # Setup logging
    args.name = f'{args.run_name}' if args.name is None else args.name
    log_base_path = os.path.join(args.logs, args.name or 'textme_eval')
    args.log_base_path = log_base_path
    os.makedirs(log_base_path, exist_ok=True)

    log_filename = f'out-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
    args.log_path = os.path.join(log_base_path, log_filename)
    args.log_level = logging.INFO

    setup_logging(args.log_path, args.log_level)
    random_seed(args.seed, args.rank)

    logging.info("=" * 60)
    logging.info("TextME Evaluation")
    logging.info("=" * 60)
    logging.info(f"Source model: {args.source_model_name}")
    logging.info(f"Target model: {args.target_model_name}")
    logging.info(f"Pivot model: {args.pivot_model_name}")
    logging.info(f"Use offset: {args.use_offset}")
    logging.info(f"Use projection: {args.use_projection}")

    # Note: Users need to implement their own data loading
    # based on the eval_datasets.py in EfficientBind
    logging.info("")
    logging.info("Note: This evaluation script requires implementing data loading.")
    logging.info("Please refer to EfficientBind/data/eval_datasets.py for dataset implementations.")
    logging.info("")
    logging.info("Example usage for retrieval:")
    logging.info("  python evaluate.py --val_al_ret_data AudioCaps --source_model_name languagebind \\")
    logging.info("    --target_model_name clap --use_projection --use_offset")
    logging.info("")
    logging.info("Example usage for classification:")
    logging.info("  python evaluate.py --val_p_cls_data modelnet40 --source_model_name languagebind \\")
    logging.info("    --target_model_name uni3d --use_projection --use_offset")

    logging.info("")
    logging.info("Evaluation script ready. Implement data loading to run evaluation.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
