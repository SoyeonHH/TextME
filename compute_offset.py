#!/usr/bin/env python3
"""
Stage 1: Offset Computation for TextME.

Computes modality-specific centroids (μ_text, μ_modal) from representative samples.
These offsets enable the interchangeable space where centered text and modal
embeddings become functionally equivalent.

Extracted from actual EfficientBind implementation (src/offset.py)

Usage:
    python compute_offset.py \
        --offset_model clip \
        --dataset_name coco \
        --data_root /path/to/captions \
        --raw_data_root /path/to/images \
        --saving_path ./offsets \
        --offset_num 5000 \
        --batch_size 64
"""

import os
import sys
import argparse
import logging
import pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def random_seed(seed=42, rank=0):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def parse_args(args=None):
    """Parse command line arguments.

    Extracted from: EfficientBind/src/offset.py
    """
    parser = argparse.ArgumentParser(description='Compute modality offsets for TextME')

    parser.add_argument("--offset_model", type=str, required=True,
                        choices=['clip', 'clap', 'languagebind', 'uni3d', 'cxr_clip', 'moleculestm', 'remoteclip'],
                        help="Model name to calculate offset.")
    parser.add_argument("--offset_num", type=int, default=5000,
                        help="Number of samples to embed and save.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for embedding.")
    parser.add_argument("--saving_path", type=str, default="./offsets",
                        help="Path to save the offset results.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding.")
    parser.add_argument("--dim", type=int, default=1024,
                        help="Dimension of the embeddings.")

    # Data
    parser.add_argument("--dataset_name", type=str, default="coco",
                        help="Name of the dataset.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to the preprocessed caption data.")
    parser.add_argument("--raw_data_root", type=str, default=None,
                        help="Path to the raw modality data (images, audio, etc.).")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples (defaults to offset_num).")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use.")

    return parser.parse_args(args)


def embed_and_save(args, dataset):
    """
    Embed text and modality data and collect embeddings.

    Extracted from: EfficientBind/src/offset.py
    """
    from textme.models import build_encoder

    all_modal = []
    all_captions = []

    # Sample subset if needed
    if args.offset_num is not None and args.offset_num < len(dataset):
        logging.info(f"Sampling {args.offset_num} samples from {len(dataset)}")
        indices = random.sample(range(len(dataset)), args.offset_num)
        offset_dataset = Subset(dataset, indices)
        logging.info(f"Sampled {len(offset_dataset)} samples")
    else:
        offset_dataset = dataset

    # Create DataLoader for batch processing
    dataloader = DataLoader(
        offset_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Build encoder
    logging.info(f"Loading {args.offset_model} encoder...")
    encoder = build_encoder(
        args.offset_model,
        device=args.device,
        use_projection=False,
        use_offset=False,
    )

    for batch in tqdm(dataloader, desc="Computing embeddings"):
        captions = batch['caption']

        with torch.no_grad():
            # Encode text
            caption_embeddings = encoder.encode_text(captions)

            # Encode modality based on model type
            if args.offset_model == "clip":
                if 'image' in batch:
                    modal_embeddings = encoder.encode_image(batch['image'])
                else:
                    modal_embeddings = caption_embeddings
            elif args.offset_model == "clap":
                if 'audio' in batch:
                    modal_embeddings = encoder.encode_audio(batch['audio'])
                else:
                    modal_embeddings = caption_embeddings
            elif args.offset_model == "languagebind":
                if 'image' in batch:
                    modal_embeddings = encoder.encode_image(batch['image'])
                else:
                    modal_embeddings = caption_embeddings
            elif args.offset_model == "uni3d":
                if 'glb_path' in batch:
                    modal_embeddings = encoder.encode_point(batch['glb_path'])
                else:
                    modal_embeddings = caption_embeddings
            elif args.offset_model == "cxr_clip":
                if 'image' in batch:
                    modal_embeddings = encoder.encode_image(batch['image'])
                else:
                    modal_embeddings = caption_embeddings
            elif args.offset_model == "moleculestm":
                if 'smiles' in batch:
                    modal_embeddings = encoder.encode_smile(batch['smiles'])
                else:
                    modal_embeddings = caption_embeddings
            elif args.offset_model == "remoteclip":
                if 'image' in batch:
                    modal_embeddings = encoder.encode_image(batch['image'])
                else:
                    modal_embeddings = caption_embeddings
            else:
                modal_embeddings = caption_embeddings

            # Collect embeddings
            for i in range(len(captions)):
                all_captions.append({
                    "caption": captions[i],
                    "embeddings": caption_embeddings[i]
                })
                all_modal.append({
                    "embeddings": modal_embeddings[i] if modal_embeddings.shape[0] > i else caption_embeddings[i]
                })

    return all_modal, all_captions


def compute_embed_means(args, all_modal, all_captions):
    """
    Compute the mean of the embeddings (centroids).

    Reference: https://github.com/yuhui-zh15/C3/blob/main/image_captioning/src/parse_data/compute_embed_means.py
    Extracted from: EfficientBind/src/offset.py
    """
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    text_mean = torch.zeros(1, args.dim, device=device)
    modal_mean = torch.zeros(1, args.dim, device=device)

    # Compute text mean
    if len(all_captions) > 0:
        for item in all_captions:
            cap_embed = item["embeddings"].to(device)
            text_mean += cap_embed / cap_embed.norm()

        text_mean = text_mean / len(all_captions)

        os.makedirs(args.saving_path, exist_ok=True)
        with open(os.path.join(args.saving_path, "text_embed_mean.pkl"), "wb") as f:
            pickle.dump(text_mean.cpu(), f)
        logging.info(f"Saved text_embed_mean to {os.path.join(args.saving_path, 'text_embed_mean.pkl')}")

    # Compute modal mean
    if len(all_modal) > 0:
        for item in all_modal:
            modal_embed = item["embeddings"].to(device)
            modal_mean += modal_embed / modal_embed.norm()

        modal_mean = modal_mean / len(all_modal)

        with open(os.path.join(args.saving_path, "img_embed_mean.pkl"), "wb") as f:
            pickle.dump(modal_mean.cpu(), f)
        logging.info(f"Saved img_embed_mean to {os.path.join(args.saving_path, 'img_embed_mean.pkl')}")

    # Compute and log modality gap
    modality_gap = torch.norm(modal_mean - text_mean).item()
    logging.info(f"Text centroid norm: {torch.norm(text_mean).item():.4f}")
    logging.info(f"Modal centroid norm: {torch.norm(modal_mean).item():.4f}")
    logging.info(f"Modality gap norm: {modality_gap:.4f}")

    # Save metadata
    import json
    metadata = {
        'model': args.offset_model,
        'dataset': args.dataset_name,
        'num_samples': len(all_captions),
        'embedding_dim': args.dim,
        'text_centroid_norm': torch.norm(text_mean).item(),
        'modal_centroid_norm': torch.norm(modal_mean).item(),
        'modality_gap_norm': modality_gap,
    }
    with open(os.path.join(args.saving_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return modal_mean, text_mean


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    random_seed(args.seed)

    # Set sample_size to offset_num if not specified
    if args.sample_size is None:
        args.sample_size = args.offset_num

    # Build dataset based on dataset_name
    from textme.data import build_offset_dataset

    logging.info(f"Loading {args.dataset_name} dataset...")
    dataset = build_offset_dataset(
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        raw_data_root=args.raw_data_root,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    logging.info("Starting Offset Calculation")
    all_modal, all_captions = embed_and_save(args, dataset)
    compute_embed_means(args, all_modal, all_captions)

    logging.info("Offset computation complete!")


if __name__ == "__main__":
    main()
