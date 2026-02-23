#!/usr/bin/env python3
"""
Stage 2: Projection Training for TextME.

Trains lightweight projection networks to align centered text embeddings
with LLM anchor space using only text descriptions.

Extracted from actual EfficientBind implementation (src/train.py)

Usage:
    python train.py \
        --model_name clip \
        --pivot_model_name qwen3_embed_4b \
        --dataset_name coco \
        --data_root /path/to/captions \
        --embed_dir /path/to/precomputed_embeds \
        --use_offset \
        --use_projection \
        --batch_size 256 \
        --epochs 50 \
        --lr 5e-4
"""

import os
import sys
import argparse
import logging
import time
import copy
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_trainable_parameters(model, model_name="Model"):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"{model_name}: Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}%"
    )


def get_autocast(precision):
    """Get autocast context manager based on precision."""
    if precision == "amp":
        return lambda: autocast(dtype=torch.float16)
    elif precision == "fp16":
        return lambda: autocast(dtype=torch.float16)
    elif precision == "bf16":
        return lambda: autocast(dtype=torch.bfloat16)
    else:
        return lambda: torch.autocast(device_type='cuda', enabled=False)


def backward(loss, scaler):
    """Backward pass with optional gradient scaling."""
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def main(args):
    """Main training function.

    Extracted from: EfficientBind/src/train.py
    """
    # Import here to avoid circular imports during development
    from textme.models import build_encoder, ProjectionHead, ENCODER_DIM
    from textme.data import CaptionDataset
    from textme.losses import HardNegativeContrastiveLoss, TextMSELoss

    # --- Setup ---
    if args.name is None:
        args.name = f"textme_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.checkpoint_dir:
        args.checkpoint_path = os.path.join(args.checkpoint_dir, args.name)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        logging.info(f"Checkpoints will be saved to: {args.checkpoint_path}")
    else:
        args.checkpoint_path = None
        logging.info("Checkpoint directory not specified, checkpoints will not be saved.")

    if args.logs:
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(log_base_path, log_filename)
        handlers = [logging.FileHandler(log_path, mode='w'), logging.StreamHandler()]
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        logging.info(f"Log file: {log_path}")

    logging.info(f"Starting run: {args.name}")
    logging.info(f"Args: {args}")

    # Device and Seed
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {args.device}")
    random_seed(args.seed)

    # --- Load Model ---
    logging.info(f"Loading {args.model_name} model...")
    model_device = args.device

    # Build encoder with projection head
    encoder = build_encoder(
        args.model_name,
        device=str(model_device),
        use_projection=args.use_projection,
        use_offset=args.use_offset,
        offset_dir=args.offset_dir,
        offset_num=args.offset_num,
        out_dim=args.out_dim,
        init_mode=args.init_mode,
        dim_act=args.dim_act,
    )

    # Freeze encoder, only train projector
    for param in encoder.model.parameters():
        param.requires_grad = False

    if hasattr(encoder, 'projector') and encoder.projector is not None:
        for param in encoder.projector.parameters():
            param.requires_grad = True
        projector = encoder.projector
    else:
        raise ValueError("Projection head is required for training. Use --use_projection flag.")

    logging.info(f"Model loaded on device: {model_device}")
    logging.info("Trainable parameters:")
    print_trainable_parameters(projector, "Projector")

    # --- Load Pivot Model (if LanguageBind or CLIP as pivot) ---
    pivot_model = None
    use_pivot_model = False

    if args.pivot_model_name in ["languagebind", "clip"]:
        logging.info(f"Loading {args.pivot_model_name} as pivot model...")
        pivot_model = build_encoder(
            args.pivot_model_name,
            device=str(model_device),
            use_projection=False,
            use_offset=False,
        )
        # Set output dimension to match pivot model
        if args.pivot_model_name == "languagebind":
            args.out_dim = 768
        elif args.pivot_model_name == "clip":
            args.out_dim = 1024
        use_pivot_model = True
        logging.info(f"Set output dimension to {args.pivot_model_name} dimension: {args.out_dim}")

    # --- Dataset and DataLoader ---
    logging.info(f"Loading text data and precomputed embeddings...")
    dataset = CaptionDataset(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        embed_dir=args.embed_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    args.total_steps = len(train_dataloader) * args.epochs
    logging.info(f"Training data: {len(dataset)} samples, {len(train_dataloader)} batches/epoch.")
    logging.info(f"Total training steps: {args.total_steps}")

    # --- Loss Function and Optimizer ---
    if args.loss_type == "contrastive":
        criterion = HardNegativeContrastiveLoss(
            temperature=args.temperature,
            top_perc_margin=args.top_perc_margin,
            bottom_perc_margin=args.bottom_perc_margin,
        ).to(model_device)
    elif args.loss_type == "mse":
        criterion = TextMSELoss().to(model_device)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    optimizer = optim.AdamW(
        projector.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps
    )

    # Scheduler
    scaler = GradScaler() if args.precision == "amp" else None
    scheduler = None
    if not args.skip_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, args.total_steps // 10),
            T_mult=1,
            eta_min=1e-6
        )

    logging.info(f"Optimizer: AdamW, LR: {args.lr}, WD: {args.weight_decay}")
    logging.info(f"Loss Function: {type(criterion).__name__}")
    logging.info(f"Precision: {args.precision}")

    # --- Initialize WandB ---
    if args.save_logs and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            config=vars(args),
            resume='allow',
        )
        wandb.watch(projector, log='all', log_freq=args.wandb_log_freq)
        logging.info("WandB initialized.")

    # --- Training Loop ---
    logging.info("Starting training...")
    autocast_fn = get_autocast(args.precision)

    best_train_loss = float('inf')
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        projector.train()
        num_batches_per_epoch = len(train_dataloader)

        loss_m = AverageMeter()
        batch_time_m = AverageMeter()
        end = time.time()

        logging.info(f"Epoch {epoch+1}/{args.epochs} Training started.")

        pbar = tqdm(
            enumerate(train_dataloader),
            total=num_batches_per_epoch,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=True
        )

        for batch_idx, batch in pbar:
            step = epoch * num_batches_per_epoch + batch_idx

            optimizer.zero_grad()

            texts = batch['caption']
            if not isinstance(texts, list):
                texts = list(texts)
            current_mini_batch_size = len(texts)

            with autocast_fn():
                # Encode text with source encoder (includes offset + projection)
                model_embeddings = encoder.encode_text(texts).to(model_device)

                if use_pivot_model and pivot_model is not None:
                    # Use pivot model for target embeddings
                    pivot_embeddings = pivot_model.encode_text(texts).to(model_device)
                    target_embeddings = F.normalize(pivot_embeddings, p=2, dim=1)
                else:
                    # Use precomputed LLM embeddings
                    precomputed_embeds = batch['llm_embedding'].to(model_device)
                    target_embeddings = F.normalize(precomputed_embeds, p=2, dim=1)

                loss = criterion(model_embeddings, target_embeddings)

            backward(loss, scaler)

            if scaler is not None:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(projector.parameters(), args.grad_clip_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            loss_m.update(loss.item(), current_mini_batch_size)
            batch_time_m.update(time.time() - end)
            end = time.time()

            # Logging
            if args.save_logs and WANDB_AVAILABLE and (step % args.wandb_log_freq == 0):
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/avg_loss": loss_m.avg,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch + (batch_idx / num_batches_per_epoch),
                }, step=step)

            pbar.set_postfix(
                Loss=f"{loss_m.avg:.4f}",
                LR=f"{optimizer.param_groups[0]['lr']:.6f}",
            )

        # --- End of Training Epoch ---
        avg_train_loss = loss_m.avg
        logging.info(f"Epoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_train_loss:.4f}")

        # Check if this is the best epoch
        is_best = avg_train_loss < best_train_loss
        if is_best:
            best_train_loss = avg_train_loss
            logging.info(f"New best training loss: {best_train_loss:.4f}")

        # --- Save Checkpoint ---
        if args.checkpoint_path:
            checkpoint_dict = {
                "epoch": epoch + 1,
                "name": args.name,
                "state_dict": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "best_train_loss": best_train_loss,
                "out_dim": args.out_dim,
                "avg_train_loss": avg_train_loss,
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            # Save latest checkpoint
            latest_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
            torch.save(checkpoint_dict, latest_path)
            logging.info(f"Saved latest checkpoint to {latest_path}")

            # Save best checkpoint
            if is_best:
                best_path = os.path.join(args.checkpoint_path, "best.pt")
                torch.save(checkpoint_dict, best_path)
                logging.info(f"Saved best checkpoint to {best_path}")

            # Save periodic epoch checkpoints
            if args.save_frequency > 0 and (epoch + 1) % args.save_frequency == 0:
                epoch_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt")
                torch.save(checkpoint_dict, epoch_path)
                logging.info(f"Saved epoch {epoch + 1} checkpoint to {epoch_path}")

    logging.info("Training finished.")
    if args.save_logs and WANDB_AVAILABLE and wandb.run:
        wandb.finish()


def parse_args():
    """Parse command line arguments.

    Extracted from: EfficientBind/src/train.py args_parser()
    """
    parser = argparse.ArgumentParser(description="Train TextME projection using precomputed LLM embeddings.")

    parser.add_argument("--name", type=str, default=None, help="Name for the training run.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--dataset_name", type=str, default="coco", help="Name of the dataset to use.")
    parser.add_argument("--embed_dir", type=str, default=None, help="Directory containing precomputed LLM embeddings.")

    # Model
    parser.add_argument("--model_name", type=str, default="clip",
                        choices=["clip", "clap", "languagebind", "uni3d", "cxr_clip", "moleculestm", "remoteclip", "viclip"],
                        help="Name of the source encoder model.")
    parser.add_argument("--pivot_model_name", type=str, default="qwen3_embed_4b",
                        choices=["qwen3_embed_4b", "qwen3_embed_0.6b", "nv_embed_v2", "languagebind", "clip"],
                        help="Name of the pivot/anchor model.")
    parser.add_argument("--out_dim", type=int, default=2560, help="Output dimension (LLM embedding dim).")
    parser.add_argument("--init_mode", type=str, default="xav", choices=["xav", "he", "eye"],
                        help="Initialization mode for projection head weights.")
    parser.add_argument("--dim_act", type=str, default="gelu", help="Activation function in projection head.")

    # Training
    parser.add_argument("--use_projection", action="store_true", help="Use projection head.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--precision", type=str, default="amp", choices=["fp32", "fp16", "amp"],
                        help="Training precision.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--sample_size", type=int, default=100000, help="Number of training samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Loss
    parser.add_argument("--loss_type", type=str, default="contrastive", choices=["contrastive", "mse"],
                        help="Loss type.")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature parameter for contrastive loss.")
    parser.add_argument("--top_perc_margin", type=float, default=0.9,
                        help="Top percentage margin for hard negative loss.")
    parser.add_argument("--bottom_perc_margin", type=float, default=0.1,
                        help="Bottom percentage margin for hard negative loss.")

    # Scheduler
    parser.add_argument("--skip_scheduler", action="store_true", help="Skip using a learning rate scheduler.")
    parser.add_argument("--warmup", type=int, default=500, help="Number of warmup steps.")

    # AdamW
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter.")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon parameter.")

    # Offset
    parser.add_argument("--use_offset", action="store_true", help="Use precomputed offset.")
    parser.add_argument("--offset_dir", type=str, default="./offsets", help="Directory containing offset results.")
    parser.add_argument("--offset_num", type=int, default=5000, help="Number of samples used to compute offset.")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="Directory to save checkpoints.")
    parser.add_argument("--save_frequency", type=int, default=10, help="Frequency (in epochs) to save checkpoints.")

    # Logging
    parser.add_argument("--logs", type=str, default="./logs/", help="Directory to save log files.")
    parser.add_argument("--save_logs", action="store_true", help="Enable saving logs to WandB.")
    parser.add_argument("--wandb_project_name", type=str, default="textme", help="WandB project name.")
    parser.add_argument("--wandb_log_freq", type=int, default=100, help="Frequency (in steps) to log to WandB.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
