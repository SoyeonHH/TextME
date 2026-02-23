"""
Evaluation configuration for TextME.

Extracted from: EfficientBind/evaluation/config.py
"""

import argparse
import ast


class ParseKwargs(argparse.Action):
    """Parse key=value pairs from command line."""
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)
        setattr(namespace, self.dest, kw)


def parse_args(args=None):
    """
    Parse evaluation arguments.

    Extracted from: EfficientBind/evaluation/config.py
    """
    parser = argparse.ArgumentParser(description="TextME Evaluation")

    # Evaluation mode
    parser.add_argument("--task", type=str, default="retrieval",
                        choices=["retrieval", "classification"],
                        help="Evaluation task type.")
    parser.add_argument("--mode", type=str, default="test",
                        help="Mode to evaluate.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of samples to test.")
    parser.add_argument("--batch_size_val", type=int, default=128,
                        help="Batch size for validation.")
    parser.add_argument("--precision", type=str, default="amp",
                        choices=["amp", "fp16", "fp32"],
                        help="Precision mode.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # Paths
    parser.add_argument("--val_data", type=str, default="./data",
                        help="Path to the validation data.")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory containing cached embeddings.")
    parser.add_argument("--logs", type=str, default="./logs",
                        help="Path to the logs.")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory containing model checkpoints.")

    # Logging
    parser.add_argument("--save_logs", type=bool, default=True,
                        help="Whether to save the logs.")
    parser.add_argument("--name", type=str, default=None,
                        help="Name of the experiment.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of the run.")
    parser.add_argument("--save_embeddings", type=bool, default=False,
                        help="Whether to save the embeddings.")

    # Model configuration
    parser.add_argument("--source_model_name", type=str, default="languagebind",
                        help="Name of the source model (text encoder).")
    parser.add_argument("--target_model_name", type=str, default="clap",
                        help="Name of the target model (modality encoder).")
    parser.add_argument("--pivot_model_name", type=str, default="qwen3_embed_4b",
                        help="Name of the pivot LLM model.")
    parser.add_argument("--out_dim", type=int, default=2560,
                        help="Output dimension of projection head.")
    parser.add_argument("--init_mode", type=str, default="xav",
                        help="Initialization mode for projection.")
    parser.add_argument("--dim_act", type=str, default="gelu",
                        help="Activation function.")

    # Projection checkpoints (per model)
    parser.add_argument("--load_projection_checkpoint", action="store_true",
                        help="Load projection checkpoint.")
    parser.add_argument("--clip_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for CLIP.")
    parser.add_argument("--clap_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for CLAP.")
    parser.add_argument("--languagebind_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for LanguageBind.")
    parser.add_argument("--uni3d_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for Uni3D.")
    parser.add_argument("--cxr_clip_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for CXR-CLIP.")
    parser.add_argument("--moleculestm_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for MoleculeSTM.")
    parser.add_argument("--remoteclip_proj_checkpoint", type=str, default=None,
                        help="Projection checkpoint for RemoteCLIP.")

    # Offset configuration
    parser.add_argument("--use_offset", action="store_true",
                        help="Use precomputed offset.")
    parser.add_argument("--offset_path", type=str, default=None,
                        help="Path to precomputed offset file.")
    parser.add_argument("--offset_num", type=int, default=5000,
                        help="Number of samples used for offset computation.")
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Standard deviation of noise to add to embeddings.")
    parser.add_argument("--uniform_noise", action="store_true",
                        help="Use uniform ball noise instead of Gaussian.")

    # Projection configuration
    parser.add_argument("--use_projection", action="store_true",
                        help="Use projection head.")
    parser.add_argument("--train_projection", action="store_true",
                        help="Train projection head.")

    # Multi-sentence evaluation
    parser.add_argument("--multi_sentence", type=bool, default=False,
                        help="Whether to use multi-sentence per sample.")

    # Image-Language Retrieval
    parser.add_argument("--val_il_ret_data", nargs="+", default=None,
                        help="Image-language retrieval datasets (e.g., coco, flickr).")
    parser.add_argument("--val_i_cls_data", nargs="+", default=None,
                        help="Image classification datasets.")

    # Audio-Language Retrieval/Classification
    parser.add_argument("--val_al_ret_data", nargs="+", default=None,
                        help="Audio-language retrieval datasets (e.g., audiocaps, clotho).")
    parser.add_argument("--val_a_cls_data", nargs="+", default=None,
                        help="Audio classification datasets (e.g., esc50, AudioSet).")
    parser.add_argument("--val_ai_ret_data", nargs="+", default=None,
                        help="Audio-image retrieval datasets.")

    # 3D Point Cloud
    parser.add_argument("--val_pi_ret_data", nargs="+", default=None,
                        help="Point-image retrieval datasets.")
    parser.add_argument("--val_p_cls_data", nargs="+", default=None,
                        help="Point classification datasets (e.g., modelnet40, scanobjnn).")

    # X-ray Classification
    parser.add_argument("--val_x_cls_data", nargs="+", default=None,
                        help="X-ray classification datasets (e.g., rsna, siim).")

    # Molecule Retrieval
    parser.add_argument("--val_ml_ret_data", nargs="+", default=None,
                        help="Molecule-language retrieval datasets.")
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20],
                        help="T values for molecule retrieval evaluation.")
    parser.add_argument("--molecule_type", type=str, default="SMILES",
                        choices=["SMILES", "Graph"],
                        help="Molecule representation type.")
    parser.add_argument("--test_mode", type=str, default="given_molecule",
                        choices=["given_text", "given_molecule"],
                        help="Test mode for molecule retrieval.")

    # Remote Sensing
    parser.add_argument("--val_remote_ret_data", nargs="+", default=None,
                        help="Remote sensing retrieval datasets.")

    # Ablation
    parser.add_argument("--ablation", type=str, default=None,
                        help="Ablation type (e.g., cross_modal).")

    # Dataset info
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of the dataset.")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to the data root.")
    parser.add_argument("--raw_data_root", type=str, default=None,
                        help="Path to the raw data root.")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")

    args = parser.parse_args(args) if args else parser.parse_args()
    return args
