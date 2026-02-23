#!/usr/bin/env python3
"""
Prepare TextME datasets for HuggingFace upload.

This script packages:
1. Caption datasets (training data for each modality)
2. Offset vectors (precomputed centroids for interchangeable space)

Usage:
    python prepare_hf_datasets.py --output_dir ./hf_datasets
    python prepare_hf_datasets.py --upload --repo_id your-username/textme-data
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm


# Dataset configurations
CAPTION_DATASETS = {
    'coco': {
        'file': 'coco_captions_preprocessed.parquet',
        'id_col': 'caption_id',
        'modality': 'image',
        'encoder': 'clip',
        'description': 'COCO 2014 image captions (~590K samples)'
    },
    'audiocaps': {
        'file': 'audiocaps_captions_preprocessed.parquet',
        'id_col': 'id',
        'modality': 'audio',
        'encoder': 'clap',
        'description': 'AudioCaps audio descriptions (~49K samples)'
    },
    'objaverse': {
        'file': 'objaverse_captions_preprocessed.parquet',
        'id_col': 'objaverse_id',
        'modality': '3d',
        'encoder': 'uni3d',
        'description': 'Objaverse 3D object descriptions (~1.5M samples)'
    },
    'chestxray': {
        'file': 'chestxray_captions_preprocessed.parquet',
        'id_col': 'image_id',
        'modality': 'xray',
        'encoder': 'cxr_clip',
        'description': 'ChestX-ray14 radiology reports (~112K samples)'
    },
    'pubchem': {
        'file': 'pubchem_captions_preprocessed.parquet',
        'id_col': 'id',
        'modality': 'molecule',
        'encoder': 'moleculestm',
        'description': 'PubChem molecular descriptions (~250K samples)'
    },
    'remoteclip': {
        'file': 'remote_captions_preprocessed.parquet',
        'id_col': 'id',
        'modality': 'remote_sensing',
        'encoder': 'remoteclip',
        'description': 'Remote sensing image captions (~68K samples)'
    }
}

# Offset configurations (encoder -> dataset mapping)
OFFSET_CONFIGS = {
    'clip_coco': {'encoder': 'clip', 'dataset': 'coco', 'dim': 1024},
    'clap_audiocaps': {'encoder': 'clap', 'dataset': 'audiocaps', 'dim': 512},
    'uni3d_objaverse': {'encoder': 'uni3d', 'dataset': 'objaverse', 'dim': 1024},
    'cxr_clip_chestxray': {'encoder': 'cxr_clip', 'dataset': 'chestxray', 'dim': 512},
    'moleculestm_pubchem': {'encoder': 'moleculestm', 'dataset': 'pubchem', 'dim': 256},
    'remoteclip_remoteclip_ret3': {'encoder': 'remoteclip', 'dataset': 'remoteclip', 'dim': 768},
    'languagebind_coco': {'encoder': 'languagebind', 'dataset': 'coco', 'dim': 768},
    'viclip_internvid': {'encoder': 'viclip', 'dataset': 'internvid', 'dim': 768},
}


def load_offset_vectors(offset_dir: str, encoder_dataset: str, num_samples: int = 5000):
    """Load offset vectors from pickle files."""
    base_path = os.path.join(offset_dir, str(num_samples), encoder_dataset)

    text_offset_path = os.path.join(base_path, 'text_embed_mean.pkl')
    modal_offset_path = os.path.join(base_path, 'img_embed_mean.pkl')

    offsets = {}

    if os.path.exists(text_offset_path):
        with open(text_offset_path, 'rb') as f:
            text_offset = pickle.load(f)
            if isinstance(text_offset, torch.Tensor):
                text_offset = text_offset.cpu().numpy()
            offsets['text_centroid'] = text_offset.squeeze()

    if os.path.exists(modal_offset_path):
        with open(modal_offset_path, 'rb') as f:
            modal_offset = pickle.load(f)
            if isinstance(modal_offset, torch.Tensor):
                modal_offset = modal_offset.cpu().numpy()
            offsets['modal_centroid'] = modal_offset.squeeze()

    return offsets


def prepare_caption_datasets(caption_dir: str, output_dir: str):
    """Prepare caption datasets in unified format."""
    output_path = os.path.join(output_dir, 'captions')
    os.makedirs(output_path, exist_ok=True)

    dataset_info = []

    for dataset_name, config in tqdm(CAPTION_DATASETS.items(), desc="Processing caption datasets"):
        src_file = os.path.join(caption_dir, config['file'])

        if not os.path.exists(src_file):
            print(f"  Skipping {dataset_name}: file not found at {src_file}")
            continue

        # Load and process
        df = pd.read_parquet(src_file)

        # Standardize columns
        df_out = pd.DataFrame({
            'id': df[config['id_col']].astype(str),
            'caption': df['caption'].astype(str),
            'dataset': dataset_name,
            'modality': config['modality'],
            'encoder': config['encoder']
        })

        # Add modality-specific columns
        if 'smiles' in df.columns:
            df_out['smiles'] = df['smiles']
        if 'filename' in df.columns:
            df_out['filename'] = df['filename']

        # Save
        out_file = os.path.join(output_path, f'{dataset_name}.parquet')
        df_out.to_parquet(out_file, index=False)

        dataset_info.append({
            'dataset': dataset_name,
            'modality': config['modality'],
            'encoder': config['encoder'],
            'num_samples': len(df_out),
            'description': config['description'],
            'file': f'{dataset_name}.parquet'
        })

        print(f"  {dataset_name}: {len(df_out)} samples")

    # Save metadata
    meta_df = pd.DataFrame(dataset_info)
    meta_df.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)

    return dataset_info


def prepare_offset_datasets(offset_dir: str, output_dir: str, num_samples: int = 5000):
    """Prepare offset vectors as individual files per encoder."""
    output_path = os.path.join(output_dir, 'offsets', str(num_samples))
    os.makedirs(output_path, exist_ok=True)

    offset_records = []

    # Scan actual directories in offset_dir
    sample_dir = os.path.join(offset_dir, str(num_samples))
    if not os.path.exists(sample_dir):
        print(f"  Offset directory not found: {sample_dir}")
        return []

    for encoder_dataset in tqdm(sorted(os.listdir(sample_dir)), desc="Processing offset vectors"):
        folder_path = os.path.join(sample_dir, encoder_dataset)
        if not os.path.isdir(folder_path):
            continue

        # Skip backup folders
        if encoder_dataset.endswith('_bk'):
            continue

        text_path = os.path.join(folder_path, 'text_embed_mean.pkl')
        modal_path = os.path.join(folder_path, 'img_embed_mean.pkl')

        if not os.path.exists(text_path) or not os.path.exists(modal_path):
            print(f"  Skipping {encoder_dataset}: offset files incomplete")
            continue

        # Load offsets
        with open(text_path, 'rb') as f:
            text_offset = pickle.load(f)
            if isinstance(text_offset, torch.Tensor):
                text_offset = text_offset.cpu().numpy()
            text_offset = text_offset.squeeze()

        with open(modal_path, 'rb') as f:
            modal_offset = pickle.load(f)
            if isinstance(modal_offset, torch.Tensor):
                modal_offset = modal_offset.cpu().numpy()
            modal_offset = modal_offset.squeeze()

        dim = len(text_offset)

        # Create output folder for this encoder
        encoder_output_dir = os.path.join(output_path, encoder_dataset)
        os.makedirs(encoder_output_dir, exist_ok=True)

        # Save as numpy files
        np.save(os.path.join(encoder_output_dir, 'text_centroid.npy'), text_offset)
        np.save(os.path.join(encoder_output_dir, 'modal_centroid.npy'), modal_offset)

        # Also save metadata
        metadata = {
            'encoder_dataset': encoder_dataset,
            'embedding_dim': dim,
            'num_samples': num_samples,
        }
        with open(os.path.join(encoder_output_dir, 'metadata.json'), 'w') as f:
            import json
            json.dump(metadata, f, indent=2)

        offset_records.append({
            'encoder_dataset': encoder_dataset,
            'embedding_dim': dim,
            'num_samples': num_samples,
        })

        print(f"  {encoder_dataset}: dim={dim}")

    # Save overall metadata
    import json
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(offset_records, f, indent=2)

    return offset_records


def create_dataset_card(output_dir: str, caption_info: list, offset_info: list):
    """Create HuggingFace dataset card (README.md)."""
    readme_content = """---
license: mit
task_categories:
- zero-shot-classification
- text-to-image
- text-to-audio
language:
- en
tags:
- multimodal
- cross-modal
- embeddings
- textme
size_categories:
- 1M<n<10M
---

# TextME: Training Datasets

This dataset contains the training data for [TextME](https://github.com/your-username/TextME), a text-only modality expansion framework.

## Dataset Description

TextME enables zero-shot cross-modal transfer by leveraging the **consistent modality gap** property of pretrained contrastive encoders. This dataset provides:

1. **Caption datasets**: Text descriptions for training projection networks
2. **Offset vectors**: Precomputed centroids for the interchangeable space

## Caption Datasets

| Dataset | Modality | Encoder | Samples | Description |
|---------|----------|---------|---------|-------------|
"""

    for info in caption_info:
        readme_content += f"| {info['dataset']} | {info['modality']} | {info['encoder']} | {info['num_samples']:,} | {info['description']} |\n"

    readme_content += """
### Usage

```python
from datasets import load_dataset

# Load specific dataset
coco = load_dataset("your-username/textme-data", data_files="captions/coco.parquet")
audiocaps = load_dataset("your-username/textme-data", data_files="captions/audiocaps.parquet")

# Load all caption datasets
all_captions = load_dataset("your-username/textme-data", data_dir="captions")
```

## Offset Vectors

Precomputed centroids (μ_text and μ_modal) for each encoder-dataset pair:

| Encoder-Dataset | Embedding Dim |
|-----------------|---------------|
"""

    for info in offset_info:
        readme_content += f"| {info['encoder_dataset']} | {info['embedding_dim']} |\n"

    readme_content += """
### Structure

```
offsets/
└── 5000/                          # Number of samples used for computation
    ├── clip_coco/
    │   ├── text_centroid.npy      # μ_text (1024-dim)
    │   ├── modal_centroid.npy     # μ_modal (1024-dim)
    │   └── metadata.json
    ├── clap_audiocaps/
    │   ├── text_centroid.npy      # μ_text (512-dim)
    │   ├── modal_centroid.npy     # μ_modal (512-dim)
    │   └── metadata.json
    └── ...
```

### Usage

```python
import numpy as np
from huggingface_hub import hf_hub_download

# Download specific offset files
text_centroid = np.load(
    hf_hub_download("SoyeonHH/textme-data", "offsets/5000/clip_coco/text_centroid.npy")
)
modal_centroid = np.load(
    hf_hub_download("SoyeonHH/textme-data", "offsets/5000/clip_coco/modal_centroid.npy")
)

# Apply centering for interchangeable space
centered_text = text_embedding - text_centroid
centered_modal = modal_embedding - modal_centroid
```

## Citation

```bibtex
@article{hong2026textme,
  title={TextME: Bridging Unseen Modalities Through Text Descriptions},
  author={Hong, Soyeon and Kim, Jinchan and You, Jaegook and Choi, Seungtaek and Kwak, Suha and Cho, Hyunsouk},
  journal={arXiv preprint arXiv:2602.03098},
  year={2026}
}
```

## License

This dataset is released under the MIT License.
"""

    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)


def upload_to_huggingface(output_dir: str, repo_id: str):
    """Upload dataset to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description='Prepare TextME datasets for HuggingFace')
    parser.add_argument('--caption_dir', type=str,
                        default='/workspace/data/pretraining_captions',
                        help='Directory containing caption parquet files')
    parser.add_argument('--offset_dir', type=str,
                        default='/data2/datasets/mm/BIND/offset_results',
                        help='Directory containing offset pickle files')
    parser.add_argument('--output_dir', type=str,
                        default='./hf_datasets',
                        help='Output directory for prepared datasets')
    parser.add_argument('--num_offset_samples', type=int, default=5000,
                        help='Number of samples used for offset calculation')
    parser.add_argument('--upload', action='store_true',
                        help='Upload to HuggingFace Hub')
    parser.add_argument('--repo_id', type=str,
                        help='HuggingFace repo ID (e.g., your-username/textme-data)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("TextME Dataset Preparation for HuggingFace")
    print("=" * 60)

    # Prepare caption datasets
    print("\n[1/3] Preparing caption datasets...")
    caption_info = prepare_caption_datasets(args.caption_dir, args.output_dir)

    # Prepare offset datasets
    print("\n[2/3] Preparing offset vectors...")
    offset_info = prepare_offset_datasets(args.offset_dir, args.output_dir, args.num_offset_samples)

    # Create dataset card
    print("\n[3/3] Creating dataset card...")
    create_dataset_card(args.output_dir, caption_info, offset_info)

    print("\n" + "=" * 60)
    print(f"Dataset prepared at: {args.output_dir}")
    print("=" * 60)

    # Upload if requested
    if args.upload:
        if not args.repo_id:
            print("Error: --repo_id required for upload")
            return
        print(f"\nUploading to HuggingFace: {args.repo_id}")
        upload_to_huggingface(args.output_dir, args.repo_id)


if __name__ == '__main__':
    main()
