"""
Encoder Wrappers for TextME.

Provides unified interfaces for various pretrained contrastive encoders.
Extracted from actual EfficientBind implementation (src/models.py)

Supported encoders:
- CLIP (Image) - 1024-dim
- ViCLIP (Video) - 768-dim
- CLAP (Audio) - 512-dim
- Uni3D (3D Point Cloud) - 1024-dim
- CXR-CLIP (X-ray) - 512-dim
- MoleculeSTM (Molecule) - 256-dim
- LanguageBind (Multi-modal) - 768-dim
- RemoteCLIP (Remote Sensing) - 768-dim
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Union, Optional, Dict
from abc import ABC, abstractmethod

from .projector import ProjectionHead


# Encoder embedding dimensions (from EfficientBind/src/models.py)
ENCODER_DIM = {
    'clip': 1024,
    'clap': 512,
    'languagebind': 768,
    'uni3d': 1024,
    'cxr_clip': 512,
    'moleculestm': 256,
    'remoteclip': 768,
    'viclip': 768,
}

# Default datasets for offset computation (from EfficientBind/src/models.py)
MODEL_OFFSET_DATA = {
    'clip': 'coco',
    'clap': 'audiocaps',
    'languagebind': 'coco',
    'uni3d': 'objaverse',
    'cxr_clip': 'chestxray',
    'moleculestm': 'pubchem',
    'remoteclip': 'remoteclip_ret3',
    'viclip': 'internvid',
}

# LLM anchor dimensions
ANCHOR_DIM = {
    'qwen3_embed_4b': 2560,
    'qwen3_embed_0.6b': 1024,
    'nv_embed_v2': 4096,
    'gte_qwen2_1.5b': 1536,
}


def generate_offset_config(
    model_name: str,
    dataset_name: str,
    offset_num: int,
    offset_dir: str = "./offsets"
) -> Dict[str, str]:
    """
    Generate offset configuration paths.

    Extracted from: EfficientBind/src/models.py

    Args:
        model_name: Model name (e.g., 'clip', 'uni3d', 'moleculestm')
        dataset_name: Dataset name for offset computation
        offset_num: Number of samples used for offset computation
        offset_dir: Base directory for offset files

    Returns:
        Dictionary mapping modality names to offset file paths
    """
    base_path = os.path.join(offset_dir, str(offset_num), f"{model_name}_{dataset_name}")

    # Base config for all models
    config = {
        'text': os.path.join(base_path, 'text_embed_mean.pkl'),
    }

    # Model-specific modality offset paths
    if model_name == 'uni3d':
        config['point'] = os.path.join(base_path, 'img_embed_mean.pkl')
        config['image'] = os.path.join(base_path, 'img_embed_mean.pkl')
        config['modal'] = os.path.join(base_path, 'img_embed_mean.pkl')
    elif model_name == 'moleculestm':
        config['smiles'] = os.path.join(base_path, 'img_embed_mean.pkl')
        config['modal'] = os.path.join(base_path, 'img_embed_mean.pkl')
    elif model_name == 'cxr_clip':
        config['image'] = os.path.join(base_path, 'img_embed_mean.pkl')
        config['modal'] = os.path.join(base_path, 'img_embed_mean.pkl')
    elif model_name == 'clap':
        config['audio'] = os.path.join(base_path, 'img_embed_mean.pkl')
        config['modal'] = os.path.join(base_path, 'img_embed_mean.pkl')
    else:
        # Default for CLIP, LanguageBind, RemoteCLIP, etc.
        config['image'] = os.path.join(base_path, 'img_embed_mean.pkl')
        config['modal'] = os.path.join(base_path, 'img_embed_mean.pkl')

    return config


def process_embeddings(
    embeddings: Tensor,
    offset: Optional[Tensor] = None,
    noise_std: float = 0.0,
    uniform_noise: bool = False,
) -> Tensor:
    """
    Apply offset subtraction and optional noise injection.

    Extracted from: EfficientBind/src/models.py

    Args:
        embeddings: Input embeddings [batch_size, dim]
        offset: Offset vector to subtract (centroid)
        noise_std: Standard deviation of noise to add
        uniform_noise: Use uniform ball noise instead of Gaussian

    Returns:
        Processed embeddings
    """
    if offset is not None:
        # Ensure offset is on same device
        if offset.device != embeddings.device:
            offset = offset.to(embeddings.device)
        embeddings = embeddings - offset

    if noise_std > 0.0:
        if uniform_noise:
            # Uniform ball noise
            noise = torch.randn_like(embeddings)
            noise = F.normalize(noise, dim=-1)
            radius = torch.rand(embeddings.shape[0], 1, device=embeddings.device)
            noise = noise * radius * noise_std
        else:
            # Gaussian noise
            noise = torch.randn_like(embeddings) * noise_std
        embeddings = embeddings + noise

    return embeddings


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for encoder wrappers.

    All encoders must implement:
    - encode_text: Encode text descriptions
    - encode_modal: Encode modality-specific inputs
    """

    def __init__(
        self,
        args=None,
        device: str = 'cuda',
        use_projection: bool = False,
        use_offset: bool = False,
        offset_dir: str = "./offsets",
        offset_num: int = 5000,
        out_dim: int = 2560,
        init_mode: str = 'xav',
        dim_act: str = 'gelu',
        noise_std: float = 0.0,
        uniform_noise: bool = False,
    ):
        super().__init__()

        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.use_projection = use_projection
        self.use_offset = use_offset
        self.offset_dir = offset_dir
        self.offset_num = offset_num
        self.out_dim = out_dim
        self.noise_std = noise_std
        self.uniform_noise = uniform_noise

        # Will be initialized by subclasses
        self.model = None
        self.tokenizer = None
        self.projector = None
        self.offset = {}

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the encoder's embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for offset lookup."""
        pass

    def _init_projector(self, in_dim: int, out_dim: int, init_mode: str, dim_act: str):
        """Initialize projection head."""
        self.projector = ProjectionHead(
            in_dim=in_dim,
            proj_dim=2 * in_dim,
            out_dim=out_dim,
            init_mode=init_mode,
            dim_act=dim_act,
        ).to(self._device)

    def _load_offset(self):
        """Load precomputed offset vectors."""
        if not self.use_offset:
            return

        dataset_name = MODEL_OFFSET_DATA.get(self.model_name, 'coco')
        offset_config = generate_offset_config(
            self.model_name, dataset_name, self.offset_num, self.offset_dir
        )

        for modality, path in offset_config.items():
            if path and os.path.exists(path):
                with open(path, 'rb') as f:
                    offset_tensor = pickle.load(f)
                if not isinstance(offset_tensor, torch.Tensor):
                    offset_tensor = torch.tensor(offset_tensor, dtype=torch.float32)
                self.offset[modality] = offset_tensor.to(self._device)
                print(f"Loaded offset for {modality} from {path}")

    def _process_text_embeddings(self, embeddings: Tensor) -> Tensor:
        """Process text embeddings with offset and projection."""
        if self.use_offset:
            offset = self.offset.get('text', None)
            embeddings = process_embeddings(
                embeddings, offset=offset,
                noise_std=self.noise_std, uniform_noise=self.uniform_noise
            )

        if self.use_projection and self.projector is not None:
            embeddings = self.projector(embeddings)

        return F.normalize(embeddings, p=2, dim=-1)

    def _process_modal_embeddings(self, embeddings: Tensor, modality: str = 'modal') -> Tensor:
        """Process modal embeddings with offset and projection."""
        if self.use_offset:
            offset = self.offset.get(modality, self.offset.get('modal', None))
            embeddings = process_embeddings(
                embeddings, offset=offset,
                noise_std=self.noise_std, uniform_noise=self.uniform_noise
            )

        if self.use_projection and self.projector is not None:
            embeddings = self.projector(embeddings)

        return F.normalize(embeddings, p=2, dim=-1)

    @abstractmethod
    def encode_text(self, texts: List[str]) -> Tensor:
        """Encode text descriptions."""
        pass

    @abstractmethod
    def encode_modal(self, inputs) -> Tensor:
        """Encode modality-specific inputs."""
        pass


class CLIPEncoder(BaseEncoder):
    """
    CLIP encoder wrapper for image-text embeddings.

    Uses OpenCLIP implementation for flexibility across model variants.
    """

    def __init__(
        self,
        model_name: str = 'ViT-L-14',
        pretrained: str = 'openai',
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(device=device, **kwargs)

        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self._device).eval()

        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                ENCODER_DIM['clip'], self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        # Load offsets if needed
        self._load_offset()

    @property
    def embedding_dim(self) -> int:
        return ENCODER_DIM['clip']

    @property
    def model_name(self) -> str:
        return 'clip'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        tokens = self.tokenizer(texts).to(self._device)
        embeddings = self.model.encode_text(tokens)
        return self._process_text_embeddings(embeddings)

    @torch.no_grad()
    def encode_modal(self, images) -> Tensor:
        return self.encode_image(images)

    @torch.no_grad()
    def encode_image(self, images) -> Tensor:
        """Encode images. Accepts file paths, PIL images, or tensors."""
        from PIL import Image

        if isinstance(images, (list, tuple)):
            if isinstance(images[0], str):
                # File paths
                images = [Image.open(p).convert('RGB') for p in images]
            # Process PIL images
            images = torch.stack([self.preprocess(img) for img in images])

        images = images.to(self._device)
        embeddings = self.model.encode_image(images)
        return self._process_modal_embeddings(embeddings, 'image')


class CLAPEncoder(BaseEncoder):
    """
    CLAP encoder wrapper for audio-text embeddings.
    """

    def __init__(
        self,
        model_name: str = 'laion/larger_clap_music_and_speech',
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(device=device, **kwargs)

        from transformers import ClapModel, ClapProcessor
        self.model = ClapModel.from_pretrained(model_name).to(self._device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)

        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                ENCODER_DIM['clap'], self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        # Load offsets if needed
        self._load_offset()

    @property
    def embedding_dim(self) -> int:
        return ENCODER_DIM['clap']

    @property
    def model_name(self) -> str:
        return 'clap'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        embeddings = self.model.get_text_features(**inputs)
        return self._process_text_embeddings(embeddings)

    @torch.no_grad()
    def encode_modal(self, audios, sampling_rate: int = 48000) -> Tensor:
        return self.encode_audio(audios, sampling_rate)

    @torch.no_grad()
    def encode_audio(self, audios, sampling_rate: int = 48000) -> Tensor:
        """Encode audio. Accepts file paths or waveform tensors."""
        import librosa

        if isinstance(audios, (list, tuple)) and isinstance(audios[0], str):
            # Load from file paths
            audios = [librosa.load(p, sr=sampling_rate)[0] for p in audios]

        inputs = self.processor(
            audios=audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        embeddings = self.model.get_audio_features(**inputs)
        return self._process_modal_embeddings(embeddings, 'audio')


class LanguageBindEncoder(BaseEncoder):
    """
    LanguageBind encoder wrapper for multi-modal embeddings.

    Supports: image, video, audio, depth, thermal modalities.
    Note: Requires LanguageBind installation.
    """

    def __init__(self, device: str = 'cuda', **kwargs):
        super().__init__(device=device, **kwargs)
        self._embedding_dim = ENCODER_DIM['languagebind']

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                self._embedding_dim, self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        print("Note: LanguageBind encoder requires separate installation.")
        print("See: https://github.com/PKU-YuanGroup/LanguageBind")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return 'languagebind'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        raise NotImplementedError("LanguageBind encoder requires separate installation.")

    @torch.no_grad()
    def encode_modal(self, inputs, modality: str = 'image') -> Tensor:
        raise NotImplementedError("LanguageBind encoder requires separate installation.")


class Uni3DEncoder(BaseEncoder):
    """
    Uni3D encoder wrapper for 3D point cloud embeddings.

    Extracted from: EfficientBind/src/models.py (Uni3D_Plus class)

    Requires:
        - Uni3D installation (https://github.com/baaivision/Uni3D)
        - EVA-CLIP for text encoding
        - Pretrained Uni3D checkpoint (LVIS, ModelNet40, or ScanObjectNN)

    Args:
        mode: Checkpoint mode - 'lvis', 'mnet', or 'scan'
        eva_clip_path: Path to EVA-CLIP checkpoint
        uni3d_checkpoint: Path to Uni3D checkpoint
    """

    def __init__(
        self,
        mode: str = 'lvis',
        eva_clip_path: str = 'laion2b_s9b_b144k',
        uni3d_checkpoint: str = None,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self._embedding_dim = ENCODER_DIM['uni3d']
        self.mode = mode

        # Store paths for later initialization
        self.eva_clip_path = eva_clip_path
        self.uni3d_checkpoint = uni3d_checkpoint

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                self._embedding_dim, self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        # Load offsets if needed
        self._load_offset()

        print("Note: Uni3D encoder requires separate installation.")
        print("See: https://github.com/baaivision/Uni3D")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return 'uni3d'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        """
        Encode text using EVA-CLIP text encoder.

        In full implementation, uses:
            tokens = self.tokenizer(texts).to(self.device)
            embeddings = self.clip.encode_text(tokens)
        """
        raise NotImplementedError(
            "Uni3D encoder requires EVA-CLIP and Uni3D installation. "
            "See: https://github.com/baaivision/Uni3D"
        )

    @torch.no_grad()
    def encode_modal(self, points, colors=None) -> Tensor:
        return self.encode_point(points, colors)

    @torch.no_grad()
    def encode_point(self, points, colors=None) -> Tensor:
        """
        Encode 3D point clouds.

        Args:
            points: Point cloud tensor [B, N, 3]
            colors: RGB colors tensor [B, N, 3] (optional)

        In full implementation:
            pc = points.to(device)
            rgb = colors.to(device) if colors else zeros
            feature = torch.cat((pc, rgb), dim=-1)
            pc_features = self.point_encoder.encode_pc(feature)
        """
        raise NotImplementedError(
            "Uni3D encoder requires Uni3D installation. "
            "See: https://github.com/baaivision/Uni3D"
        )


class CXRCLIPEncoder(BaseEncoder):
    """
    CXR-CLIP encoder wrapper for chest X-ray image embeddings.

    Extracted from: EfficientBind/src/models.py (CXR_CLIP_Plus class)

    Requires:
        - CXR-CLIP installation
        - Pretrained CXR-CLIP checkpoint

    Reference: https://github.com/kakaobrain/cxr-clip
    """

    def __init__(
        self,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self._embedding_dim = ENCODER_DIM['cxr_clip']

        # Store paths for later initialization
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                self._embedding_dim, self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        # Load offsets if needed
        self._load_offset()

        print("Note: CXR-CLIP encoder requires separate installation.")
        print("See: https://github.com/kakaobrain/cxr-clip")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return 'cxr_clip'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        """
        Encode text descriptions for X-ray images.

        In full implementation:
            tokens = self.tokenizer(texts, max_length=256, padding='max_length',
                                   truncation=True, return_tensors='pt')
            embeddings = self.model.encode_text(tokens)
            if self.model.projection:
                embeddings = self.model.text_projection(embeddings)
        """
        raise NotImplementedError(
            "CXR-CLIP encoder requires CXR-CLIP installation. "
            "See: https://github.com/kakaobrain/cxr-clip"
        )

    @torch.no_grad()
    def encode_modal(self, images) -> Tensor:
        return self.encode_image(images)

    @torch.no_grad()
    def encode_image(self, images) -> Tensor:
        """
        Encode chest X-ray images.

        Args:
            images: List of image paths or PIL images

        In full implementation:
            images_data = [np.array(Image.open(img).convert("RGB")) for img in images]
            processed = [self.transform_image(self.transform, img) for img in images_data]
            inputs = torch.stack(processed).to(self.device)
            embeddings = self.model.encode_image(inputs)
            if self.model.projection:
                embeddings = self.model.image_projection(embeddings)
        """
        raise NotImplementedError(
            "CXR-CLIP encoder requires CXR-CLIP installation. "
            "See: https://github.com/kakaobrain/cxr-clip"
        )


class MoleculeSTMEncoder(BaseEncoder):
    """
    MoleculeSTM encoder wrapper for molecular structure embeddings.

    Extracted from: EfficientBind/src/models.py (MoleculeSTM_Plus class)

    Requires:
        - MoleculeSTM installation
        - SciBERT for text encoding
        - MegaMolBART for molecule encoding
        - Pretrained MoleculeSTM checkpoints

    Reference: https://github.com/chao1224/MoleculeSTM
    """

    def __init__(
        self,
        molecule_path: str = None,
        vocab_path: str = None,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self._embedding_dim = ENCODER_DIM['moleculestm']

        # Store paths for later initialization
        self.molecule_path = molecule_path
        self.vocab_path = vocab_path

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                self._embedding_dim, self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        # Load offsets if needed
        self._load_offset()

        print("Note: MoleculeSTM encoder requires separate installation.")
        print("See: https://github.com/chao1224/MoleculeSTM")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return 'moleculestm'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        """
        Encode text descriptions for molecules.

        In full implementation:
            tokens_ids, masks = prepare_text_tokens(
                device=self.device, description=texts,
                tokenizer=self.tokenizer, max_seq_len=256
            )
            text_output = self.text_model(tokens_ids, masks)
            text_repr = text_output["pooler_output"]
            text_repr = self.text2latent(text_repr)
        """
        raise NotImplementedError(
            "MoleculeSTM encoder requires MoleculeSTM installation. "
            "See: https://github.com/chao1224/MoleculeSTM"
        )

    @torch.no_grad()
    def encode_modal(self, smiles) -> Tensor:
        return self.encode_smile(smiles)

    @torch.no_grad()
    def encode_smile(self, smiles: List[str]) -> Tensor:
        """
        Encode SMILES molecular representations.

        Args:
            smiles: List of SMILES strings

        In full implementation:
            molecule_repr = get_molecule_repr_MoleculeSTM(
                molecule_data=smiles, mol2latent=self.mol2latent,
                molecule_type="SMILES", MegaMolBART_wrapper=self.molecule_model
            )
        """
        raise NotImplementedError(
            "MoleculeSTM encoder requires MoleculeSTM installation. "
            "See: https://github.com/chao1224/MoleculeSTM"
        )


class RemoteCLIPEncoder(BaseEncoder):
    """
    RemoteCLIP encoder wrapper for remote sensing image embeddings.

    Extracted from: EfficientBind/src/models.py (RemoteCLIP_Plus class)

    Requires:
        - OpenCLIP
        - Pretrained RemoteCLIP checkpoint

    Reference: https://github.com/ChenDelong1999/RemoteCLIP
    """

    def __init__(
        self,
        model_name: str = 'ViT-L-14',
        checkpoint_path: str = None,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self._embedding_dim = ENCODER_DIM['remoteclip']

        # Store paths for later initialization
        self._model_variant = model_name
        self.checkpoint_path = checkpoint_path

        # Initialize projector if needed
        if self.use_projection:
            self._init_projector(
                self._embedding_dim, self.out_dim,
                kwargs.get('init_mode', 'xav'), kwargs.get('dim_act', 'gelu')
            )

        # Load offsets if needed
        self._load_offset()

        print("Note: RemoteCLIP encoder requires pretrained checkpoint.")
        print("See: https://github.com/ChenDelong1999/RemoteCLIP")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        return 'remoteclip'

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        """
        Encode text descriptions for remote sensing images.

        In full implementation:
            tokens = open_clip.tokenize(texts).to(self.device)
            embeddings = self.clip.encode_text(tokens)
        """
        raise NotImplementedError(
            "RemoteCLIP encoder requires pretrained checkpoint. "
            "See: https://github.com/ChenDelong1999/RemoteCLIP"
        )

    @torch.no_grad()
    def encode_modal(self, images) -> Tensor:
        return self.encode_image(images)

    @torch.no_grad()
    def encode_image(self, images) -> Tensor:
        """
        Encode remote sensing images.

        In full implementation:
            processed = [self.preprocess(img) for img in images]
            inputs = torch.stack(processed).to(self.device)
            embeddings = self.clip.encode_image(inputs)
        """
        raise NotImplementedError(
            "RemoteCLIP encoder requires pretrained checkpoint. "
            "See: https://github.com/ChenDelong1999/RemoteCLIP"
        )


def build_encoder(
    encoder_name: str,
    device: str = 'cuda',
    **kwargs
) -> BaseEncoder:
    """
    Factory function to build encoder by name.

    Extracted from: EfficientBind/src/models.py

    Args:
        encoder_name: One of 'clip', 'clap', 'languagebind', 'uni3d',
                      'cxr_clip', 'moleculestm', 'remoteclip'
        device: Device to load model on
        **kwargs: Additional arguments for specific encoders

    Returns:
        Encoder instance
    """
    encoder_name = encoder_name.lower()

    if encoder_name == 'clip':
        return CLIPEncoder(device=device, **kwargs)
    elif encoder_name == 'clap':
        return CLAPEncoder(device=device, **kwargs)
    elif encoder_name == 'languagebind':
        return LanguageBindEncoder(device=device, **kwargs)
    elif encoder_name == 'uni3d':
        return Uni3DEncoder(device=device, **kwargs)
    elif encoder_name == 'cxr_clip':
        return CXRCLIPEncoder(device=device, **kwargs)
    elif encoder_name == 'moleculestm':
        return MoleculeSTMEncoder(device=device, **kwargs)
    elif encoder_name == 'remoteclip':
        return RemoteCLIPEncoder(device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Available: {list(ENCODER_DIM.keys())}"
        )
