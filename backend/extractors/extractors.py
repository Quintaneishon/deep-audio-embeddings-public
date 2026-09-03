"""Concrete embedding extractors and registry."""
from typing import Dict, List, Optional
import numpy as np
from backend import config
import torch
import torch.nn as nn
from .base import (
    ModelConfig,
    EmbeddingExtractor,
    ConvolutionalExtractor,
    TransformerExtractor
)
from backend import utils
from pathlib import Path
from backend.DL.models.MusiCNN import Musicnn
from backend.DL.models.VGG import VGG_Res
from backend.DL.models.VGGish import VGGish
from backend.DL.models.Whisper import WhisperEmbedding, pad_or_trim, log_mel_spectrogram
from backend.DL.models.MERT import MERTEmbedding, resample_audio
from backend.DL.models.WhisperContrastive import WhisperContrastive
from backend.DL.models.WhisperContrastiveMultiLabel import WhisperContrastiveMultiLabel
from backend.DL.models.WhisperContrastiveHybrid import WhisperContrastiveHybrid
from backend.DL.models.MusiCNNContrastive import MusiCNNContrastive
from backend.DL.models.VGGContrastive import VGGContrastive
from backend.DL.models.VGGWhisperDual import VGGWhisperDual
from backend.DL.models.MERTContrastiveLora import MERTContrastiveLora
from backend.DL.models.MERTContrastiveMultiLabel import MERTContrastiveMultiLabel

DC = config.CUDA_DEVICE
if isinstance(DC, str) and DC.startswith('cuda') and not torch.cuda.is_available():
    print(f'Warning: CUDA not available, falling back from {DC} to cpu')
    DC = 'cpu'
print('Using device: ', DC)

def set_device(device: str):
    global DC
    if isinstance(device, str) and device.startswith('cuda') and not torch.cuda.is_available():
        print(f'Warning: CUDA not available, ignoring {device}, using cpu')
        device = 'cpu'
    DC = device
    print('Device switched to:', DC)

# =============================================================================
# Convolutional Models
# =============================================================================

class MusiCNNExtractor(ConvolutionalExtractor):
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'musicnn'

    def extract(self, audio_path: str, weights_path: str, dataset: str, **kwargs) -> np.ndarray:
        model = Musicnn(n_class=config.N_TAGS, dataset=dataset)
        model.load_state_dict(torch.load(weights_path, map_location=DC, weights_only=True))
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)

class VGGExtractor(ConvolutionalExtractor):
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'vgg'

    def extract(self, audio_path: str, weights_path: str, dataset: str, **kwargs) -> np.ndarray:
        model = VGG_Res(n_class=config.N_TAGS, use_simple_res=dataset == 'mtat')
        model.load_state_dict(torch.load(weights_path, map_location=DC, weights_only=True))
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)

            # Extract at layer7_pool (before Dense compression) — empirically
            # better nDCG and agreement than dense1 for music similarity graphs.
            x = model.spec(x)
            x = model.to_db(x)
            x = x.unsqueeze(1)
            x = model.spec_bn(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.layer5(x)
            x = model.layer6(x)
            x = model.layer7(x)
            x = x.squeeze(2)
            if x.size(-1) != 1:
                x = nn.MaxPool1d(x.size(-1))(x)
            layer7_pool = x.squeeze(2)  # [1, 512]

            embeddings = layer7_pool.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)

class VGGishExtractor(TransformerExtractor):
    def __init__(self):
        super().__init__('pretrained')

    @property
    def model_name(self) -> str:
        return 'vggish'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        model = VGGish(pretrained=True, n_class=config.N_TAGS)
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            _, embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)

# =============================================================================
# Transformer Models
# =============================================================================

class WhisperExtractor(TransformerExtractor):
    def __init__(self):
        super().__init__('base')

    @property
    def model_name(self) -> str:
        return 'whisper'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        model = WhisperEmbedding(model_name=self.model_size, device=DC)
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)


class MERTExtractor(TransformerExtractor):
    def __init__(self):
        super().__init__('95m')

    @property
    def model_name(self) -> str:
        return 'mert'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        model_id = config.MERT_MODEL_IDS.get(self.model_size, 'm-a-p/MERT-v1-95M')
        model = MERTEmbedding(model_name=model_id, cache_dir=weights_path, device=DC)
        model.to(DC)
        model.eval()

        y, loaded_sr = utils.load_audio_safe(audio_path, sr=None)
        y_tensor = torch.from_numpy(y).float()
        y_tensor = resample_audio(y_tensor, loaded_sr)

        with torch.no_grad():
            x = y_tensor.unsqueeze(0).to(DC)
            embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)

class WhisperContrastiveExtractor(TransformerExtractor):
    def __init__(self):
        super().__init__('base')

    @property
    def model_name(self) -> str:
        return 'whisper_contrastive'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        model = WhisperContrastive(model_name=self.model_size, projection_dim=config.N_DIM, device=DC)

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if not k.startswith('taggram_projection')}
        model.load_state_dict(state_dict)
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)


class WhisperContrastiveMultiLabelExtractor(TransformerExtractor):
    """Extractor for models trained with multi-label supervised contrastive loss."""

    def __init__(self):
        super().__init__('base')

    @property
    def model_name(self) -> str:
        return 'whisper_contrastive_multilabel'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        model = WhisperContrastiveMultiLabel(model_name=self.model_size, projection_dim=config.N_DIM, device=DC)

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if not k.startswith('taggram_projection')}
        model.load_state_dict(state_dict)
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)


class WhisperContrastiveHybridExtractor(TransformerExtractor):
    """Extractor for models trained with hybrid (genre + BPM) contrastive loss."""

    def __init__(self):
        super().__init__('base')

    @property
    def model_name(self) -> str:
        return 'whisper_contrastive_hybrid'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        model = WhisperContrastiveHybrid(model_name=self.model_size, projection_dim=config.N_DIM, device=DC)

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                      if not k.startswith('taggram_projection')}
        model.load_state_dict(state_dict)
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)

# =============================================================================
# MusiCNN Contrastive Models
# =============================================================================

class MusiCNNContrastiveHybridExtractor(ConvolutionalExtractor):
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'musicnn_contrastive_hybrid'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        model = MusiCNNContrastive(projection_dim=config.N_DIM)

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)

        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy()

        return embeddings.reshape(1, -1)


class MusiCNNContrastiveExtractor(ConvolutionalExtractor):
    """MusiCNN fine-tuned with SupConLoss (genre-only contrastive)."""
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'musicnn_contrastive'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        model = MusiCNNContrastive(projection_dim=config.N_DIM)
        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)
        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)
        return embeddings


class MusiCNNMultiSignalExtractor(ConvolutionalExtractor):
    """MusiCNN fine-tuned with MultiSignalLoss (genre+BPM+instrument+timesig+squareness)."""
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'musicnn_multisignal'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        model = MusiCNNContrastive(projection_dim=config.N_DIM)
        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)
        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)
        return embeddings

class VGGContrastiveExtractor(ConvolutionalExtractor):
    """VGG fine-tuned with contrastive loss + projection head."""
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'vgg_contrastive'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        model = VGGContrastive(projection_dim=config.N_DIM)
        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)
        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)
        return embeddings

class VGGContrastiveProj512Extractor(ConvolutionalExtractor):
    """VGG fine-tuned with MultiLabel SupCon loss, projection_dim=512 (thesis model)."""
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'vgg_contrastive_proj512'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        model = VGGContrastive(projection_dim=512)
        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)
        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)
        return embeddings


class VGGAudioPureExtractor(ConvolutionalExtractor):
    """VGG fine-tuned with Audio-Pure contrastive loss (BPM + squareness + time-sig, no genre labels)."""

    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'vgg_audio_pure'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        model = VGGContrastive(projection_dim=config.N_DIM)
        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)
        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)
        return embeddings


class MERTContrastiveLoraExtractor(TransformerExtractor):
    """MERT-v1-95M fine-tuned with LoRA + MultiSignalLoss."""

    def __init__(self):
        super().__init__('95m')

    @property
    def model_name(self) -> str:
        return 'mert_lora'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model_cfg  = checkpoint.get('model_config', {})

        model = MERTContrastiveLora(
            model_name=config.MERT_MODEL_IDS['95m'],
            projection_dim=model_cfg.get('projection_dim', config.N_DIM),
            lora_r=model_cfg.get('lora_r', 8),
            lora_alpha=model_cfg.get('lora_alpha', 16),
            device=torch.device(DC),
            cache_dir=config.MODEL_WEIGHTS['mert']['95m'],
        )

        model_state = model.state_dict()
        model_state.update(checkpoint['trainable_state_dict'])
        model.load_state_dict(model_state)
        model.eval()

        y, loaded_sr = utils.load_audio_safe(audio_path, sr=None)
        y_tensor = torch.from_numpy(y).float()
        from backend.DL.models.MERT import resample_audio
        y_tensor = resample_audio(y_tensor, loaded_sr, target_sr=24000)

        with torch.no_grad():
            x = y_tensor.unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)

        return embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)

class MERTContrastiveMultiLabelExtractor(TransformerExtractor):
    """MERT-v1-95M frozen at layer 6 + projection head, trained with MultiLabelSupConLoss."""

    def __init__(self):
        super().__init__('95m')

    @property
    def model_name(self) -> str:
        return 'mert_contrastive_multilabel'

    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model_cfg = checkpoint.get('model_config', {})

        model = MERTContrastiveMultiLabel(
            model_name=config.MERT_MODEL_IDS['95m'],
            projection_dim=model_cfg.get('projection_dim', config.N_DIM),
            backbone_layer=model_cfg.get('backbone_layer', 6),
            cache_dir=config.MODEL_WEIGHTS['mert']['95m'],
            device=torch.device(DC),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load audio and resample to 24 kHz for MERT
        y, loaded_sr = utils.load_audio_safe(audio_path, sr=None)
        y_tensor = torch.from_numpy(y).float()
        y_tensor = resample_audio(y_tensor, loaded_sr, target_sr=24000)

        with torch.no_grad():
            x = y_tensor.unsqueeze(0).to(DC)
            embeddings_tensor = model(x)  # [1, projection_dim], L2-normalized

        return embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)


class VGGWhisperDualExtractor(ConvolutionalExtractor):
    """Dual-backbone VGG+Whisper model trained with DualBackboneLoss."""
    def __init__(self):
        super().__init__(datasets=['msd'])

    @property
    def model_name(self) -> str:
        return 'vgg_whisper_dual'

    def extract(self, audio_path: str, weights_path: str, dataset: str = 'msd', **kwargs) -> np.ndarray:
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=DC, weights_only=True)
        model_cfg = checkpoint.get('model_config', {})

        model = VGGWhisperDual(
            projection_dim=model_cfg.get('projection_dim', config.N_DIM),
            whisper_size=model_cfg.get('whisper_size', 'base'),
            device=torch.device(DC),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DC)
        model.eval()

        y, _ = utils.load_audio_safe(audio_path, sr=config.SAMPLE_RATE)
        with torch.no_grad():
            x = torch.from_numpy(y).float().unsqueeze(0).to(DC)
            embeddings_tensor = model.forward_extract(x)
            embeddings = embeddings_tensor.squeeze(0).cpu().numpy().reshape(1, -1)
        return embeddings


# =============================================================================
# Registry
# =============================================================================

class ExtractorRegistry:
    """Registry of all available embedding extractors."""

    def __init__(self):
        self._extractors: Dict[str, EmbeddingExtractor] = {}

    def register(self, extractor: EmbeddingExtractor) -> 'ExtractorRegistry':
        self._extractors[extractor.model_name] = extractor
        return self

    def get(self, model_name: str) -> Optional[EmbeddingExtractor]:
        return self._extractors.get(model_name)

    def get_all_configs(self) -> List[ModelConfig]:
        configs = []
        for extractor in self._extractors.values():
            configs.extend(extractor.get_configs())
        return configs

    def __iter__(self):
        return iter(self._extractors.values())

    def __len__(self):
        return len(self._extractors)

    @classmethod
    def create_default(cls) -> 'ExtractorRegistry':
        return (
            cls()
            .register(MusiCNNExtractor())
            .register(VGGExtractor())
            .register(WhisperExtractor())
            .register(VGGishExtractor())
            .register(WhisperContrastiveExtractor())
            .register(WhisperContrastiveMultiLabelExtractor())
            .register(WhisperContrastiveHybridExtractor())
            .register(MusiCNNContrastiveHybridExtractor())
            .register(MusiCNNContrastiveExtractor())
            .register(MusiCNNMultiSignalExtractor())
            .register(VGGContrastiveExtractor())
            .register(VGGContrastiveProj512Extractor())
        )

    @classmethod
    def create_visual_extractor(cls) -> 'ExtractorRegistry':
        return (cls()
        .register(MusiCNNContrastiveHybridExtractor())
        .register(MusiCNNMultiSignalExtractor())
        .register(MusiCNNContrastiveExtractor())
        )
