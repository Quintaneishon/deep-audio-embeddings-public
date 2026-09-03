"""
MERT-based multi-label contrastive learning model.

Frozen MERT-v1-95M backbone at layer 6 (768-dim timbral/harmonic features)
+ trainable MLP projection head (768→768→128, L2-normalized).

Trained with MultiLabelSupConLoss using Jaccard-weighted positive pairs
from all genre tags per track (multi-hot labels).

Why layer 6:
  - probe_layers showed layer_06 nDCG=0.532 vs layer_12 nDCG=0.415
  - Earlier layers preserve timbral/harmonic features expert recommended
  - Projection head trained on top fixes hubness geometry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# MERT audio parameters
SAMPLE_RATE = 24000
MAX_DURATION = 30
MAX_SAMPLES = MAX_DURATION * SAMPLE_RATE

BACKBONE_LAYER = 6  # Best DJ balance from layer probe results


class MERTContrastiveMultiLabel(nn.Module):
    """
    MERT encoder (frozen at layer 6) + trainable projection head.

    Architecture:
      1. Frozen MERT-v1-95M backbone (12 transformer layers)
      2. Extract hidden_states[6+1] — layer 6 output, [B, T, 768]
      3. Mean-pool over time → [B, 768]
      4. Projection head: Linear(768→768) + ReLU + Linear(768→128)
      5. L2-normalize → unit-sphere embeddings

    The backbone is fully frozen; only the projection head trains.
    """

    def __init__(
        self,
        model_name: str = 'm-a-p/MERT-v1-95M',
        projection_dim: int = 128,
        backbone_layer: int = BACKBONE_LAYER,
        cache_dir: Optional[str] = None,
        device=None,
    ):
        super().__init__()

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.backbone_layer = backbone_layer
        self.sample_rate = SAMPLE_RATE

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Feature extractor (preprocessor — not a nn.Module, no params)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # MERT backbone
        self.backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.backbone.to(self.device)

        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # 768

        # Trainable projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )
        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.projection_head.to(self.device)

    def _preprocess(self, audio_batch: torch.Tensor) -> dict:
        """
        Preprocess a batch of raw waveforms for MERT.

        Args:
            audio_batch: [B, num_samples] at 24kHz

        Returns:
            dict with input_values tensor ready for the backbone
        """
        audios = []
        for i in range(audio_batch.shape[0]):
            a = audio_batch[i].cpu().numpy()
            # Pad or trim to MAX_SAMPLES
            if len(a) > MAX_SAMPLES:
                a = a[:MAX_SAMPLES]
            elif len(a) < MAX_SAMPLES:
                a = np.pad(a, (0, MAX_SAMPLES - len(a)))
            audios.append(a)

        inputs = self.processor(
            audios,
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding=True,
        )
        return inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized projection embeddings.

        Args:
            x: Raw audio waveform [B, num_samples] at 24kHz

        Returns:
            embeddings: [B, projection_dim] L2-normalized
        """
        inputs = self._preprocess(x)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Backbone is frozen — no grad needed through it
        with torch.no_grad():
            outputs = self.backbone(**inputs)
            # hidden_states: tuple of (num_layers+1) tensors [B, T, 768]
            # index 0 = embedding layer, 1..12 = transformer layer outputs
            layer_features = outputs.hidden_states[self.backbone_layer + 1]
            # Mean pool over time
            pooled = layer_features.mean(dim=1)  # [B, 768]

        # Projection head (trainable)
        projected = self.projection_head(pooled)  # [B, projection_dim]

        # L2 normalize for cosine contrastive loss
        embeddings = F.normalize(projected, p=2, dim=1)
        return embeddings

    def get_trainable_parameters(self):
        """Return only projection head parameters."""
        return self.projection_head.parameters()
