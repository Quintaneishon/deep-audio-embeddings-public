"""
MERT-based audio embedding extractor.

This module wraps the MERT (Music Understanding Model with Large-Scale
Self-supervised Training) from Hugging Face to extract audio embeddings.

MERT is specifically pre-trained on music data and captures both acoustic
and musical features through self-supervised learning on 160k hours of music.
Like Whisper, it uses Transformer-based encoding to capture temporal patterns,
but it's specialized for music understanding tasks.

Reference: https://huggingface.co/m-a-p/MERT-v1-95M
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
import warnings
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torchaudio

# MERT audio parameters
SAMPLE_RATE = 24000  # MERT uses 24kHz
MAX_DURATION = 30  # Maximum duration in seconds
MAX_SAMPLES = MAX_DURATION * SAMPLE_RATE


def pad_or_trim_audio(audio: Union[np.ndarray, torch.Tensor],
                     target_length: int = MAX_SAMPLES) -> torch.Tensor:
    """
    Pad or trim audio to target length.

    Args:
        audio: Audio waveform (1D array/tensor)
        target_length: Target length in samples

    Returns:
        Processed audio tensor
    """
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    if audio.dim() > 1:
        audio = audio.squeeze()

    current_length = audio.shape[0]

    if current_length > target_length:
        # Trim
        audio = audio[:target_length]
    elif current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = torch.nn.functional.pad(audio, (0, padding), value=0)

    return audio


def resample_audio(audio: torch.Tensor,
                   orig_sr: int,
                   target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio tensor
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio tensor
    """
    if orig_sr == target_sr:
        return audio

    try:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        return resampler(audio)
    except ImportError:
        warnings.warn(
            "Audio sample rate should match MERT's expected rate (24kHz)."
        )
        return audio


class MERTEmbedding(nn.Module):
    """
    MERT-based embedding extractor for music audio.

    Extracts final layer embeddings (high-level musical features),
    temporally pooled to get fixed-size representations.
    """

    def __init__(self,
                 model_name: str = 'm-a-p/MERT-v1-95M',
                 intermediate_layer: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 trust_remote_code: bool = True,
                 device=None):
        """
        Initialize MERT embedding extractor.

        Args:
            model_name: Hugging Face model identifier
                       Options: 'm-a-p/MERT-v1-95M' (95M params)
                               'm-a-p/MERT-v1-330M' (330M params)
            intermediate_layer: Which layer to use for intermediate features
                              (default: layer at 1/3 depth)
            cache_dir: Directory to cache downloaded models (default: ~/.cache/huggingface)
            trust_remote_code: Whether to trust remote code from HF
        """
        super(MERTEmbedding, self).__init__()

        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = SAMPLE_RATE

        # Load feature extractor (preprocessor)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # Load MERT model
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            output_hidden_states=True,  # Need intermediate layers
        )
        self.model.to(self.device)
        self.model.eval()

        # Get model configuration
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers


    def preprocess_audio(self, audio: torch.Tensor) -> dict:
        """
        Preprocess audio for MERT model.

        Args:
            audio: Audio tensor [batch_size, num_samples] or [num_samples]

        Returns:
            Dictionary with preprocessed inputs
        """
        # Handle batch dimension
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        batch_size = audio.shape[0]
        processed_audios = []

        for i in range(batch_size):
            # Get single audio sample
            single_audio = audio[i].cpu().numpy()

            # Pad or trim to max length
            single_audio = pad_or_trim_audio(
                torch.from_numpy(single_audio),
                MAX_SAMPLES
            ).numpy()

            processed_audios.append(single_audio)

        # Process with feature extractor
        inputs = self.processor(
            processed_audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )

        return inputs

    def extract_features(self, audio: torch.Tensor):
        """
        Extract features from MERT encoder at multiple layers.

        Args:
            audio: Audio waveform tensor [batch, num_samples]

        Returns:
            final_features: Final layer output [batch, seq_len, hidden_size]
            intermediate_features: Intermediate layer output [batch, seq_len, hidden_size]
        """
        # Preprocess audio
        inputs = self.preprocess_audio(audio)

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get hidden states from all layers
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is the embedding output, 1 to num_layers are the layer outputs
        hidden_states = outputs.hidden_states

        # Final layer features (last hidden state)
        final_features = hidden_states[-1]  # [batch, seq_len, hidden_size]

        # Intermediate layer features
        # Add 1 because index 0 is embedding layer
        intermediate_layer = getattr(self, 'intermediate_layer', None)
        if intermediate_layer is not None:
            intermediate_features = hidden_states[intermediate_layer + 1]
        else:
            intermediate_features = None

        return final_features, intermediate_features

    def temporal_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal pooling over sequence dimension.

        Two strategies:
        1. Mean pooling (used for final embedding)
        2. Max pooling (alternative)

        Args:
            features: [batch, seq_len, feature_dim]

        Returns:
            pooled: [batch, feature_dim]
        """
        # Mean pooling over time dimension
        pooled = torch.mean(features, dim=1)  # [batch, feature_dim]

        return pooled

    def temporal_pooling_concat(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal pooling with concatenation of max and mean.

        Args:
            features: [batch, seq_len, feature_dim]

        Returns:
            pooled: [batch, feature_dim * 2]
        """
        # Max pooling over time
        max_pool = torch.max(features, dim=1)[0]  # [batch, feature_dim]

        # Mean pooling over time
        mean_pool = torch.mean(features, dim=1)  # [batch, feature_dim]

        # Concatenate
        pooled = torch.cat([max_pool, mean_pool], dim=1)  # [batch, feature_dim * 2]

        return pooled

    def forward(self, x: torch.Tensor):
        """
        Extract embeddings from audio.

        Args:
            x: Raw audio waveform tensor [batch, num_samples]
               Expected sample rate: 24kHz (MERT's native rate)
               If your audio is at 16kHz, resample to 24kHz first

        Returns:
            embeddings: [batch, hidden_size] - final layer features
        """
        # Extract features from multiple layers
        final_features, _ = self.extract_features(x)

        # Apply temporal pooling to get fixed-size representations
        embeddings = self.temporal_pooling(final_features)  # [batch, hidden_size]

        return embeddings

    def get_embedding_dim(self) -> int:
        """
        Get the dimension of the output embedding.

        Returns:
            Embedding dimension
        """
        return self.hidden_size


