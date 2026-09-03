"""
Whisper-based contrastive learning model for genre classification.

This module extends the WhisperEmbedding model by freezing the encoder
and adding a trainable projection head for supervised contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.DL.models.Whisper import WhisperEmbedding, log_mel_spectrogram, pad_or_trim

class WhisperContrastive(WhisperEmbedding):
    """
    Whisper-based contrastive learning model.

    Architecture:
    1. Frozen Whisper encoder (pre-trained)
    2. Raw intermediate layer features (after pooling)
    3. Trainable projection head: intermediate_dim → 128
    4. L2 normalization for contrastive learning
    """

    def __init__(self, model_name='base', projection_dim=128, intermediate_layer=None, device=None):
        """
        Initialize Whisper contrastive learning model.

        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium')
            projection_dim: Dimension of the projection head output (default: 128)
            intermediate_layer: Which encoder layer to extract for intermediate features
            device: Device to load model on (default: cuda if available, else cpu)
        """
        super(WhisperContrastive, self).__init__(
            model_name=model_name,
            intermediate_layer=intermediate_layer,
            device=device
        )

        self.projection_dim = projection_dim

        # Create trainable projection head BEFORE freezing
        # Input: n_audio_state (512 for base model)
        # Output: projection_dim (128 by default)
        self.projection_head = nn.Sequential(
            nn.Linear(self.n_audio_state, self.n_audio_state),
            nn.ReLU(),
            nn.Linear(self.n_audio_state, projection_dim)
        )

        # Initialize projection head
        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.projection_head.to(self.device)

        # Freeze all Whisper encoder parameters (after creating projection head)
        self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze all parameters in the Whisper encoder."""
        for param in self.whisper_model.parameters():
            param.requires_grad = False

    def forward_features(self, x):
        """
        Extract normalized embeddings from audio for contrastive learning.
        Uses raw intermediate layer features.

        Args:
            x: Raw audio waveform tensor [batch, num_samples] at 16kHz

        Returns:
            embeddings: [batch, projection_dim] - L2 normalized embeddings
        """
        batch_size = x.shape[0]

        # Convert audio to mel spectrogram (Whisper format)
        mel_list = []
        for i in range(batch_size):
            # Pad or trim to 30 seconds
            audio_padded = pad_or_trim(x[i].cpu().numpy())

            # Convert to mel spectrogram
            mel = log_mel_spectrogram(audio_padded, n_mels=self.n_mels)
            mel_list.append(mel)

        # Stack into batch
        mel = torch.stack(mel_list).to(self.device)  # [batch, n_mels, n_frames]

        # Extract features from intermediate encoder layer (frozen)
        with torch.no_grad():
            _, intermediate_features = self.extract_encoder_features(mel)
            # intermediate_features: [batch, time_frames, n_audio_state]

            # Average pooling over time to get fixed-size representation
            pooled_features = torch.mean(intermediate_features, dim=1)  # [batch, n_audio_state]

        # Apply projection head to raw intermediate features (trainable)
        embeddings = self.projection_head(pooled_features)  # [batch, projection_dim]

        # L2 normalization: z_i = z_i / ||z_i||
        # This ensures that cosine similarity = dot product
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, x):
        """
        Forward pass for contrastive learning.

        Args:
            x: Raw audio waveform tensor [batch, num_samples] at 16kHz

        Returns:
            embeddings: [batch, projection_dim] - L2 normalized embeddings
        """
        return self.forward_features(x)

    def get_trainable_parameters(self):
        """Return only the trainable parameters (projection head)."""
        return self.projection_head.parameters()

