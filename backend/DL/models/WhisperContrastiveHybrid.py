"""
Whisper-based hybrid contrastive learning model (genre + BPM).

Trained with HybridSupConLoss which weights positive pairs by a convex
combination of genre-match similarity and BPM Gaussian similarity:
    similarity = alpha * genre_match + (1 - alpha) * exp(-|bpm_i - bpm_j|^2 / (2*sigma^2))

Architecturally identical to WhisperContrastive; the distinct class name
allows checkpoints from hybrid training to be loaded unambiguously.
"""

from backend.DL.models.WhisperContrastive import WhisperContrastive


class WhisperContrastiveHybrid(WhisperContrastive):
    """
    Whisper encoder + projection head trained with hybrid SupCon loss.

    The architecture is the same as WhisperContrastive:
      - Frozen Whisper encoder
      - Trainable MLP projection head: n_audio_state → projection_dim
      - L2-normalised output embeddings

    The distinction from the base class is purely for checkpoint identification:
    weights saved by train_contrastive_hybrid.py are loaded here so that
    the extractor registry can tell them apart from single-label or multi-label runs.
    """

    def __init__(self, model_name='base', projection_dim=128,
                 intermediate_layer=None, device=None):
        super().__init__(
            model_name=model_name,
            projection_dim=projection_dim,
            intermediate_layer=intermediate_layer,
            device=device,
        )
