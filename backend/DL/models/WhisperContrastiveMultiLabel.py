"""
Whisper-based multi-label contrastive learning model.

Trained with MultiLabelSupConLoss using Jaccard-weighted positive pairs
from all genre tags per track (multi-hot labels).
Architecturally identical to WhisperContrastive; the distinct class name
allows checkpoints from multi-label training to be loaded unambiguously.
"""

from backend.DL.models.WhisperContrastive import WhisperContrastive


class WhisperContrastiveMultiLabel(WhisperContrastive):
    """
    Whisper encoder + projection head trained with multi-label SupCon loss.

    The architecture is the same as WhisperContrastive:
      - Frozen Whisper encoder
      - Trainable MLP projection head: n_audio_state → projection_dim
      - L2-normalised output embeddings

    The distinction from the base class is purely for checkpoint identification:
    weights saved by train_contrastive_multilabel.py are loaded here so that
    the extractor registry can tell them apart from single-label or hybrid runs.
    """

    def __init__(self, model_name='base', projection_dim=128,
                 intermediate_layer=None, device=None):
        super().__init__(
            model_name=model_name,
            projection_dim=projection_dim,
            intermediate_layer=intermediate_layer,
            device=device,
        )
