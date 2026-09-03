"""
VGGish Model with Pre-trained Weights from Google AudioSet

Based on the official VGGish implementation:
https://github.com/tensorflow/models/tree/master/research/audioset/vggish

This implementation uses torchvggish package for pretrained weights.
"""

import torch
import torchaudio
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import os
import urllib.request
import torchvggish
from backend import config


class VGGish(nn.Module):
    """
    VGGish model with Google's pretrained weights.

    Architecture matches the original VGGish from AudioSet.
    Outputs 128-dimensional embeddings.
    """

    # VGGish parameters (from original implementation)
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = 64
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset for log mel spectrogram
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example is 0.96 seconds
    EXAMPLE_HOP_SECONDS = 0.96     # No overlap

    def __init__(self,
                 pretrained: bool = True,
                 weights_path: Optional[str] = None,
                 pca_path: Optional[str] = None,
                 use_pca: bool = False,
                 n_class: int = 50):
        """
        Initialize VGGish model.

        Args:
            pretrained: If True, load pretrained weights from torchvggish
            weights_path: Path to custom weights file (if not using pretrained)
            pca_path: Path to PCA parameters file
            use_pca: If True, apply PCA postprocessing to embeddings
            n_class: Number of output classes (for optional classifier head)
        """
        super(VGGish, self).__init__()

        self.use_pca = use_pca
        self.n_class = n_class

        # Mel spectrogram parameters
        self.n_fft = int(self.STFT_WINDOW_LENGTH_SECONDS * self.SAMPLE_RATE)
        self.hop_length = int(self.STFT_HOP_LENGTH_SECONDS * self.SAMPLE_RATE)

        # Build the model
        self._build_model()

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained_weights(weights_path, pca_path)

        # Optional classifier head for fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_class)
        )

    def _build_model(self):
        """Build VGGish architecture."""

        # Mel spectrogram layer
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.NUM_MEL_BINS,
            f_min=self.MEL_MIN_HZ,
            f_max=self.MEL_MAX_HZ
        )

        # Convolutional layers (matching original VGGish)
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 6))

        # Fully connected layers (embeddings)
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),  # For 64x96 input
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def _download_weights(self, url: str, filename: str) -> str:
        """Download pretrained weights if not present."""
        weights_dir = config.WEIGHTS_DIR / 'vggish'
        os.makedirs(weights_dir, exist_ok=True)

        weights_path = weights_dir / filename

        if not os.path.exists(weights_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, weights_path)
                print(f"Downloaded to {weights_path}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return None

        return weights_path

    def _load_pretrained_weights(self, weights_path: Optional[str] = None, pca_path: Optional[str] = None):
        """Load pretrained VGGish weights."""
        try:
            # Load pretrained VGGish model
            vggish_model = torchvggish.vggish()

            # Copy weights from pretrained model
            self.features.load_state_dict(vggish_model.features.state_dict())
            self.embeddings.load_state_dict(vggish_model.embeddings.state_dict())

            # Load PCA parameters if requested
            if self.use_pca and hasattr(vggish_model, 'pca'):
                self.pca = vggish_model.pca
                print("✓ Loaded PCA parameters")

        except ImportError:
            print("torchvggish not installed. Attempting manual download...")

    def forward(self, x: torch.Tensor, return_embedding: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VGGish.

        Args:
            x: Input audio waveform [batch, samples] or spectrogram [batch, 1, freq, time]
            return_embedding: If True, return embeddings. If False, return classifier output.

        Returns:
            embeddings: [batch, 128] embedding vectors
            classifier_output: [batch, n_class] classification logits (if classifier is trained)
        """
        # If input is waveform, convert to mel spectrogram
        if x.dim() == 2:  # [batch, samples]
            x = self._audio_to_mel(x)

        # Convolutional features
        x = self.features(x)

        # Apply adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Embeddings (128-dimensional)
        embedding = self.embeddings(x)

        # Apply PCA if requested
        if self.use_pca and hasattr(self, 'pca'):
            embedding = self.pca(embedding)

        # Optional classifier
        classifier_output = self.classifier(embedding)

        if return_embedding:
            return classifier_output, embedding
        else:
            return classifier_output

    def _audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio waveform to log mel spectrogram.

        Args:
            audio: [batch, samples]

        Returns:
            mel: [batch, 1, n_mels, time_frames]
        """
        # Compute mel spectrogram
        mel = self.mel_spec(audio)

        # Convert to log scale (with offset to avoid log(0))
        mel = torch.log(mel + self.LOG_OFFSET)

        # Add channel dimension
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        return mel

    def extract_features(self, audio_path: str, device: str = 'cpu') -> np.ndarray:
        """
        Extract VGGish embeddings from an audio file.

        Args:
            audio_path: Path to audio file
            device: Device to run inference on ('cpu' or 'cuda')

        Returns:
            embeddings: [n_segments, 128] array of embeddings
        """
        import librosa

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        # Extract embeddings
        self.eval()
        with torch.no_grad():
            _, embeddings = self.forward(audio_tensor)

        return embeddings.cpu().numpy()


def load_vggish_pretrained(pretrained: bool = True,
                          n_class: int = 50,
                          device: str = 'cpu') -> VGGish:
    """
    Convenience function to load pretrained VGGish model.

    Args:
        pretrained: Load pretrained weights
        n_class: Number of output classes
        device: Device to load model on

    Returns:
        VGGish model
    """
    model = VGGish(pretrained=pretrained, n_class=n_class)
    model = model.to(device)
    model.eval()
    return model

