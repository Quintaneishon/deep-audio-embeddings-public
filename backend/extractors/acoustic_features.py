"""
Acoustic Features Extractor for Lightweight Adapter.

Extracts 6 quantitative acoustic features from audio:
1. Spectral Centroid (brightness)
2. Spectral Bandwidth (timbral richness)
3. Spectral Rolloff (high-frequency content)
4. Zero-Crossing Rate (percussiveness)
5. RMS Energy (loudness)
6. Tempo (BPM - critical for electronic music)
"""

import numpy as np
import librosa
import torch
import warnings
from pathlib import Path
import h5py
import json

warnings.filterwarnings('ignore', category=FutureWarning)


class AcousticFeatureExtractor:
    """
    Extracts and normalizes acoustic features from audio.

    Features are normalized using z-score normalization based on
    statistics computed from the training set.
    """

    def __init__(self, sample_rate=16000, cache_dir=None):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Sample rate for audio processing
            cache_dir: Directory to cache computed features
        """
        self.sample_rate = sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Normalization statistics (will be computed from training set)
        self.feature_mean = None
        self.feature_std = None

        # Feature names for reference
        self.feature_names = [
            'spectral_centroid',
            'spectral_bandwidth',
            'spectral_rolloff',
            'zero_crossing_rate',
            'rms_energy',
            'tempo'
        ]

    def extract_features(self, audio, normalize=True):
        """
        Extract 6 acoustic features from audio.

        Args:
            audio: Audio waveform (numpy array or torch tensor) or path to file
            normalize: Whether to apply z-score normalization

        Returns:
            features: Tensor of shape [6] with extracted features
        """
        # Load audio if path is provided
        if isinstance(audio, (str, Path)):
            y, sr = librosa.load(audio, sr=self.sample_rate, mono=True, res_type='kaiser_fast')
        elif isinstance(audio, torch.Tensor):
            y = audio.cpu().numpy()
            sr = self.sample_rate
        else:
            y = audio
            sr = self.sample_rate

        # Extract features
        features = []

        # 1. Spectral Centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(centroid))

        # 2. Spectral Bandwidth (timbral richness)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.append(np.mean(bandwidth))

        # 3. Spectral Rolloff (high-frequency content)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        features.append(np.mean(rolloff))

        # 4. Zero-Crossing Rate (percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))

        # 5. RMS Energy (loudness)
        rms = librosa.feature.rms(y=y)[0]
        features.append(np.mean(rms))

        # 6. Tempo (BPM - critical for electronic music)
        # Use onset_envelope for more robust tempo detection
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        features.append(tempo)

        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)

        # Normalize if requested and stats are available
        if normalize and self.feature_mean is not None:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        return features

    def extract_batch(self, audio_batch, normalize=True):
        """
        Extract features from batch of audio.

        Args:
            audio_batch: Tensor of shape [batch, num_samples]
            normalize: Whether to normalize features

        Returns:
            features_batch: Tensor of shape [batch, 6]
        """
        batch_size = audio_batch.shape[0]
        features_list = []

        for i in range(batch_size):
            features = self.extract_features(audio_batch[i], normalize=normalize)
            features_list.append(features)

        return torch.stack(features_list)

    def compute_normalization_stats(self, audio_files):
        """
        Compute mean and std for normalization from training set.

        Args:
            audio_files: List of audio file paths

        Returns:
            feature_mean: Mean of each feature
            feature_std: Std of each feature
        """
        print(f"Computing normalization statistics from {len(audio_files)} files...")

        all_features = []

        for i, audio_file in enumerate(audio_files):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")

            try:
                features = self.extract_features(audio_file, normalize=False)
                all_features.append(features.numpy())
            except Exception as e:
                print(f"  Warning: Failed to extract features from {audio_file}: {e}")
                continue

        # Compute statistics
        all_features = np.array(all_features)
        self.feature_mean = torch.tensor(np.mean(all_features, axis=0), dtype=torch.float32)
        self.feature_std = torch.tensor(np.std(all_features, axis=0), dtype=torch.float32)

        print(f"\nNormalization statistics computed:")
        for i, name in enumerate(self.feature_names):
            print(f"  {name:25s}: mean={self.feature_mean[i]:.2f}, std={self.feature_std[i]:.2f}")

        return self.feature_mean, self.feature_std

    def save_stats(self, filepath):
        """Save normalization statistics to file."""
        stats = {
            'feature_mean': self.feature_mean.tolist(),
            'feature_std': self.feature_std.tolist(),
            'feature_names': self.feature_names
        }

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Normalization statistics saved to {filepath}")

    def load_stats(self, filepath):
        """Load normalization statistics from file."""
        with open(filepath, 'r') as f:
            stats = json.load(f)

        self.feature_mean = torch.tensor(stats['feature_mean'], dtype=torch.float32)
        self.feature_std = torch.tensor(stats['feature_std'], dtype=torch.float32)

    def cache_features(self, audio_files, cache_file):
        """
        Pre-compute and cache features for all audio files.

        Args:
            audio_files: List of audio file paths
            cache_file: Path to HDF5 cache file
        """
        print(f"Caching features for {len(audio_files)} files to {cache_file}...")

        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache_file, 'w') as f:
            for i, audio_file in enumerate(audio_files):
                if (i + 1) % 100 == 0:
                    print(f"  Cached {i + 1}/{len(audio_files)} files")

                try:
                    features = self.extract_features(audio_file, normalize=False)

                    # Use filename as key
                    key = Path(audio_file).name
                    f.create_dataset(key, data=features.numpy())

                except Exception as e:
                    print(f"  Warning: Failed to cache {audio_file}: {e}")
                    continue

        print(f"Feature caching complete: {cache_file}")

    def load_cached_features(self, filename, cache_file, normalize=True):
        """
        Load pre-computed features from cache.

        Args:
            filename: Name of audio file
            cache_file: Path to HDF5 cache file
            normalize: Whether to normalize features

        Returns:
            features: Tensor of shape [6]
        """
        key = Path(filename).name

        with h5py.File(cache_file, 'r') as f:
            if key not in f:
                raise KeyError(f"Features for {filename} not found in cache")

            features = torch.tensor(f[key][:], dtype=torch.float32)

        # Normalize if requested
        if normalize and self.feature_mean is not None:
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)

        return features


def extract_features_from_directory(audio_dir, output_file, sample_rate=16000):
    """
    Helper function to extract features from all audio files in a directory.

    Args:
        audio_dir: Directory containing audio files
        output_file: Output HDF5 file for cached features
        sample_rate: Sample rate for processing
    """
    from glob import glob

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.ogg', '*.flac']:
        audio_files.extend(glob(str(Path(audio_dir) / '**' / ext), recursive=True))

    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    extractor = AcousticFeatureExtractor(sample_rate=sample_rate)

    # Compute normalization stats
    extractor.compute_normalization_stats(audio_files)

    # Save stats
    stats_file = Path(output_file).parent / 'feature_stats.json'
    extractor.save_stats(stats_file)

    # Cache features
    extractor.cache_features(audio_files, output_file)

    print(f"\nFeature extraction complete!")
    print(f"  Cache: {output_file}")
    print(f"  Stats: {stats_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract acoustic features from audio files')
    parser.add_argument('--audio-dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--output', type=str, default='ML/features_cache/acoustic_features.h5',
                       help='Output HDF5 file for cached features')
    parser.add_argument('--sample-rate', type=int, default=16000,
                       help='Sample rate for audio processing')

    args = parser.parse_args()

    extract_features_from_directory(args.audio_dir, args.output, args.sample_rate)

