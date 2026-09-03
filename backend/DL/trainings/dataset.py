"""
PyTorch Dataset for MTG-Jamendo audio genre classification.

Handles loading audio files and genre labels from the MTG-Jamendo dataset
for supervised contrastive learning.
"""

import torch
import librosa
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

# Suppress librosa and soundfile warnings
warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')


class MTGJamendoDataset(Dataset):
    """
    MTG-Jamendo Dataset for audio genre classification.

    Loads audio files and their corresponding genre labels from the
    MTG-Jamendo dataset splits.
    """

    def __init__(
        self,
        tsv_file: str,
        audio_dir: str,
        sample_rate: int = 16000,
        duration: Optional[float] = None,
        transform=None
    ):
        """
        Initialize MTG-Jamendo dataset.

        Args:
            tsv_file: Path to TSV file with track metadata and labels
            audio_dir: Root directory containing audio files
            sample_rate: Target sample rate for audio (default: 16000 Hz)
            duration: Maximum duration in seconds (None = use full track)
            transform: Optional transform to apply to audio
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform

        # Load metadata from TSV file
        # Some rows have multiple tags separated by tabs, so we need to handle variable columns
        self.metadata = self._load_tsv_with_variable_tags(tsv_file)

        # Parse genre labels
        self._parse_genres()

        # Create genre to index mapping
        self.unique_genres = sorted(list(set(self.all_genres)))
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.unique_genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}

        print(f"Loaded {len(self)} tracks with {len(self.unique_genres)} unique genres")
        print(f"Genres: {', '.join(self.unique_genres[:10])}...")

    def _load_tsv_with_variable_tags(self, tsv_file):
        """
        Load TSV file where the TAGS column may contain tab-separated values.

        Returns:
            DataFrame with columns: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION, TAGS
        """
        data = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            # Read header
            f.readline().strip().split('\t')

            # Read data rows
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    # First 5 columns: TRACK_ID, ARTIST_ID, ALBUM_ID, PATH, DURATION
                    # Remaining columns are all tags (may be multiple, tab-separated)
                    row = {
                        'TRACK_ID': parts[0],
                        'ARTIST_ID': parts[1],
                        'ALBUM_ID': parts[2],
                        'PATH': parts[3],
                        'DURATION': float(parts[4]),
                        'TAGS': '\t'.join(parts[5:])  # Join all tag columns
                    }
                    data.append(row)

        return pd.DataFrame(data)

    def _parse_genres(self):
        """Parse genre tags from the TAGS column."""
        self.genres = []
        self.all_genres = []

        for tags in self.metadata['TAGS']:
            # Tags can be tab-separated, e.g., "genre---rock\tgenre---pop"
            tag_list = str(tags).split('\t') if isinstance(tags, str) else [str(tags)]

            # Extract genres (remove "genre---" prefix)
            track_genres = []
            for tag in tag_list:
                if tag.startswith('genre---'):
                    genre = tag.replace('genre---', '')
                    track_genres.append(genre)
                    self.all_genres.append(genre)

            # For contrastive learning, use the first genre as primary label
            # TODO: Implement multi-label approach
            self.genres.append(track_genres[0] if track_genres else 'unknown')

    def __len__(self) -> int:
        """Return the number of tracks in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single item from the dataset.

        Args:
            idx: Index of the track

        Returns:
            audio: Audio waveform tensor [num_samples]
            label: Genre label as integer index
        """
        # Get track metadata
        row = self.metadata.iloc[idx]
        path = row['PATH']

        # Construct full audio path
        audio_path = self.audio_dir / path
        if not audio_path.exists():
            # Try with .low.mp3 extension
            audio_path_low = self.audio_dir / path.replace('.mp3', '.low.mp3')
            if audio_path_low.exists():
                audio_path = audio_path_low
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_path} or {audio_path_low}")

        # Load audio
        try:
            waveform, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,  # Target sample rate
                mono=True  # Convert to mono
            )
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zero tensor on error
            num_samples = int(self.duration * self.sample_rate) if self.duration else 480000
            return torch.zeros(num_samples), 0

        # Trim or pad to fixed duration if specified
        if self.duration is not None:
            target_length = int(self.duration * self.sample_rate)
            if waveform.shape[0] > target_length:
                waveform = waveform[:target_length]
            elif waveform.shape[0] < target_length:
                padding = target_length - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Get genre label
        genre = self.genres[idx]
        label = self.genre_to_idx.get(genre, 0)

        return waveform, label

    def get_genre_distribution(self) -> dict:
        """Get the distribution of genres in the dataset."""
        from collections import Counter
        return Counter(self.genres)

    def get_weights_for_balanced_sampling(self) -> torch.Tensor:
        """
        Calculate sample weights for balanced sampling.

        Useful for creating a WeightedRandomSampler to handle class imbalance.

        Returns:
            weights: Tensor of weights for each sample
        """
        genre_counts = self.get_genre_distribution()
        weights = []

        for genre in self.genres:
            # Weight inversely proportional to class frequency
            weight = 1.0 / genre_counts[genre]
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float)


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function to handle variable-length audio.

    Pads all audio to the same length within the batch.

    Args:
        batch: List of (audio, label) tuples

    Returns:
        audios: Batched audio tensor [batch_size, max_length]
        labels: Batched label tensor [batch_size]
    """
    audios, labels = zip(*batch)

    # Find max length in batch
    max_length = max(audio.shape[0] for audio in audios)

    # Pad all audio to max length
    padded_audios = []
    for audio in audios:
        if audio.shape[0] < max_length:
            padding = max_length - audio.shape[0]
            audio = torch.nn.functional.pad(audio, (0, padding))
        padded_audios.append(audio)

    # Stack into batch tensors
    audios_batch = torch.stack(padded_audios)
    labels_batch = torch.tensor(labels, dtype=torch.long)

    return audios_batch, labels_batch


class MTGJamendoMultiLabelDataset(MTGJamendoDataset):
    """
    MTG-Jamendo dataset returning multi-hot genre vectors instead of single-label integers.

    Inherits all audio loading from MTGJamendoDataset.
    __getitem__ returns (audio, multi_hot_vector) where multi_hot_vector is float [num_classes].
    """

    def _parse_genres(self):
        """Parse ALL genre tags per track (not just the first)."""
        self.genres = []       # primary label (kept for compat)
        self.all_genres = []
        self.track_genres = []  # list of lists: all genres per track

        for tags in self.metadata['TAGS']:
            tag_list = str(tags).split('\t') if isinstance(tags, str) else [str(tags)]
            track_genre_list = [
                tag.replace('genre---', '')
                for tag in tag_list
                if tag.startswith('genre---')
            ]
            self.track_genres.append(track_genre_list)
            self.all_genres.extend(track_genre_list)
            self.genres.append(track_genre_list[0] if track_genre_list else 'unknown')

    def __getitem__(self, idx: int):
        """
        Returns:
            audio:     [num_samples] float tensor
            label_vec: [num_classes] multi-hot float tensor
        """
        audio, _ = super().__getitem__(idx)

        # Build multi-hot vector
        label_vec = torch.zeros(len(self.unique_genres), dtype=torch.float32)
        for genre in self.track_genres[idx]:
            if genre in self.genre_to_idx:
                label_vec[self.genre_to_idx[genre]] = 1.0

        return audio, label_vec


def collate_fn_multilabel(batch):
    """Collate for multi-hot float labels."""
    audios, labels = zip(*batch)
    max_length = max(a.shape[0] for a in audios)
    padded = []
    for a in audios:
        if a.shape[0] < max_length:
            a = torch.nn.functional.pad(a, (0, max_length - a.shape[0]))
        padded.append(a)
    return torch.stack(padded), torch.stack(labels)


def create_dataloaders_multilabel(
    train_tsv: str,
    val_tsv: str,
    test_tsv: str,
    audio_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 30.0,
    balanced_sampling: bool = False
):
    """
    Create dataloaders returning multi-hot genre vectors for multi-label contrastive learning.
    Same interface as create_dataloaders.
    """
    train_dataset = MTGJamendoMultiLabelDataset(train_tsv, audio_dir, sample_rate, duration)
    val_dataset   = MTGJamendoMultiLabelDataset(val_tsv,   audio_dir, sample_rate, duration)
    test_dataset  = MTGJamendoMultiLabelDataset(test_tsv,  audio_dir, sample_rate, duration)

    train_sampler = None
    shuffle = True
    if balanced_sampling:
        weights = train_dataset.get_weights_for_balanced_sampling()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        shuffle = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=train_sampler, num_workers=num_workers,
        collate_fn=collate_fn_multilabel, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_multilabel, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn_multilabel, pin_memory=True
    )

    dataset_info = {
        'num_classes': len(train_dataset.unique_genres),
        'genre_to_idx': train_dataset.genre_to_idx,
        'idx_to_genre': train_dataset.idx_to_genre,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }
    return train_loader, val_loader, test_loader, dataset_info


def create_dataloaders(
    train_tsv: str,
    val_tsv: str,
    test_tsv: str,
    audio_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 30.0,
    balanced_sampling: bool = False
):
    """
    Create train, validation, and test dataloaders.

    Args:
        train_tsv: Path to training split TSV
        val_tsv: Path to validation split TSV
        test_tsv: Path to test split TSV
        audio_dir: Root directory containing audio files
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        sample_rate: Target sample rate
        duration: Duration to crop/pad audio to (seconds)
        balanced_sampling: Whether to use balanced sampling for training

    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    # Create datasets
    train_dataset = MTGJamendoDataset(
        train_tsv, audio_dir, sample_rate, duration
    )
    val_dataset = MTGJamendoDataset(
        val_tsv, audio_dir, sample_rate, duration
    )
    test_dataset = MTGJamendoDataset(
        test_tsv, audio_dir, sample_rate, duration
    )

    # Create sampler for balanced training if requested
    train_sampler = None
    shuffle = True
    if balanced_sampling:
        weights = train_dataset.get_weights_for_balanced_sampling()
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(weights), replacement=True
        )
        shuffle = False  # Mutually exclusive with sampler

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Dataset info for reference
    dataset_info = {
        'num_classes': len(train_dataset.unique_genres),
        'genre_to_idx': train_dataset.genre_to_idx,
        'idx_to_genre': train_dataset.idx_to_genre,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_distribution': train_dataset.get_genre_distribution()
    }

    return train_loader, val_loader, test_loader, dataset_info

