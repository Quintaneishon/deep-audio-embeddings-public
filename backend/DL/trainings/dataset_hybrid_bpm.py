"""
PyTorch Dataset for MTG-Jamendo with multi-hot genres and BPM labels.

Returns (audio, genre_multi_hot, bpm) for use with HybridJaccardBPMLoss.
"""

import json
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')


class MTGJamendoHybridDataset(Dataset):
    """
    MTG-Jamendo Dataset returning (audio, genre_multi_hot, bpm).

    All genre tags per track are encoded as a multi-hot vector.
    BPM is loaded from a pre-calculated cache JSON.
    """

    def __init__(
        self,
        tsv_file: str,
        audio_dir: str,
        bpm_cache_path: str,
        sample_rate: int = 16000,
        duration: Optional[float] = None,
        transform=None,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.transform = transform

        self.metadata = self._load_tsv(tsv_file)

        with open(bpm_cache_path, 'r') as f:
            self.bpm_cache = json.load(f)

        self._parse_genres()
        self._parse_bpms()

        print(f"[HybridDataset] {len(self)} tracks | {self.num_genres} genres | "
              f"BPM hit rate: {self._bpm_hit_rate:.1%}")

    # ------------------------------------------------------------------
    # TSV loading
    # ------------------------------------------------------------------

    def _load_tsv(self, tsv_file: str) -> pd.DataFrame:
        data = []
        with open(tsv_file, 'r', encoding='utf-8') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    data.append({
                        'TRACK_ID': parts[0],
                        'ARTIST_ID': parts[1],
                        'ALBUM_ID': parts[2],
                        'PATH': parts[3],
                        'DURATION': float(parts[4]),
                        'TAGS': '\t'.join(parts[5:]),
                    })
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Genre parsing -> multi-hot
    # ------------------------------------------------------------------

    def _parse_genres(self):
        all_genre_tags = set()
        self._track_genres: List[List[str]] = []

        for tags in self.metadata['TAGS']:
            tag_list = str(tags).split('\t') if isinstance(tags, str) else [str(tags)]
            track_genres = [
                t.replace('genre---', '')
                for t in tag_list
                if t.startswith('genre---')
            ]
            if not track_genres:
                track_genres = ['unknown']
            self._track_genres.append(track_genres)
            all_genre_tags.update(track_genres)

        self.unique_genres = sorted(all_genre_tags)
        self.genre_to_idx = {g: i for i, g in enumerate(self.unique_genres)}
        self.num_genres = len(self.unique_genres)

        self.genre_vectors = torch.zeros(len(self.metadata), self.num_genres)
        for i, genres in enumerate(self._track_genres):
            for g in genres:
                self.genre_vectors[i, self.genre_to_idx[g]] = 1.0

    # ------------------------------------------------------------------
    # BPM parsing
    # ------------------------------------------------------------------

    def _parse_bpms(self):
        bpm_values = []
        self._bpm_hits = 0

        for path in self.metadata['PATH']:
            val = self.bpm_cache.get(path)
            if val is not None:
                bpm_values.append(float(val))
                self._bpm_hits += 1

        self.default_bpm = float(np.median(bpm_values)) if bpm_values else 120.0
        self._bpm_hit_rate = self._bpm_hits / len(self.metadata) if len(self.metadata) else 0.0

        bpms = []
        for path in self.metadata['PATH']:
            val = self.bpm_cache.get(path)
            bpms.append(float(val) if val is not None else self.default_bpm)
        self.bpms = torch.tensor(bpms, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        row = self.metadata.iloc[idx]
        path = row['PATH']

        audio_path = self.audio_dir / path
        if not audio_path.exists():
            alt = self.audio_dir / path.replace('.mp3', '.low.mp3')
            if alt.exists():
                audio_path = alt
            else:
                raise FileNotFoundError(f"Audio file not found: {audio_path} or {alt}")

        try:
            waveform, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            waveform = torch.from_numpy(waveform).float()
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            num_samples = int(self.duration * self.sample_rate) if self.duration else 480000
            waveform = torch.zeros(num_samples)

        if self.duration is not None:
            target_len = int(self.duration * self.sample_rate)
            if waveform.shape[0] > target_len:
                waveform = waveform[:target_len]
            elif waveform.shape[0] < target_len:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))

        if self.transform:
            waveform = self.transform(waveform)

        genre_vec = self.genre_vectors[idx]
        bpm = self.bpms[idx].item()

        return waveform, genre_vec, bpm


def collate_fn_hybrid(
    batch: List[Tuple[torch.Tensor, torch.Tensor, float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad audio to max length in batch, stack genre vectors and BPMs."""
    audios, genres, bpms = zip(*batch)

    max_len = max(a.shape[0] for a in audios)
    padded = []
    for a in audios:
        if a.shape[0] < max_len:
            a = torch.nn.functional.pad(a, (0, max_len - a.shape[0]))
        padded.append(a)

    return (
        torch.stack(padded),
        torch.stack(genres),
        torch.tensor(bpms, dtype=torch.float32),
    )


def create_dataloaders_hybrid(
    train_tsv: str,
    val_tsv: str,
    test_tsv: str,
    audio_dir: str,
    bpm_cache_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 30.0,
):
    """Create train/val/test loaders returning (audio, genre_multi_hot, bpm)."""

    datasets = {}
    for split, tsv in [('train', train_tsv), ('val', val_tsv), ('test', test_tsv)]:
        datasets[split] = MTGJamendoHybridDataset(
            tsv_file=tsv,
            audio_dir=audio_dir,
            bpm_cache_path=bpm_cache_path,
            sample_rate=sample_rate,
            duration=duration,
        )

    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_hybrid,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_hybrid,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_hybrid,
        pin_memory=True,
    )

    dataset_info = {
        'num_genres': datasets['train'].num_genres,
        'genre_to_idx': datasets['train'].genre_to_idx,
        'unique_genres': datasets['train'].unique_genres,
        'default_bpm': datasets['train'].default_bpm,
        'train_size': len(datasets['train']),
        'val_size': len(datasets['val']),
        'test_size': len(datasets['test']),
    }

    return train_loader, val_loader, test_loader, dataset_info
