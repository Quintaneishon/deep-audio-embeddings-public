"""
PyTorch Dataset for MTG-Jamendo with multi-signal structural features.

Returns (audio, genre_multi_hot, instrument_multi_hot, bpm,
         time_signature, squareness_score) per track.

Requires a precomputed features cache produced by:
    precompute_structure_features.py

Features in the cache:
    bpm              float
    time_signature   int  (3 or 4)
    squareness_score float in [0, 1]
    status           str  ('ok' | 'error' | 'too_short' | ...)
"""

import json
import warnings
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple

warnings.filterwarnings('ignore', message='PySoundFile failed')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')

# Tags that start with these prefixes are parsed as the corresponding signal
_GENRE_PREFIX      = 'genre---'
_INSTRUMENT_PREFIX = 'instrument---'


class MTGJamendoMultiSignalDataset(Dataset):
    """
    MTG-Jamendo dataset returning all signals needed for MultiSignalLoss.

    Items: (audio, genre_vec, instrument_vec, bpm, time_sig, squareness)
    """

    def __init__(
        self,
        tsv_file: str,
        audio_dir: str,
        features_cache_path: str,
        # Shared vocabularies — pass from the training split so val/test match
        genre_to_idx: Optional[dict] = None,
        instrument_to_idx: Optional[dict] = None,
        sample_rate: int = 16000,
        duration: Optional[float] = 30.0,
        transform=None,
    ):
        self.audio_dir   = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration    = duration
        self.transform   = transform

        self.metadata = self._load_tsv(tsv_file)

        with open(features_cache_path, 'r') as f:
            self.features_cache = json.load(f)

        self._parse_tags(genre_to_idx, instrument_to_idx)
        self._build_feature_tensors()

        cache_hits = sum(
            1 for p in self.metadata['PATH']
            if self.features_cache.get(p, {}).get('status') == 'ok'
        )
        print(
            f"[MultiSignalDataset] {len(self)} tracks | "
            f"{self.num_genres} genres | {self.num_instruments} instruments | "
            f"cache hit rate: {cache_hits / len(self):.1%}"
        )

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
                        'PATH':     parts[3],
                        'DURATION': float(parts[4]),
                        'TAGS':     '\t'.join(parts[5:]),
                    })
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Tag parsing  →  multi-hot genre + instrument vectors
    # ------------------------------------------------------------------

    def _parse_tags(self, genre_to_idx, instrument_to_idx):
        all_genres      = set()
        all_instruments = set()
        track_genres:      List[List[str]] = []
        track_instruments: List[List[str]] = []

        for tags in self.metadata['TAGS']:
            tag_list = str(tags).split('\t') if isinstance(tags, str) else [str(tags)]

            genres = [
                t.replace(_GENRE_PREFIX, '')
                for t in tag_list if t.startswith(_GENRE_PREFIX)
            ] or ['unknown']

            instruments = [
                t.replace(_INSTRUMENT_PREFIX, '')
                for t in tag_list if t.startswith(_INSTRUMENT_PREFIX)
            ] or ['unknown']

            track_genres.append(genres)
            track_instruments.append(instruments)
            all_genres.update(genres)
            all_instruments.update(instruments)

        # Build or reuse shared vocabularies
        if genre_to_idx is None:
            self.genre_to_idx = {g: i for i, g in enumerate(sorted(all_genres))}
        else:
            self.genre_to_idx = genre_to_idx
            # Add unseen genres from this split so we never get KeyErrors
            for g in all_genres:
                if g not in self.genre_to_idx:
                    self.genre_to_idx[g] = len(self.genre_to_idx)

        if instrument_to_idx is None:
            self.instrument_to_idx = {inst: i for i, inst in enumerate(sorted(all_instruments))}
        else:
            self.instrument_to_idx = instrument_to_idx
            for inst in all_instruments:
                if inst not in self.instrument_to_idx:
                    self.instrument_to_idx[inst] = len(self.instrument_to_idx)

        self.num_genres      = len(self.genre_to_idx)
        self.num_instruments = len(self.instrument_to_idx)

        # Build multi-hot matrices (stored on CPU)
        N = len(self.metadata)
        self.genre_vectors      = torch.zeros(N, self.num_genres)
        self.instrument_vectors = torch.zeros(N, self.num_instruments)

        for i, (genres, instruments) in enumerate(zip(track_genres, track_instruments)):
            for g in genres:
                if g in self.genre_to_idx:
                    self.genre_vectors[i, self.genre_to_idx[g]] = 1.0
            for inst in instruments:
                if inst in self.instrument_to_idx:
                    self.instrument_vectors[i, self.instrument_to_idx[inst]] = 1.0

    # ------------------------------------------------------------------
    # Structural feature tensors from cache
    # ------------------------------------------------------------------

    def _build_feature_tensors(self):
        bpms, time_sigs, squareness = [], [], []

        default_bpm = 120.0
        bpm_values = [
            v['bpm'] for v in self.features_cache.values()
            if v.get('status') == 'ok' and v.get('bpm') is not None
        ]
        if bpm_values:
            default_bpm = float(np.median(bpm_values))

        for path in self.metadata['PATH']:
            entry = self.features_cache.get(path, {})
            if entry.get('status') == 'ok':
                bpms.append(float(entry.get('bpm') or default_bpm))
                time_sigs.append(int(entry.get('time_signature') or 4))
                squareness.append(float(entry.get('squareness_score') or 0.5))
            else:
                bpms.append(default_bpm)
                time_sigs.append(4)
                squareness.append(0.5)   # neutral fallback

        self.bpms             = torch.tensor(bpms,      dtype=torch.float32)
        self.time_signatures  = torch.tensor(time_sigs, dtype=torch.long)
        self.squareness_scores = torch.tensor(squareness, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.metadata.iloc[idx]
        audio_path = self.audio_dir / row['PATH']

        if not audio_path.exists():
            alt = self.audio_dir / row['PATH'].replace('.mp3', '.low.mp3')
            audio_path = alt if alt.exists() else audio_path

        try:
            waveform, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            waveform = torch.from_numpy(waveform).float()
        except Exception:
            n = int(self.duration * self.sample_rate) if self.duration else 480000
            waveform = torch.zeros(n)

        if self.duration is not None:
            target = int(self.duration * self.sample_rate)
            if waveform.shape[0] > target:
                waveform = waveform[:target]
            elif waveform.shape[0] < target:
                waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[0]))

        if self.transform:
            waveform = self.transform(waveform)

        return (
            waveform,
            self.genre_vectors[idx],
            self.instrument_vectors[idx],
            self.bpms[idx].item(),
            self.time_signatures[idx].item(),
            self.squareness_scores[idx].item(),
        )


# ------------------------------------------------------------------
# Collate + DataLoader factory
# ------------------------------------------------------------------

def collate_fn_multisignal(batch):
    audios, genres, instruments, bpms, time_sigs, squareness = zip(*batch)

    max_len = max(a.shape[0] for a in audios)
    padded = [
        torch.nn.functional.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a
        for a in audios
    ]

    return (
        torch.stack(padded),
        torch.stack(genres),
        torch.stack(instruments),
        torch.tensor(bpms,       dtype=torch.float32),
        torch.tensor(time_sigs,  dtype=torch.long),
        torch.tensor(squareness, dtype=torch.float32),
    )


def create_dataloaders_multisignal(
    train_tsv: str,
    val_tsv: str,
    test_tsv: str,
    audio_dir: str,
    features_cache_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000,
    duration: float = 30.0,
):
    """Create train/val/test loaders sharing vocabulary from the training split."""

    train_ds = MTGJamendoMultiSignalDataset(
        tsv_file=train_tsv,
        audio_dir=audio_dir,
        features_cache_path=features_cache_path,
        sample_rate=sample_rate,
        duration=duration,
    )
    # Share vocabulary so val/test indices match training
    val_ds = MTGJamendoMultiSignalDataset(
        tsv_file=val_tsv,
        audio_dir=audio_dir,
        features_cache_path=features_cache_path,
        genre_to_idx=train_ds.genre_to_idx,
        instrument_to_idx=train_ds.instrument_to_idx,
        sample_rate=sample_rate,
        duration=duration,
    )
    test_ds = MTGJamendoMultiSignalDataset(
        tsv_file=test_tsv,
        audio_dir=audio_dir,
        features_cache_path=features_cache_path,
        genre_to_idx=train_ds.genre_to_idx,
        instrument_to_idx=train_ds.instrument_to_idx,
        sample_rate=sample_rate,
        duration=duration,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn_multisignal,
        pin_memory=True,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
        {
            'num_genres':       train_ds.num_genres,
            'num_instruments':  train_ds.num_instruments,
            'genre_to_idx':     train_ds.genre_to_idx,
            'instrument_to_idx': train_ds.instrument_to_idx,
            'train_size':       len(train_ds),
            'val_size':         len(val_ds),
            'test_size':        len(test_ds),
        },
    )
