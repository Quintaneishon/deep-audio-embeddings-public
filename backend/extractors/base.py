from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import os
import numpy as np
from tqdm import tqdm


@dataclass
class ModelConfig:
    """Configuration for a specific model/dataset combination."""
    model_name: str
    dataset: str
    weights_path: str

    @property
    def key(self) -> str:
        return f"{self.model_name}_{self.dataset}"


class EmbeddingExtractor(ABC):
    """Abstract base class for all embedding extractors."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def extract(self, audio_path: str, weights_path: str, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def get_configs(self) -> List[ModelConfig]:
        pass


class ConvolutionalExtractor(EmbeddingExtractor):
    """Base for convolutional models that use multiple datasets."""

    def __init__(self, datasets: List[str] = None):
        self.datasets = datasets

    def get_configs(self) -> List[ModelConfig]:
        import config
        return [ModelConfig(
            self.model_name,
            dataset=ds,
            weights_path=config.MODEL_WEIGHTS[self.model_name][ds])
            for ds in self.datasets
        ]


class TransformerExtractor(EmbeddingExtractor):
    """Base for transformer models with a single model size."""

    def __init__(self, model_size: str):
        self.model_size = model_size

    def get_configs(self) -> List[ModelConfig]:
        import config
        return [ModelConfig(
            self.model_name,
            dataset=self.model_size,
            weights_path=config.MODEL_WEIGHTS[self.model_name][self.model_size]
        )]


@dataclass
class ProcessingStats:
    """Track processing statistics with automatic counting."""
    tracks_processed: int = 0
    embeddings_created: int = 0
    errors: int = 0
    skipped: int = 0

    def record_success(self) -> None:
        self.embeddings_created += 1

    def record_error(self) -> None:
        self.errors += 1

    def record_skip(self) -> None:
        self.skipped += 1

    def record_track_completed(self) -> None:
        self.tracks_processed += 1

    def to_dict(self) -> Dict[str, int]:
        return {
            'tracks_processed': self.tracks_processed,
            'embeddings_created': self.embeddings_created,
            'errors': self.errors,
            'skipped': self.skipped
        }

    def __str__(self) -> str:
        return (f"Processed: {self.tracks_processed} tracks, "
                f"{self.embeddings_created} embeddings created, "
                f"{self.skipped} skipped, {self.errors} errors")


class TrackProcessor:
    """Processes tracks to extract embeddings."""

    def __init__(self, registry, db, audio_dir: str):
        self.registry = registry
        self.db = db
        self.audio_dir = audio_dir

    def process_track(self, track_id: int, model_config: ModelConfig, stats: ProcessingStats) -> Optional[int]:
        """Process single track with specific model config."""
        # Check if already processed
        existing = self.db.get_embedding(track_id, model_config.model_name, model_config.dataset)
        if existing:
            stats.record_skip()
            return None

        # Get track info by ID
        track = self.db.get_track_by_id(track_id)
        if not track:
            stats.record_error()
            return None

        audio_path = os.path.join(self.audio_dir, track['filename'])
        if not os.path.exists(audio_path):
            stats.record_error()
            return None

        # Get extractor
        extractor = self.registry.get(model_config.model_name)
        if not extractor:
            stats.record_error()
            return None

        try:
            embeddings = extractor.extract(
                audio_path,
                model_config.weights_path,
                dataset=model_config.dataset
            )

            embedding_id = self.db.insert_embedding(
                track_id, model_config.model_name, model_config.dataset,
                embeddings
            )
            stats.record_success()
            return embedding_id

        except Exception as e:
            self._log_error(track_id, model_config, e)
            stats.record_error()
            return None

    def process_all_tracks(self) -> ProcessingStats:
        """Process all tracks with all model configurations."""
        stats = ProcessingStats()
        tracks = self.db.get_all_tracks()

        if not tracks:
            print("No tracks found in database.")
            return stats

        configs = self.registry.get_all_configs()
        total_operations = len(tracks) * len(configs)

        with tqdm(total=total_operations, desc="Processing tracks") as pbar:
            for track in tracks:
                for model_config in configs:
                    self.process_track(track['id'], model_config, stats)
                    pbar.update(1)

                self.db.mark_track_processed(track['id'])
                stats.record_track_completed()

        return stats

    def _log_error(self, track_id: int, model_config: ModelConfig, error: Exception) -> None:
        import traceback
        print(f"\nERROR processing track {track_id} with {model_config.key}")
        print(f"Error: {type(error).__name__}: {error}")
        traceback.print_exc()
