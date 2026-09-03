import os
from pathlib import Path
from backend import config, utils
from backend.db import database
from backend.extractors.extractors import ExtractorRegistry
from backend.extractors.base import TrackProcessor, ProcessingStats

def index_audio_files():
    """
    Scan audio directory and add all audio files to the database.

    Returns:
        Number of new tracks indexed
    """
    audio_dir = Path(config.AUDIO_DIR)
    if not audio_dir.exists():
        print(f"Error: Audio directory {config.AUDIO_DIR} does not exist")
        return 0

    # Get all audio files
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith(tuple(audio_extensions))
    ]

    indexed_count = 0
    for filename in audio_files:
        # Check if already in database
        existing = database.get_track_by_filename(filename)
        if existing is None:
            # Get audio duration
            try:
                audio_path = audio_dir / filename
                duration = utils.get_audio_duration(audio_path)
            except Exception as e:
                print(f"Warning: Could not get duration for {filename}: {e}")
                duration = None

            # Insert into database
            database.insert_track(filename, duration)
            indexed_count += 1

    print(f"\nIndexed {indexed_count} new tracks")
    return indexed_count

def process_all_tracks():
    """CLI command to process all tracks."""
    registry = ExtractorRegistry.create_default()
    processor = TrackProcessor(registry, database, config.AUDIO_DIR)

    stats = processor.process_all_tracks()
    print(stats)
    return stats.to_dict()

def process_single_track(filename):
    """Process a single track with all model configurations.

    Args:
        filename: Name of the audio file to process

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get track from database by filename
        track = database.get_track_by_filename(filename)

        if track is None:
            print(f"Track {filename} not found in database. Adding it now...")
            # Get audio duration
            audio_path = Path(config.AUDIO_DIR) / filename
            if not audio_path.exists():
                print(f"Error: Audio file not found at {audio_path}")
                return False

            duration = utils.get_audio_duration(audio_path)
            database.insert_track(filename, duration)
            track = database.get_track_by_filename(filename)

            if track is None:
                print(f"Error: Failed to insert track {filename}")
                return False

        track_id = track['id']
        print(f"Processing track ID {track_id}: {filename}")

        # Create processor and stats
        registry = ExtractorRegistry.create_default()
        processor = TrackProcessor(registry, database, config.AUDIO_DIR)
        stats = ProcessingStats()

        # Process with all model configurations
        configs = registry.get_all_configs()
        for model_config in configs:
            print(f"  Extracting {model_config.key}...")
            processor.process_track(track_id, model_config, stats)

        # Mark track as processed
        database.mark_track_processed(track_id)

        print(f"Completed: {stats}")
        return stats.errors == 0

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False
