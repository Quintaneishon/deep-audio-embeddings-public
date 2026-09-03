#!/usr/bin/env python3
"""
Script to extract 1000 songs from FMA small dataset with balanced genre distribution.
Creates a CSV with song names and genres from the FMA metadata.
"""

import csv
import random
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from backend import config

FMA_SMALL_DIR = Path(config.FMA_SMALL_DIR)
FMA_METADATA_DIR = Path(config.FMA_METADATA_DIR)
TRACKS_CSV = FMA_METADATA_DIR / 'tracks.csv'
OUTPUT_DIR = Path(config.AUDIO_OUTPUT_DIR)
OUTPUT_CSV = OUTPUT_DIR / 'selected_songs.csv'

# Select 1000 songs with balanced genre distribution
NUM_SONGS = 8000
RANDOM_SEED = 42


def load_fma_tracks():
    """Load FMA tracks.csv with proper multi-index handling."""
    print(f"Loading tracks metadata from: {TRACKS_CSV}")

    # FMA tracks.csv has a multi-level header
    tracks = pd.read_csv(TRACKS_CSV, index_col=0, header=[0, 1])

    return tracks


def find_audio_file(track_id):
    """
    Find the audio file for a given track_id in FMA small dataset.
    Files are organized as: fma_small/NNN/NNNNNN.mp3
    where NNN is the first 3 digits of the 6-digit track_id
    """
    # Convert track_id to 6-digit string with leading zeros
    track_id_str = str(track_id).zfill(6)

    # Get the directory (first 3 digits)
    dir_name = track_id_str[:3]

    # Build the expected path
    audio_path = FMA_SMALL_DIR / dir_name / f"{track_id_str}.mp3"

    if audio_path.exists():
        return audio_path

    return None


def main():
    random.seed(RANDOM_SEED)
    print("=" * 60)
    print("FMA Small Dataset Song Extraction (Balanced by Genre)")
    print("=" * 60)

    # Load tracks metadata
    tracks = load_fma_tracks()

    print(f"Total tracks in metadata: {len(tracks)}")

    # Extract genre information
    # In FMA, the genre info is typically in ('track', 'genre_top')
    try:
        # Try to get the top-level genre
        if ('track', 'genre_top') in tracks.columns:
            genre_col = ('track', 'genre_top')
        elif 'genre_top' in tracks.columns:
            genre_col = 'genre_top'
        else:
            # Fallback: print available columns to help debug
            print("\nAvailable columns:")
            print(tracks.columns.tolist()[:20])  # Print first 20 columns
            raise ValueError("Could not find genre column")

        # Filter tracks that have a genre
        tracks_with_genre = tracks[tracks[genre_col].notna()].copy()
        tracks_with_genre['genre'] = tracks_with_genre[genre_col]

        print(f"Tracks with genre info: {len(tracks_with_genre)}")

    except Exception as e:
        print(f"Error accessing genre column: {e}")
        print("Trying alternative approach...")
        # Alternative: check if columns have different structure
        tracks_with_genre = tracks.copy()
        tracks_with_genre['genre'] = 'Unknown'

    # Step 1: Group tracks by genre and check which audio files exist
    print("\nStep 1: Scanning available audio files and grouping by genre...")
    genre_tracks = defaultdict(list)
    not_found_count = 0

    pbar = tqdm(tracks_with_genre.iterrows(),
                total=len(tracks_with_genre),
                desc="Scanning files",
                unit="track")

    for track_id, track_data in pbar:
        # Check if audio file exists
        source_file = find_audio_file(track_id)

        if source_file and source_file.exists():
            # Get genre
            try:
                genre_value = track_data.get('genre', None)
                if genre_value is not None and not (isinstance(genre_value, float) and pd.isna(genre_value)):
                    genre = str(genre_value).strip().split('\n')[0]
                else:
                    genre = 'Unknown'
            except:
                genre = 'Unknown'

            genre_tracks[genre].append((track_id, source_file))
        else:
            not_found_count += 1

    pbar.close()

    # Print genre statistics
    print(f"\nFound audio files: {sum(len(tracks) for tracks in genre_tracks.values())}")
    print(f"Audio files not found: {not_found_count}")
    print(f"\nGenres found: {len(genre_tracks)}")

    print("\nAvailable songs per genre:")
    for genre, tracks in sorted(genre_tracks.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {genre}: {len(tracks)} songs")

    # Step 2: Calculate balanced selection
    print(f"\nStep 2: Selecting {NUM_SONGS} songs with balanced genre distribution...")
    print(f"Random seed: {RANDOM_SEED}")

    # Calculate songs per genre (balanced)
    total_genres = len(genre_tracks)
    songs_per_genre = NUM_SONGS // total_genres
    remaining_songs = NUM_SONGS % total_genres

    print(f"Target: ~{songs_per_genre} songs per genre ({total_genres} genres)")

    selected_tracks = []

    # First pass: take songs_per_genre from each genre
    for genre, tracks in genre_tracks.items():
        available = len(tracks)
        to_select = min(songs_per_genre, available)
        selected = random.sample(tracks, to_select)
        selected_tracks.extend([(track_id, source_file, genre) for track_id, source_file in selected])

    # Second pass: distribute remaining songs to genres that have more available
    if remaining_songs > 0 and len(selected_tracks) < NUM_SONGS:
        # Get genres that have more songs available
        genres_with_extra = [(genre, tracks) for genre, tracks in genre_tracks.items()
                             if len(tracks) > songs_per_genre]

        if genres_with_extra:
            # Sort by number of available songs (descending)
            genres_with_extra.sort(key=lambda x: len(x[1]), reverse=True)

            for genre, tracks in genres_with_extra:
                if len(selected_tracks) >= NUM_SONGS:
                    break

                # Get track_ids already selected from this genre
                already_selected = {track_id for track_id, _, g in selected_tracks if g == genre}

                # Get remaining tracks from this genre
                remaining = [(track_id, source_file) for track_id, source_file in tracks
                             if track_id not in already_selected]

                if remaining:
                    # Select one more from this genre
                    track_id, source_file = random.choice(remaining)
                    selected_tracks.append((track_id, source_file, genre))

    # Shuffle to mix genres
    random.shuffle(selected_tracks)

    print(f"Selected {len(selected_tracks)} songs total")

    # Step 3: Copy files and create CSV
    print(f"\nStep 3: Copying selected files to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful_copies = []

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'genre'])

        pbar = tqdm(selected_tracks, desc="Copying files", unit="file")

        for track_id, source_file, genre in pbar:
            dest_filename = f"{str(track_id).zfill(6)}.mp3"
            dest_path = OUTPUT_DIR / dest_filename

            try:
                shutil.copy2(source_file, dest_path)
                writer.writerow([dest_filename, genre])
                successful_copies.append(dest_filename)
                pbar.set_postfix({'copied': len(successful_copies)})
            except Exception as e:
                tqdm.write(f"Error copying {source_file}: {e}")

        pbar.close()

    print(f"\n" + "=" * 60)
    print(f"Completed!")
    print(f"Successfully copied {len(successful_copies)} songs to {OUTPUT_DIR}")
    print(f"CSV file created: {OUTPUT_CSV}")
    print("=" * 60)

    # Print final genre distribution
    genre_counts = {}
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            genre = row['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    print("\nFinal genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} songs")


if __name__ == '__main__':
    main()
