#!/usr/bin/env python3
"""
Script to collect .wav files from personal tracks folder organized by genre (subfolder name).
Copies all .wav files to the audio output directory and creates a selected_songs.csv.
"""

import argparse
import csv
import shutil
from pathlib import Path
from tqdm import tqdm

from backend import config

SOURCE_DIR = Path(config.MY_TRACKS_DIR)
OUTPUT_DIR = Path(config.AUDIO_OUTPUT_DIR)
OUTPUT_CSV = OUTPUT_DIR / 'selected_songs.csv'


def main():
    parser = argparse.ArgumentParser(description='Collect .wav files from personal tracks folder.')
    parser.add_argument('--csv-only', action='store_true', help='Only generate the CSV, do not copy audio files to OUTPUT_DIR')
    args = parser.parse_args()

    print("=" * 60)
    print("Personal Tracks Collection")
    print("=" * 60)

    # Gather .wav/.mp3 files recursively; genre = first-level subfolder name (empty for root)
    tracks = []
    for audio in sorted(f for ext in ('**/*.wav', '**/*.mp3') for f in SOURCE_DIR.glob(ext)):
        relative = audio.relative_to(SOURCE_DIR)
        genre = relative.parts[0] if len(relative.parts) > 1 else 'Unknown'
        tracks.append((audio, genre))

    print(f"Found {len(tracks)} files across {len(set(g for _, g in tracks))} genres")

    if len(tracks) == 0:
        print("No tracks found", SOURCE_DIR)
        exit(0)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not args.csv_only:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful_copies = []

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'genre'])

        desc = "Generating CSV" if args.csv_only else "Copying files"
        for source_file, genre in tqdm(tracks, desc=desc, unit="file"):
            if args.csv_only:
                writer.writerow([source_file.name, genre])
                successful_copies.append(source_file.name)
            else:
                dest_path = OUTPUT_DIR / source_file.name
                try:
                    shutil.copy(source_file, dest_path)
                    writer.writerow([source_file.name, genre])
                    successful_copies.append(source_file.name)
                except Exception as e:
                    print(f"Error copying {source_file}: {e}")

    print("\n" + "=" * 60)
    if args.csv_only:
        print(f"CSV-only mode: no files copied")
    else:
        print(f"Copied {len(successful_copies)} files to {OUTPUT_DIR}")
    print(f"CSV created: {OUTPUT_CSV}")
    print("=" * 60)

    genre_counts = {}
    with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            genre_counts[row['genre']] = genre_counts.get(row['genre'], 0) + 1

    print("\nGenre distribution:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre}: {count} songs")


if __name__ == '__main__':
    main()
