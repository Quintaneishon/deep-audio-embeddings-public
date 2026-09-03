#!/usr/bin/env python3
"""
Script to extract songs from MTG-Jamendo dataset with instrument, mood/theme, and genre tags.
Creates a CSV with song information and their tags for evaluation.

Uses stratified sampling by tag diversity to ensure representative evaluation
across easy (many tags) and hard (few tags) cases.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from backend import config

# Stratified sampling thresholds for tag diversity
# Low: 1-2 tags, Medium: 3-5 tags, High: 6+ tags
TAG_DIVERSITY_LOW_MAX = 2
TAG_DIVERSITY_MED_MAX = 5


def load_tags_from_tsv(tsv_file):
    """
    Load tags from a TSV file.

    Returns:
        dict: {track_id: {'path': str, 'tags': [tag1, tag2, ...]}}
    """
    tags_dict = {}

    with open(tsv_file, 'r', encoding='utf-8') as f:
        # Skip header
        f.readline()

        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                track_id = parts[0]
                path = parts[3]

                # Get all tags (column 5 onwards)
                tag_str = '\t'.join(parts[5:])
                tag_list = tag_str.split('\t') if tag_str else []

                # Clean tags (remove prefix)
                clean_tags = []
                for tag in tag_list:
                    if '---' in tag:
                        clean_tag = tag.split('---', 1)[1]
                        clean_tags.append(clean_tag)

                if clean_tags:
                    tags_dict[track_id] = {
                        'path': path,
                        'tags': clean_tags
                    }

    return tags_dict


def find_audio_file(relative_path, audio_dir):
    """
    Find the audio file in the MTG-Jamendo dataset.
    Tries both .mp3 and .low.mp3 extensions.
    """
    audio_path = audio_dir / relative_path

    if audio_path.exists():
        return audio_path

    # Try .low.mp3 version
    if relative_path.endswith('.mp3'):
        low_path = audio_dir / relative_path.replace('.mp3', '.low.mp3')
        if low_path.exists():
            return low_path

    return None


def main():
    parser = argparse.ArgumentParser(
        description='Select a reproducible, tag-diverse MTG-Jamendo subset.'
    )
    parser.add_argument('--data-root', default=config.MTG_JAMENDO_ROOT,
                        help='MTG-Jamendo root containing data/ and songs/')
    parser.add_argument('--split', default='split-0', help='Official split directory name')
    parser.add_argument('--output-dir', default='backend/data/audio_mtg',
                        help='Private output directory for copied audio and the CSV manifest')
    parser.add_argument('--num-songs', type=int, default=2000,
                        help='Maximum number of songs to copy')
    parser.add_argument('--seed', type=int, default=42, help='Sampling seed')
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    audio_dir = data_root / 'songs'
    splits_dir = data_root / 'data' / 'splits' / args.split
    output_dir = Path(args.output_dir).expanduser()
    output_csv = output_dir / 'selected_songs.csv'
    num_songs = args.num_songs
    rng = random.Random(args.seed)

    print("=" * 60)
    print("MTG-Jamendo Song Extraction (Instrument, Mood/Theme & Genre Tags)")
    print("=" * 60)

    # Load instrument and mood/theme tags from all splits
    print("\nLoading tags...")

    instrument_files = [
        splits_dir / 'autotagging_instrument-train.tsv',
        splits_dir / 'autotagging_instrument-validation.tsv',
        splits_dir / 'autotagging_instrument-test.tsv'
    ]

    moodtheme_files = [
        splits_dir / 'autotagging_moodtheme-train.tsv',
        splits_dir / 'autotagging_moodtheme-validation.tsv',
        splits_dir / 'autotagging_moodtheme-test.tsv'
    ]

    genre_files = [
        splits_dir / 'autotagging_genre-train.tsv',
        splits_dir / 'autotagging_genre-validation.tsv',
        splits_dir / 'autotagging_genre-test.tsv'
    ]

    # Load all tags
    all_tracks = {}

    print("Loading instrument tags...")
    for tsv_file in instrument_files:
        if tsv_file.exists():
            tags = load_tags_from_tsv(tsv_file)
            for track_id, data in tags.items():
                if track_id not in all_tracks:
                    all_tracks[track_id] = {
                        'path': data['path'],
                        'instrument': set(data['tags']),
                        'mood': set(),
                        'genre': set()
                    }
                else:
                    all_tracks[track_id]['instrument'].update(data['tags'])

    print("Loading mood/theme tags...")
    for tsv_file in moodtheme_files:
        if tsv_file.exists():
            tags = load_tags_from_tsv(tsv_file)
            for track_id, data in tags.items():
                if track_id not in all_tracks:
                    all_tracks[track_id] = {
                        'path': data['path'],
                        'instrument': set(),
                        'mood': set(data['tags']),
                        'genre': set()
                    }
                else:
                    all_tracks[track_id]['mood'].update(data['tags'])

    print("Loading genre tags...")
    for tsv_file in genre_files:
        if tsv_file.exists():
            tags = load_tags_from_tsv(tsv_file)
            for track_id, data in tags.items():
                if track_id not in all_tracks:
                    all_tracks[track_id] = {
                        'path': data['path'],
                        'instrument': set(),
                        'mood': set(),
                        'genre': set(data['tags'])
                    }
                else:
                    all_tracks[track_id]['genre'].update(data['tags'])

    print(f"\nTotal tracks with instrument/mood/genre tags: {len(all_tracks)}")

    # Count tag frequencies
    instrument_counts = Counter()
    mood_counts = Counter()
    genre_counts = Counter()

    for track_id, data in all_tracks.items():
        instrument_counts.update(data['instrument'])
        mood_counts.update(data['mood'])
        genre_counts.update(data['genre'])

    print(f"\nUnique instrument tags: {len(instrument_counts)}")
    print(f"Unique mood/theme tags: {len(mood_counts)}")
    print(f"Unique genre tags: {len(genre_counts)}")

    # Show top tags
    print("\nTop 10 instrument tags:")
    for tag, count in instrument_counts.most_common(10):
        print(f"  {tag}: {count} tracks")

    print("\nTop 10 mood/theme tags:")
    for tag, count in mood_counts.most_common(10):
        print(f"  {tag}: {count} tracks")

    print("\nTop 10 genre tags:")
    for tag, count in genre_counts.most_common(10):
        print(f"  {tag}: {count} tracks")

    # Step 1: Check which audio files exist
    print("\nStep 1: Scanning available audio files...")
    available_tracks = []
    not_found_count = 0

    pbar = tqdm(all_tracks.items(),
                total=len(all_tracks),
                desc="Scanning files",
                unit="track")

    for track_id, data in pbar:
        audio_path = find_audio_file(data['path'], audio_dir)

        if audio_path and audio_path.exists():
            # Calculate tag diversity score (prefer tracks with multiple tags)
            tag_diversity = len(data['instrument']) + len(data['mood']) + len(data['genre'])

            available_tracks.append({
                'track_id': track_id,
                'path': data['path'],
                'audio_path': audio_path,
                'instrument_tags': data['instrument'],
                'mood_tags': data['mood'],
                'genre_tags': data['genre'],
                'tag_diversity': tag_diversity
            })
        else:
            not_found_count += 1

    pbar.close()

    print(f"\nAudio files found: {len(available_tracks)}")
    print(f"Audio files not found: {not_found_count}")

    # Step 2: Select diverse songs using STRATIFIED SAMPLING by tag diversity
    if num_songs is None:
        print(f"\nStep 2: Selecting ALL available songs (maximum possible)...")
        selected_tracks = available_tracks
        print(f"Selected all {len(selected_tracks)} available tracks")
    elif len(available_tracks) <= num_songs:
        print(f"\nStep 2: Selecting {num_songs} songs with STRATIFIED sampling by tag diversity...")
        print(f"  (Random seed: {args.seed} for reproducibility)")
        selected_tracks = available_tracks
        print(f"Selected all {len(selected_tracks)} available tracks (less than target)")
    else:
        print(f"\nStep 2: Selecting {num_songs} songs with STRATIFIED sampling by tag diversity...")
        print(f"  (Random seed: {args.seed} for reproducibility)")
        # STRATIFIED SAMPLING: Group tracks by tag diversity level
        # This ensures we test models on easy, medium, AND hard cases

        low_diversity = [t for t in available_tracks if t['tag_diversity'] <= TAG_DIVERSITY_LOW_MAX]
        med_diversity = [t for t in available_tracks if
                         TAG_DIVERSITY_LOW_MAX < t['tag_diversity'] <= TAG_DIVERSITY_MED_MAX]
        high_diversity = [t for t in available_tracks if t['tag_diversity'] > TAG_DIVERSITY_MED_MAX]

        print(f"\n  Tag diversity distribution in available tracks:")
        print(f"    Low (1-{TAG_DIVERSITY_LOW_MAX} tags):     {len(low_diversity)} tracks")
        print(f"    Medium ({TAG_DIVERSITY_LOW_MAX + 1}-{TAG_DIVERSITY_MED_MAX} tags):  {len(med_diversity)} tracks")
        print(f"    High ({TAG_DIVERSITY_MED_MAX + 1}+ tags):    {len(high_diversity)} tracks")

        # Calculate proportional allocation (aim for equal representation with fallback)
        # Target: ~1/3 from each stratum, adjusted for availability
        n_per_stratum = num_songs // 3

        # Determine how many to sample from each stratum
        n_low = min(n_per_stratum, len(low_diversity))
        n_med = min(n_per_stratum, len(med_diversity))
        n_high = min(n_per_stratum, len(high_diversity))

        # Redistribute any shortfall to other strata
        total_allocated = n_low + n_med + n_high
        shortfall = num_songs - total_allocated

        if shortfall > 0:
            # Try to fill from strata with remaining capacity
            remaining_low = len(low_diversity) - n_low
            remaining_med = len(med_diversity) - n_med
            remaining_high = len(high_diversity) - n_high

            # Distribute shortfall proportionally to remaining capacity
            total_remaining = remaining_low + remaining_med + remaining_high
            if total_remaining > 0:
                add_low = min(remaining_low, int(shortfall * remaining_low / total_remaining))
                add_med = min(remaining_med, int(shortfall * remaining_med / total_remaining))
                add_high = min(remaining_high, shortfall - add_low - add_med)

                n_low += add_low
                n_med += add_med
                n_high += add_high

        print(f"\n  Stratified sampling allocation:")
        print(f"    Low diversity:    {n_low} tracks")
        print(f"    Medium diversity: {n_med} tracks")
        print(f"    High diversity:   {n_high} tracks")

        selected_tracks = []

        # Sample from each stratum (no sorting bias - pure random sampling)
        if low_diversity and n_low > 0:
            selected_tracks.extend(rng.sample(low_diversity, n_low))

        if med_diversity and n_med > 0:
            selected_tracks.extend(rng.sample(med_diversity, n_med))

        if high_diversity and n_high > 0:
            selected_tracks.extend(rng.sample(high_diversity, n_high))

        # If we still need more (edge case), add randomly from remaining
        if len(selected_tracks) < num_songs:
            selected_ids = {t['track_id'] for t in selected_tracks}
            remaining = [t for t in available_tracks if t['track_id'] not in selected_ids]
            needed = num_songs - len(selected_tracks)
            if remaining:
                selected_tracks.extend(rng.sample(remaining, min(needed, len(remaining))))

        # Shuffle to avoid any ordering bias
        rng.shuffle(selected_tracks)

        print(f"\n  Selected {len(selected_tracks)} songs total (stratified)")

    # Step 3: Copy files and create CSV
    print(f"\nStep 3: Copying selected files to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_copies = []

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'filename', 'instrument_tags', 'mood_tags', 'genre', 'all_tags'])

        pbar = tqdm(selected_tracks, desc="Copying files", unit="file")

        for track_data in pbar:
            track_id = track_data['track_id']
            source_file = track_data['audio_path']

            # Create a clean filename
            dest_filename = f"{track_id}.mp3"
            dest_path = output_dir / dest_filename

            try:
                shutil.copy2(source_file, dest_path)

                # Format tags for CSV
                instrument_str = '|'.join(sorted(track_data['instrument_tags']))
                mood_str = '|'.join(sorted(track_data['mood_tags']))
                genre_str = '|'.join(sorted(track_data['genre_tags']))
                all_tags_str = '|'.join(
                    sorted(track_data['instrument_tags'] | track_data['mood_tags'] | track_data['genre_tags']))

                writer.writerow([
                    track_id,
                    dest_filename,
                    instrument_str,
                    mood_str,
                    genre_str,
                    all_tags_str
                ])

                successful_copies.append(dest_filename)
                pbar.set_postfix({'copied': len(successful_copies)})

            except Exception as e:
                tqdm.write(f"Error copying {source_file}: {e}")

        pbar.close()

    print(f"\n" + "=" * 60)
    print(f"Completed!")
    print(f"  Successfully copied {len(successful_copies)} songs to {output_dir}")
    print(f"  CSV file created: {output_csv}")
    print("=" * 60)

    # Print final statistics
    print("\nFinal statistics:")

    # Count tag distributions in selected songs
    selected_instrument = Counter()
    selected_mood = Counter()
    selected_genre = Counter()
    tracks_with_all = 0
    tracks_with_instrument_only = 0
    tracks_with_mood_only = 0
    tracks_with_genre_only = 0
    tracks_with_multiple = 0

    # Count by diversity stratum
    final_low = 0
    final_med = 0
    final_high = 0

    for track_data in selected_tracks:
        has_instrument = len(track_data['instrument_tags']) > 0
        has_mood = len(track_data['mood_tags']) > 0
        has_genre = len(track_data['genre_tags']) > 0

        tag_type_count = sum([has_instrument, has_mood, has_genre])

        if tag_type_count == 3:
            tracks_with_all += 1
        elif tag_type_count == 2:
            tracks_with_multiple += 1
        elif has_instrument:
            tracks_with_instrument_only += 1
        elif has_mood:
            tracks_with_mood_only += 1
        elif has_genre:
            tracks_with_genre_only += 1

        selected_instrument.update(track_data['instrument_tags'])
        selected_mood.update(track_data['mood_tags'])
        selected_genre.update(track_data['genre_tags'])

        # Count diversity strata
        div = track_data['tag_diversity']
        if div <= TAG_DIVERSITY_LOW_MAX:
            final_low += 1
        elif div <= TAG_DIVERSITY_MED_MAX:
            final_med += 1
        else:
            final_high += 1

    print(f"\nStratified sampling result:")
    print(
        f"  Low diversity (1-{TAG_DIVERSITY_LOW_MAX} tags):     {final_low} tracks ({100 * final_low / len(selected_tracks):.1f}%)")
    print(
        f"  Medium diversity ({TAG_DIVERSITY_LOW_MAX + 1}-{TAG_DIVERSITY_MED_MAX} tags):  {final_med} tracks ({100 * final_med / len(selected_tracks):.1f}%)")
    print(
        f"  High diversity ({TAG_DIVERSITY_MED_MAX + 1}+ tags):    {final_high} tracks ({100 * final_high / len(selected_tracks):.1f}%)")

    print(f"\nTag type coverage:")
    print(f"  Tracks with all 3 tag types (instrument & mood & genre): {tracks_with_all}")
    print(f"  Tracks with 2 tag types: {tracks_with_multiple}")
    print(f"  Tracks with only instrument tags: {tracks_with_instrument_only}")
    print(f"  Tracks with only mood tags: {tracks_with_mood_only}")
    print(f"  Tracks with only genre tags: {tracks_with_genre_only}")
    print(f"  Total unique instrument tags: {len(selected_instrument)}")
    print(f"  Total unique mood tags: {len(selected_mood)}")
    print(f"  Total unique genre tags: {len(selected_genre)}")

    print(f"\nTop 5 selected instrument tags:")
    for tag, count in selected_instrument.most_common(5):
        print(f"  {tag}: {count} tracks")

    print(f"\nTop 5 selected mood/theme tags:")
    for tag, count in selected_mood.most_common(5):
        print(f"  {tag}: {count} tracks")

    print(f"\nTop 5 selected genre tags:")
    for tag, count in selected_genre.most_common(5):
        print(f"  {tag}: {count} tracks")


if __name__ == '__main__':
    main()
