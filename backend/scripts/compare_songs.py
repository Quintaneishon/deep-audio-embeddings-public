#!/usr/bin/env python3
"""
Script to compare two audio files using Essentia algorithms.

Usage:
    python compare_songs.py path/to/song1.mp3 path/to/song2.mp3

Metrics reported:
- Serra09 HPCP cross-similarity distance (cover song detection)
- MFCC timbral cosine distance (mean 20-coeff MFCCs, full track)
- Key compatibility distance on the circle of fifths (0–6)
"""

import sys
import argparse
from pathlib import Path
import essentia.standard as es
import numpy as np

# Circle-of-fifths position for each key name (0 = C, 1 = G, ..., 11 = F)
_COF_POSITIONS = {
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5,
    'F#': 6, 'Gb': 6, 'C#': 7, 'Db': 7, 'G#': 8, 'Ab': 8,
    'D#': 9, 'Eb': 9, 'A#': 10, 'Bb': 10, 'F': 11,
}


def load_audio(filepath):
    """Load audio file and return mono signal at 44.1kHz."""
    loader = es.MonoLoader(filename=str(filepath), sampleRate=44100)
    audio = loader()
    return audio


def extract_chroma_features(audio, sample_rate=44100):
    """
    Extract chroma features from audio signal.

    Returns:
        numpy.ndarray: Chroma features (12 x frames)
    """
    windowing = es.Windowing(type='blackmanharris62')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks()
    hpcp = es.HPCP()

    chroma_frames = []

    frame_size = 4096
    hop_size = 2048

    for frame_start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[frame_start:frame_start + frame_size]
        windowed_frame = windowing(frame)
        spec = spectrum(windowed_frame)
        freqs, mags = spectral_peaks(spec)
        hpcp_frame = hpcp(freqs, mags)
        chroma_frames.append(hpcp_frame)

    return np.array(chroma_frames).T  # Shape: (12, n_frames)


def extract_mean_mfcc(audio, sample_rate=44100, n_coefficients=20):
    """
    Extract mean MFCC vector (n_coefficients coefficients) over the full track.

    Returns:
        numpy.ndarray: Mean MFCC vector of shape (n_coefficients,)
    """
    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc_algo = es.MFCC(numberCoefficients=n_coefficients, sampleRate=sample_rate)

    frame_size = 2048
    hop_size = 1024
    mfcc_frames = []

    for frame_start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[frame_start:frame_start + frame_size]
        windowed = windowing(frame)
        spec = spectrum(windowed)
        _, coeffs = mfcc_algo(spec)
        mfcc_frames.append(coeffs)

    return np.mean(np.array(mfcc_frames), axis=0)


def cosine_distance(a, b):
    """Cosine distance between two vectors (0 = identical, 1 = orthogonal, 2 = opposite)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return float('nan')
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def extract_key(audio, sample_rate=44100):
    """
    Extract key using Essentia KeyExtractor.

    Returns:
        key (str): e.g. 'C', 'F#', 'Bb'
        scale (str): 'major' or 'minor'
        strength (float)
    """
    key_extractor = es.KeyExtractor(sampleRate=sample_rate)
    key, scale, strength = key_extractor(audio)
    return key, scale, strength


def key_cof_distance(key1, key2):
    """
    Circular distance between two keys on the circle of fifths.

    Returns an integer in [0, 6]: 0 = same key, 6 = tritone.
    Returns None if either key is unrecognised.
    """
    pos1 = _COF_POSITIONS.get(key1)
    pos2 = _COF_POSITIONS.get(key2)
    if pos1 is None or pos2 is None:
        return None
    diff = abs(pos1 - pos2)
    return min(diff, 12 - diff)


def compute_similarity(song1_path, song2_path):
    """
    Compute similarity between two songs.

    Returns:
        dict with keys: serra09_distance, mfcc_cosine_distance, key_cof_distance,
                        key1, key2, scale1, scale2
    """
    print(f"\nComparing songs:")
    print(f"  Song 1: {song1_path.name}")
    print(f"  Song 2: {song2_path.name}")
    print()

    print("Loading audio files...")
    audio1 = load_audio(song1_path)
    audio2 = load_audio(song2_path)

    print(f"  Song 1 duration: {len(audio1)/44100:.2f} seconds")
    print(f"  Song 2 duration: {len(audio2)/44100:.2f} seconds")

    # ── Serra09 HPCP cover-song distance ─────────────────────────────────────
    print("\nExtracting chroma features (HPCP)...")
    chroma1 = extract_chroma_features(audio1)
    chroma2 = extract_chroma_features(audio2)

    print(f"  Song 1 chroma shape: {chroma1.shape}")
    print(f"  Song 2 chroma shape: {chroma2.shape}")

    print("Computing Serra09 CoverSongSimilarity score...")
    chroma1_t = chroma1.T
    chroma2_t = chroma2.T

    cross_sim_binarized = es.CrossSimilarityMatrix(binarize=True, binarizePercentile=0.095)
    binary_matrix = cross_sim_binarized(chroma1_t, chroma2_t)

    cover_similarity = es.CoverSongSimilarity(alignmentType='serra09')
    _, serra09_distance = cover_similarity(binary_matrix)

    # ── MFCC timbral cosine distance ─────────────────────────────────────────
    print("\nExtracting MFCC features...")
    mfcc1 = extract_mean_mfcc(audio1)
    mfcc2 = extract_mean_mfcc(audio2)
    mfcc_dist = cosine_distance(mfcc1, mfcc2)

    # ── Key compatibility (circle of fifths) ──────────────────────────────────
    print("Extracting keys...")
    key1, scale1, strength1 = extract_key(audio1)
    key2, scale2, strength2 = extract_key(audio2)
    cof_dist = key_cof_distance(key1, key2)

    return {
        'serra09_distance': float(serra09_distance),
        'mfcc_cosine_distance': mfcc_dist,
        'key_cof_distance': cof_dist,
        'key1': key1, 'scale1': scale1, 'strength1': strength1,
        'key2': key2, 'scale2': scale2, 'strength2': strength2,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare two audio files using Essentia algorithms'
    )
    parser.add_argument('song1', type=str, help='Path to first audio file')
    parser.add_argument('song2', type=str, help='Path to second audio file')

    args = parser.parse_args()

    song1_path = Path(args.song1)
    song2_path = Path(args.song2)

    if not song1_path.exists():
        print(f"Error: File not found: {song1_path}")
        sys.exit(1)

    if not song2_path.exists():
        print(f"Error: File not found: {song2_path}")
        sys.exit(1)

    result = compute_similarity(song1_path, song2_path)

    print()
    print("=" * 50)
    print("SIMILARITY RESULTS")
    print("=" * 50)

    print(f"\n1. Serra09 HPCP Cover-Song Distance: {result['serra09_distance']:.4f}")
    print(f"   (Lower = more similar; based on chroma/HPCP alignment)")

    print(f"\n2. MFCC Timbral Cosine Distance:     {result['mfcc_cosine_distance']:.4f}")
    print(f"   (Higher = less similar timbre; mean of 20 MFCC coefficients)")

    cof = result['key_cof_distance']
    cof_str = str(cof) if cof is not None else 'n/a'
    print(f"\n3. Key Compatibility (Circle of Fifths): {cof_str}")
    print(f"   Song 1: {result['key1']} {result['scale1']} (strength={result['strength1']:.3f})")
    print(f"   Song 2: {result['key2']} {result['scale2']} (strength={result['strength2']:.3f})")
    print(f"   (0 = same key, 6 = tritone / maximally distant)")

    print()


if __name__ == '__main__':
    main()
