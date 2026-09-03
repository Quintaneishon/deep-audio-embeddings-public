"""
Precompute structural audio features for contrastive training.

Pipeline per track:
  1. Run MSAF to detect segment boundaries (intro, body, outro, ...)
  2. Remove first segment (intro) and last segment (outro)
  3. Crop audio to the core body
  4. On the core: estimate BPM, time signature, squareness score,
     and phrase lengths in bars

Output: JSON cache  {relative_path: {bpm, time_signature, squareness_score,
                                     phrase_lengths_bars, core_start_sec,
                                     core_end_sec, num_segments_total,
                                     num_segments_core, status}}

Usage:
    python precompute_structure_features.py \\
        --tsv   /data/mtg-jamendo/splits/autotagging_genre-train.tsv \\
        --audio /data/mtg-jamendo/songs \\
        --output features_cache.json \\
        --workers 4
"""

import json
import argparse
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import librosa

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# MSAF import — fail loudly so the user knows to install it
# ---------------------------------------------------------------------------
try:
    import msaf
    MSAF_AVAILABLE = True
except ImportError:
    MSAF_AVAILABLE = False
    print("[WARNING] msaf not found. Install with:  pip install msaf")
    print("          Falling back to librosa-only segmentation (less accurate).")


# ===========================================================================
# Core feature extraction
# ===========================================================================

def _segment_with_msaf(audio_path: str, algorithm: str = 'scluster'):
    """
    Run MSAF on a file and return segment boundary times in seconds.

    Falls back to 'foote' if the primary algorithm fails.
    Returns an empty list on total failure.
    """
    try:
        boundaries, _ = msaf.process(audio_path, boundaries_id=algorithm)
        return list(boundaries)
    except Exception:
        pass

    # Fallback algorithm
    fallback = 'foote' if algorithm != 'foote' else 'sf'
    try:
        boundaries, _ = msaf.process(audio_path, boundaries_id=fallback)
        return list(boundaries)
    except Exception:
        return []


def _segment_with_librosa(y: np.ndarray, sr: int) -> list:
    """
    Lightweight structural segmentation using librosa's recurrence matrix.
    Used when MSAF is not available.
    """
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        R = librosa.segment.recurrence_matrix(mfcc, mode='affinity', sym=True)
        df = librosa.segment.timelag_filter(scipy_ndimage_label=False)(R)
        frames = librosa.segment.agglomerative(df, k=8)
        times = librosa.frames_to_time(frames, sr=sr)
        # prepend 0.0 and append duration
        boundaries = [0.0] + list(times) + [len(y) / sr]
        return sorted(set(boundaries))
    except Exception:
        return []


def _crop_core(y: np.ndarray, sr: int, boundaries: list):
    """
    Given segment boundary times, remove the first and last segment
    (intro and outro) and return the cropped waveform plus timing info.

    If fewer than 3 boundaries exist (< 2 segments), the full audio is used.
    """
    if len(boundaries) < 3:
        # Can't safely remove intro/outro — use everything
        return y, 0.0, len(y) / sr, False

    core_start = boundaries[1]    # end of intro
    core_end   = boundaries[-2]   # start of outro

    # Sanity check: core must be at least 10 seconds
    if core_end - core_start < 10.0:
        return y, 0.0, len(y) / sr, False

    start_sample = int(core_start * sr)
    end_sample   = int(core_end   * sr)
    return y[start_sample:end_sample], core_start, core_end, True


def _estimate_bpm(y: np.ndarray, sr: int) -> float:
    """
    Estimate BPM using librosa's beat tracker.
    Uses onset strength envelope for robustness on diverse genres.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, trim=False)
    # beat_track can return an array in newer librosa versions
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
    return float(tempo)


def _estimate_time_signature(y: np.ndarray, sr: int, beat_times: np.ndarray) -> int:
    """
    Estimate time signature (beats per bar) by testing groupings of 3 and 4.

    Method: compute onset strength at beat positions, then evaluate the
    autocorrelation energy at lags corresponding to 3-beat and 4-beat bars.
    Whichever lag shows stronger periodicity wins.

    Returns 3 or 4 (the two overwhelmingly common signatures).
    """
    if len(beat_times) < 8:
        return 4  # not enough data, default to 4/4

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_frames = beat_frames[beat_frames < len(onset_env)]

    # Sample onset strength at beat positions
    beat_strength = onset_env[beat_frames]

    # Autocorrelation at lag=3 vs lag=4
    def acf_at_lag(signal, lag):
        if len(signal) <= lag:
            return 0.0
        return float(np.corrcoef(signal[:-lag], signal[lag:])[0, 1])

    acf3 = acf_at_lag(beat_strength, 3)
    acf4 = acf_at_lag(beat_strength, 4)

    return 3 if acf3 > acf4 else 4


def _compute_squareness(
    segment_times: list,
    core_start: float,
    core_end: float,
    bpm: float,
    time_sig: int,
) -> tuple:
    """
    Compute squareness score and phrase lengths in bars for segments
    within the core region.

    Square phrase = length in bars is a multiple of 4 (regardless of time sig).
    We allow ±15% tolerance to account for BPM estimation error.

    Returns:
        squareness_score    float in [0, 1]
        phrase_lengths_bars list of floats
    """
    if bpm <= 0:
        return 0.0, []

    seconds_per_beat = 60.0 / bpm
    seconds_per_bar  = seconds_per_beat * time_sig

    # Keep only boundaries within [core_start, core_end]
    core_bounds = [t for t in segment_times if core_start <= t <= core_end]

    # Ensure core edges are included
    if not core_bounds or core_bounds[0] > core_start + 0.5:
        core_bounds = [core_start] + core_bounds
    if core_bounds[-1] < core_end - 0.5:
        core_bounds = core_bounds + [core_end]

    if len(core_bounds) < 2:
        return 0.0, []

    phrase_lengths_bars = []
    for i in range(len(core_bounds) - 1):
        duration_sec = core_bounds[i + 1] - core_bounds[i]
        length_bars  = duration_sec / seconds_per_bar
        phrase_lengths_bars.append(round(length_bars, 2))

    # A phrase is "square" if its bar length rounds to a multiple of 4
    # with ±15% tolerance
    square_count = 0
    for length in phrase_lengths_bars:
        nearest_multiple_of_4 = round(length / 4) * 4
        if nearest_multiple_of_4 == 0:
            continue
        deviation = abs(length - nearest_multiple_of_4) / nearest_multiple_of_4
        if deviation <= 0.15:
            square_count += 1

    total = len(phrase_lengths_bars)
    squareness = square_count / total if total > 0 else 0.0

    return round(squareness, 4), phrase_lengths_bars


# ===========================================================================
# Per-track worker
# ===========================================================================

def process_track(args) -> tuple:
    """
    Worker function — processes one track and returns (path, feature_dict).
    Designed to run in a multiprocessing Pool.
    """
    audio_path, rel_path, sample_rate, msaf_algorithm = args

    result = {
        'bpm': None,
        'time_signature': None,
        'squareness_score': None,
        'phrase_lengths_bars': [],
        'core_start_sec': None,
        'core_end_sec': None,
        'num_segments_total': 0,
        'num_segments_core': 0,
        'intro_removed': False,
        'status': 'error',
    }

    try:
        # ── 1. Load audio ──────────────────────────────────────────────────
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

        if len(y) < sr * 10:
            result['status'] = 'too_short'
            return rel_path, result

        # ── 2. Segment ─────────────────────────────────────────────────────
        if MSAF_AVAILABLE:
            boundaries = _segment_with_msaf(audio_path, algorithm=msaf_algorithm)
        else:
            boundaries = _segment_with_librosa(y, sr)

        result['num_segments_total'] = max(0, len(boundaries) - 1)

        # ── 3. Crop to core (remove intro + outro) ─────────────────────────
        y_core, core_start, core_end, intro_removed = _crop_core(y, sr, boundaries)

        result['core_start_sec'] = round(core_start, 3)
        result['core_end_sec']   = round(core_end, 3)
        result['intro_removed']  = intro_removed

        if len(y_core) < sr * 10:
            result['status'] = 'core_too_short'
            return rel_path, result

        # ── 4. BPM on core ─────────────────────────────────────────────────
        bpm = _estimate_bpm(y_core, sr)
        result['bpm'] = round(bpm, 2)

        # ── 5. Beat times on core ──────────────────────────────────────────
        onset_env = librosa.onset.onset_strength(y=y_core, sr=sr, aggregate=np.median)
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, bpm=bpm, trim=False
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # ── 6. Time signature ──────────────────────────────────────────────
        time_sig = _estimate_time_signature(y_core, sr, beat_times)
        result['time_signature'] = time_sig

        # ── 7. Squareness on core segments ─────────────────────────────────
        # Re-express segment boundaries relative to the core
        core_segment_times = [
            t for t in boundaries
            if core_start <= t <= core_end
        ]
        result['num_segments_core'] = max(0, len(core_segment_times) - 1)

        squareness, phrase_bars = _compute_squareness(
            core_segment_times, core_start, core_end, bpm, time_sig
        )
        result['squareness_score']    = squareness
        result['phrase_lengths_bars'] = phrase_bars
        result['status'] = 'ok'

    except Exception as e:
        result['status'] = f'error: {type(e).__name__}: {str(e)[:120]}'

    return rel_path, result


# ===========================================================================
# TSV loading
# ===========================================================================

def load_track_paths(tsv_files: list, audio_dir: Path) -> list:
    """
    Parse one or more MTG-Jamendo TSV files and return a list of
    (absolute_path, relative_path) tuples for existing audio files.
    """
    seen = set()
    tracks = []

    for tsv_file in tsv_files:
        with open(tsv_file, 'r', encoding='utf-8') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                rel_path = parts[3]  # PATH column
                if rel_path in seen:
                    continue
                seen.add(rel_path)

                abs_path = audio_dir / rel_path
                if not abs_path.exists():
                    alt = audio_dir / rel_path.replace('.mp3', '.low.mp3')
                    if alt.exists():
                        abs_path = alt
                    else:
                        continue
                tracks.append((str(abs_path), rel_path))

    return tracks


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Precompute structural features (BPM, time sig, squareness) for MTG-Jamendo'
    )
    parser.add_argument(
        '--tsv', nargs='+', required=True,
        help='One or more MTG-Jamendo split TSV files'
    )
    parser.add_argument('--audio', required=True, help='Root audio directory')
    parser.add_argument('--output', required=True, help='Output JSON cache path')
    parser.add_argument(
        '--algorithm', default='scluster',
        choices=['scluster', 'foote', 'olda', 'sf'],
        help='MSAF segmentation algorithm (default: scluster)'
    )
    parser.add_argument('--sample-rate', type=int, default=22050,
                        help='Sample rate for analysis (default: 22050)')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 1),
                        help='Parallel workers (default: cpu_count - 1)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip tracks already present in the output file')
    args = parser.parse_args()

    audio_dir = Path(args.audio)
    output_path = Path(args.output)

    # Load existing cache if resuming
    cache = {}
    if args.resume and output_path.exists():
        with open(output_path, 'r') as f:
            cache = json.load(f)
        print(f"[resume] Loaded {len(cache)} existing entries from {output_path}")

    # Collect tracks
    all_tracks = load_track_paths(args.tsv, audio_dir)
    print(f"Found {len(all_tracks)} unique tracks across {len(args.tsv)} TSV file(s)")

    # Filter already-done
    if args.resume:
        all_tracks = [(a, r) for a, r in all_tracks if r not in cache]
        print(f"Remaining to process: {len(all_tracks)}")

    if not all_tracks:
        print("Nothing to process.")
        return

    # Build worker args
    worker_args = [
        (abs_path, rel_path, args.sample_rate, args.algorithm)
        for abs_path, rel_path in all_tracks
    ]

    # Progress tracking
    done = 0
    errors = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(worker_args)} tracks with {args.workers} worker(s)...")
    print(f"MSAF available: {MSAF_AVAILABLE}  |  Algorithm: {args.algorithm}\n")

    def _save():
        with open(output_path, 'w') as f:
            json.dump(cache, f, indent=2)

    with Pool(processes=args.workers) as pool:
        for rel_path, features in pool.imap_unordered(process_track, worker_args, chunksize=1):
            done += 1
            cache[rel_path] = features

            status = features['status']
            if status != 'ok':
                errors += 1

            # Progress line
            bpm_str = f"{features['bpm']:.1f} BPM" if features['bpm'] else 'BPM?'
            ts_str  = f"{features['time_signature']}/4" if features['time_signature'] else 'TS?'
            sq_str  = f"sq={features['squareness_score']:.2f}" if features['squareness_score'] is not None else 'sq=?'
            intro   = 'trimmed' if features['intro_removed'] else 'full'
            print(
                f"[{done:>5}/{len(worker_args)}] {status:<8} | {bpm_str:<12} {ts_str:<6} {sq_str:<8} "
                f"| {intro} | {rel_path}"
            )

            # Save every 50 tracks so progress isn't lost on crash
            if done % 50 == 0:
                _save()
                print(f"  → checkpoint saved ({done} done, {errors} errors)")

    _save()

    ok_count = sum(1 for v in cache.values() if v['status'] == 'ok')
    print(f"\n{'='*60}")
    print(f"Done. {ok_count}/{len(cache)} tracks processed successfully.")
    print(f"Errors: {errors}")
    print(f"Cache saved to: {output_path}")


if __name__ == '__main__':
    main()
