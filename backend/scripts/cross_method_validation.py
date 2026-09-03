#!/usr/bin/env python3
"""
Cross-method validation: compare Serra09 HPCP distance vs embedding cosine distance
for a random sample of track pairs, then compute Spearman ρ between the two rankings.

Runs over a single model/dataset or all registered models at once (--all-models).

Usage (single model):
    python cross_method_validation.py \
        --model vgg_whisper_dual \
        --dataset msd \
        --n-pairs 200

Usage (all models):
    python cross_method_validation.py \
        --all-models \
        --n-pairs 200 \
        --output reports/cross_method_validation.txt \
        --csv    reports/cross_method_validation.csv
"""

import csv
import argparse
import logging
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import essentia.standard as es
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine as cosine_distance

from backend import config
from backend.extractors.extractors import ExtractorRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serra09 / HPCP helpers
# ---------------------------------------------------------------------------

def _load_audio_essentia(filepath: str) -> np.ndarray:
    loader = es.MonoLoader(filename=filepath, sampleRate=44100)
    return loader()


def _extract_hpcp(audio: np.ndarray) -> np.ndarray:
    windowing = es.Windowing(type="blackmanharris62")
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks()
    hpcp = es.HPCP()

    frames = []
    frame_size, hop_size = 4096, 2048
    for start in range(0, len(audio) - frame_size, hop_size):
        frame = audio[start : start + frame_size]
        spec = spectrum(windowing(frame))
        freqs, mags = spectral_peaks(spec)
        frames.append(hpcp(freqs, mags))

    return np.array(frames)  # (n_frames, 12)


def serra09_distance(path1: str, path2: str) -> float:
    """Return the Serra09 cover-song distance (lower = more similar)."""
    audio1 = _load_audio_essentia(path1)
    audio2 = _load_audio_essentia(path2)

    chroma1 = _extract_hpcp(audio1)
    chroma2 = _extract_hpcp(audio2)

    cross_sim = es.CrossSimilarityMatrix(binarize=True, binarizePercentile=0.095)
    binary_matrix = cross_sim(chroma1, chroma2)

    cover_sim = es.CoverSongSimilarity(alignmentType="serra09")
    _, distance = cover_sim(binary_matrix)
    return float(distance)


# ---------------------------------------------------------------------------
# Embedding cosine distance
# ---------------------------------------------------------------------------

def embedding_cosine_distance(
    extractor, weights_path: str, dataset: str, path1: str, path2: str
) -> float:
    """Return cosine distance (1 - cosine_similarity) between two tracks."""
    emb1 = extractor.extract(path1, weights_path=weights_path, dataset=dataset)
    emb2 = extractor.extract(path2, weights_path=weights_path, dataset=dataset)
    v1 = emb1.reshape(-1).astype(np.float64)
    v2 = emb2.reshape(-1).astype(np.float64)
    return float(cosine_distance(v1, v2))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_tracks(audio_dir: str) -> list[Path]:
    audio_dir = Path(audio_dir)
    tracks = sorted(
        p for p in audio_dir.rglob("*")
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    )
    return tracks


def sample_pairs(tracks: list[Path], n: int, seed: int) -> list[tuple[Path, Path]]:
    rng = random.Random(seed)
    all_pairs = list(combinations(tracks, 2))
    if len(all_pairs) <= n:
        log.warning("Only %d unique pairs available; using all of them.", len(all_pairs))
        return all_pairs
    return rng.sample(all_pairs, n)


def weights_available(weights_path: str) -> bool:
    """Return True if the weights path is a real file or a known non-file token (e.g. 'base')."""
    p = Path(weights_path)
    # Non-path tokens like 'base' or '95m' are used by Whisper / MERT directly
    if not p.is_absolute() and not p.exists():
        return True   # treat as a valid token
    return p.exists()


def build_model_list(registry: ExtractorRegistry) -> list[tuple]:
    """Return list of (model_key, dataset_key, weights_path, extractor) for available models."""
    entries = []
    for model_key, variants in config.MODEL_WEIGHTS.items():
        extractor = registry.get(model_key)
        if extractor is None:
            log.debug("No extractor registered for '%s' — skipping.", model_key)
            continue
        for dataset_key, weights_path in variants.items():
            if not weights_available(weights_path):
                log.info("Weights not found for %s/%s — skipping.", model_key, dataset_key)
                continue
            entries.append((model_key, dataset_key, weights_path, extractor))
    return entries


# ---------------------------------------------------------------------------
# Core evaluation for one model
# ---------------------------------------------------------------------------

def evaluate_model(
    extractor,
    weights_path: str,
    dataset: str,
    pairs: list[tuple[Path, Path]],
    cached_serra09: dict[tuple, float],
) -> tuple[list[dict], dict[tuple, float]]:
    """
    Evaluate one model against pre-sampled pairs.

    Returns
    -------
    rows : list of dicts with keys track1, track2, serra09_dist, emb_dist
    cached_serra09 : updated cache (so Serra09 is computed once across all models)
    """
    rows = []
    skipped = 0

    for i, (t1, t2) in enumerate(pairs, 1):
        key = (str(t1), str(t2))

        # Serra09 — compute once and cache
        if key not in cached_serra09:
            try:
                cached_serra09[key] = serra09_distance(str(t1), str(t2))
            except Exception as exc:
                log.warning("Serra09 failed (%s, %s): %s — skipping pair.", t1.name, t2.name, exc)
                cached_serra09[key] = None

        s09 = cached_serra09[key]
        if s09 is None:
            skipped += 1
            continue

        # Embedding distance
        try:
            emb = embedding_cosine_distance(extractor, weights_path, dataset, str(t1), str(t2))
        except Exception as exc:
            log.warning("Embedding failed (%s, %s): %s — skipping pair.", t1.name, t2.name, exc)
            skipped += 1
            continue

        rows.append({
            'track1': t1.name,
            'track2': t2.name,
            'serra09_dist': s09,
            'emb_dist': emb,
        })

    if skipped:
        log.warning("  Skipped %d pairs.", skipped)

    return rows, cached_serra09


def spearman_summary(rows: list[dict]) -> tuple[float, float]:
    if len(rows) < 2:
        return float('nan'), float('nan')
    s09  = np.array([r['serra09_dist'] for r in rows])
    embs = np.array([r['emb_dist']     for r in rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, pval = spearmanr(s09, embs)
    return float(rho), float(pval)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    all_results: list[dict],   # [{model, dataset, rows, rho, pval}, ...]
    audio_dir: str,
    n_pairs_requested: int,
    seed: int,
) -> str:
    col_w = 45
    header = f"{'Track 1':<{col_w}}  {'Track 2':<{col_w}}  {'Serra09':>10}  {'EmbCos':>10}"
    sep = "-" * len(header)
    wide_sep = "=" * len(header)

    lines = [
        wide_sep,
        "Cross-Method Validation Report  (Serra09 HPCP vs Embedding Cosine Distance)",
        f"  Audio dir : {audio_dir}",
        f"  Pairs     : {n_pairs_requested} requested  (seed={seed})",
        wide_sep,
        "",
        "SUMMARY",
        "-------",
        f"  {'Model/Dataset':<45}  {'N eval':>6}  {'Spearman ρ':>11}  {'p-value':>10}",
        "  " + "-" * 77,
    ]

    for res in all_results:
        label = f"{res['model']}/{res['dataset']}"
        n = len(res['rows'])
        rho_s  = f"{res['rho']:+.4f}" if not np.isnan(res['rho']) else "   n/a"
        pval_s = f"{res['pval']:.4e}"  if not np.isnan(res['pval']) else "      n/a"
        lines.append(f"  {label:<45}  {n:>6}  {rho_s:>11}  {pval_s:>10}")

    lines += ["", wide_sep, ""]

    for res in all_results:
        label = f"{res['model']} [{res['dataset']}]"
        lines += [
            f"--- {label} ---",
            f"  Evaluated : {len(res['rows'])} pairs",
            f"  Spearman ρ: {res['rho']:+.4f}  (p={res['pval']:.4e})"
            if not np.isnan(res['rho']) else "  Spearman ρ: n/a",
            "",
            header,
            sep,
        ]
        for r in res['rows']:
            t1_s = r['track1'][:col_w]
            t2_s = r['track2'][:col_w]
            lines.append(
                f"{t1_s:<{col_w}}  {t2_s:<{col_w}}  {r['serra09_dist']:>10.4f}  {r['emb_dist']:>10.4f}"
            )
        lines += [sep, ""]

    lines += [
        "Interpretation",
        "--------------",
        "  High ρ (|ρ| > 0.5) : model agrees with signal-level harmonic similarity.",
        "  Low  ρ (|ρ| < 0.3) : model captures different (timbral / semantic) structure.",
        "  Moderate ρ          : partial overlap with harmonic cues.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extremes summary
# ---------------------------------------------------------------------------

def build_extremes_summary(all_results: list[dict]) -> str:
    """
    Find the highest and lowest Serra09 pair across all evaluated rows,
    then for each extreme report:
      - best  EmbCos model: the one whose EmbCos most agrees with Serra09
        (highest EmbCos for the highest Serra09 pair; lowest EmbCos for the lowest)
      - worst EmbCos model: the one that most disagrees with Serra09
    Returns a formatted string ready to print / append to the report.
    """
    # Collect all (serra09, emb, model/dataset label, track1, track2)
    flat: list[tuple[float, float, str, str, str]] = []
    for res in all_results:
        label = f"{res['model']}/{res['dataset']}"
        for r in res['rows']:
            flat.append((r['serra09_dist'], r['emb_dist'], label, r['track1'], r['track2']))

    if not flat:
        return ""

    # Unique Serra09 values keyed by pair; each pair has the same Serra09 across models
    serra09_by_pair: dict[tuple[str, str], float] = {}
    for s09, _emb, _lbl, t1, t2 in flat:
        serra09_by_pair[(t1, t2)] = s09

    highest_pair = max(serra09_by_pair, key=lambda k: serra09_by_pair[k])
    lowest_pair  = min(serra09_by_pair, key=lambda k: serra09_by_pair[k])
    highest_s09  = serra09_by_pair[highest_pair]
    lowest_s09   = serra09_by_pair[lowest_pair]

    # For each extreme pair gather (emb_dist, label)
    def emb_for_pair(pair: tuple[str, str]) -> list[tuple[float, str]]:
        return [(emb, lbl) for s09, emb, lbl, t1, t2 in flat if (t1, t2) == pair]

    high_embs = emb_for_pair(highest_pair)
    low_embs  = emb_for_pair(lowest_pair)

    # best/worst for highest Serra09 pair
    high_best_emb,  high_best_model  = max(high_embs, key=lambda x: x[0])   # highest EmbCos agrees
    high_worst_emb, high_worst_model = min(high_embs, key=lambda x: x[0])   # lowest EmbCos disagrees

    # best/worst for lowest Serra09 pair
    low_best_emb,  low_best_model    = min(low_embs,  key=lambda x: x[0])   # lowest EmbCos agrees
    low_worst_emb, low_worst_model   = max(low_embs,  key=lambda x: x[0])   # highest EmbCos disagrees

    wide_sep = "=" * 80
    lines = [
        "",
        wide_sep,
        "EXTREMES SUMMARY  —  Best & Worst EmbCos agreement with Serra09",
        wide_sep,
        "",
        f"  Highest Serra09 pair  (Serra09 = {highest_s09:.4f}  →  most dissimilar harmonically)",
        f"    Track 1 : {highest_pair[0]}",
        f"    Track 2 : {highest_pair[1]}",
        f"    Best  EmbCos model  (highest EmbCos, agrees) : {high_best_model:<45}  EmbCos = {high_best_emb:.4f}",
        f"    Worst EmbCos model  (lowest  EmbCos, disagrees) : {high_worst_model:<45}  EmbCos = {high_worst_emb:.4f}",
        "",
        f"  Lowest Serra09 pair   (Serra09 = {lowest_s09:.4f}  →  most similar harmonically)",
        f"    Track 1 : {lowest_pair[0]}",
        f"    Track 2 : {lowest_pair[1]}",
        f"    Best  EmbCos model  (lowest  EmbCos, agrees)    : {low_best_model:<45}  EmbCos = {low_best_emb:.4f}",
        f"    Worst EmbCos model  (highest EmbCos, disagrees) : {low_worst_model:<45}  EmbCos = {low_worst_emb:.4f}",
        "",
        wide_sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(all_results: list[dict], csv_path: str) -> None:
    fieldnames = ['model', 'dataset', 'track1', 'track2', 'serra09_dist', 'emb_cosine_dist']
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in all_results:
            for r in res['rows']:
                writer.writerow({
                    'model':            res['model'],
                    'dataset':          res['dataset'],
                    'track1':           r['track1'],
                    'track2':           r['track2'],
                    'serra09_dist':     f"{r['serra09_dist']:.6f}",
                    'emb_cosine_dist':  f"{r['emb_dist']:.6f}",
                })
    log.info("CSV written to %s  (%d rows)", csv_path,
             sum(len(r['rows']) for r in all_results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-method validation: Serra09 HPCP vs embedding cosine distance"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all-models", action="store_true",
                      help="Run validation for every registered model with available weights")
    mode.add_argument("--model", help="Single model key (e.g. vgg_whisper_dual)")

    parser.add_argument("--dataset", default=None,
                        help="Dataset key for single-model mode (e.g. msd, mtat, base)")
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Number of random track pairs (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--audio-dir", default=None,
                        help="Audio directory (default: config.AUDIO_DIR)")
    parser.add_argument("--output", default=None,
                        help="Path for the text report (default: stdout only)")
    parser.add_argument("--csv", default=None,
                        help="Path for the CSV output (default: <output>.csv or reports/cross_method_validation.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    registry = ExtractorRegistry.create_default()

    # Build list of (model, dataset, weights, extractor) to evaluate
    if args.all_models:
        model_list = build_model_list(registry)
        if not model_list:
            log.error("No models with available weights found.")
            sys.exit(1)
        log.info("Running validation for %d model/dataset combinations.", len(model_list))
    else:
        if args.model not in config.MODEL_WEIGHTS:
            log.error("Unknown model '%s'. Available: %s", args.model, list(config.MODEL_WEIGHTS))
            sys.exit(1)
        variants = config.MODEL_WEIGHTS[args.model]
        dataset = args.dataset
        if dataset is None:
            if len(variants) == 1:
                dataset = next(iter(variants))
                log.info("--dataset not specified; using '%s'.", dataset)
            else:
                log.error("--dataset required for model '%s'. Options: %s", args.model, list(variants))
                sys.exit(1)
        if dataset not in variants:
            log.error("Unknown dataset '%s' for model '%s'. Options: %s", dataset, args.model, list(variants))
            sys.exit(1)
        extractor = registry.get(args.model)
        if extractor is None:
            log.error("No extractor registered for '%s'.", args.model)
            sys.exit(1)
        model_list = [(args.model, dataset, variants[dataset], extractor)]

    # Discover tracks and sample pairs
    audio_dir = args.audio_dir or config.AUDIO_DIR
    log.info("Scanning audio directory: %s", audio_dir)
    tracks = discover_tracks(audio_dir)
    if len(tracks) < 2:
        log.error("Need at least 2 audio files in '%s'.", audio_dir)
        sys.exit(1)
    log.info("Found %d tracks.", len(tracks))

    pairs = sample_pairs(tracks, args.n_pairs, args.seed)
    log.info("Sampled %d pairs (seed=%d).", len(pairs), args.seed)

    # Evaluate — Serra09 is computed once and cached across models
    all_results = []
    cached_serra09: dict[tuple, float] = {}

    for model_key, dataset_key, weights_path, extractor in model_list:
        log.info("Evaluating %s / %s ...", model_key, dataset_key)
        rows, cached_serra09 = evaluate_model(
            extractor, weights_path, dataset_key, pairs, cached_serra09
        )
        rho, pval = spearman_summary(rows)
        log.info("  ρ = %+.4f  (p = %.4e, n = %d)", rho, pval, len(rows))
        all_results.append({
            'model':   model_key,
            'dataset': dataset_key,
            'rows':    rows,
            'rho':     rho,
            'pval':    pval,
        })

    if not all_results:
        log.error("No results produced.")
        sys.exit(1)

    # Build and print report
    report = build_report(all_results, audio_dir, args.n_pairs, args.seed)
    print(report)

    # Build and print extremes summary
    extremes = build_extremes_summary(all_results)
    print(extremes)

    # Write text report (includes extremes summary appended)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + extremes + "\n")
        log.info("Report written to %s", out_path)

    # Write CSV
    csv_path = args.csv
    if csv_path is None:
        if args.output:
            csv_path = str(Path(args.output).with_suffix('.csv'))
        else:
            csv_path = str(Path(config.REPORTS_DIR) / 'cross_method_validation.csv')
    write_csv(all_results, csv_path)


if __name__ == "__main__":
    main()
