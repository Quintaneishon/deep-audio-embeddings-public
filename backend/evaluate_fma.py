"""
FMA-based evaluation of audio embedding models.

Uses FMA-small (8,000 tracks) as an independent test set — never seen during training.

Metrics
-------
1. Hubness (skewness)
       Geometric quality of the embedding space.
       < 0.5 is healthy; > 2.0 indicates severe hub concentration.

2. Genre nDCG@10  (8 genres, 100 % coverage)
       Retrieval quality measured against FMA genre_top labels.
       Fair baseline: all models were trained with a genre signal.

3. Echonest Spearman ρ  (~16 % of FMA tracks, ~160 selected songs)
       Spearman correlation between pairwise cosine distance and
       pairwise |Δfeature|.  Reports ρ for:
           tempo        (BPM proxy — advantage expected for Hybrid / MultiSignal)
           energy       (never in training — tests generalization)
           danceability (never in training — tests generalization)
           instrumentalness (never in training — tests generalization)
       Higher ρ = the embedding space is better organized along that dimension.
       Comparing Hybrid vs SupCon on tempo shows what BPM supervision adds.
       Comparing MultiSignal vs Hybrid on instrumentalness shows instrument
       supervision adds value beyond BPM.
"""

import csv
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from backend import config, utils
from backend.db import database
from backend.extractors.extractors import ExtractorRegistry


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loading
# ─────────────────────────────────────────────────────────────────────────────

def load_fma_metadata() -> Dict[str, dict]:
    """
    Load genre and Echonest features for the songs in selected_songs.csv.

    Returns
    -------
    dict  filename → {genre: str,
                      tempo: float|None, energy: float|None,
                      danceability: float|None, instrumentalness: float|None,
                      valence: float|None, speechiness: float|None,
                      acousticness: float|None}
    """
    csv_path = Path(config.CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"selected_songs.csv not found: {csv_path}")

    # Load selected filenames + genre from project CSV
    selected = {}
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            selected[row['filename']] = {'genre': row['genre']}

    # Load FMA tracks.csv for genre_top (fallback / cross-check)
    fma_dir = Path(config.FMA_METADATA_DIR)
    tracks_csv = fma_dir / 'tracks.csv'
    echonest_csv = fma_dir / 'echonest.csv'

    # Parse multi-level header CSVs
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    echonest = pd.read_csv(echonest_csv, index_col=0, header=[0, 1, 2])

    echonest_features = ['tempo', 'energy', 'danceability',
                         'instrumentalness', 'valence', 'speechiness', 'acousticness']

    result = {}
    for filename, meta in selected.items():
        track_id = int(filename.replace('.mp3', ''))
        entry = dict(meta)

        # Override genre from tracks.csv if present (more reliable)
        if track_id in tracks.index:
            genre_top = tracks.loc[track_id, ('track', 'genre_top')]
            if pd.notna(genre_top):
                entry['genre'] = str(genre_top)

        # Echonest features
        for feat in echonest_features:
            val = None
            if track_id in echonest.index:
                v = echonest.loc[track_id, ('echonest', 'audio_features', feat)]
                if pd.notna(v):
                    val = float(v)
            entry[feat] = val

        result[filename] = entry

    n_echo = sum(1 for v in result.values() if v.get('tempo') is not None)
    print(f"  Loaded {len(result)} tracks | Echonest coverage: {n_echo}/{len(result)}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise Spearman correlation
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_spearman(embeddings: np.ndarray,
                       values: np.ndarray) -> Tuple[float, float, int]:
    """
    Spearman ρ between pairwise cosine distances and pairwise |Δvalue|.

    Returns (rho, p_value, n_pairs).
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    n = len(embeddings)
    sim = cos_sim(embeddings)

    dists, diffs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1.0 - float(sim[i, j]))
            diffs.append(abs(float(values[i]) - float(values[j])))

    rho, pval = spearmanr(dists, diffs)
    return float(rho), float(pval), len(dists)


# ─────────────────────────────────────────────────────────────────────────────
# Genre centroid similarity
# ─────────────────────────────────────────────────────────────────────────────

def _genre_centroid_metrics(embeddings: np.ndarray,
                            genre_labels: List[str]) -> Tuple[float, float]:
    """
    Centroid-based genre evaluation.

    For each song: cosine similarity to its own genre centroid vs all centroids.

    Returns
    -------
    mean_sim_to_own_genre : float
    agreement_rate        : float  (fraction where nearest centroid == own genre)
    """
    unique_genres = list(set(genre_labels))
    if len(unique_genres) < 2:
        return 0.0, 0.0

    centroids = {
        g: embeddings[[i for i, gl in enumerate(genre_labels) if gl == g]].mean(axis=0)
        for g in unique_genres
    }
    centroid_matrix = np.array([centroids[g] for g in unique_genres])  # (G, D)

    sims = cos_sim(embeddings, centroid_matrix)  # (N, G)

    own_sims, agreements = [], 0
    for i, genre in enumerate(genre_labels):
        genre_idx = unique_genres.index(genre)
        own_sims.append(float(sims[i, genre_idx]))
        if int(sims[i].argmax()) == genre_idx:
            agreements += 1

    return float(np.mean(own_sims)), agreements / len(genre_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_fma_metrics() -> Dict[str, dict]:
    """
    Compute FMA-based evaluation metrics for all registered extractors.

    Returns
    -------
    dict  combo_key → metric_dict
    """
    print("\n" + "=" * 60)
    print("FMA EVALUATION METRICS")
    print("(Independent test set — not used in training)")
    print("=" * 60)

    # Load metadata once
    print("\nLoading FMA metadata...")
    try:
        fma_data = load_fma_metadata()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}

    all_filenames = list(fma_data.keys())

    registry = ExtractorRegistry.create_default()
    model_configs = registry.get_all_configs()
    results = {}

    for model_config in model_configs:
        combo_key = model_config.key
        print(f"\n--- {combo_key} ---")

        # ── Load embeddings from database ──────────────────────────────
        embeddings_list, valid_filenames = [], []
        for filename in all_filenames:
            data = database.get_embedding_by_filename(
                filename, model_config.model_name, model_config.dataset)
            if data is None or data['embedding'] is None:
                continue
            embeddings_list.append(data['embedding'])
            valid_filenames.append(filename)

        if len(embeddings_list) == 0:
            print(f"  No embeddings — skipping (run preprocess-all first)")
            continue

        embeddings = np.vstack(embeddings_list)
        print(f"  Embeddings: {embeddings.shape}")

        # ── 1. Hubness ────────────────────────────────────────────────
        hubness_skew, k_occ = utils.compute_hubness_skewness(embeddings, k=10)

        # ── 2. Genre nDCG@10 ─────────────────────────────────────────
        genre_labels_set = [{fma_data[f]['genre']} for f in valid_filenames]
        genre_ndcg = utils.compute_ndcg_at_k(embeddings, genre_labels_set, k=10)

        # ── 3. Genre centroid similarity & Recall ─────────────────────
        genre_labels_str = [fma_data[f]['genre'] for f in valid_filenames]
        genre_centroid_sim, genre_agreement = _genre_centroid_metrics(embeddings, genre_labels_str)

        recall_at_1  = utils.compute_recall_at_k(embeddings, genre_labels_str, k=1)
        recall_at_10 = utils.compute_recall_at_k(embeddings, genre_labels_str, k=10)

        # ── 4. Echonest Spearman correlations ─────────────────────────
        echonest_features = ['tempo', 'energy', 'danceability',
                             'instrumentalness', 'valence']
        echo_results = {}

        # Indices with echonest data
        echo_idx = [i for i, f in enumerate(valid_filenames)
                    if fma_data[f].get('tempo') is not None]

        if len(echo_idx) >= 20:
            echo_emb = embeddings[echo_idx]
            for feat in echonest_features:
                vals = np.array([fma_data[valid_filenames[i]][feat]
                                 for i in echo_idx], dtype=float)
                rho, pval, n_pairs = _pairwise_spearman(echo_emb, vals)
                echo_results[feat] = {'rho': rho, 'pval': pval, 'n_pairs': n_pairs}
                print(f"    Echonest {feat:20s} ρ={rho:+.4f}  (p={pval:.3f}, {n_pairs} pairs)")
        else:
            print(f"  Echonest: only {len(echo_idx)} tracks with data — skipping")

        results[combo_key] = {
            'n_samples': len(valid_filenames),
            'n_echonest': len(echo_idx),
            'hubness_skewness': float(hubness_skew),
            'mean_k_occurrence': float(np.mean(k_occ)),
            'max_k_occurrence': float(np.max(k_occ)),
            'genre_ndcg_at_10': float(genre_ndcg),
            'genre_centroid_sim': genre_centroid_sim,
            'genre_agreement_rate': genre_agreement,
            'recall_at_1': float(recall_at_1),
            'recall_at_10': float(recall_at_10),
            'echonest': echo_results,
        }
        print(f"  Hubness skew: {hubness_skew:.4f}  |  Genre nDCG@10: {genre_ndcg:.4f}  "
              f"|  Centroid sim: {genre_centroid_sim:.4f}  |  Agreement: {genre_agreement:.2%}  "
              f"|  R@1: {recall_at_1:.4f}  R@10: {recall_at_10:.4f}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
