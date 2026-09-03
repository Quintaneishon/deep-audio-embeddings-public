import subprocess
import json
import librosa
import numpy as np
from sklearn.manifold import TSNE
import umap.umap_ as umap
import soundfile as sf
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Tuple
from scipy.stats import skew

def get_audio_duration(audio_path):
    """
    Get audio duration using ffprobe (supports all formats: MP3, M4A, WAV, FLAC, OGG, etc.)
    This avoids the librosa soundfile/audioread deprecation warning.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds (float) or None if failed
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception:
        # Fallback to librosa if ffprobe fails
        try:
            return librosa.get_duration(path=str(audio_path))
        except:
            return None

def proyectar_embeddings(embeddings, metodo='umap', std_normalize=False, random_state=42, n_components=2):
    """
    Project embeddings to 2D or 3D using UMAP or t-SNE.

    Args:
        embeddings: 2D array of embeddings (n_samples, n_dimensions)
        metodo: Projection method ('umap' or 'tsne')
        std_normalize: Whether to apply standard normalization
        random_state: Random seed for reproducibility
        n_components: Number of dimensions for projection (2 or 3)

    Returns:
        coords: 2D or 3D coordinates (n_samples, n_components)
    """
    if embeddings is None or len(embeddings.shape) != 2:
        raise ValueError("Embeddings debe ser una matriz 2D (n_muestras, n_dimensiones).")

    if n_components not in [2, 3]:
        raise ValueError("n_components debe ser 2 o 3.")

    X = embeddings.copy()
    n_samples = X.shape[0]

    # Check if we have enough samples for dimensionality reduction
    min_samples = n_components if n_components == 2 else 3
    if n_samples < min_samples:
        raise ValueError(f"Se requieren al menos {min_samples} muestras para proyección {n_components}D, pero solo hay {n_samples}. "
                        "Procesa más archivos de audio antes de calcular proyecciones.")

    # STD-Normalization (optional)
    if std_normalize:
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1e-8
        X = X / stds

    if metodo == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=random_state, n_iter=1000)
        coords = reducer.fit_transform(X)

    elif metodo == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=15, min_dist=0.1)
        coords = reducer.fit_transform(X)

    else:
        raise ValueError("Método no válido. Usa: 'tsne' o 'umap'.")

    return coords

def load_audio_safe(audio_path, sr=None):
    """
    Load audio safely without soundfile/audioread warnings.
    Tries soundfile first (supports WAV/FLAC/OGG), falls back to librosa for MP3/M4A.
    Always converts stereo to mono.

    Args:
        audio_path: Path to audio file
        sr: Target sample rate (None = keep original)

    Returns:
        y: Audio waveform as numpy array (mono, 1D)
        sr: Sample rate
    """
    try:
        # Try soundfile first for supported formats (WAV, FLAC, OGG)
        y, orig_sr = sf.read(audio_path, dtype='float32')

        # Convert stereo to mono if needed
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Resample if needed
        if sr is not None and sr != orig_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            return y, sr
        return y, orig_sr
    except:
        # Fall back to librosa for MP3/M4A (suppress warnings)
        # librosa.load automatically converts to mono by default
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            y, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
        return y, loaded_sr


def compute_hubness_skewness(embeddings: np.ndarray, k: int = 10) -> Tuple[float, np.ndarray]:
    """
    Compute hubness as skewness of k-occurrence distribution.

    Hubness is a phenomenon in high-dimensional spaces where some points
    become hubs that appear in many k-nearest neighbor lists.

    Args:
        embeddings: [N, D] embedding matrix
        k: Number of nearest neighbors to consider

    Returns:
        skewness: Hubness measure (higher = more hubness problem, <0.2 is good)
        k_occurrences: Array of k-occurrence counts for each point
    """
    n_samples = len(embeddings)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # For each sample, find k nearest neighbors
    k_occurrences = np.zeros(n_samples)

    for i in range(n_samples):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self

        # Get k nearest neighbors
        top_k_indices = np.argsort(sims)[-k:]

        # Count occurrences
        k_occurrences[top_k_indices] += 1

    # Compute skewness
    skewness_value = skew(k_occurrences)

    return skewness_value, k_occurrences

def compute_ndcg_at_k(embeddings: np.ndarray, labels: List[Set], k: int = 10) -> float:
    """
    Compute nDCG@K for multi-label tag-based retrieval using GLOBAL IDCG.

    For each query, retrieves k nearest neighbors and computes relevance
    based on tag overlap (Jaccard similarity). The Ideal DCG is computed
    using the true top-k most relevant items from the ENTIRE corpus,
    not just re-ranking the retrieved items.

    Args:
        embeddings: [N, D] embedding matrix
        labels: List of N sets, where each set contains tags for that sample
        k: Number of results to consider

    Returns:
        mean_ndcg: Average nDCG@K across all queries (0-1, higher is better)
    """
    n_samples = len(embeddings)
    sim_matrix = cosine_similarity(embeddings)

    # Precompute all pairwise Jaccard similarities for efficiency
    # This avoids redundant computation in the inner loop
    jaccard_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                continue
            union_size = len(labels[i] | labels[j])
            if union_size > 0:
                jaccard_matrix[i, j] = len(labels[i] & labels[j]) / union_size

    ndcg_scores = []

    for i in range(n_samples):
        # Get query labels
        query_labels = labels[i]

        if len(query_labels) == 0:
            continue

        # Get similarities from embedding space
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # Exclude self

        # Get top-k results based on embedding similarity
        top_k_indices = np.argsort(sims)[-k:][::-1]

        # Compute relevance scores for retrieved items (Jaccard similarity)
        relevance = jaccard_matrix[i, top_k_indices]

        # DCG@K: Discounted Cumulative Gain of the retrieved results
        dcg = np.sum(relevance / np.log2(np.arange(2, k + 2)))

        # GLOBAL Ideal DCG: Use the TRUE best-k items from the entire corpus
        # This is the key fix - we compare against the best possible retrieval
        all_relevances = jaccard_matrix[i].copy()
        all_relevances[i] = -np.inf  # Exclude self

        # Get the k highest relevance scores from the entire corpus
        ideal_relevance = np.sort(all_relevances)[-k:][::-1]
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, k + 2)))

        # nDCG = DCG / IDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_recall_at_k(embeddings: np.ndarray, genre_labels: List[str], k: int) -> float:
    """
    Recall@K: fraction of queries where at least 1 same-genre track appears in top-k results.

    Args:
        embeddings: [N, D] embedding matrix
        genre_labels: list of N genre strings (one per track)
        k: cutoff rank

    Returns:
        recall: float in [0, 1]
    """
    n_samples = len(embeddings)
    sim_matrix = cosine_similarity(embeddings)

    hits = 0
    for i in range(n_samples):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        top_k_indices = np.argsort(sims)[-k:]
        query_genre = genre_labels[i]
        if any(genre_labels[idx] == query_genre for idx in top_k_indices):
            hits += 1

    return hits / n_samples
