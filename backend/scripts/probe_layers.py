#!/usr/bin/env python3
"""
probe_layers.py — Layer depth probing for audio embedding models

Evaluates FMA metrics (hubness, nDCG@10, genre agreement, Spearman ρ) at
each layer of MERT, Whisper, MusiCNN and VGG to find the optimal extraction
depth for the DJ similarity graph — without retraining anything.

Key insight: the last layer is over-specialized for the pretraining task
(genre classification / speech). Intermediate layers often capture richer
representations for music similarity (timbre, rhythm, harmony).

Usage (from the repository root):
    FMA_METADATA_DIR=/path/to/fma_metadata python -m backend.scripts.probe_layers
    python -m backend.scripts.probe_layers --models mert whisper
    python -m backend.scripts.probe_layers --models all --max-tracks 2000 --cuda 1
    python -m backend.scripts.probe_layers --output backend/reports/layer_probe.txt

Requires:
    FMA_METADATA_DIR env var pointing to fma_metadata/ directory
    (same as required by evaluate_fma.py)
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend import config, utils
from backend.evaluate_fma import (
    load_fma_metadata,
    _pairwise_spearman,
    _genre_centroid_metrics,
)

ECHONEST_FEATURES = ['tempo', 'energy', 'danceability', 'instrumentalness', 'valence']


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(embeddings: np.ndarray,
                    genre_labels: List[str],
                    echo_idx: List[int],
                    fma_data: dict,
                    filenames: List[str]) -> dict:
    """Compute full FMA metric suite on a given embedding matrix."""
    hubness_skew, _ = utils.compute_hubness_skewness(embeddings, k=10)
    genre_ndcg = utils.compute_ndcg_at_k(embeddings, [{g} for g in genre_labels], k=10)
    centroid_sim, agreement = _genre_centroid_metrics(embeddings, genre_labels)

    echo_rho = {}
    if len(echo_idx) >= 20:
        echo_emb = embeddings[echo_idx]
        for feat in ECHONEST_FEATURES:
            vals = np.array([fma_data[filenames[i]][feat] for i in echo_idx], dtype=float)
            rho, _, _ = _pairwise_spearman(echo_emb, vals)
            echo_rho[feat] = rho

    return {
        'hubness':   float(hubness_skew),
        'ndcg10':    float(genre_ndcg),
        'cent_sim':  float(centroid_sim),
        'agreement': float(agreement),
        'echo':      echo_rho,
    }


def fmt_row(label: str, dim: int, m: dict, current: bool = False) -> str:
    e = m['echo']
    marker = ' ← current' if current else ''
    return (
        f"  {label:<20} [{dim:>4}d]  "
        f"hub={m['hubness']:+.4f}  nDCG={m['ndcg10']:.4f}  "
        f"agree={m['agreement']:.1%}  "
        f"ρ_tempo={e.get('tempo', float('nan')):+.4f}  "
        f"ρ_energy={e.get('energy', float('nan')):+.4f}  "
        f"ρ_dance={e.get('danceability', float('nan')):+.4f}  "
        f"ρ_instr={e.get('instrumentalness', float('nan')):+.4f}"
        f"{marker}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spatial pooling helper
# ─────────────────────────────────────────────────────────────────────────────

def pool_spatial(x: torch.Tensor) -> torch.Tensor:
    """Mean pool over all dims after batch and channel → [B, C]."""
    if x.dim() > 2:
        return x.flatten(start_dim=2).mean(dim=-1)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# MERT layer prober
# ─────────────────────────────────────────────────────────────────────────────

def probe_mert(audio_paths: List[str],
               filenames: List[str],
               device: torch.device) -> Tuple[Dict[str, np.ndarray], List[bool]]:
    """
    Extract MERT embeddings at layers 0, 3, 6, 9, 12.
    Layer 0 = after feature projection (before transformer blocks).
    Layer N = after Nth transformer block.
    Current extractor uses layer 12 (last).
    """
    print("  Loading MERT-v1-95M...")
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    model_name = config.MERT_MODEL_IDS['95m']
    cache_dir   = config.MODEL_WEIGHTS['mert']['95m']

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True,
        output_hidden_states=True).to(device)
    model.eval()

    num_layers    = model.config.num_hidden_layers   # 12 for 95M
    probe_indices = sorted(set(range(0, num_layers + 1, 3)) | {num_layers})
    names         = [f"layer_{i:02d}" for i in probe_indices]
    buffers       = {n: [] for n in names}
    valid_mask    = []

    import librosa

    for i, (path, fn) in enumerate(zip(audio_paths, filenames)):
        if i % 200 == 0:
            print(f"    MERT {i}/{len(audio_paths)}")
        try:
            y, sr = utils.load_audio_safe(path)
            if sr != 24000:
                y = librosa.resample(y, orig_sr=sr, target_sr=24000)

            from backend.DL.models.MERT import pad_or_trim_audio
            y_t = pad_or_trim_audio(torch.from_numpy(y).float())
            inputs = processor(y_t.numpy(), sampling_rate=24000,
                               return_tensors='pt', padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                # hidden_states: tuple of (num_layers+1) × [1, seq_len, hidden_size]
                for name, idx in zip(names, probe_indices):
                    h = out.hidden_states[idx]          # [1, seq, D]
                    pooled = h.mean(dim=1).squeeze(0).cpu().numpy()
                    buffers[name].append(pooled)
            valid_mask.append(True)

        except Exception as e:
            for name in names:
                buffers[name].append(None)
            valid_mask.append(False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {}
    for name in names:
        arrs = [x for x in buffers[name] if x is not None]
        result[name] = np.stack(arrs) if arrs else np.zeros((0, 1))
    return result, valid_mask, probe_indices, num_layers


# ─────────────────────────────────────────────────────────────────────────────
# Whisper layer prober
# ─────────────────────────────────────────────────────────────────────────────

def probe_whisper(audio_paths: List[str],
                  filenames: List[str],
                  device: torch.device) -> Tuple[Dict[str, np.ndarray], List[bool]]:
    """
    Extract Whisper-base encoder embeddings at each of the 6 encoder blocks
    plus the pre-transformer representation.
    Current contrastive extractor uses layer 2 (n_audio_layer // 3).
    """
    print("  Loading Whisper base...")
    import whisper as whisper_lib

    wmodel  = whisper_lib.load_model('base', device=device)
    wmodel.eval()
    encoder = wmodel.encoder
    n_blocks = len(encoder.blocks)   # 6 for base

    # Probe every layer: pre-transformer + each block
    names  = ['pre_transformer'] + [f"layer_{i:02d}" for i in range(1, n_blocks + 1)]
    buffers = {n: [] for n in names}
    valid_mask = []

    for i, (path, fn) in enumerate(zip(audio_paths, filenames)):
        if i % 200 == 0:
            print(f"    Whisper {i}/{len(audio_paths)}")
        try:
            audio = whisper_lib.load_audio(path)
            audio = whisper_lib.pad_or_trim(audio)
            mel   = whisper_lib.log_mel_spectrogram(
                audio, n_mels=wmodel.dims.n_mels).unsqueeze(0).to(device)

            with torch.no_grad():
                x = F.gelu(encoder.conv1(mel))
                x = F.gelu(encoder.conv2(x))
                x = x.permute(0, 2, 1)
                x = (x + encoder.positional_embedding).to(x.dtype)

                # layer 0 = pre-transformer
                buffers['pre_transformer'].append(x.mean(dim=1).squeeze(0).cpu().numpy())

                for b_idx, block in enumerate(encoder.blocks):
                    x = block(x)
                    name = f"layer_{b_idx + 1:02d}"
                    feat = x if b_idx < n_blocks - 1 else encoder.ln_post(x)
                    buffers[name].append(feat.mean(dim=1).squeeze(0).cpu().numpy())

            valid_mask.append(True)

        except Exception as e:
            for name in names:
                buffers[name].append(None)
            valid_mask.append(False)

    del wmodel
    gc.collect()
    torch.cuda.empty_cache()

    result = {}
    for name in names:
        arrs = [x for x in buffers[name] if x is not None]
        result[name] = np.stack(arrs) if arrs else np.zeros((0, 1))

    # Current contrastive models use layer n_blocks//3
    current_layer_idx = n_blocks // 3
    return result, valid_mask, names, current_layer_idx


# ─────────────────────────────────────────────────────────────────────────────
# MusiCNN layer prober
# ─────────────────────────────────────────────────────────────────────────────

def probe_musicnn(audio_paths: List[str],
                  filenames: List[str],
                  device: torch.device) -> Tuple[Dict[str, np.ndarray], List[bool]]:
    """
    Extract MusiCNN (MSD) features at 5 stages:
      front_end   — after parallel branch concat (561-ch, before backend convs)
      res1        — after 1st backend 1D-conv
      res2        — after 2nd backend 1D-conv + residual
      res3        — after 3rd backend 1D-conv + residual
      global_pool — after max+avg global pooling of [front+res1+res2+res3]
      dense1      — after Dense+BN+ReLU  ← current embedding
    """
    print("  Loading MusiCNN (MSD)...")
    from backend.DL.models.MusiCNN import Musicnn

    model = Musicnn(n_class=config.N_TAGS, dataset='msd')
    model.load_state_dict(
        torch.load(config.MODEL_WEIGHTS['musicnn']['msd'],
                   map_location=device, weights_only=True))
    model.to(device).eval()

    stage_names = ['front_end', 'res1', 'res2', 'res3', 'global_pool', 'dense1']
    buffers     = {n: [] for n in stage_names}
    valid_mask  = []

    for i, (path, fn) in enumerate(zip(audio_paths, filenames)):
        if i % 200 == 0:
            print(f"    MusiCNN {i}/{len(audio_paths)}")
        try:
            y, _ = utils.load_audio_safe(path, sr=16000)
            x = torch.from_numpy(y).float().unsqueeze(0).to(device)

            with torch.no_grad():
                # Spectrogram
                x = model.spec(x)
                x = model.to_db(x)
                x = x.unsqueeze(1)
                x = model.spec_bn(x)       # [1, 1, n_mels, T]

                # Front-end (5 parallel branches)
                out = torch.cat([layer(x) for layer in model.layers], dim=1)
                out = out.squeeze(2)       # [1, 561, T]
                length = out.size(2)

                buffers['front_end'].append(out.mean(dim=-1).squeeze(0).cpu().numpy())

                # Backend residual convs
                res1 = model.layer1(out)
                buffers['res1'].append(res1.mean(dim=-1).squeeze(0).cpu().numpy())

                res2 = model.layer2(res1) + res1
                buffers['res2'].append(res2.mean(dim=-1).squeeze(0).cpu().numpy())

                res3 = model.layer3(res2) + res2
                buffers['res3'].append(res3.mean(dim=-1).squeeze(0).cpu().numpy())

                # Global max+avg pooling
                cat = torch.cat([out, res1, res2, res3], dim=1)   # [1, 561+3*512, T]
                mp   = nn.MaxPool1d(length)(cat).squeeze(2)
                avgp = nn.AvgPool1d(length)(cat).squeeze(2)
                pool = torch.cat([mp, avgp], dim=1)                # [1, (561+3*512)*2]
                buffers['global_pool'].append(pool.squeeze(0).cpu().numpy())

                # Dense1 — current embedding
                emb = model.relu(model.bn(model.dense1(pool)))
                buffers['dense1'].append(emb.squeeze(0).cpu().numpy())

            valid_mask.append(True)

        except Exception as e:
            for name in stage_names:
                buffers[name].append(None)
            valid_mask.append(False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {}
    for name in stage_names:
        arrs = [x for x in buffers[name] if x is not None]
        result[name] = np.stack(arrs) if arrs else np.zeros((0, 1))
    return result, valid_mask, stage_names, stage_names.index('dense1')


# ─────────────────────────────────────────────────────────────────────────────
# VGG layer prober
# ─────────────────────────────────────────────────────────────────────────────

def probe_vgg(audio_paths: List[str],
              filenames: List[str],
              device: torch.device) -> Tuple[Dict[str, np.ndarray], List[bool]]:
    """
    Extract VGG_Res (MSD) features at 5 stages.
    Each ResBlock uses stride=2, so spatial dims halve each time.
    After layer7: freq dim = 1 → MaxPool1d over time → [B, 512].
    Current embedding = dense1 output.
    """
    print("  Loading VGG_Res (MSD)...")
    from backend.DL.models.VGG import VGG_Res

    model = VGG_Res(n_class=config.N_TAGS, use_simple_res=False)
    model.load_state_dict(
        torch.load(config.MODEL_WEIGHTS['vgg']['msd'],
                   map_location=device, weights_only=True))
    model.to(device).eval()

    stage_names = ['layer2', 'layer4', 'layer6', 'layer7_pool', 'dense1']
    buffers     = {n: [] for n in stage_names}
    valid_mask  = []

    for i, (path, fn) in enumerate(zip(audio_paths, filenames)):
        if i % 200 == 0:
            print(f"    VGG {i}/{len(audio_paths)}")
        try:
            y, _ = utils.load_audio_safe(path, sr=16000)
            x = torch.from_numpy(y).float().unsqueeze(0).to(device)

            with torch.no_grad():
                x = model.spec(x)
                x = model.to_db(x)
                x = x.unsqueeze(1)
                x = model.spec_bn(x)       # [1, 1, 128, T]

                x = model.layer1(x)
                x = model.layer2(x)        # [1, 128, 32, T/4]
                buffers['layer2'].append(pool_spatial(x).squeeze(0).cpu().numpy())

                x = model.layer3(x)
                x = model.layer4(x)        # [1, 256, 8, T/16]
                buffers['layer4'].append(pool_spatial(x).squeeze(0).cpu().numpy())

                x = model.layer5(x)
                x = model.layer6(x)        # [1, 256, 2, T/64]
                buffers['layer6'].append(pool_spatial(x).squeeze(0).cpu().numpy())

                x = model.layer7(x)
                x = x.squeeze(2)           # [1, 512, T/128] (freq dim = 1 after 7 blocks)
                if x.size(-1) != 1:
                    x = nn.MaxPool1d(x.size(-1))(x)
                x = x.squeeze(2)           # [1, 512]
                buffers['layer7_pool'].append(x.squeeze(0).cpu().numpy())

                emb = model.relu(model.bn(model.dense1(x)))
                buffers['dense1'].append(emb.squeeze(0).cpu().numpy())

            valid_mask.append(True)

        except Exception as e:
            for name in stage_names:
                buffers[name].append(None)
            valid_mask.append(False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    result = {}
    for name in stage_names:
        arrs = [x for x in buffers[name] if x is not None]
        result[name] = np.stack(arrs) if arrs else np.zeros((0, 1))
    return result, valid_mask, stage_names, stage_names.index('dense1')


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PROBERS = {
    'mert':    probe_mert,
    'whisper': probe_whisper,
    'musicnn': probe_musicnn,
    'vgg':     probe_vgg,
}

MODEL_DESCRIPTIONS = {
    'mert':    'MERT-v1-95M (12 transformer layers, 768-dim)',
    'whisper': 'Whisper base encoder (6 blocks, 512-dim)',
    'musicnn': 'MusiCNN MSD (CNN front-end + 3 residual backend blocks)',
    'vgg':     'VGG_Res MSD (7 residual blocks, 512-dim)',
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=['all'],
                        choices=['all', 'mert', 'whisper', 'musicnn', 'vgg'])
    parser.add_argument('--max-tracks', type=int, default=None,
                        help='Subsample N tracks for faster runs (default: all ~8000). '
                             '2000 is a good balance for speed vs reliable Spearman ρ.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path (default: reports/layer_probe_TIMESTAMP.txt)')
    parser.add_argument('--cuda', type=int, default=1,
                        help='CUDA device index (default: 1, leaving GPU 0 for desktop)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    models_to_probe = list(MODEL_PROBERS.keys()) if 'all' in args.models else args.models
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    timestamp = datetime.now().strftime('%Y%m%dT%H-%M-%S')
    out_path  = args.output or str(
        Path(config.REPORTS_DIR) / f'layer_probe_{timestamp}.txt')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────────
    print("\nLoading FMA metadata...")
    fma_data = load_fma_metadata()

    # Resolve existing audio files
    audio_dir      = Path(config.AUDIO_DIR)
    audio_paths    = []
    valid_filenames = []
    for fn in fma_data:
        p = audio_dir / fn
        if p.exists():
            audio_paths.append(str(p))
            valid_filenames.append(fn)

    print(f"Audio files found: {len(audio_paths)}")

    if args.max_tracks and args.max_tracks < len(audio_paths):
        import random
        random.seed(args.seed)
        idx = random.sample(range(len(audio_paths)), args.max_tracks)
        audio_paths     = [audio_paths[i]     for i in idx]
        valid_filenames = [valid_filenames[i]  for i in idx]
        print(f"Subsampled to: {len(audio_paths)} tracks")

    with open(out_path, 'w') as report:

        header = (
            f"=== Layer Probe Results — {device} ===\n"
            f"Generated: {timestamp}\n"
            f"Tracks: {len(audio_paths)} / {len(fma_data)} (FMA-small)\n"
        )
        print(header)
        report.write(header + '\n')

        for model_name in models_to_probe:
            sec_header = (
                f"\n{'='*70}\n"
                f"{model_name.upper()}  —  {MODEL_DESCRIPTIONS[model_name]}\n"
                f"{'='*70}"
            )
            print(sec_header)
            report.write(sec_header + '\n')

            t0 = time.time()
            prober = MODEL_PROBERS[model_name]
            layer_embs, valid_mask, layer_names, current_idx = prober(
                audio_paths, valid_filenames, device)

            elapsed = time.time() - t0
            n_valid = sum(valid_mask)
            print(f"  Extracted {n_valid}/{len(audio_paths)} tracks in {elapsed:.0f}s")

            # Filter filenames and labels to valid tracks
            vf = [valid_filenames[i] for i, ok in enumerate(valid_mask) if ok]
            gl = [fma_data[fn]['genre'] for fn in vf]
            ei = [j for j, fn in enumerate(vf) if fma_data[fn].get('tempo') is not None]

            col_header = (
                f"\n  {'Layer':<20} {'Dim':>5}  "
                f"{'Hubness':>8}  {'nDCG@10':>8}  {'Agree':>7}  "
                f"{'ρ_tempo':>8}  {'ρ_energy':>9}  {'ρ_dance':>8}  {'ρ_instr':>8}"
            )
            divider = f"  {'-'*110}"
            print(col_header)
            print(divider)
            report.write(col_header + '\n' + divider + '\n')

            best_ndcg, best_hub, best_layer_ndcg, best_layer_hub = -1, 99, None, None
            results = {}

            for idx_l, (name, emb) in enumerate(layer_embs.items()):
                if emb.shape[0] == 0:
                    continue
                is_current = (idx_l == current_idx)
                print(f"  Computing metrics for {name} (dim={emb.shape[1]})...", end='', flush=True)
                m = compute_metrics(emb, gl, ei, fma_data, vf)
                results[name] = m

                row = fmt_row(name, emb.shape[1], m, current=is_current)
                print('\r' + row)
                report.write(row + '\n')

                if m['ndcg10'] > best_ndcg:
                    best_ndcg, best_layer_ndcg = m['ndcg10'], name
                if abs(m['hubness']) < abs(best_hub):
                    best_hub, best_layer_hub = m['hubness'], name

            # Summary for this model
            summary = (
                f"\n  >> Best nDCG@10 : {best_layer_ndcg} ({best_ndcg:.4f})\n"
                f"  >> Best hubness : {best_layer_hub} ({best_hub:+.4f})\n"
            )
            print(summary)
            report.write(summary)

            gc.collect()
            torch.cuda.empty_cache()

        footer = f"\nResults saved to: {out_path}\n"
        print(footer)
        report.write(footer)

    print(f"Done. Report: {out_path}")


if __name__ == '__main__':
    main()
