#!/usr/bin/env python3
"""
Selección de tripletes para el experimento perceptual humano (eval game).

Este script reconstruye fielmente la metodología con la que se generaron los
10 tripletes (anchor + opción A + opción B) que se insertaron en la tabla
`eval_triplets` el 2026-04-17 y que se usaron para medir el Human-Model
Agreement (HMA) de la tesis.

────────────────────────────────────────────────────────────────────────────
METODOLOGÍA (resumen — ver docs/eval_triplet_selection.md para el detalle)
────────────────────────────────────────────────────────────────────────────
La dificultad de cada triplete NO se eligió por género ni a mano. Se construyó
con un "oráculo" de distancia y un filtro de desacuerdo entre modelos:

  1. ORÁCULO DE DISTANCIA = promedio de distancia coseno sobre 4 modelos
     simultáneamente (no un solo modelo de referencia):
         musicnn/msd, vgg/msd,
         whisper_contrastive_multilabel/base, musicnn_multisignal/msd

  2. POOLS POR PERCENTIL (la "perilla" de dificultad). Para cada anchor se
     ordenan todos los demás tracks por distancia promedio al anchor:
         close_pool  = ranked[: 15%]        → candidato "positivo" (similar)
         medium_pool = ranked[15% : 35%]    → distractor difícil (medio-lejano)
     Una opción se toma de cada pool. Tomar el distractor de la banda 15–35%
     (en vez de muy lejos) es lo que vuelve los tripletes NO triviales.

  3. FILTRO DE DESACUERDO (el criterio más informativo): un triplete se
     conserva SOLO si los 4 modelos NO coinciden unánimemente en cuál opción
     está más cerca. Si los 4 votan igual, se descarta (poco informativo).

  4. RANDOMIZACIÓN A/B 50/50 para evitar sesgo de posición.
     random.seed(42) para reproducibilidad.

────────────────────────────────────────────────────────────────────────────
Uso:
    cd backend
    python scripts/select_eval_triplets.py \
        --db db/data.db \
        --n-triplets 10 \
        --max-anchors 80 \
        --seed 42 \
        --dry-run            # imprime sin insertar; quita --dry-run para insertar
"""

import argparse
import io
import random
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# Solo nombres de archivo FMA: 6 dígitos + .mp3 (evita pirate_audio / mis-tracks)
FMA_FILENAME_RE = re.compile(r"^\d{6}\.mp3$")

# Los 4 modelos que forman el oráculo de distancia promedio.
ORACLE_MODELS = [
    ("musicnn", "msd"),
    ("vgg", "msd"),
    ("whisper_contrastive_multilabel", "base"),
    ("musicnn_multisignal", "msd"),
]

# Bandas de percentil para construir los pools de candidatos.
CLOSE_FRAC = 0.15   # 15% más similar
MEDIUM_LO = 0.15
MEDIUM_HI = 0.35    # banda 15–35% para el distractor difícil


def load_embeddings_for_model(con, model, dataset):
    """{filename: np.ndarray [D]} para un (model, dataset). Usa np.load (BLOB)."""
    rows = con.execute(
        """
        SELECT t.filename, e.embedding_data
        FROM embeddings e JOIN tracks t ON e.track_id = t.id
        WHERE e.model = ? AND e.dataset = ?
        """,
        (model, dataset),
    ).fetchall()
    out = {}
    for filename, blob in rows:
        if not FMA_FILENAME_RE.match(filename):
            continue
        arr = np.load(io.BytesIO(blob), allow_pickle=False).reshape(-1)
        out[filename] = arr
    return out


def main():
    ap = argparse.ArgumentParser(description="Selecciona tripletes para el eval HMA.")
    ap.add_argument("--db", required=True, help="Ruta a data.db (con tracks + embeddings).")
    ap.add_argument("--n-triplets", type=int, default=10)
    ap.add_argument("--max-anchors", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Imprime los tripletes sin insertarlos en eval_triplets.")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        sys.exit(f"DB no encontrada: {db_path}")

    random.seed(args.seed)
    con = sqlite3.connect(str(db_path))

    # Cargar embeddings de los 4 modelos del oráculo.
    print("Cargando embeddings de los 4 modelos del oráculo...")
    model_embs = {}
    for model, ds in ORACLE_MODELS:
        emb = load_embeddings_for_model(con, model, ds)
        if not emb:
            sys.exit(f"Sin embeddings para {model}/{ds}. ¿Corriste preprocess-all?")
        model_embs[(model, ds)] = emb
        print(f"  {model}/{ds}: {len(emb)} tracks")

    # Universo común: tracks presentes en LOS 4 modelos.
    common = set.intersection(*[set(e.keys()) for e in model_embs.values()])
    common = sorted(common)
    print(f"\nTracks comunes a los 4 modelos: {len(common)}")
    if len(common) < 10:
        sys.exit("Muy pocos tracks comunes para muestrear tripletes.")

    def avg_cos_dist(anchor, other):
        """Distancia coseno promedio anchor↔other sobre los 4 modelos."""
        ds = []
        for emb in model_embs.values():
            a = emb[anchor].reshape(1, -1)
            o = emb[other].reshape(1, -1)
            ds.append(float(cosine_distances(a, o)))
        return sum(ds) / len(ds)

    def model_votes(anchor, pos, neg):
        """Voto de cada modelo: 'pos' si pos está más cerca del anchor, si no 'neg'."""
        votes = []
        for emb in model_embs.values():
            a = emb[anchor].reshape(1, -1)
            dp = float(cosine_distances(a, emb[pos].reshape(1, -1)))
            dn = float(cosine_distances(a, emb[neg].reshape(1, -1)))
            votes.append("pos" if dp <= dn else "neg")
        return votes

    anchors = random.sample(common, min(args.max_anchors, len(common)))
    triplets = []  # (anchor, option_a, option_b, votes)

    for anchor in anchors:
        if len(triplets) >= args.n_triplets:
            break
        others = [f for f in common if f != anchor]
        ranked = sorted(others, key=lambda f: avg_cos_dist(anchor, f))
        n = len(ranked)
        close_pool = ranked[: max(1, int(n * CLOSE_FRAC))]
        medium_pool = ranked[int(n * MEDIUM_LO): int(n * MEDIUM_HI)]
        if not close_pool or not medium_pool:
            continue

        pos = random.choice(close_pool)
        neg = random.choice(medium_pool)

        votes = model_votes(anchor, pos, neg)
        # FILTRO DE DESACUERDO: descartar si los 4 modelos coinciden.
        if len(set(votes)) < 2:
            continue

        # Randomizar posición A/B para evitar sesgo posicional.
        if random.random() < 0.5:
            option_a, option_b = pos, neg
        else:
            option_a, option_b = neg, pos

        triplets.append((anchor, option_a, option_b, votes))

    print(f"\nTripletes seleccionados: {len(triplets)} (objetivo {args.n_triplets})\n")
    print(f"{'#':>2}  {'anchor':>12}  {'option_a':>12}  {'option_b':>12}  votos(4 modelos)")
    print("-" * 78)
    for i, (anc, a, b, votes) in enumerate(triplets, 1):
        print(f"{i:>2}  {anc:>12}  {a:>12}  {b:>12}  {votes}")

    if args.dry_run:
        print("\n[dry-run] No se insertó nada. Quita --dry-run para escribir en eval_triplets.")
        con.close()
        return

    cur = con.cursor()
    for anc, a, b, _ in triplets:
        cur.execute(
            "INSERT INTO eval_triplets (anchor_filename, option_a_filename, option_b_filename) "
            "VALUES (?, ?, ?)",
            (anc, a, b),
        )
    con.commit()
    print(f"\nInsertados {len(triplets)} tripletes en eval_triplets.")
    con.close()


if __name__ == "__main__":
    main()
