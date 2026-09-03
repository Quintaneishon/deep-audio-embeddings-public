from pathlib import Path
import sqlite3
from backend import config
from flask import g
import csv
import numpy as np
from backend.utils import proyectar_embeddings
from sklearn.neighbors import NearestNeighbors
import io


def init_db():
    """Initialize the database schema with BLOB columns for vectors."""
    # Ensure cache directory exists
    Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(config.DATABASE_PATH)
    db.execute('PRAGMA foreign_keys = ON')

    # Create tracks table
    db.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            duration REAL,
            spectral_centroid REAL,
            tempo REAL,
            processed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create embeddings table - storing actual vectors as BLOBs
    db.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            track_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            embedding_data BLOB NOT NULL,
            embedding_shape TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (track_id, model, dataset),
            FOREIGN KEY (track_id) REFERENCES tracks (id) ON DELETE CASCADE
        )
    ''')

    # Create indexes for faster lookups
    db.execute('CREATE INDEX IF NOT EXISTS idx_tracks_filename ON tracks(filename)')
    db.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_lookup ON embeddings(track_id, model, dataset)')

    # Eval tables
    db.execute('''
        CREATE TABLE IF NOT EXISTS eval_triplets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            anchor_filename   TEXT NOT NULL,
            option_a_filename TEXT NOT NULL,
            option_b_filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    db.execute('''
        CREATE TABLE IF NOT EXISTS eval_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT NOT NULL,
            triplet_id       INTEGER NOT NULL,
            choice           TEXT NOT NULL CHECK(choice IN ('a','b')),
            response_time_ms INTEGER NOT NULL,
            respondent_type  TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (triplet_id) REFERENCES eval_triplets(id)
        )
    ''')

    # Backfill: add columns if upgrading an older schema (idempotent)
    existing_cols = {row[1] for row in db.execute('PRAGMA table_info(eval_responses)').fetchall()}
    if 'respondent_type' not in existing_cols:
        db.execute('ALTER TABLE eval_responses ADD COLUMN respondent_type TEXT')

    db.execute('CREATE INDEX IF NOT EXISTS idx_eval_responses_session ON eval_responses(session_id)')
    db.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_response_once '
        'ON eval_responses(session_id, triplet_id)'
    )

    db.commit()
    db.close()
    print(f"Database initialized at {config.DATABASE_PATH}")

def get_db():
    """Get database connection, creating it if it doesn't exist in flask context."""
    if 'db' not in g:
        Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(config.DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db

def get_hma_db():
    """Get database connection, creating it if it doesn't exist in flask context."""

    # Ensure cache directory exists
    if 'hma_db' not in g:
        Path(config.DATABASE_EVAL_PATH).parent.mkdir(parents=True, exist_ok=True)
        g.hma_db = sqlite3.connect(config.DATABASE_EVAL_PATH)
        g.hma_db.row_factory = sqlite3.Row
        g.hma_db.execute('PRAGMA foreign_keys = ON')
    return g.hma_db

def close_db(e=None):
    """Close database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()
    hma_db = g.pop('hma_db', None)
    if hma_db is not None:
        hma_db.close()

def clean_db(drop_tables=False):
    """
    Clean the database by removing all data.

    Args:
        drop_tables: If True, drop and recreate all tables (full reset).
                    If False, only delete all records (preserve schema).

    Returns:
        dict: Statistics about deleted records
    """
    # Connect directly (not through Flask context)
    db = sqlite3.connect(config.DATABASE_PATH)
    cursor = db.cursor()

    try:
        if drop_tables:
            cursor.execute('DROP TABLE IF EXISTS embeddings')
            cursor.execute('DROP TABLE IF EXISTS tracks')
            db.commit()
            db.close()
            init_db()

            return {
                'status': 'success',
                'action': 'full_reset',
                'message': 'Database dropped and reinitialized'
            }
        else:
            # Get counts before deletion
            embeddings_count = cursor.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0]
            tracks_count = cursor.execute('SELECT COUNT(*) FROM tracks').fetchone()[0]

            # Delete all records (respect foreign key constraints by deleting in order)
            cursor.execute('DELETE FROM embeddings')
            cursor.execute('DELETE FROM tracks')

            # Reset autoincrement counters
            cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("embeddings", "tracks")')

            db.commit()
            db.close()

            return {
                'status': 'success',
                'action': 'clear_data',
                'deleted': {
                    'embeddings': embeddings_count,
                    'tracks': tracks_count
                },
                'message': f'Deleted {tracks_count} tracks, {embeddings_count} embeddings'
            }
    except Exception as e:
        db.rollback()
        db.close()
        return {
            'status': 'error',
            'message': str(e)
        }

# ============================================================================
# Track Functions
# ============================================================================

def get_track_by_filename(filename):
    """Get track record by filename."""
    db = get_db()
    track = db.execute(
        'SELECT * FROM tracks WHERE filename = ?',
        (filename,)
    ).fetchone()
    return track

def get_track_by_id(track_id):
    """Get track record by ID."""
    db = get_db()
    track = db.execute(
        'SELECT * FROM tracks WHERE id = ?',
        (track_id,)
    ).fetchone()
    return track

def get_embedding_by_filename(filename, model, dataset):
    """
    Get embedding vector directly by filename.

    Returns:
        dict with 'embedding_id', 'embedding' or None
    """
    db = get_db()
    result = db.execute('''
        SELECT e.track_id, e.embedding_data, e.embedding_shape
        FROM embeddings e
        JOIN tracks t ON e.track_id = t.id
        WHERE t.filename = ? AND e.model = ? AND e.dataset = ?
    ''', (filename, model, dataset)).fetchone()

    if result:
        embedding = _blob_to_numpy(result['embedding_data'])
        return {
            'embedding_id': result['track_id'],
            'embedding': embedding,
        }

    return None

def insert_track(filename, duration=None):
    """Insert a new track into the database."""
    db = get_db()
    cursor = db.execute(
        'INSERT OR IGNORE INTO tracks (filename, duration) VALUES (?, ?)',
        (filename, duration)
    )
    db.commit()
    # Handle case where track already exists
    if cursor.lastrowid == 0:
        track = get_track_by_filename(filename)
        return track['id'] if track else None
    return cursor.lastrowid

def get_all_tracks():
    """Get all tracks from the database."""
    db = get_db()
    tracks = db.execute('SELECT id, filename FROM tracks').fetchall()
    return [dict(track) for track in tracks]

def get_tags():
    """
    Get all tags from the csv file if exists.
    """
    # Load genre mappings from CSV
    csv_path = Path(config.CSV_PATH)
    genre_map = {}
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                genre_map[row['filename']] = row['genre']
    else:
        print(f"Warning: CSV file not found at {csv_path}")
    return genre_map

def get_embedding_coords(red, dataset, metodo, dimensions):
    """
    Load the actual numpy arrays for an embedding and include genre tags.

    Returns:
        list: Array of dictionaries with 'data' (projected coordinates) and 'tag' (genre)
    """
    db = get_db()
    result = db.execute(
        '''SELECT e.embedding_data, t.filename
           FROM embeddings e
           JOIN tracks t ON e.track_id = t.id
           WHERE e.model = ? AND e.dataset = ?''',
        (red, dataset)
    ).fetchall()

    if not result:
        return None

    genre_map = get_tags()

    embeddings_list = []
    filenames = []
    for row in result:
        embedding = _blob_to_numpy(row['embedding_data'])
        embeddings_list.append(embedding)
        filenames.append(row['filename'])

    # Stack all embeddings into a single 2D array
    embeddings_array = np.vstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]

    # Project embeddings
    projected = proyectar_embeddings(embeddings_array, metodo, n_components=dimensions)

    # Build result array with genre tags
    result_array = []
    for i, filename in enumerate(filenames):
        genre = genre_map.get(filename, 'Unknown')
        result_array.append({
            'coords': projected[i].tolist(),
            'tag': genre,
            'name': filename,
            'audio': filename
        })

    return result_array

def get_embedding_graph(red, dataset, k=5):
    """
    Compute K-nearest neighbors graph from raw embeddings (no projection).

    Returns:
        dict: {nodes: [...], links: [...]} or None if no data.
        Each node: {id, name, tag, audio}
        Each link: {source, target, distance, sameGenre}
    """
    db = get_db()
    result = db.execute(
        '''SELECT e.embedding_data, t.filename
           FROM embeddings e
           JOIN tracks t ON e.track_id = t.id
           WHERE e.model = ? AND e.dataset = ?''',
        (red, dataset)
    ).fetchall()

    if not result:
        return None

    genre_map = get_tags()

    embeddings_list = []
    filenames = []
    for row in result:
        embedding = _blob_to_numpy(row['embedding_data'])
        embeddings_list.append(embedding)
        filenames.append(row['filename'])

    embeddings_array = np.vstack(embeddings_list) if len(embeddings_list) > 1 else embeddings_list[0]
    n_samples = embeddings_array.shape[0]

    # Clamp k to available samples (need at least k+1 for self-exclusion)
    effective_k = min(k, n_samples - 1)
    if effective_k < 1:
        effective_k = 1

    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric='cosine')
    nn.fit(embeddings_array)
    distances, indices = nn.kneighbors(embeddings_array)

    nodes = []
    for i, filename in enumerate(filenames):
        genre = genre_map.get(filename, 'Unknown')
        nodes.append({
            'id': i,
            'name': filename,
            'tag': genre,
            'audio': filename
        })

    # Build deduplicated edges (only keep i < j)
    seen = set()
    links = []
    for i in range(n_samples):
        genre_i = nodes[i]['tag']
        for j_pos in range(1, effective_k + 1):
            neighbor = int(indices[i][j_pos])
            edge_key = (min(i, neighbor), max(i, neighbor))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            genre_j = nodes[neighbor]['tag']
            links.append({
                'source': i,
                'target': neighbor,
                'distance': float(distances[i][j_pos]),
                'sameGenre': genre_i == genre_j
            })

    return {'nodes': nodes, 'links': links}



def mark_track_processed(track_id):
    """Mark a track as processed."""
    db = get_db()
    db.execute(
        'UPDATE tracks SET processed_at = CURRENT_TIMESTAMP WHERE id = ?',
        (track_id,)
    )
    db.commit()

def get_embedding(track_id, model, dataset):
    """
    Get embedding record for a track (metadata only, no vectors).
    """
    db = get_db()
    embedding = db.execute(
        'SELECT track_id, model, dataset, embedding_shape, created_at FROM embeddings WHERE track_id = ? AND model = ? AND dataset = ?',
        (track_id, model, dataset)
    ).fetchone()
    return embedding

def insert_embedding(track_id, model, dataset, embedding_array):
    """
    Insert a new embedding record with vector stored in database.

    Args:
        track_id: Track ID
        model: Model name
        dataset: Dataset name
        embedding_array: Numpy array of embeddings

    Returns:
        embedding_id
    """
    db = get_db()

    embedding_blob, embedding_shape = _numpy_to_blob(embedding_array)

    cursor = db.execute(
        '''INSERT OR REPLACE INTO embeddings
           (track_id, model, dataset, embedding_data, embedding_shape)
           VALUES (?, ?, ?, ?, ?)''',
        (track_id, model, dataset, embedding_blob, embedding_shape)
    )
    db.commit()
    return cursor.lastrowid


def delete_track(filename):
    """Delete a track from the database."""
    track = get_track_by_filename(filename)
    if track:
        db = get_db()
        db.execute('DELETE FROM embeddings WHERE track_id = ?', (track['id'],))
        db.execute('DELETE FROM tracks WHERE id = ?', (track['id'],))
        db.commit()
        return True
    return False

# ============================================================================
# Eval Functions
# ============================================================================

def insert_eval_triplet(anchor_filename, option_a_filename, option_b_filename):
    db = get_db()
    cur = db.execute(
        'INSERT INTO eval_triplets (anchor_filename, option_a_filename, option_b_filename) VALUES (?,?,?)',
        (anchor_filename, option_a_filename, option_b_filename)
    )
    db.commit()
    return cur.lastrowid

def get_random_triplet(exclude_ids=None):
    db = get_db()
    if exclude_ids:
        ph = ','.join('?' * len(exclude_ids))
        row = db.execute(
            f'SELECT * FROM eval_triplets WHERE id NOT IN ({ph}) ORDER BY RANDOM() LIMIT 1',
            tuple(exclude_ids)
        ).fetchone()
    else:
        row = db.execute(
            'SELECT * FROM eval_triplets ORDER BY RANDOM() LIMIT 1'
        ).fetchone()
    return dict(row) if row else None

def insert_eval_response(session_id, triplet_id, choice, response_time_ms,
                         respondent_type=None):
    db = get_db()
    if db.execute('SELECT 1 FROM eval_triplets WHERE id = ?', (triplet_id,)).fetchone() is None:
        raise ValueError('triplet does not exist')
    try:
        cur = db.execute(
            '''INSERT INTO eval_responses
               (session_id, triplet_id, choice, response_time_ms, respondent_type)
               VALUES (?,?,?,?,?)''',
            (session_id, triplet_id, choice, response_time_ms, respondent_type or None)
        )
        db.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError as exc:
        db.rollback()
        raise ValueError('response already recorded or invalid') from exc

# ============================================================================
# Helper Functions for Numpy <-> BLOB conversion
# ============================================================================

def _numpy_to_blob(array):
    """Convert numpy array to BLOB (bytes) and shape string."""
    # Serialize to bytes
    out = io.BytesIO()
    np.save(out, array, allow_pickle=False)
    out.seek(0)
    blob = out.read()

    shape_str = str(array.shape)

    return blob, shape_str


def _blob_to_numpy(blob):
    """Convert BLOB (bytes) and shape string back to numpy array."""
    in_bytes = io.BytesIO(blob)
    array = np.load(in_bytes, allow_pickle=False)
    return array
