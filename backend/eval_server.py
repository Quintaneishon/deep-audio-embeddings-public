"""Minimal Flask server for SoundMatch eval — no ML dependencies."""
import sqlite3
import random
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
try:
    from .eval_validation import (
        MAX_JSON_BYTES,
        normalize_session_id,
        parse_shown_ids,
        validate_eval_response,
    )
except ImportError:  # Support ``python eval_server.py`` in the minimal container.
    from eval_validation import (
        MAX_JSON_BYTES,
        normalize_session_id,
        parse_shown_ids,
        validate_eval_response,
    )

DB_PATH    = os.environ.get('DB_PATH',   os.path.join(os.path.dirname(__file__), 'db', 'data.db'))
AUDIO_DIR  = os.environ.get('AUDIO_DIR', os.path.join(os.path.dirname(__file__), 'data', 'audio_fma'))

app = Flask(__name__)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'ALLOWED_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000',
    ).split(',')
    if origin.strip()
]
EVAL_COLLECTION_OPEN = os.environ.get('EVAL_COLLECTION_OPEN', 'false').lower() == 'true'
CORS(app, origins=ALLOWED_ORIGINS)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_eval_response_once '
        'ON eval_responses(session_id, triplet_id)'
    )
    return conn

@app.route('/eval/triplet')
def get_triplet():
    shown_raw  = request.args.get('shown', '')
    try:
        exclude = parse_shown_ids(shown_raw)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    conn       = get_db()
    if exclude:
        ph  = ','.join('?' * len(exclude))
        row = conn.execute(
            f'SELECT * FROM eval_triplets WHERE id NOT IN ({ph}) ORDER BY RANDOM() LIMIT 1',
            tuple(exclude)
        ).fetchone()
    else:
        row = conn.execute(
            'SELECT * FROM eval_triplets ORDER BY RANDOM() LIMIT 1'
        ).fetchone()
    conn.close()
    return jsonify({'triplet': dict(row) if row else None})

@app.route('/eval/response', methods=['POST'])
def post_response():
    if not EVAL_COLLECTION_OPEN:
        return jsonify({'error': 'evaluation collection is closed'}), 403
    if request.content_length and request.content_length > MAX_JSON_BYTES:
        return jsonify({'error': 'payload too large'}), 413
    conn = None
    try:
        response = validate_eval_response(request.get_json(silent=True))
        conn = get_db()
        if conn.execute(
            'SELECT 1 FROM eval_triplets WHERE id = ?',
            (response.triplet_id,),
        ).fetchone() is None:
            conn.close()
            raise ValueError('triplet does not exist')
        cur = conn.execute(
            'INSERT INTO eval_responses '
            '(session_id, triplet_id, choice, response_time_ms, respondent_type) '
            'VALUES (?,?,?,?,?)',
            (
                response.session_id,
                response.triplet_id,
                response.choice,
                response.response_time_ms,
                response.respondent_type,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
    except sqlite3.IntegrityError:
        if conn is not None:
            conn.rollback()
            conn.close()
        return jsonify({'error': 'response already recorded or invalid'}), 409
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    return jsonify({'id': row_id}), 201

@app.route('/eval/status')
def get_status():
    try:
        session_id = normalize_session_id(request.args.get('session_id', ''))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    conn  = get_db()
    total = conn.execute('SELECT COUNT(*) FROM eval_triplets').fetchone()[0]
    done  = 0
    done = conn.execute(
        'SELECT COUNT(DISTINCT triplet_id) FROM eval_responses WHERE session_id=?',
        (session_id,)
    ).fetchone()[0]
    conn.close()
    return jsonify({'total_triplets': total, 'session_responses': done})

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    try:
        resp = send_from_directory(AUDIO_DIR, filename)
        resp.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        return resp
    except Exception:
        return jsonify({'error': f'not found: {filename}'}), 404

if __name__ == '__main__':
    app.run(port=5000, debug=True)
