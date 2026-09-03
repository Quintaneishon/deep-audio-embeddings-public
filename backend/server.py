from flask import Flask, request, send_from_directory
from flask_cors import CORS
from pathlib import Path
from backend.db import database
from backend.eval_validation import (
    MAX_JSON_BYTES,
    normalize_session_id,
    parse_shown_ids,
    validate_eval_response,
)
from werkzeug.utils import secure_filename
from backend import preprocessing
from backend import evaluate_fma
import click
from backend import config
import os
from datetime import datetime
import csv

app = Flask(__name__)
CORS(app, origins=config.ALLOWED_ORIGINS)

# Register database teardown
app.teardown_appcontext(database.close_db)

# ============================================================================
# CSV Helper Functions
# ============================================================================

def add_tag_to_csv(filename, genre):
    """Add a filename and genre tag to the CSV file.

    Args:
        filename: Name of the audio file
        genre: Genre tag to associate with the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(config.CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([filename, genre.strip()])
            print(f"Added {filename} with genre '{genre.strip()}' to CSV")
        return True
    except Exception as e:
        print(f"Warning: Failed to write to CSV: {e}")
        return False


def remove_tag_from_csv(filename):
    """Remove a filename and its genre tag from the CSV file.

    Args:
        filename: Name of the audio file to remove

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(config.CSV_PATH):
            return True  # Nothing to remove

        # Read all rows from CSV
        rows = []
        with open(config.CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader if row and row[0] != filename]

        # Write back all rows except the deleted one
        with open(config.CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
            print(f"Removed {filename} from CSV tags file")
        return True
    except Exception as e:
        print(f"Warning: Failed to remove from CSV: {e}")
        return False

# ============================================================================
# Flask Routes
# ============================================================================

def _audio_path(filename, *, must_exist=False):
    """Resolve an audio path and reject traversal and symlink escapes."""
    root = Path(config.AUDIO_DIR).resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError('invalid audio filename') from exc
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(filename)
    return candidate


def _mutations_are_enabled():
    if config.MUTATIONS_ENABLED:
        return None
    return {
        'success': False,
        'error': 'upload and delete operations are disabled',
    }, 403

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/config")
def get_config():
    """Expose available model architectures and their supported datasets."""
    models = {
        arch: {
            'label': config.MODEL_DISPLAY_NAMES.get(arch, arch),
            'datasets': [
                {'value': ds, 'label': config.DATASET_DISPLAY_NAMES.get(ds, ds.upper())}
                for ds in datasets.keys()
            ],
        }
        for arch, datasets in config.MODEL_WEIGHTS.items()
    }
    return {'models': models}

@app.route("/audios")
def listar_audios():
    tracks = database.get_all_tracks()
    return tracks

@app.route("/tags")
def listar_tags():
    genre_map = database.get_tags()
    return list(set(genre_map.values()))

@app.route('/embeddings')
def embedding():
    try:
        red = request.args.get('red', '').lower()
        dataset = request.args.get('dataset', '').lower()
        metodo = request.args.get('metodo', '').lower()
        dimensiones = request.args.get('dimensions', '').lower()
    except:
        print('No se encontro param de:',red,dataset)

    # Normalize parameters
    if red == '':
        red = 'musicnn'
    if dataset == '':
        dataset = 'msd'
    if metodo == '':
        metodo = 'umap'
    if dimensiones == '':
        dimensions = 2
    else:
        dimensions = int(dimensiones)

    print(f'Computing embeddings for ({red}/{dataset}/{metodo}/{dimensions})')
    embeddings = database.get_embedding_coords(red, dataset, metodo, dimensions)

    return {
        'name': 'embeddings_'+red+"_"+dataset+'_'+metodo+'_'+str(dimensions),
        'data': embeddings,
    }


@app.route('/graph')
def graph():
    red = request.args.get('red', 'whisper_contrastive').lower()
    dataset = request.args.get('dataset', 'base').lower()
    k_param = request.args.get('k', '').strip()

    # Count songs to determine adaptive K
    tracks = database.get_all_tracks()
    n_songs = len(tracks) if tracks else 0

    if k_param:
        k = int(k_param)
    else:
        k = 3 if n_songs > 1000 else 5

    print(f'Computing graph for ({red}/{dataset}) with k={k}, songs={n_songs}')
    result = database.get_embedding_graph(red, dataset, k=k)

    if result is None:
        return {'nodes': [], 'links': [], 'k': k}, 200

    result['k'] = k
    return result


def _safe_float(val):
    """Return None for inf/nan so they serialize as valid JSON null."""
    import math
    if val is None:
        return None
    try:
        if math.isinf(val) or math.isnan(val):
            return None
    except (TypeError, ValueError):
        return None
    return float(val)


@app.route('/compare')
def compare_songs_api():
    """Compare two songs: Serra09 + MFCC distances via Essentia and embedding cosine distance."""
    song1 = request.args.get('song1', '')
    song2 = request.args.get('song2', '')
    model = request.args.get('model', 'musicnn').lower()
    dataset = request.args.get('dataset', 'msd').lower()

    if not song1 or not song2:
        return {'error': 'song1 and song2 are required'}, 400

    try:
        path1 = str(_audio_path(song1, must_exist=True))
        path2 = str(_audio_path(song2, must_exist=True))
    except FileNotFoundError as exc:
        missing = str(exc).strip("'")
        return {'error': f'Audio file not found: {missing}'}, 404
    except ValueError:
        return {'error': 'invalid audio filename'}, 400

    if not os.path.exists(path1):
        return {'error': f'Audio file not found: {song1}'}, 404
    if not os.path.exists(path2):
        return {'error': f'Audio file not found: {song2}'}, 404

    try:
        from pathlib import Path as _Path
        from backend.scripts.compare_songs import compute_similarity
        similarity = compute_similarity(_Path(path1), _Path(path2))
    except Exception as exc:
        print(f'compare_songs error: {exc}')
        return {'error': f'Signal analysis failed: {str(exc)}'}, 500

    emb_dist = None
    emb_error = None
    if model in config.MODEL_WEIGHTS and dataset in config.MODEL_WEIGHTS.get(model, {}):
        weights_path = config.MODEL_WEIGHTS[model][dataset]
        try:
            from backend.scripts.cross_method_validation import embedding_cosine_distance
            from backend.extractors.extractors import ExtractorRegistry
            registry = ExtractorRegistry.create_default()
            extractor = registry.get(model)
            if extractor:
                emb_dist = embedding_cosine_distance(extractor, weights_path, dataset, path1, path2)
            else:
                emb_error = f'No extractor registered for model: {model}'
        except Exception as exc:
            emb_error = str(exc)
            print(f'Embedding cosine error: {exc}')
    else:
        emb_error = f'Model {model}/{dataset} not found in config'

    return {
        'song1': song1,
        'song2': song2,
        'model': model,
        'dataset': dataset,
        'serra09_dist': _safe_float(similarity['serra09_distance']),
        'mfcc_cosine_dist': _safe_float(similarity['mfcc_cosine_distance']),
        'key1': similarity['key1'],
        'scale1': similarity['scale1'],
        'key2': similarity['key2'],
        'scale2': similarity['scale2'],
        'key_cof_distance': similarity['key_cof_distance'],
        'emb_cosine_dist': _safe_float(emb_dist),
        'emb_error': emb_error,
    }


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    """Serve audio files from the audio directory with CORS headers."""
    try:
        response = send_from_directory(config.AUDIO_DIR, filename)
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        return response
    except Exception as e:
        print(f"Error serving audio file {filename}: {e}")
        return {"error": f"Audio file not found: {filename}"}, 404

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Upload and process a new audio file."""
    disabled = _mutations_are_enabled()
    if disabled:
        return disabled
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return {'success': False, 'error': 'No file provided'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'success': False, 'error': 'No file selected'}, 400

        # Validate MP3 format
        extensions = ('.mp3','.wav','.ogg','.flac','.m4a')
        if not file.filename.lower().endswith(extensions):
            return {'success': False, 'error': 'Only audio files are supported'}, 400

        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = str(_audio_path(filename))

        # Save file (overwrite if exists)
        file.save(filepath)
        print(f"Saved uploaded file: {filename}")

        genre = request.form.get('genre', '')

        # Save name and genre to CSV if genre is provided
        if genre and genre.strip():
            add_tag_to_csv(filename, genre)

        # Process the track to extract embeddings
        print(f"Starting embedding extraction for: {filename}")
        success = preprocessing.process_single_track(filename)

        if success:
            print(f"Successfully processed: {filename}")
            return {'success': True, 'filename': filename}, 200
        else:
            print(f"Failed to process: {filename}")
            return {'success': False, 'error': 'Failed to process audio file'}, 500

    except Exception as e:
        print(f"Error in upload_audio: {e}")
        return {'success': False, 'error': str(e)}, 500

@app.route('/audio/<path:filename>', methods=['DELETE'])
def delete_audio(filename):
    """Delete audio file, database entry, and CSV tag entry."""
    disabled = _mutations_are_enabled()
    if disabled:
        return disabled
    try:
        # Delete the audio file
        safe_path = _audio_path(filename, must_exist=True)
        safe_path.unlink()

        # Delete from database
        database.delete_track(filename)

        # Delete from CSV tags file
        remove_tag_from_csv(filename)

        return {"success": True}, 200
    except ValueError:
        return {"success": False, "error": "invalid audio filename"}, 400
    except FileNotFoundError:
        return {"success": False, "error": "audio file not found"}, 404
    except Exception as e:
        print(f"Error deleting audio file {filename}: {e}")
        return {"success": False, "error": "could not delete audio file"}, 500


# ============================================================================
# Eval Routes
# ============================================================================

@app.route('/eval/triplet')
def get_eval_triplet():
    shown_raw = request.args.get('shown', '')
    try:
        exclude_ids = parse_shown_ids(shown_raw)
    except ValueError as exc:
        return {'error': str(exc)}, 400
    triplet = database.get_random_triplet(exclude_ids=exclude_ids or None)
    from flask import jsonify
    return jsonify({'triplet': triplet})

@app.route('/eval/response', methods=['POST'])
def post_eval_response():
    from flask import jsonify
    if not config.EVAL_COLLECTION_OPEN:
        return jsonify({'error': 'evaluation collection is closed'}), 403
    if request.content_length and request.content_length > MAX_JSON_BYTES:
        return jsonify({'error': 'payload too large'}), 413
    try:
        response = validate_eval_response(request.get_json(silent=True))
        row_id = database.insert_eval_response(
            response.session_id,
            response.triplet_id,
            response.choice,
            response.response_time_ms,
            respondent_type=response.respondent_type,
        )
    except ValueError as exc:
        message = str(exc)
        status = 409 if 'already recorded' in message else 400
        return jsonify({'error': message}), status
    return jsonify({'id': row_id}), 201

@app.route('/eval/status')
def get_eval_status():
    from flask import jsonify
    try:
        session_id = normalize_session_id(request.args.get('session_id', ''))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    db = database.get_db()
    total = db.execute('SELECT COUNT(*) FROM eval_triplets').fetchone()[0]
    done = 0
    done = db.execute(
        'SELECT COUNT(DISTINCT triplet_id) FROM eval_responses WHERE session_id=?',
        (session_id,)
    ).fetchone()[0]
    return jsonify({'total_triplets': total, 'session_responses': done})

# ============================================================================
# Flask CLI Commands for Preprocessing
# ============================================================================

@app.cli.command('init-db')
def init_db_command():
    """Initialize the database schema."""
    database.init_db()
    click.echo('Database initialized successfully.')

@app.cli.command('clean-db')
@click.option('--drop-tables', is_flag=True, default=False, help='Drop all tables before cleaning')
def clean_db_command(drop_tables):
    """Clean the database."""
    with app.app_context():
        database.clean_db(drop_tables)
        click.echo('Database cleaned successfully.')

@app.cli.command('index-audio')
def index_audio_command():
    """Scan audio directory and index all audio files."""
    with app.app_context():
        count = preprocessing.index_audio_files()
        click.echo(f'Indexed {count} new audio files.')

@app.cli.command('preprocess-all')
@click.option('--cuda', default=None, help='CUDA device (e.g. cuda:0, cuda:1, cpu)')
def preprocess_all_command(cuda):
    """Extract embeddings for all tracks."""
    if cuda is not None:
        from backend.extractors.extractors import set_device
        set_device(cuda)
    with app.app_context():
        click.echo('Starting batch preprocessing...')
        stats = preprocessing.process_all_tracks()
        click.echo('\nPreprocessing complete!')
        click.echo(f"  Tracks processed: {stats['tracks_processed']}")
        click.echo(f"  Embeddings created: {stats['embeddings_created']}")
        click.echo(f"  Errors: {stats['errors']}")

@app.cli.command('preprocess-track')
@click.argument('filename')
def preprocess_track_command(filename):
    """Process a single audio file."""
    with app.app_context():
        click.echo(f'Processing {filename}...')
        success = preprocessing.process_single_track(filename)
        if success:
            click.echo(f'Successfully processed {filename}')
        else:
            click.echo(f'Failed to process {filename}')

@app.cli.command('compute-hma')
def compute_hma_command():
    """Compute Human-Model Agreement for ALL models in the DB, sorted by HMA%."""
    from sklearn.metrics.pairwise import cosine_distances
    from collections import Counter
    import numpy as np

    with app.app_context():
        db = database.get_hma_db()
        triplets  = db.execute('SELECT * FROM eval_triplets').fetchall()
        responses = db.execute('SELECT triplet_id, choice FROM eval_responses').fetchall()

        if not responses:
            click.echo('No human responses yet.')
            return

        # Majority vote per triplet (ties: skip that triplet)
        votes = {}
        for r in responses:
            votes.setdefault(r['triplet_id'], []).append(r['choice'])
        human = {}
        for tid, cs in votes.items():
            top2 = Counter(cs).most_common(2)
            if len(top2) == 1 or top2[0][1] != top2[1][1]:
                human[tid] = top2[0][0]
            # else: tied — excluded from scoring

        click.echo(f'Triplets with unambiguous majority vote: {len(human)}\n')

        # Discover all (model, dataset) pairs present in the DB
        db = database.get_db()
        all_models = db.execute(
            'SELECT DISTINCT model, dataset FROM embeddings ORDER BY model, dataset'
        ).fetchall()

        results = []
        for row in all_models:
            model, ds = row['model'], row['dataset']
            agree = total = skipped = 0
            for t in triplets:
                if t['id'] not in human:
                    continue
                e0 = database.get_embedding_by_filename(t['anchor_filename'],   model, ds)
                ea = database.get_embedding_by_filename(t['option_a_filename'], model, ds)
                eb = database.get_embedding_by_filename(t['option_b_filename'], model, ds)
                if not (e0 and ea and eb):
                    skipped += 1
                    continue
                anc = e0['embedding'].reshape(1, -1)
                da  = float(cosine_distances(anc, ea['embedding'].reshape(1, -1)))
                db_ = float(cosine_distances(anc, eb['embedding'].reshape(1, -1)))
                model_choice = 'a' if da <= db_ else 'b'
                if model_choice == human[t['id']]:
                    agree += 1
                total += 1
            if total == 0:
                continue
            pct = agree / total * 100
            results.append((pct, model, ds, agree, total, skipped))

        results.sort(reverse=True)

        click.echo(f"  {'Model':<50} {'HMA%':>6} {'Agree':>6} {'N':>4} {'Skipped':>7}")
        click.echo('  ' + '-' * 76)
        for pct, model, ds, agree, total, skipped in results:
            key = f'{model}/{ds}'
            click.echo(f"  {key:<50} {pct:6.1f}% {agree:>6} {total:>4} {skipped:>8}")

        if results:
            best = results[0]
            click.echo(f"\nBest model: {best[1]}/{best[2]}  →  {best[0]:.1f}% HMA  "
                       f"({best[3]}/{best[4]} triplets correct)")

            reports_dir = config.REPORTS_DIR
            os.makedirs(reports_dir, exist_ok=True)
            output_file = os.path.join(reports_dir, f'hma_{datetime.now().strftime("%Y%m%dT%H-%M-%S")}.txt')

            with open(output_file, 'w') as f:
                f.write('=== Human-Model Agreement (HMA) ===\n')
                f.write(f'Triplets with unambiguous majority vote: {len(human)}\n')
                f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write('\n' + '=' * 76 + '\n\n')
                f.write(f"  {'Model':<50} {'HMA%':>6} {'Agree':>6} {'N':>4} {'Skipped':>7}\n")
                f.write('  ' + '-' * 76 + '\n')
                for pct, model, ds, agree, total, skipped in results:
                    key = f'{model}/{ds}'
                    f.write(f"  {key:<50} {pct:6.1f}% {agree:>6} {total:>4} {skipped:>8}\n")
                f.write('\n' + '=' * 76 + '\n\n')
                f.write(f"Best model: {best[1]}/{best[2]}  →  {best[0]:.1f}% HMA  "
                        f"({best[3]}/{best[4]} triplets correct)\n")

            click.echo(f'\n✓ Report saved to: {output_file}')

@app.cli.command('compute-fma-metrics')
def compute_fma_metrics_command():
    """Evaluate all models on FMA-small: Hubness, Genre nDCG@10, Echonest Spearman correlations."""
    with app.app_context():
        click.echo('Computing FMA evaluation metrics...')
        click.echo('Dataset: FMA-small (independent test set, not used in training)')
        click.echo('Metrics: Hubness | Genre nDCG@10 | Genre Centroid Sim | Agreement Rate | Echonest Spearman')

        results = evaluate_fma.compute_fma_metrics()

        if not results:
            click.echo('No results. Run preprocess-all first to extract embeddings for FMA songs.')
            return

        reports_dir = config.REPORTS_DIR
        os.makedirs(reports_dir, exist_ok=True)
        output_file = os.path.join(reports_dir, f'fma_metrics_{datetime.now().strftime("%Y%m%dT%H-%M-%S")}.txt')

        with open(output_file, 'w') as f:
            f.write('=== FMA Evaluation Metrics ===\n')
            f.write('Independent test set (FMA-small, never seen during training)\n')
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('\n' + '='*70 + '\n\n')

            # Summary table header
            f.write(f"{'Model':<40} {'Hubness':>8} {'nDCG@10':>8} {'CentSim':>8} {'Agree':>7} {'R@1':>6} {'R@10':>6} {'ρ_tempo':>8} {'ρ_energy':>9} {'ρ_dance':>8} {'ρ_instr':>8}\n")
            f.write('-'*124 + '\n')

            for combo_key, m in results.items():
                echo = m.get('echonest', {})
                rho_tempo  = f"{echo['tempo']['rho']:+.4f}"        if 'tempo'           in echo else '  n/a  '
                rho_energy = f"{echo['energy']['rho']:+.4f}"       if 'energy'          in echo else '  n/a  '
                rho_dance  = f"{echo['danceability']['rho']:+.4f}" if 'danceability'     in echo else '  n/a  '
                rho_instr  = f"{echo['instrumentalness']['rho']:+.4f}" if 'instrumentalness' in echo else '  n/a  '

                f.write(f"{combo_key:<40} {m['hubness_skewness']:>8.4f} {m['genre_ndcg_at_10']:>8.4f} "
                        f"{m['genre_centroid_sim']:>8.4f} {m['genre_agreement_rate']:>7.2%} "
                        f"{m['recall_at_1']:>6.4f} {m['recall_at_10']:>6.4f} "
                        f"{rho_tempo:>8} {rho_energy:>9} {rho_dance:>8} {rho_instr:>8}\n")

            f.write('\n' + '='*70 + '\n\n')

            # Detailed section
            for combo_key, m in results.items():
                f.write(f'{combo_key}:\n')
                f.write(f"  Tracks evaluated: {m['n_samples']} | Echonest subset: {m['n_echonest']}\n")
                f.write(f"  Hubness skewness:  {m['hubness_skewness']:.4f}  (mean k-occ: {m['mean_k_occurrence']:.1f}, max: {m['max_k_occurrence']:.0f})\n")
                f.write(f"  Genre nDCG@10:     {m['genre_ndcg_at_10']:.4f}\n")
                f.write(f"  Genre centroid sim:{m['genre_centroid_sim']:.4f}  |  Agreement rate: {m['genre_agreement_rate']:.2%}\n")
                f.write(f"  Recall@1:          {m['recall_at_1']:.4f}  |  Recall@10: {m['recall_at_10']:.4f}\n")
                if m.get('echonest'):
                    f.write(f"  Echonest Spearman ρ (cosine_dist vs |Δfeature|):\n")
                    for feat, v in m['echonest'].items():
                        f.write(f"    {feat:<20} ρ={v['rho']:+.4f}  p={v['pval']:.4f}  ({v['n_pairs']} pairs)\n")
                f.write('\n')

        click.echo(f'\n✓ Report saved to: {output_file}')
        click.echo(f'✓ Evaluated {len(results)} model/dataset combinations')
        click.echo('\nQuick summary:')
        click.echo(f"  {'Model':<40} {'Hubness':>8} {'nDCG@10':>8} {'CentSim':>8} {'Agree':>7} {'R@1':>6} {'R@10':>6} {'ρ_tempo':>8}")
        for combo_key, m in results.items():
            echo = m.get('echonest', {})
            rho_t = f"{echo['tempo']['rho']:+.4f}" if 'tempo' in echo else '  n/a  '
            click.echo(f"  {combo_key:<40} {m['hubness_skewness']:>8.4f} {m['genre_ndcg_at_10']:>8.4f} "
                       f"{m['genre_centroid_sim']:>8.4f} {m['genre_agreement_rate']:>7.2%} "
                       f"{m['recall_at_1']:>6.4f} {m['recall_at_10']:>6.4f} {rho_t:>8}")
