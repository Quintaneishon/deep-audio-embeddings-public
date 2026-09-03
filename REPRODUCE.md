# Reproducing Deep Audio Embeddings

This guide separates what can be reproduced from public inputs from what is
intentionally private.

## Reproducibility scope

Publicly reproducible:

- installation and command-line interfaces;
- FMA indexing and embedding extraction after the user downloads FMA;
- model training from MTG-Jamendo's official split files;
- frontend build and local visualization;
- inspection of the HMA analysis code.

Not distributed:

- FMA or MTG-Jamendo audio;
- pretrained or thesis-trained weights;
- SQLite databases and participant-level responses;
- generated result figures and tables.

The exact HMA numbers therefore cannot be regenerated from this repository
alone. They require the frozen private research databases. Aggregate results are
reported in [RESULTS.md](RESULTS.md), and the sanitized notebook documents the
analysis.

## 1. Environment

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

Edit `backend/.env` with local dataset paths. The file is ignored by Git.

## 2. Download datasets yourself

Use only the official sources and comply with each dataset's license:

- [FMA repository and download instructions](https://github.com/mdeff/fma)
- [MTG-Jamendo repository and download instructions](https://github.com/MTG/mtg-jamendo-dataset)

Expected layouts:

```text
/path/to/fma_metadata/tracks.csv
/path/to/fma_small/000/000002.mp3

/path/to/mtg-jamendo-dataset/data/splits/split-0/
/path/to/mtg-jamendo-dataset/songs/
```

Set `FMA_METADATA_DIR`, `FMA_SMALL_DIR`, and `MTG_JAMENDO_ROOT` in
`backend/.env`. All copied audio and generated manifests remain ignored.

## 3. Build the local FMA workspace

```bash
python -m backend.scripts.select_songs_fma
python -m flask --app backend.server init-db
python -m flask --app backend.server index-audio
```

Download any required public baseline weights from the original model projects
and place them at the paths documented in `backend/README.md`. Then extract
embeddings:

```bash
python -m flask --app backend.server preprocess-all --cuda cpu
```

For a CUDA machine, replace `cpu` with the desired device such as `cuda:0`.

## 4. Prepare MTG-Jamendo structural features

The hybrid and MultiSignal trainers need one feature cache covering the official
training, validation, and test partitions:

```bash
MTG_ROOT=/path/to/mtg-jamendo-dataset

python -m backend.DL.trainings.precompute_structure_features \
  --tsv \
    "$MTG_ROOT/data/splits/split-0/autotagging_genre-train.tsv" \
    "$MTG_ROOT/data/splits/split-0/autotagging_genre-validation.tsv" \
    "$MTG_ROOT/data/splits/split-0/autotagging_genre-test.tsv" \
  --audio "$MTG_ROOT/songs" \
  --output backend/data/features_cache_split0.json \
  --resume
```

The output is local generated data and is not committed.

## 5. Train a representative model

This command now uses `--data-root` and `--split` directly:

```bash
python -m backend.DL.trainings.train_contrastive_musicnn_multisignal \
  --data-root "$MTG_ROOT" \
  --split split-0 \
  --features-cache backend/data/features_cache_split0.json \
  --pretrained-weights backend/DL/weights/msd/musicnn.pth \
  --num-epochs 50 \
  --batch-size 16 \
  --cuda-number 0
```

Use `--help` on any module in `backend/DL/trainings/` for its model-specific
options. Checkpoints are written to ignored directories. Checkpoint loading uses
PyTorch's weights-only mode; do not load untrusted serialized model files.

## 6. Run the frontend

```bash
cd frontend
npm ci
cp .env.example .env
npm test -- --watchAll=false
npm run build
npm start
```

Use Node.js 20 (`nvm use` reads `frontend/.nvmrc`). The frontend expects the API
at `http://localhost:5000` unless `REACT_APP_API_BASE_URL` is changed.

Audio is served from your local ignored data directory. No recording is supplied
by this repository.

## 7. Inspect the HMA analysis

The public notebook is `backend/reports/hma_analysis.ipynb`. It has no stored
outputs and accepts private database locations through:

- `HMA_EVAL_DB`: frozen evaluation-response database;
- `HMA_EMBEDDINGS_DB`: frozen embedding database;
- `DEEP_AUDIO_REPORTS_DIR`: optional local output directory.

Authorized researchers can run:

```bash
HMA_EVAL_DB=/private/path/evaluation.db \
HMA_EMBEDDINGS_DB=/private/path/embeddings.db \
jupyter lab backend/reports/hma_analysis.ipynb
```

Participant-level databases must not be copied into Git. Generated notebook
figures and tables stay ignored.

## 8. Verification before reporting results

Record the Git commit, dataset versions, split name, environment, seed, model
configuration, and checkpoint hash for every run. A result should be called an
exact reproduction only when it uses the same frozen inputs; otherwise describe
it as a rerun or replication.
