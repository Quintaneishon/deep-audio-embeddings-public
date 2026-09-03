# Backend

The Flask backend indexes locally downloaded audio, extracts embeddings, serves
visualization data, and supports the SoundMatch triplet study. It also contains
the thesis training and evaluation modules.

Run all Python commands below from the repository root.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env
```

For only the small evaluation server, install
`backend/eval_requirements.txt` instead.

## Private local inputs

Set local paths in `backend/.env`:

- `FMA_METADATA_DIR`: downloaded FMA metadata directory;
- `FMA_SMALL_DIR`: downloaded FMA-small audio directory;
- `MTG_JAMENDO_ROOT`: downloaded MTG-Jamendo root;
- `MY_TRACKS_DIR`: optional personal recordings, for private local use only.

The repository does not distribute datasets or recordings. Download FMA from
[mdeff/fma](https://github.com/mdeff/fma) and MTG-Jamendo from
[MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset).
Generated audio subsets, manifests, databases, caches, reports, and weights are
ignored by Git.

## Baseline weights

Locally downloaded baseline files are expected under `backend/DL/weights/`, for
example:

```text
backend/DL/weights/msd/musicnn.pth
backend/DL/weights/mtat/musicnn.pth
backend/DL/weights/msd/vgg.pth
backend/DL/weights/mtat/vgg.pth
```

MusiCNN weights originate from
[jordipons/musicnn](https://github.com/jordipons/musicnn). VGG weights originate
from [minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models).
Whisper, VGGish, and MERT dependencies may download their public model assets to
the user's normal framework cache. Verify the license and checksum of every
download. Do not commit weights.

## FMA workflow

After configuring the FMA paths:

```bash
python -m backend.scripts.select_songs_fma
python -m flask --app backend.server init-db
python -m flask --app backend.server index-audio
python -m flask --app backend.server preprocess-all --cuda cpu
python -m flask --app backend.server run
```

Other useful commands are `preprocess-track`, `compute-fma-metrics`, and
`compute-hma`. Use `python -m flask --app backend.server --help` for the current
CLI list.

## API overview

Read-only routes include `/config`, `/audios`, `/tags`, `/embeddings`, `/graph`,
`/compare`, and `/audio/<filename>`.

`POST /upload` and `DELETE /audio/<filename>` are disabled unless
`MUTATIONS_ENABLED=true`. They are intended only for a trusted environment; this
flag is not authentication.

The triplet-study routes are:

- `GET /eval/triplet?shown=1,2,3`;
- `POST /eval/response`;
- `GET /eval/status?session_id=<uuid>`.

Response collection is disabled unless `EVAL_COLLECTION_OPEN=true`. Accepted
responses require a UUID session, an existing positive triplet ID, choice `a` or
`b`, a bounded non-negative response time, and respondent type `public`. A
session can submit only one response for a triplet. Names are not
accepted or stored by the current UI.

Set `ALLOWED_ORIGINS` to a comma-separated allowlist. The default permits only
the local React development origins.

## MTG-Jamendo training

Training modules use the official files under
`data/splits/<split>/autotagging_genre-{train,validation,test}.tsv` and audio
under `songs/`. Every trainer's `--data-root` and `--split` options control these
locations.

Hybrid and MultiSignal models first require a structural-feature cache:

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

Representative training command:

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

Use `--help` for other modules in `backend/DL/trainings/`. Generated checkpoints
and logs are ignored. Load only trusted checkpoints; code paths use
`weights_only=True` when reading PyTorch state.

## HMA analysis and privacy

`backend/reports/hma_analysis.ipynb` contains the analysis method with execution
outputs removed. It reads database locations from `HMA_EVAL_DB` and
`HMA_EMBEDDINGS_DB`. Those frozen databases are private and are not part of the
public reproducibility bundle. Generated notebook figures and tables are also
ignored; aggregate findings are in the root `RESULTS.md`.

## Minimal deployment image

The Docker image intentionally excludes databases and audio. At runtime, mount
private data outside the image and set `DB_PATH`, `AUDIO_DIR`,
`ALLOWED_ORIGINS`, and `EVAL_COLLECTION_OPEN` deliberately. `start.sh` refuses
to boot if its database file is absent. Review `SECURITY.md` before exposing any
service to the internet.
