# Deep Audio Embeddings

Research software developed for a master's degree in Computer Science and
Engineering at UNAM. The project compares deep audio representations and
studies how well their similarity judgments agree with human listeners.

The repository contains:

- a Flask backend for indexing audio, extracting embeddings, and serving an API;
- a React frontend for interactive 2D/3D exploration and the SoundMatch
  perceptual-evaluation interface;
- contrastive-training pipelines for Whisper, MusiCNN, MERT, and VGG variants;
- a sanitized analysis notebook describing the Human-Model Agreement (HMA)
  methodology.

## Public-data policy

This public version intentionally does **not** contain audio, datasets, model
weights, SQLite databases, participant names, individual responses, or generated
result figures/tables. Those files are ignored by Git.

Download the source datasets from their maintainers:

- [FMA (Free Music Archive dataset)](https://github.com/mdeff/fma)
- [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset)

Dataset and recording licenses remain with their respective owners. Do not add
downloaded audio to this repository.

The aggregate thesis findings are reported in [RESULTS.md](RESULTS.md). The
notebook at `backend/reports/hma_analysis.ipynb` is included with outputs removed.
Its private input databases are not published because they contain research and
participant-level records. This means the analysis method is inspectable, while
exact regeneration of the HMA results requires authorized access to the frozen,
de-identified research snapshots.

## Quick start

Use Python 3.11 and Node.js 20. Run backend commands from the repository root so
package imports behave consistently.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example backend/.env

python -m flask --app backend.server init-db
python -m flask --app backend.server index-audio
python -m flask --app backend.server run
```

In another terminal:

```bash
cd frontend
npm ci
cp .env.example .env
npm start
```

The API runs at `http://localhost:5000` and the frontend at
`http://localhost:3000`. Dataset download, feature preparation, training, and
verification commands are in [REPRODUCE.md](REPRODUCE.md).

## Security defaults

Human-response collection and upload/delete endpoints are disabled by default.
To run a deliberate local study, set `EVAL_COLLECTION_OPEN=true`. To enable
audio upload/delete in a trusted local environment, set
`MUTATIONS_ENABLED=true`. Configure `ALLOWED_ORIGINS` explicitly for any
deployment.

Evaluation responses are validated, only one response per session and triplet
is accepted, and the current frontend does not collect respondent names. See
[SECURITY.md](SECURITY.md) before deploying the application.

## Project documentation

- [REPRODUCE.md](REPRODUCE.md): reproducibility scope and commands
- [backend/README.md](backend/README.md): API, data preparation, and training
- [frontend/README.md](frontend/README.md): frontend setup and verification
- [PUBLIC_RELEASE.md](PUBLIC_RELEASE.md): clean-history publication checklist
- [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md): detailed architecture reference

## Testing status

The frontend currently has a minimal application smoke test. Comprehensive
frontend interaction tests and broader backend test coverage are planned work;
see [frontend/README.md](frontend/README.md). Until those tests exist, treat this
as research software rather than a production service.

## Copyright and license

This project was created as part of a UNAM master's degree, but academic purpose
does not remove copyright. No open-source license has been selected yet, so
copyright applies automatically and reuse permission is not granted merely by
making the repository public.

Before publishing, add a `LICENSE` file. MIT is a simple choice when broad reuse
with attribution is desired; Apache-2.0 adds an explicit patent grant. The code
must not be described as "without copyright" or "public domain" unless the
copyright holder deliberately adopts an appropriate dedication.

## Citation

If you use this work, cite the associated UNAM master's thesis. Add the final
thesis title, author, year, and repository URL here after the defense.
