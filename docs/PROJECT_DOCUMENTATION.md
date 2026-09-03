# Project Architecture

This document gives a public-safe technical overview. It deliberately excludes
private infrastructure, database snapshots, participant records, recordings,
and trained weights.

## System boundaries

The system has four major parts:

1. `backend/server.py` provides the full Flask API used for local embedding
   exploration and preprocessing.
2. `backend/eval_server.py` provides a small deployment option for SoundMatch
   without the machine-learning dependency stack.
3. `backend/DL/` contains model definitions, datasets, losses, feature
   preparation, and training entry points.
4. `frontend/` contains the React visualization and perceptual-evaluation UI.

SQLite is a local implementation detail, not a public data artifact. The full
server stores track metadata and embedding arrays. The evaluation service stores
triplets and minimal response records. Public releases contain schema/code only,
never populated database files.

## Data flow

For local visualization, a user downloads FMA, copies a selected local subset
into an ignored directory, indexes it, and extracts one or more model embeddings.
The API projects or compares those embeddings and the React application renders
the response. Recordings remain on the user's machine.

For model training, a user downloads MTG-Jamendo and invokes a training module
with `--data-root` and `--split`. Hybrid and MultiSignal variants also read an
ignored structural-feature cache. Training checkpoints and logs are ignored.

For SoundMatch, the browser requests a triplet, plays the three recordings from
the configured backend, and submits the minimal choice record. The server
validates the request and enforces one response per session/triplet pair. The
frontend advances only after the save succeeds.

For HMA analysis, the sanitized notebook reads explicitly supplied frozen
private snapshots, derives majority choices, computes model choices from cosine
distance, and performs descriptive and paired analyses. Notebook outputs and
generated artifacts are not versioned. Only aggregate prose findings appear in
`RESULTS.md`.

## Evaluation schema

The current public schema records:

- triplet: identifier plus anchor, option A, and option B filenames;
- response: generated identifier, anonymous session UUID, triplet identifier,
  choice, response time, respondent category, and timestamp.

It does not require or collect a respondent name. A foreign key links each
response to an existing triplet, and a unique index prevents a session from
voting twice on the same triplet.

## Security model

The repository is safe-by-default, not production-authenticated:

- evaluation collection is disabled unless `EVAL_COLLECTION_OPEN=true`;
- upload/delete is disabled unless `MUTATIONS_ENABLED=true`;
- allowed browser origins are explicitly configured;
- audio paths are resolved within the configured audio root;
- evaluation payloads are size-bounded and type-checked;
- PyTorch checkpoints are read in weights-only mode.

The feature flags are operational safeguards, not identity controls. A public
service that enables mutations needs real authentication, authorization, rate
limiting, monitoring, and an abuse response process. A participant study also
needs the applicable review, consent/privacy information, minimization, and
retention plan.

## Reproducibility model

The public repository supports rerunning code from source datasets that users
obtain under the original licenses. It does not claim that participant-level HMA
results are independently reproducible without the private frozen inputs. The
root `REPRODUCE.md` states this boundary and supplies executable commands.

For every reported run, retain outside Git the code commit, dataset version,
official split, seed, dependency environment, model configuration, and
checkpoint hash. This metadata distinguishes an exact reproduction from a new
replication.

## Public release process

The historical development repository remains private. The public repository is
created from a reviewed tracked-file snapshot, producing a single clean initial
commit with no old objects or private branches. See `PUBLIC_RELEASE.md` for the
release and GitHub security checklist.
