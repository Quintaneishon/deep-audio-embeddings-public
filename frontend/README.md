# Frontend

React interface for exploring audio embeddings and running the SoundMatch
triplet task. Audio and visualization data come from a separately running local
backend; no music is included in this repository.

## Setup

Use Node.js 20. If `nvm` is installed, `nvm use` reads `.nvmrc`.

```bash
cd frontend
nvm use
npm ci
cp .env.example .env
npm start
```

`REACT_APP_API_BASE_URL` defaults to `http://localhost:5000`. Point it only to a
backend you control.

## Main routes

- `/`: Plotly-based 2D/3D embedding exploration;
- `/visualize`: Deck.gl exploration;
- `/graph`: nearest-neighbor graph;
- `/particles`: Three.js audio-reactive visualization;
- `/eval`: SoundMatch perceptual evaluation.

The visualization routes need a populated local backend database and locally
downloaded audio. Follow the root `REPRODUCE.md`.

## SoundMatch integrity and privacy

The current public evaluation UI creates a random UUID in browser storage. It
sends only the session UUID, triplet ID, `a`/`b` choice, response time, and fixed
`public` respondent type. It does not ask for or submit participant names.

The old query-string expert label was removed because a browser-controlled label
cannot prove expertise. Any future expert cohort needs a separately authenticated
recruitment flow; it is listed as future work rather than presented as secure.

The UI waits for a successful server response before advancing. A failed save is
shown to the participant and can be retried, so a network failure no longer
silently drops a vote. The backend rejects malformed votes and duplicate
session/triplet pairs. Response collection is closed by default on the server.

## Verification

```bash
cd frontend
npm test -- --watchAll=false
npm run build
```

`src/App.test.js` is currently a minimal route-shell smoke test. Tests for vote
submission, failed-request retry, audio controls, and visualization interaction
are planned as a later feature. This limitation is explicit so a successful
smoke test is not mistaken for comprehensive coverage.

## Production notes

The repository contains no Vercel account metadata and no production API URL.
Set deployment variables in the hosting platform rather than committing them.
Restrict the backend's `ALLOWED_ORIGINS`, leave collection disabled when a study
is not active, and never bake databases or recordings into a frontend or backend
image.
