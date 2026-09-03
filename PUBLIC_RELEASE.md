# Public Release Checklist

The safest publication design is a **new public repository created from one
audited snapshot**. Keep the existing repository private as the historical
archive. Do not change the existing repository's visibility, and do not delete
its `.git` directory.

## 1. Finish and freeze the private release candidate

- Commit the reviewed fixes on `main` in the private repository.
- Do not merge or push the private `feature/DJ` branch.
- Choose and add an open-source `LICENSE` if reuse should be allowed.
- Replace the citation placeholder after the thesis metadata is final.
- Run the backend checks and frontend test/build in `REPRODUCE.md`.
- Confirm the notebook has no outputs or absolute private paths.

## 2. Audit exactly what is tracked

```bash
git status --short
git ls-files
git grep -nEi 'password|passwd|secret|token|api[_-]?key|BEGIN .*PRIVATE KEY|/Users/|10\.[0-9]+\.|192\.168\.'
git ls-files | grep -Ei '\.(db|sqlite3?|mp3|wav|flac|m4a|ogg|pth|pt|ckpt|safetensors)$'
```

The last command should print nothing. Review every match from the credential and
private-path search; a match is a lead, not proof of a secret.

## 3. Create the clean public snapshot in a sibling directory

After the private release commit is final:

```bash
mkdir ../deep-audio-embeddings-public
git archive HEAD | tar -x -C ../deep-audio-embeddings-public
cd ../deep-audio-embeddings-public
git init -b main
git config --local user.name "YOUR PUBLIC NAME"
git config --local user.email "COPY YOUR GITHUB NOREPLY ADDRESS FROM SETTINGS"
git add .
git commit -m "Initial public release"
```

This preserves the private repository and exports only tracked files from the
selected commit. The new public repository will have one commit and cannot carry
the private branch or old object history. Use the exact no-reply address shown in
GitHub **Settings → Emails** so the public commit does not expose a work or
personal email address.

Inspect the new repository again before adding a remote. Only then create an
empty public GitHub repository and push `main` to its URL.

## 4. Turn on GitHub protections

Under **Settings → Security → Code security and analysis**, enable:

- secret scanning;
- push protection;
- dependency graph and Dependabot alerts;
- Dependabot security updates;
- private vulnerability reporting.

The repository includes Dependabot configuration and a CodeQL workflow. Require
the CodeQL check on `main` after its first successful run. Consider branch
protection requiring pull requests once collaborators are added.

## 5. Final public inspection

- Open the repository in a logged-out/private browser window.
- Check the commit list contains only the intended public commit.
- Open the commit and verify its displayed author email is the intended GitHub
  no-reply address.
- Check branches and tags contain no private work.
- Browse the notebook in GitHub and verify there are no rendered outputs.
- Download the repository ZIP and confirm it contains no datasets, databases,
  recordings, weights, participant data, environment files, or generated
  figures/tables.
- Verify all dataset links and clean-install commands from a fresh directory.
