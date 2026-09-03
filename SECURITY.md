# Security Policy

## Supported version

Only the latest public commit is supported. This is research software and has
not undergone a production security certification.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature rather than a public issue.
Repository maintainers should enable it under **Settings → Security → Code
security and analysis → Private vulnerability reporting** before publication.

Do not include participant data, database contents, credentials, private URLs,
or copyrighted recordings in a report. If private reporting is unavailable,
open a public issue containing only a request for a private contact channel.

## Deployment boundaries

- Human-response collection is closed unless `EVAL_COLLECTION_OPEN=true`.
- Upload/delete endpoints are closed unless `MUTATIONS_ENABLED=true`.
- Neither flag provides authentication. Do not enable mutation endpoints on an
  untrusted public network without adding real authentication and authorization.
- Set `ALLOWED_ORIGINS` to the exact frontend origins.
- Mount databases and audio at runtime; never bake them into an image or commit
  them to Git.
- Use only checkpoints and dependencies from trusted sources.

Before a real participant study, obtain any required institutional approval,
publish an appropriate privacy notice/consent flow, define retention rules, and
collect only the minimum necessary data.
