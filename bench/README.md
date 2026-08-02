# Gantry Bench, the product

Three deployables in one repo, per `webui/ARCHITECTURE.md`.

    bench/api      FastAPI + SQLAlchemy: state, uploads, jobs, SSE
    bench/worker   claims jobs, runs gates, reports verdicts
    bench/web      React + TypeScript + TanStack Query

## Run it

    # 1. API
    .venv/bin/python -m uvicorn app.main:app --app-dir bench/api --port 7910

    # 2. Worker (one per gate class; g0 is CPU-only)
    cd bench/worker && ../../.venv/bin/python run.py --gates g0

    # 3. Web
    cd bench/web && npm install && npm run dev     # http://localhost:7911

SQLite and local disk stand in for Postgres and S3. The seam is `api/app/db.py`
plus two storage helpers; nothing else knows the difference.

## Something to upload

`bench/data/` is gitignored, so a fresh clone starts with an empty bench. Two
real datasets are committed at the repository root for this:

    samples/two_handed_58clips.zip
    samples/one_handed_58clips.zip

Upload both. Separately they show the pipeline runs; together they show the
thing the product exists to detect. See `samples/README.md`.

## Deploying it

    bench/deploy/deploy.sh ubuntu@HOST /path/to/key.pem

Builds the frontend, copies the API, the bundle and the worker, and installs
both as systemd units with `Restart=always`. The API serves the built SPA
itself, so there is one process and no proxy to misconfigure.

It binds to loopback and does not open a port. Expose it with

    cloudflared tunnel --url http://127.0.0.1:8090

The token in `/home/ubuntu/gantry_bench/env` is generated on the host and is
never in this repository.

**There is no authentication.** `viewer()` reads an `x-demo-user` header and
trusts it, so anyone who can reach the deployment can read every submission and
upload their own. That function is the single seam real auth goes through.

## What is built

All four gates, and the screens over them.

    Gate 0 · Intake         can this be read at all. Seconds, free, CPU
    Gate 1 · Data report    what is in it, from every installed check. A minute, free, CPU
    Gate 2 · Signal check   does a probe learn anything from it. Minutes, GPU
    Gate 3 · Robot test     does a policy trained on it do better. Hours, GPU and simulator

Around them: submissions and versions, upload with progress, the job queue, the
event log, live gate progress over SSE that survives a reload, the budget panel
that says what a trial count can and cannot detect, retry that distinguishes our
failure from a judgement on the data, the resubmit loop, the leaderboard with a
compact letter display and a pair chooser, and export.

What is not built is in `HANDOFF.md` section 9. The largest gap: no submission
has yet produced its own shuffled control end to end through the product, so
every live verdict is a ranking rather than an attribution.
