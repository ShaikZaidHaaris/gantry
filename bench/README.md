# Gantry Bench — the product

Three deployables in one repo, per `webui/ARCHITECTURE.md`.

    bench/api      FastAPI + SQLAlchemy — state, uploads, jobs, SSE
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

## What is built (step 1 of 6)

Submissions, upload with progress, the job queue, the event log, SSE, and
**Gate 0 · Intake** — the free readability check. A user can submit a dataset
and see their own numbers, or a refusal that names what to fix.

Next: Gate 1 (the free data report), then the live timeline over SSE, then the
paid gates.
