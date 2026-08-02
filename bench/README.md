# Gantry Bench, the product

Three deployables in one repo, per `webui/ARCHITECTURE.md`.

    bench/api      FastAPI + SQLAlchemy: state, uploads, jobs, SSE
    bench/worker   claims jobs, runs gates, reports verdicts
    bench/web      React + TypeScript + TanStack Query

## Run it

    bench/.demo/run-local.sh          # all three; stop with `run-local.sh stop`

API on `127.0.0.1:7910`, worker on gates g0–g2, UI on **http://localhost:7911**.
Or by hand:

    # 1. API
    .venv/bin/python -m uvicorn app.main:app --app-dir bench/api --port 7910

    # 2. Worker (one per gate class; g0 is CPU-only)
    cd bench/worker && ../../.venv/bin/python run.py --gates g0

    # 3. Web
    cd bench/web && npm install && npm run dev     # http://localhost:7911

Two things that will otherwise cost an afternoon:

**Use a venv holding *this* checkout's packages.** A sibling clone has its own
editable installs, and a worker started on the system interpreter picks those up
instead, then dies on an import this branch added. The traceback never reaches
the UI — it shows as a gate that "failed" with no stated cause, which reads as a
problem with the uploaded data.

**Do not run two workers on one queue.** Both poll, both claim, and the loser
writes its result over the winner's gate. `run-local.sh` clears stragglers
before starting for exactly this reason.

`run-local.sh` also runs only `g0,g1,g2` on purpose. Every gate is priced at
zero and `finish_job` stops auto-advancing only when `cost_cents > 0`, so a
passing signal check would otherwise enqueue the robot test with no human step —
a multi-hour training run on any host where `BENCH_RUNNER` is set.

SQLite and local disk stand in for Postgres and S3. The seam is `api/app/db.py`
plus two storage helpers; nothing else knows the difference.

## Something to upload

`bench/data/` is gitignored, so a fresh clone starts with an empty bench. The
experiment's three training sets are committed at the repository root for this:

    samples/baseline.zip                        50 clips
    samples/baseline_plus_ego_two_handed.zip    the same 50, plus one half of the ego footage
    samples/baseline_plus_ego_one_handed.zip    the same 50, plus the other half

Upload all three. Any one of them shows the pipeline runs; together they show
the thing the product exists to detect. See `samples/README.md`.

## Deploying it

    bench/deploy/deploy.sh ubuntu@HOST /path/to/key.pem

Builds the frontend, copies the API, the bundle and the worker, and installs
both as systemd units with `Restart=always`. The API serves the built SPA
itself, so there is one process and no proxy to misconfigure.

It binds to loopback and does not open a port. Expose it with

    cloudflared tunnel --url http://127.0.0.1:8090

The token in `/home/ubuntu/gantry_bench/env` is generated on the host and is
never in this repository.

## Who can see what

Every visiting IP address gets its own org. Submissions are scoped to it, so two
visitors cannot see each other's uploads, and a result reaches the shared
leaderboard only when its owner publishes it.

**Read `deploy/IDENTITY.md` before putting this on the internet.** Behind
Cloudflare the client's real address arrives in a *header*, and a header is
something the client writes — so an origin that believes `CF-Connecting-IP`
unconditionally lets anyone who can reach it directly claim any address and read
that address's uploads. Identity here *is* the access control.

A forwarding header is therefore trusted only when the request carries a shared
secret the edge attaches. With none configured, forwarding headers are ignored
and every visitor collapses into one org — loud, harmless, and reported by
`/api/me`. The opposite failure is silent. Check which you are in:

    curl -s https://your-host/api/me | jq .identity     # "mode": "edge" | "direct"

Addresses are salted-hashed at rest and never displayed; the UI says
"Visitor 4c85".

This **partitions** visitors, it does not authenticate them. Anyone behind the
same NAT is one visitor, and a changed address is a new one who cannot reach
what they uploaded an hour ago. Both are inherent to naming people by address. A
signed cookie fixes both and slots into the same seam: `viewer()` in
`api/app/main.py` is still the only place that decides who is asking.

### Publishing

Off by default. `POST /api/submissions/{id}/listed` toggles both ways, refuses
without a finished robot test, and 404s on somebody else's submission. You can
always see your own unlisted entry on the board — deciding whether to publish
requires seeing where it stands.

## What is built

All four gates, and the screens over them.

    Gate 0 · Intake         can this be read at all. Seconds, CPU
    Gate 1 · Data report    what is in it, from every installed check. A minute, CPU
    Gate 2 · Signal check   does a probe learn anything from it. Minutes, GPU
    Gate 3 · Robot test     does a policy trained on it do better. Hours, GPU and simulator

Only gate 0 can refuse outright. Gate 1 describes and never passes or fails your
data; gate 2 is the one that can say no, by fitting a probe on your clips and
again on a copy with the actions attached to the wrong episodes, then comparing
both on footage neither has seen. It needs at least 10 episodes —
`smallest_conclusive() + FIT_FLOOR`, derived from the test rather than picked.

Verdicts come from a fixed vocabulary, and the distinctions are load-bearing:

    Passed        the check ran and produced its result — not always "good"
    Refused       we read your data and something in it is wrong
    Can't tell    it ran and could not conclude. NOT a no
    Our error     our machinery broke; your data was never judged

Note the prices are gone from this list. Every gate is `cost_cents: 0`, so the
word "free" was the UI stating a pricing policy nobody has decided.

Around them: submissions and versions, upload with progress, the job queue, the
event log, live gate progress over SSE that survives a reload, the budget panel
that says what a trial count can and cannot detect, retry that distinguishes our
failure from a judgement on the data, the resubmit loop, the leaderboard with a
compact letter display and a pair chooser, and export.

What is not built is in `HANDOFF.md` section 9. The largest gap: no submission
has yet produced its own shuffled control end to end through the product, so
every live verdict is a ranking rather than an attribution.

## Tests

    (cd bench/api    && python -m pytest tests/ -q)     # 32
    (cd bench/worker && python -m pytest tests/ -q)     #  6
    (cd bench/web    && npm run build)                  # tsc -b, strict

Two suites earn their place by covering configurations the others structurally
could not:

- **`api/tests/test_claim.py`** claims jobs from *threads*. Sequential requests
  never race — the first commits `running` and the second's SELECT no longer
  matches — which is why an unguarded claim passed every test written before it.
- **`api/tests/test_migration.py`** builds the *old* schema by hand and migrates
  it. Every other test starts from an empty directory, where `create_all` builds
  the tables from the models and the migration path never runs — the one
  configuration no deployment is ever in. It hid a bug that published every
  submission to the shared leaderboard.

## Rough edges worth knowing

- **`bench/runner` has never run end to end**, and `deploy.sh` does not ship it,
  so g3 is inert on any host that script produces.
- **No billing exists.** All four gates are `cost_cents: 0`.
- **g3 auto-advances**, per the note under "Run it".
- **`viewer()` partitions, it does not authenticate.** See above.
