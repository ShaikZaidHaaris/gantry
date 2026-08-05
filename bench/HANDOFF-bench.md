# Handoff — gantry-closed-loop/bench

Last updated 2026-08-04. Everything below is verified state, not intention;
where something is pending it says whose move it is.

**What this is:** the FastAPI + React + worker bench app, *not* the static
RoboBench site. Branch `closed-loop`; `bench-local-run` is kept identical.

## Live right now

    https://gantry.gurasees.com

Caddy on the L40S box (`i-0edd3c6d798336706`, Elastic IP `35.172.172.202`),
TLS from Let's Encrypt, renewed by Caddy itself. `gurasees.com/gantry.html`
is a GitHub Pages redirect pointing at it.

| | |
|---|---|
| units | `caddy`, `gantry-api`, `gantry-worker`, all enabled, survive reboot |
| identity | signed cookie (see below); `edge` mode behind Caddy |
| uploads | capped at 1 GB, enforced on bytes received, reported on `/api/me` |
| checkpoints | `/opt/dlami/nvme/openpi-checkpoints` (391 GB ephemeral NVMe) |
| gate 3 | armed: `BENCH_RUNNER=/home/ubuntu/gantry_runner/run.sh` |
| ssh | `newgpu-key-useast.pem` (Downloads), user `ubuntu` |

## ⚠ A ROBOT TEST IS LIKELY RUNNING — check before touching anything

`sub_00c71a3c78a1`, attempt 4. The first full end-to-end run this product has
ever had. Check state:

    ssh ... 'tail -3 /home/ubuntu/bench_work/sub_00c71a3c78a1/rollouts/progress.jsonl'
    ssh ... 'ps -eo pid,etime,cmd | grep -E "[r]unner.py|[t]rain.py|[r]un_ablation"'

As of this writing: arm 1 (the visitor's data) trained 3000 steps and evaluated
50 scenes, **2/50 successes, record written**. Arm 2 (shuffled control) was
mid-training. When the runner finishes, the worker posts the result and
`finish_job` **overwrites** the gate — including the bogus "stopped responding"
failure currently on it (see The sweep incident).

**Restarting `gantry-worker` kills the run** — the runner is its child. Do not
run `deploy.sh` while a run is live; it restarts the worker. The pattern used
all day for shipping web/API changes without touching the worker:

    tar -czf - -C bench/web/dist . | ssh ... 'tar -xzf - -C /home/ubuntu/gantry_bench/web/dist'
    # api likewise, then: sudo systemctl restart gantry-api   (only the api)

## Committed but NOT yet on the box

- `fb1c0fe` — `STALE_AFTER` 90s → 600s. Deploy after the run completes
  (full `deploy.sh` is fine then; it also refreshes the worker + runner).
  Everything else on `closed-loop` is deployed and verified live.

## The sweep incident (why the gate says failed while the run is alive)

Deploying at 16:07 UTC on 08-03 (DB backup holds the write lock, API restart,
startup migration) stalled heartbeat *writes* past the 90s sweep threshold, and
`sweep_stale` declared a healthy worker dead three hours into the run. The
worker never noticed and kept going. `finish_job` overwrites by job id without
checking status, so the honest result lands when the runner completes. The
margin fix is `fb1c0fe`. Lesson: heartbeats are writes; anything that stalls
writes (backup, deploy, migration) eats sweep margin.

## Identity is now a signed cookie (16edc75)

MAC addresses were asked for and are impossible (link-layer, rewritten at every
hop). The cookie does what was wanted: survives IP/network changes, separates
people behind one NAT.

- Cookie `gantry_visitor`, HttpOnly, SameSite=lax, Secure when behind a proxy
  (`cookie_secure()` follows identity mode), 400-day max-age. HMAC-signed;
  `token_key` hashes the whole cookie; org column `token_hash` (unique).
- **Migration**: an org with `token_hash IS NULL` is adopted by `ip_hash` on its
  owner's next visit *from the same address*, then carries a cookie. Adoption
  never touches an org that already has a cookie (mutation-tested).
- `BENCH_COOKIE_SECRET` falls back to `BENCH_IP_SALT`. **Rotating the salt now
  invalidates every cookie as well as every org key. Do not.**

### Pending: reattach the founder's own submissions (needs one fact from them)

They uploaded `sub_00c71a3c78a1` (and probably `sub_ea8f6e2d0a16`) under IP
identity, then changed networks *before ever revisiting*, so adoption had
nothing to match and their browser now holds a fresh empty org. The data is
fine. Fix, once they report the Visitor label shown top-right on the site:

    UPDATE submissions SET org_id = '<their cookie org id>'
    WHERE id IN ('sub_00c71a3c78a1', 'sub_ea8f6e2d0a16');

(Find the org id by label suffix: `SELECT id,name FROM orgs WHERE name LIKE
'Visitor XXXX'`.) Back up first: `python3 -c "import sqlite3; ..."` with the
`.backup()` API, never `cp` (WAL).

### Known wart: the mint burst

A first-time browser fires several cookie-less API calls in parallel; each
mints its own org (three in one second observed). The browser settles on the
last Set-Cookie; the rest are empty litter. Bounded, not dangerous. Fix worth
doing: bind a valid-but-unknown cookie to the presented value instead of
re-minting, and sweep empty token-orgs later.

## The g3 debugging campaign (all fixed, all on the box)

Every failure below presented as the *next* layer's error. In order found:

| Root cause | Fix | Commit |
|---|---|---|
| `deploy.sh` never shipped the runner; box copy was hand-placed, drifted | ship + replace | `c838ea7` |
| checkpoints landed on 12 GB root; NVMe unused | `BENCH_CHECKPOINTS` + st_dev guard | `1834954` |
| error cards showed carets, not the exception (JAX footer pushed it out) | `_why()` | `c7a60ce` |
| norm-stats/eval output went nowhere ("exit 1" undiagnosable) | per-stage logs | `9d77162` |
| logind `RemoveIPC` purged DataLoader semaphores when the last SSH session closed (my own polling killed my runs) | `loginctl enable-linger ubuntu` (box-side) | — |
| trainer looked for checkpoints on NVMe, wrote to openpi default | `--checkpoint-base-dir` from the same constant | `6b3b05b` |
| **worker env has no `LD_LIBRARY_PATH` → `CUDNN_STATUS_NOT_SUPPORTED` at step 0**; every hand-run repro "worked" because a shell has one | `CUDA_LIBS` prepended in `run()` | `12f79be` |
| **`LeRobotAlohaDataConfig` trains 14-wide actions from 16-wide EE data**; silent until eval refuses the chunk 1.6h later | `LeRobotPoseDataConfig`, and `register()` now replaces stale blocks instead of skipping | `7f9291d` |

Diagnostics that lied along the way: probes run from an SSH shell have
`LD_LIBRARY_PATH` (the worker does not); `pkill -f` matches your own ssh
command line (bracket the pattern: `[r]unner`); `systemctl show systemd-logind
-p RemoveIPC` reports the wrong object (`busctl get-property org.freedesktop.login1
... RemoveIPC` is authoritative).

## Attempts bookkeeping

g3 for `sub_00c71a3c78a1` has 6 job rows; `MAX_ATTEMPTS = 3`, so the Run-again
button will refuse. Requeue script used all day: `/tmp/requeue.py` on the box
(resets the gate, adds a job, keeps history). If the live run lands `passed`,
none of this matters.

## When the run lands: the path to the leaderboard

1. `finish_job` overwrites the gate (automatic).
2. Board ranks only a passed g3 (automatic).
3. Owner presses Publish — opt-in, `listed=1` (needs the reattach above first).

## Security debt

- `BENCH_TRUST_SECRET` and `BENCH_WORKER_TOKEN` were printed into a Claude
  transcript on 08-03 (unmasked env dump). **Rotate both**: new values into
  `/home/ubuntu/gantry_bench/env` *and* `/etc/caddy/Caddyfile` (trust secret
  must match the `header_up` line), then restart caddy + api (+ worker for the
  token — wait for the run).
- `BENCH_IP_SALT` was exposed too but **must not be rotated** (orphans every
  org and cookie). Exposure is low-severity: forging a signature only mints a
  fresh empty org. At the next quiet moment, set an explicit
  `BENCH_COOKIE_SECRET` so cookie signing stops depending on the exposed salt.
- An AWS access key was pasted in chat on 08-03 (`AKIA...RBGI`). It is the
  `default` profile's only key: **create new key → reconfigure → then delete**,
  not deactivate-first.

## Tests

76 green: 70 API (`bench/api`), 6 worker. Identity tests model two *browsers*
(cookie jars), not two addresses. Mutation-testing is the house habit: every
guard added this week was verified to bite by breaking it and watching the
suite fail; three tests were rewritten after passing against broken code.

## Other open threads

- Root disk 98% full; five 8.5 GB orphaned checkpoints in
  `~/openpi/checkpoints` await a keep/delete decision (owner's call).
- Three Lab Lab submissions have passed robot tests, unpublished; also invisible
  (org has neither ip_hash nor token_hash) — publish or adopt, owner's call.
- Gate 3 is free (`cost_cents: 0`) on a public URL: ~4h of L40S per click.
  Gating to one org is a small change to `start`/`retry` if the bill matters.
- `deploy-smoke-test` leftovers and empty burst orgs could use a sweep.
