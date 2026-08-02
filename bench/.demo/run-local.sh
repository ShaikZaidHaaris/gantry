#!/usr/bin/env bash
# Bring the bench up locally: API, worker, web.
#
# Everything runs out of this checkout's own .venv, which has the core and the
# plugins installed. That isolation is the point: a sibling clone has its own
# editable installs resolving to a different tree, and a worker started on the
# system interpreter picks those up instead and dies on an import this branch
# added. That failure is silent in the log and surfaces as a gate that "failed"
# for no stated reason, which reads as a problem with the uploaded data.
#
#   ./run-local.sh          start all three
#   ./run-local.sh stop     stop them
#
# Paths are derived here, never written down. An earlier version hardcoded one
# machine's checkout and its Windows interpreter, so it ran on exactly one
# computer and failed on every clone of the repository it shipped in.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
API_PORT="${BENCH_API_PORT:-7910}"
WEB_PORT="${BENCH_WEB_PORT:-7911}"
WORKER_NAME=local-dev

# POSIX and Windows lay a virtualenv out differently, and which one this is is a
# question the filesystem can answer.
if   [ -x "$ROOT/.venv/bin/python" ];         then PY="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/.venv/Scripts/python.exe" ]; then PY="$ROOT/.venv/Scripts/python.exe"
else
  echo "no virtualenv at $ROOT/.venv" >&2
  echo "make one:  python3 -m venv .venv && .venv/bin/pip install -e ." >&2
  echo "then:      for p in plugins/*/; do .venv/bin/pip install -e \"\$p\"; done" >&2
  exit 1
fi

export BENCH_DATA="${BENCH_DATA:-$ROOT/bench/.localdata}"
export BENCH_WORKER_TOKEN="${BENCH_WORKER_TOKEN:-dev-token}"

# Who you are is a hash of your address and a salt, and an unset salt is
# generated fresh per process. Locally that means every restart of the API makes
# you a new visitor: the submissions you uploaded a minute ago are still in the
# database, owned by an org nobody is any more, and the UI shows an empty list.
# It reads exactly like data loss and is not. A fixed salt here is what makes a
# local run survive a restart; it stays local, and production sets its own.
export BENCH_IP_SALT="${BENCH_IP_SALT:-local-dev-fixed-salt}"

stop() {
  # The API is found by the port it holds, not by a command-line match. A
  # pattern like `pkill -f uvicorn` matches this script's own command line as
  # well as any unrelated uvicorn on the machine, and both have happened.
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$API_PORT" -sTCP:LISTEN 2>/dev/null | xargs kill 2>/dev/null || true
  fi
  # The worker is found by the name it was started with. The bracket stops the
  # pattern matching the pkill process itself.
  pkill -f -- "[-]-name $WORKER_NAME" 2>/dev/null || true
  echo "stopped api + worker (the web server belongs to the terminal that started it)"
}

if [ "${1:-start}" = "stop" ]; then stop; exit 0; fi

# A second worker on the same queue is not a spare. Both poll, both claim, and
# the loser writes its result over the winner's gate.
stop
sleep 1

echo "api    -> http://127.0.0.1:$API_PORT"
( cd "$ROOT/bench/api" && "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" ) &

# Wait for it to answer rather than sleeping a guessed number of seconds. The
# worker's first poll against a socket that is not up yet is a connection error
# in the log on every start, and on a slow machine it is more than one.
for _ in $(seq 1 40); do
  curl -sf -m 1 -o /dev/null "http://127.0.0.1:$API_PORT/api/me" && break
  sleep 0.5
done

# g3 is left out on purpose. Every gate is priced at 0, and finish_job only
# stops auto-advancing when cost_cents > 0, so a passing signal check enqueues
# the robot test with no human step. On a host with BENCH_RUNNER set that is a
# multi-hour two-arm training run nobody asked for.
echo "worker -> gates g0,g1,g2 (g3 deliberately excluded)"
( cd "$ROOT/bench/worker" && "$PY" run.py --api "http://127.0.0.1:$API_PORT" \
    --gates g0,g1,g2 --name "$WORKER_NAME" --work "$ROOT/bench/.localwork" ) &

echo "web    -> http://localhost:$WEB_PORT"
cd "$ROOT/bench/web"
[ -d node_modules ] || npm install
BENCH_API="http://127.0.0.1:$API_PORT" BENCH_WEB_PORT="$WEB_PORT" npm run dev
