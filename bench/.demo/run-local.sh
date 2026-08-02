#!/usr/bin/env bash
# Bring the bench up locally: API, worker, web.
#
# Everything runs out of the worktree's own .venv, which has closed-loop's core
# and plugins installed. That isolation is the point -- the editable installs in
# the sibling checkout at C:/robotics/gantry resolve to a different tree, and a
# worker started on the system interpreter picks those up instead and dies on
# an import the branch added. That failure is silent in the log and shows up as
# a gate that "failed" for no stated reason, so it is worth not reproducing.
#
#   ./run-local.sh          start all three
#   ./run-local.sh stop     stop them
#
set -euo pipefail

ROOT="C:/robotics/gantry-closed-loop"
PY="$ROOT/.venv/Scripts/python.exe"
API_PORT=7910
WEB_PORT=7911

export BENCH_DATA="$ROOT/bench/.localdata"
export BENCH_WORKER_TOKEN="dev-token"

stop() {
  # Only ours: match on the port and the worker's own --name.
  powershell -NoProfile -Command "
    Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
      Where-Object { \$_.CommandLine -match 'port $API_PORT|--name local-dev' } |
      ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }
  " 2>/dev/null || true
  echo "stopped api + worker (leave the web server to its own terminal)"
}

if [ "${1:-start}" = "stop" ]; then stop; exit 0; fi

# A second worker on the same queue is not a spare -- both poll, both claim, and
# the loser's crash is written onto the gate as though the data were at fault.
stop

echo "api    -> http://127.0.0.1:$API_PORT"
( cd "$ROOT/bench/api" && "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" ) &

sleep 3

# g3 is left out on purpose. Every gate is priced at 0, and finish_job only
# stops auto-advancing when cost_cents > 0, so a passing signal check enqueues
# the robot test with no human step. On a host with BENCH_RUNNER set that is a
# multi-hour two-arm training run nobody asked for.
echo "worker -> gates g0,g1,g2 (g3 deliberately excluded)"
( cd "$ROOT/bench/worker" && "$PY" run.py --api "http://127.0.0.1:$API_PORT" \
    --gates g0,g1,g2 --name local-dev --work "$ROOT/bench/.localwork" ) &

echo "web    -> http://localhost:$WEB_PORT"
cd "$ROOT/bench/web" && BENCH_API="http://127.0.0.1:$API_PORT" BENCH_WEB_PORT="$WEB_PORT" npm run dev
