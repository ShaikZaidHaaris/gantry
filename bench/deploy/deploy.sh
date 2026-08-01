#!/usr/bin/env bash
# Put the bench on a host and keep it there.
#
# Usage:  bench/deploy/deploy.sh ubuntu@HOST /path/to/key.pem
#
# What this does, and what it deliberately does not
# -------------------------------------------------
# Builds the frontend, copies the API, the built bundle and the worker to the
# host, and installs both as systemd units. It does *not* open a port: the API
# binds to loopback and something else is responsible for exposing it (see
# "Reaching it" below). A deploy script that punches a hole in a firewall is a
# deploy script that does it on a day nobody was watching.
#
# Why systemd and not nohup
# -------------------------
# Every earlier attempt backgrounded the worker with nohup, and every time it
# died quietly with the shell that launched it. A worker that is gone looks
# exactly like a queue that is slow -- the submissions sit at "queued" and
# nothing anywhere says why. Restart=always is the difference between an outage
# and a blip.
set -euo pipefail

HOST="${1:?usage: deploy.sh user@host /path/to/key.pem}"
KEY="${2:?usage: deploy.sh user@host /path/to/key.pem}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no "$HOST")
RSYNC_E="ssh -i $KEY -o StrictHostKeyChecking=no"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE=/home/ubuntu/gantry_bench

echo "==> building the frontend"
(cd "$ROOT/bench/web" && npm ci --silent && npm run build)

echo "==> copying"
"${SSH[@]}" "mkdir -p $REMOTE/api $REMOTE/web/dist $REMOTE/worker $REMOTE/data /home/ubuntu/bench_work"
rsync -az --delete -e "$RSYNC_E" --exclude '__pycache__' --exclude data \
  "$ROOT/bench/api/"    "$HOST:$REMOTE/api/"
rsync -az --delete -e "$RSYNC_E" \
  "$ROOT/bench/web/dist/" "$HOST:$REMOTE/web/dist/"
rsync -az --delete -e "$RSYNC_E" --exclude '__pycache__' \
  "$ROOT/bench/worker/" "$HOST:$REMOTE/worker/"
rsync -az -e "$RSYNC_E" "$ROOT/bench/deploy/" "$HOST:$REMOTE/deploy/"

echo "==> environment"
# Generated on the host and never in the repo. The worker token is the only
# thing standing between the job queue and anything that can route to the port.
"${SSH[@]}" bash -s <<'REMOTE_ENV'
set -e
ENV=/home/ubuntu/gantry_bench/env
if [ ! -f "$ENV" ]; then
  TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(16))")
  {
    echo "BENCH_DATA=/home/ubuntu/gantry_bench/data"
    echo "BENCH_WORKER_TOKEN=$TOKEN"
    # Every feedback module this host has. A worker with fewer installed returns
    # a shorter report on the same footage, and a shorter report reads as a
    # cleaner dataset -- so the gate names anything missing rather than being
    # quietly thinner.
    echo "BENCH_REQUIRED_MODULES=$(/home/ubuntu/gantry/.venv/bin/python -c "
from importlib.metadata import entry_points
print(','.join(sorted(e.name for e in entry_points(group='gantry.feedback'))))")"
  } > "$ENV"
  chmod 600 "$ENV"
  echo "  wrote a fresh $ENV"
else
  echo "  keeping the existing $ENV"
fi
REMOTE_ENV

echo "==> services"
"${SSH[@]}" bash -s <<'REMOTE_SVC'
set -e
sudo cp /home/ubuntu/gantry_bench/deploy/gantry-api.service /etc/systemd/system/
sudo cp /home/ubuntu/gantry_bench/deploy/gantry-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gantry-api gantry-worker
sleep 6
systemctl is-active gantry-api gantry-worker
curl -s -o /dev/null -m 5 -w "  api: %{http_code}\n" http://127.0.0.1:8090/api/me
curl -s -o /dev/null -m 5 -w "  spa: %{http_code}\n" http://127.0.0.1:8090/
REMOTE_SVC

cat <<'NOTE'

==> done. Reaching it:

    cloudflared tunnel --url http://127.0.0.1:8090

  which prints a https://<name>.trycloudflare.com URL. That URL is ephemeral --
  it changes whenever cloudflared restarts. A stable one needs a Cloudflare
  account and a named tunnel.

  The alternative is opening a port in the host's firewall, which this script
  will not do for you.

==> before anyone outside your team gets the link:

  There is no authentication. `viewer()` in the API reads an `x-demo-user`
  header and trusts it, so anybody who can reach this can see every submission
  and upload their own. That seam is the one place real auth goes; nothing else
  needs to change.

==> training

  The robot test only *produces* rollouts when BENCH_RUNNER points at a runner
  (see bench/runner). Left unset, the gate reads rollouts already on disk and
  otherwise reports plainly that it cannot make them -- which is the right
  default for anything shared, because a stray click should not start a run
  that costs a day of GPU.
NOTE
