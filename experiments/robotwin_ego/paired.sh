#!/bin/bash
# Both checkpoints, closed-loop in RoboTwin, then the comparison.
#
# Sequential rather than side by side: a pi05 LoRA server holds ~22.5 GB and the
# L40S has 46, so two at once would leave the simulator's renderer nothing.
TASK=${1:-pick_dual_bottles}
SCENES=${2:-10}
PORT=8000

# uv lives in ~/.local/bin, which a non-interactive shell does not pick up, so
# `nohup uv ...` fails with "No such file or directory" and the server never
# starts. The port probe catches it in ten seconds rather than twenty minutes.
export PATH="$HOME/.local/bin:$PATH"

serve() {   # serve <config> <checkpoint dir>
  cd ~/openpi || exit 1
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 nohup uv run scripts/serve_policy.py \
    --port $PORT policy:checkpoint --policy.config="$1" --policy.dir="$2" \
    > ~/serve_rt_$3.log 2>&1 &
  echo $! > /tmp/serve.pid
  echo "[paired] serving $1 from $2"
  # Probe the port rather than grep the log: the wording of openpi's ready line
  # is not a contract, and a grep that never matches fails twenty minutes later
  # as a timeout instead of immediately as a refused connection.
  for i in $(seq 1 180); do
    if python3 -c "import socket,sys; s=socket.create_connection(('127.0.0.1',$PORT),2); s.close()" 2>/dev/null; then
      echo "[paired] port $PORT open after ${i}0s"
      break
    fi
    if ! kill -0 "$(cat /tmp/serve.pid)" 2>/dev/null; then
      echo "[paired] the server died; last lines:"; tail -30 ~/serve_rt_$3.log; return 1
    fi
    sleep 10
  done
  sleep 20   # the first infer compiles a graph; let it settle
}

stop() {
  [ -f /tmp/serve.pid ] && kill "$(cat /tmp/serve.pid)" 2>/dev/null
  pkill -f "serve_policy.py" 2>/dev/null
  sleep 15
}

run() {     # run <arm>
  cd ~/RoboTwin || exit 1
  . .venv/bin/activate
  python ~/egorun/run_robotwin.py "$TASK" "$SCENES" $PORT "$1" 2>&1 | tail -60
  deactivate
}

set -x
stop

serve pi05_ego /home/ubuntu/openpi/checkpoints/pi05_ego/v2/1999 ego
run ego
stop

serve pi05_ego_shuffled /home/ubuntu/openpi/checkpoints/pi05_ego_shuffled/v2/1999 shuffled
run shuffled
stop

cd ~/RoboTwin && . .venv/bin/activate
python ~/egorun/compare.py "$TASK" 2>&1 | tail -60
echo "PAIRED_OK"
