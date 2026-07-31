#!/bin/bash
# Run A vs B as soon as it can possibly run, without dropping the gate.
#
# The original order trained all three ablation arms before evaluating any, so
# the baseline's score -- the thing everything else is gated on -- arrived last.
# This waits for the baseline that is already training, scores it immediately,
# and if it can do the task at all, goes straight to A vs B. The ego/shuffled
# arms follow after, because they answer a question that is no longer the one
# being asked first.
#
# The gate stays. A policy that cannot do the task scores zero on every arm, and
# four more zeros is not a ranking. That check costs nothing here: the baseline
# is training anyway.
set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
TASK=${1:-pick_dual_bottles}
SCENES=${2:-10}
PORT=8000
CKPT=~/openpi/checkpoints

AB="rt_two_handed rt_one_handed rt_two_handed_shuf rt_one_handed_shuf"
REST="rt_ego rt_shuffled"

train () {  # train <arm> <exp>
  cd ~/openpi || exit 1
  [ -d ~/openpi/assets/pi05_$1 ] || uv run scripts/compute_norm_stats.py --config-name "pi05_$1" 2>&1 | tail -2
  [ -d $CKPT/pi05_$1/$2/2999 ] || XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py "pi05_$1" --exp-name=$2 --overwrite --no-wandb-enabled 2>&1 | tail -12
}

evaluate () {  # evaluate <arm> <exp>
  pkill -f "serve_policy.py"; sleep 15
  cd ~/openpi || exit 1
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 nohup uv run scripts/serve_policy.py \
    --port $PORT policy:checkpoint --policy.config="pi05_$1" \
    --policy.dir="$CKPT/pi05_$1/$2/2999" > ~/serve_$1.log 2>&1 &
  echo $! > /tmp/serve.pid
  for i in $(seq 1 180); do
    python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),2); s.close()" 2>/dev/null && break
    kill -0 "$(cat /tmp/serve.pid)" 2>/dev/null || { echo "server died"; tail -20 ~/serve_$1.log; return 1; }
    sleep 10
  done
  sleep 20
  cd ~/RoboTwin || exit 1
  . .venv/bin/activate
  python -u ~/egorun/run_ablation.py "$TASK" "$SCENES" $PORT "$1" 2>&1 | tail -35
  deactivate
  pkill -f "serve_policy.py"; sleep 10
}

# --- take over from the ablation once its baseline has trained ----------------
echo "[priority] waiting for the baseline already training..."
while [ ! -d $CKPT/pi05_rt_base/abl/2999 ]; do sleep 60; done
echo "[priority] baseline trained at $(date -u +%H:%M:%S)Z"

# Stop the old orchestration before it starts training an arm we no longer want
# next. The baseline checkpoint is on disk, so nothing is lost.
pkill -f "ablation.sh"
pkill -f "ab.sh"
sleep 5
pkill -f "scripts/train.py"
sleep 20

evaluate rt_base abl

# --- the gate -----------------------------------------------------------------
BASE=~/egorun/abl_rt_base_$TASK.json
WINS=$(python3 -c "import json;print(json.load(open('$BASE'))['successes'])" 2>/dev/null || echo 0)
echo "[priority] baseline scored $WINS/$SCENES"
if [ "$WINS" -eq 0 ]; then
  echo "PRIORITY_GATE_FAILED: the baseline cannot do the task, so no arm can be"
  echo "  ranked against another -- every one would score zero. Fix the baseline"
  echo "  first: the 500-episode randomized demo set, or more training steps."
  echo "  Not running A vs B; it would cost six hours and produce four zeros."
  exit 0
fi

# --- A vs B, first ------------------------------------------------------------
for arm in $AB; do train $arm ab; done
for arm in $AB; do evaluate $arm ab; done
cd ~/RoboTwin && . .venv/bin/activate
python -u ~/egorun/feedback_ab.py "$TASK" 2>&1 | tail -100
deactivate
echo "AB_DONE"

# --- then the original ego/shuffled question ----------------------------------
for arm in $REST; do train $arm abl; done
for arm in $REST; do evaluate $arm abl; done
cd ~/RoboTwin && . .venv/bin/activate
python -u ~/egorun/feedback_all.py "$TASK" 2>&1 | tail -100
echo "PRIORITY_OK"
