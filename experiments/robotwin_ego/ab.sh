#!/bin/bash
# A vs B, gated on the three-arm ablation having produced something to compare.
#
#   rt_two_handed   base + clips where both hands were tracked and moving
#   rt_one_handed   base + clips where one hand was mostly absent
#   *_shuf          each with its own detached-action control
#
# The gate is the point. If base scores zero, none of these can be separated
# either, and six more GPU-hours would buy four more zeros. The fix then is more
# demonstrations or more training -- not more arms.
set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
TASK=${1:-pick_dual_bottles}
SCENES=${2:-10}
PORT=8000
ARMS="rt_two_handed rt_one_handed rt_two_handed_shuf rt_one_handed_shuf"

while pgrep -f "ablation.sh" > /dev/null; do sleep 120; done

BASE=~/egorun/abl_rt_base_$TASK.json
if [ ! -f "$BASE" ]; then
  echo "AB_ABORT: the ablation produced no base arm; nothing to gate on"
  exit 1
fi
WINS=$(python3 -c "import json;print(json.load(open('$BASE'))['successes'])")
echo "[ab] base scored $WINS/$SCENES"
if [ "$WINS" -eq 0 ]; then
  echo "AB_ABORT: base scored 0. A policy that cannot do the task at all cannot"
  echo "  show which data helped it -- every arm would score zero and the"
  echo "  comparison would have no power. Fix the baseline first: the 500-episode"
  echo "  randomized demo set, or more training steps."
  exit 0
fi

for arm in $ARMS; do
  cd ~/openpi || exit 1
  [ -d ~/openpi/assets/pi05_$arm ] || uv run scripts/compute_norm_stats.py --config-name "pi05_$arm" 2>&1 | tail -2
  [ -d ~/openpi/checkpoints/pi05_$arm/ab/2999 ] || XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py "pi05_$arm" --exp-name=ab --overwrite --no-wandb-enabled 2>&1 | tail -15
done

for arm in $ARMS; do
  pkill -f "serve_policy.py"; sleep 15
  cd ~/openpi || exit 1
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 nohup uv run scripts/serve_policy.py \
    --port $PORT policy:checkpoint --policy.config="pi05_$arm" \
    --policy.dir="/home/ubuntu/openpi/checkpoints/pi05_$arm/ab/2999" > ~/serve_$arm.log 2>&1 &
  echo $! > /tmp/serve.pid
  for i in $(seq 1 180); do
    python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),2); s.close()" 2>/dev/null && break
    kill -0 "$(cat /tmp/serve.pid)" 2>/dev/null || { echo "server died"; tail -20 ~/serve_$arm.log; exit 1; }
    sleep 10
  done
  sleep 20
  cd ~/RoboTwin && . .venv/bin/activate
  python -u ~/egorun/run_ablation.py "$TASK" "$SCENES" $PORT "$arm" 2>&1 | tail -35
  deactivate
done
pkill -f "serve_policy.py"; sleep 10

cd ~/RoboTwin && . .venv/bin/activate
python -u ~/egorun/feedback_ab.py "$TASK" 2>&1 | tail -100
echo "AB_OK"
