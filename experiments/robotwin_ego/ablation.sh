#!/bin/bash
# The three-arm ablation, end to end.
#
#   rt_base      RoboTwin's own demonstrations
#   rt_ego       the same, plus the ego data
#   rt_shuffled  the same, plus the ego data with its actions detached
#
# Sequential throughout: a pi05 LoRA holds ~22.5 GB of the L40S's 46, and the
# simulator's renderer needs the rest. Each arm trains, then serves, then runs,
# then gets out of the way.
set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
TASK=${1:-pick_dual_bottles}
SCENES=${2:-10}
PORT=8000
ARMS="rt_base rt_ego rt_shuffled"

# Wait for anything still holding the GPU from the previous experiment.
while pgrep -f "run_robotwin.py|serve_policy.py" > /dev/null; do sleep 60; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# --- norm stats, then training ------------------------------------------------
for arm in $ARMS; do
  cd ~/openpi || exit 1
  if [ ! -d ~/openpi/assets/pi05_$arm ]; then
    uv run scripts/compute_norm_stats.py --config-name "pi05_$arm" 2>&1 | tail -3
  fi
  if [ ! -d ~/openpi/checkpoints/pi05_$arm/abl/2999 ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      uv run scripts/train.py "pi05_$arm" --exp-name=abl --overwrite --no-wandb-enabled \
      2>&1 | tail -25
  fi
  ls -d ~/openpi/checkpoints/pi05_$arm/abl/* 2>/dev/null | tail -2
done

# --- serve and evaluate each arm ----------------------------------------------
for arm in $ARMS; do
  pkill -f "serve_policy.py"; sleep 15
  cd ~/openpi || exit 1
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 nohup uv run scripts/serve_policy.py \
    --port $PORT policy:checkpoint --policy.config="pi05_$arm" \
    --policy.dir="/home/ubuntu/openpi/checkpoints/pi05_$arm/abl/2999" \
    > ~/serve_$arm.log 2>&1 &
  echo $! > /tmp/serve.pid

  for i in $(seq 1 180); do
    python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),2); s.close()" 2>/dev/null && break
    kill -0 "$(cat /tmp/serve.pid)" 2>/dev/null || { echo "server died"; tail -25 ~/serve_$arm.log; exit 1; }
    sleep 10
  done
  sleep 20

  cd ~/RoboTwin || exit 1
  . .venv/bin/activate
  python -u ~/egorun/run_ablation.py "$TASK" "$SCENES" $PORT "$arm" 2>&1 | tail -40
  deactivate
done
pkill -f "serve_policy.py"; sleep 10

# --- the whole feedback layer, not just the control ---------------------------
cd ~/RoboTwin && . .venv/bin/activate
python -u ~/egorun/feedback_all.py "$TASK" 2>&1 | tail -120
echo "ABLATION_OK"
