#!/bin/bash
# Dataset A against dataset B, scored on the ladder rather than on the binary.
#
#   rt_two_handed   base + clips where both hands were tracked and moving
#   rt_one_handed   base + clips where one hand was mostly absent
#
# Fifty scenes each, drawn from the same screened hundred, so the two arms face
# identical arrangements and can be paired scene by scene. Fifty is enough
# because the measurement is no longer a coin flip: reaching, disturbing and
# lifting each happen far more often than solving, and the baseline solves 12
# times in 100. On the binary alone, separating these two would have needed 326
# trials per arm.
#
# A is already trained. B is not, and trains first so that both evaluations run
# against a quiet card.
set -x
export PATH="$HOME/.local/bin:$PATH"
export WANDB_MODE=disabled
TASK=${1:-pick_dual_bottles}
SCENES=${2:-50}
PORT=8000
CKPT=~/openpi/checkpoints

train () {
  cd ~/openpi || exit 1
  [ -d ~/openpi/assets/pi05_$1 ] || uv run scripts/compute_norm_stats.py --config-name "pi05_$1" 2>&1 | tail -2
  [ -d $CKPT/pi05_$1/ab/2999 ] || XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py "pi05_$1" --exp-name=ab --overwrite --no-wandb-enabled 2>&1 | tail -10
}

evaluate () {
  pkill -f serve_policy.py; sleep 15
  for _ in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 2000 ] && break
    sleep 15
  done
  cd ~/openpi || exit 1
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 setsid uv run scripts/serve_policy.py --port $PORT \
    policy:checkpoint --policy.config="pi05_$1" --policy.dir="$CKPT/pi05_$1/ab/2999" \
    > ~/serve_$1.log 2>&1 &
  for i in $(seq 1 120); do
    python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),2); s.close()" 2>/dev/null && break
    sleep 10
  done
  sleep 25
  cd ~/RoboTwin || exit 1
  . .venv/bin/activate
  # tee, not tail: piping through tail buffers everything until the process
  # exits, which has hidden progress on every long run today.
  python -u ~/egorun/run_ablation.py "$TASK" "$SCENES" $PORT "$1" 2>&1 | tee ~/eval_$1.log | tail -25
  deactivate
  pkill -f serve_policy.py; sleep 10
}

train rt_one_handed
evaluate rt_two_handed
evaluate rt_one_handed

cd ~/RoboTwin && . .venv/bin/activate
python -u ~/egorun/compare_ab.py "$TASK" 2>&1 | tail -80
echo "AB_STAGES_OK"
