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
  # 0.35, not 0.55. JAX preallocates its fraction of the whole card up front,
  # and curobo builds CUDA graphs for the motion planner afterwards -- at 0.55
  # there was nothing left and the evaluation died with an out-of-memory inside
  # newton_base, after the policy server had already come up healthy.
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 nohup uv run scripts/serve_policy.py \
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
# Wait for the card to actually drain rather than assuming twenty seconds is
# enough. A trainer still holding 40 GB while the server preallocates is the
# same out-of-memory by a different route.
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 2000 ] && break
  sleep 10
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader

evaluate rt_base abl

# --- the gate -----------------------------------------------------------------
#
# Three outcomes, not two. The first version had `|| echo 0`, which turned "the
# evaluation never produced a record" into "the baseline scored zero" -- and
# then reported the baseline cannot do the task when in fact it had never been
# asked. An absent measurement is not a measurement of absence, and this whole
# project exists to keep those apart.
BASE=~/egorun/abl_rt_base_$TASK.json
if [ ! -f "$BASE" ]; then
  echo "PRIORITY_NO_RESULT: the baseline was never evaluated -- there is no record at"
  echo "  $BASE. That is not a score of zero and must not be read as one; the"
  echo "  evaluation failed before it could answer. Check the log above for why."
  exit 2
fi
WINS=$(python3 -c "import json;print(json.load(open('$BASE'))['successes'])")
SCORED=$(python3 -c "import json;print(json.load(open('$BASE'))['episodes'])")
echo "[priority] baseline scored $WINS/$SCORED"
if [ "$WINS" -eq 0 ]; then
  echo "PRIORITY_GATE_FAILED: the baseline was evaluated on $SCORED scenes and solved"
  echo "  none of them, so no arm can be ranked against another -- every one would"
  echo "  score zero. Fix the baseline first: the 500-episode randomized demo set,"
  echo "  or more training steps. Not running A vs B; it would produce four zeros."
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
