#!/bin/bash
# One hundred trials on the baseline, because ten told us almost nothing.
#
# 1/10 has a 95% interval of roughly [0%, 45%]: enough to say the policy is not
# dead, not enough to size anything against. At a hundred trials a true rate
# near 10% comes back as roughly [5%, 18%], which a comparison can be built on.
#
# This runs before the rest of A/B on purpose. If the baseline really is near
# 10%, four arms at ten trials each cannot separate from one another whatever
# the data does, and the four hours spent finding that out would buy a verdict
# we can already predict.
set -x
export PATH="$HOME/.local/bin:$PATH"
TASK=${1:-pick_dual_bottles}
SCENES=${2:-100}
PORT=8000

# Let whatever is training finish rather than throwing away a part-trained arm.
while pgrep -f "scripts/train.py" > /dev/null; do sleep 120; done
pkill -f serve_policy.py; sleep 10
for _ in $(seq 1 40); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 2000 ] && break
  sleep 15
done

cd ~/openpi || exit 1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 setsid uv run scripts/serve_policy.py --port $PORT \
  policy:checkpoint --policy.config=pi05_rt_base \
  --policy.dir=/home/ubuntu/openpi/checkpoints/pi05_rt_base/abl/2999 > ~/serve_base100.log 2>&1 &
for i in $(seq 1 120); do
  python3 -c "import socket; s=socket.create_connection(('127.0.0.1',$PORT),2); s.close()" 2>/dev/null && break
  sleep 10
done
sleep 25

cd ~/RoboTwin || exit 1
. .venv/bin/activate
python -u ~/egorun/run_ablation.py "$TASK" "$SCENES" $PORT rt_base100 2>&1 | tail -50
pkill -f serve_policy.py
echo "BASE100_OK"
