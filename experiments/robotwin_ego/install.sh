#!/bin/bash
# Finish the RoboTwin install, minus the part that was never needed.
#
# pytorch3d is imported in exactly one place — farthest-point sampling of point
# clouds in envs/camera/camera.py — behind a try/except that prints "missing
# pytorch3d" and carries on. This run has pointcloud: false. Ninety minutes of
# CUDA compile for a code path that never executes.
#
# curobo, on the other hand, IS required: ee-mode planning goes through
# CuroboPlanner, not mplib. mplib only does the TOPP pass in qpos mode.
set -x
pkill -9 -f "pip-req-build" 2>/dev/null
pkill -9 -f "facebookresearch/pytorch3d" 2>/dev/null
sleep 2

cd ~/RoboTwin || exit 1
. .venv/bin/activate

# RoboTwin's own two patches, which the install script applies after pytorch3d.
SAPIEN=$(pip show sapien | grep Location | awk '{print $2}')/sapien
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$SAPIEN/wrapper/urdf_loader.py"
echo "sapien encoding patches: $(grep -c 'encoding="utf-8"' "$SAPIEN/wrapper/urdf_loader.py")"

MPLIB=$(pip show mplib | grep Location | awk '{print $2}')/mplib
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$MPLIB/planner.py"
grep -n "delta_twist) < 1e-4" "$MPLIB/planner.py"

cd envs || exit 1
[ -d curobo ] || git clone -q --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git
cd curobo || exit 1
pip install -e . --no-build-isolation || exit 13
pip install -q warp-lang==1.12.0 setuptools==69.5.1

cd ~/RoboTwin || exit 1
python -c "from curobo.wrap.reacher.motion_gen import MotionGen; print('CUROBO OK')" || exit 14
echo "FINISH_OK"
