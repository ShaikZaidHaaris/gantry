# The 3 x 7 x 14 matrix, proven reachable

Every checkpoint, on every body, at every task. **One trial at an eight-step
horizon** — this establishes that all 294 cells are reachable and classify
correctly. It is not a measurement of any policy, and each cell says so in its
own `smoke` field.

```
294 cells: 234 ran, 60 refused, 0 failed, 0 in-trial errors
```

`failed` is the category that would mean a defect here. It is empty.

## What ran

234 = 3 checkpoints x 6 single-arm bodies x 13 tasks. Every one of the five
distinct gripper conversions was exercised — Panda, Rethink, Robotiq85,
Robotiq140 and Jaco three-finger all read into the 8-wide state the
checkpoints were trained on.

## What refused, and why it will keep refusing

**Baxter, 42 cells.** robosuite's own words: `Expected one single-armed robot!
Got Bimanual`. These fourteen tasks are single-arm tasks. Baxter is described
in full here — 16-wide state, 14-wide action, both RethinkGrippers measured at
both stops — so this is a refusal about the task, not a gap in the description.
Each cell also records the second, independent reason: a two-handed body has no
non-arbitrary mapping onto a one-handed policy.

**Wipe, 18 cells.** `Tried to specify gripper other than WipingGripper`. That
environment hard-asserts its own gripper, so it cannot host a declared body at
all. It is the one task here that no embodiment axis can cross.

Neither is a bug to fix. Both are the simulator answering a question correctly,
in its own words, recorded next to the cells that ran.

## Reproducing

```bash
CKPTS=ph,mh,mg ARMS=panda,sawyer,iiwa,kinova3,jaco,ur5e,baxter \
TRIALS=1 HORIZON=8 EXECUTE=16 MUJOCO_GL=egl python sweep.py
```

Drop `HORIZON` and raise `TRIALS` to turn this into a measurement. The sweep is
resumable: a cell with a file on disk is skipped, so a long run survives a
disconnect.
