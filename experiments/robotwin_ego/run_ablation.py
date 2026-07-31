"""The ego checkpoint, and its control, closed-loop in RoboTwin.

This is the run the report has been refusing to write. `feedback_control` would
not say whether ego data helped because no evaluator could execute what the
policy emits: everything installed was single-arm and joint-space, and
`retargeter_hands` refuses to produce joint positions. RoboTwin takes absolute
end-effector poses, so the chain runs with no IK step.

Two arms of the comparison, and the second is the whole point:

  ego        fine-tuned on the real hand trajectories
  shuffled   the same frames and the same actions, with the actions detached
             from the frames they belong to

A model fine-tuned on `shuffled` has had exactly as much fine-tuning, on exactly
as many frames, with the same action distribution, and no relationship between
what it saw and what it did. Beating it is evidence the ego actions carried
information. Not beating it means the gain was from fine-tuning at all.

The frames, which the first run got wrong
-----------------------------------------
The ego pipeline solves hand poses against the camera's own intrinsics, so its
poses are in the **camera** frame, and ``Mount.aligned()`` passes them through
unchanged. RoboTwin executes end-effector poses in its **world** frame. Nothing
in the widths, encodings or labels disagreed, and the first run duly commanded
positions 0.30 m (left) and 0.65 m (right) from where the arms actually work.

``PoseInFrame`` closes that, using the transform RoboTwin publishes about its
own head camera rather than one fitted to the commands — fitting an alignment
here and then scoring under it would tune the correction on the very episodes
being scored. Through the published extrinsics the training distribution lands
0.109 m (left) and 0.171 m (right) from where the arms work.

The nesting is deliberate: the frame shift is outermost, so it works in 16-wide
quaternion space on both sides and the rotation adapter underneath never sees a
frame it did not expect.

The reachability probe is still reported next to the success rate, because a
success rate whose commands were mostly unreachable is a statement about
workspaces rather than about data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/ubuntu/RoboTwin")

from gantry_adapters_core import adapt_policy  # noqa: E402
from gantry_evaluator_robotwin import RoboTwinEvaluator, width_of  # noqa: E402
from gantry_policy_pi0 import Layout, Pi0Policy  # noqa: E402

from gantry.contracts.evaluator import Protocol  # noqa: E402
from gantry.spine import ChannelSpec  # noqa: E402
from gantry.store import write_run  # noqa: E402

PROMPT = "pick up both bottles"
TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"
SCENES = int(sys.argv[2]) if len(sys.argv) > 2 else 10
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
ARM = sys.argv[4] if len(sys.argv) > 4 else "ego"
OUT = Path(f"/home/ubuntu/egorun/abl_{ARM}_{TASK}.json")
#: The full record, so the two arms can be compared by the feedback layer rather
#: than by eye off two summary numbers.
RUN = Path(f"/home/ubuntu/egorun/abl_run_{ARM}_{TASK}")

#: What the trained server reads. The config it was fine-tuned under nests the
#: cameras under "images" and takes a 14-wide state; the camera it saw was a
#: single head-mounted view, which is what RoboTwin's head camera is too.
#: The dimension names of what the checkpoint emits: a pose per arm with Euler
#: angles and a gripper. The only written record of which half is which arm.
RT_LABELS = tuple(
    f"{arm}_{part}"
    for arm in ("left", "right")
    for part in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
)

#: The encoding the checkpoint speaks. Declared so the conversion to RoboTwin's
#: quaternions is planned rather than assumed -- without it the two 14-wide and
#: 16-wide channels are just numbers of different lengths.
LAYOUT_METADATA = {
    "rotation_repr": "quat_wxyz",
    "rotation_offset": (3, 11),
    "semantics": "action.eef_abs_pose",
}

LAYOUT = Layout(
    name="robotwin_ee",
    images={"observation.head_camera.rgb": "cam_high"},
    state=16,
    action=16,
    state_key="state",
    state_from="endpose.vector",
    images_key="images",
    channels_first=True,
    labels=RT_LABELS,
    arms=2,
    metadata=LAYOUT_METADATA,
    discriminators=("rotation_repr", "rotation_offset"),
    state_semantics="observation.eef_abs_pose",
)

#: The state the checkpoint was trained on: a pose per arm with Euler angles,
#: which is what the ego retargeter emitted. Declared here rather than inferred
#: so the conversion from RoboTwin's quaternions is planned, not assumed.


def reachability(commanded: np.ndarray, reach: float = 0.75) -> dict:
    """How much of what the policy asked for is even inside the arms' reach.

    Measured per arm from that arm's own base, because a two-armed command is
    two independent reaching problems and averaging them hides the case where
    one arm is fine and the other is a metre away.
    """
    out = {}
    per_arm = width_of("ee") // 2
    for index, arm in enumerate(("left", "right")):
        block = commanded[:, index * per_arm : index * per_arm + 3]
        distance = np.linalg.norm(block, axis=1)
        out[arm] = {
            "median_distance_m": round(float(np.median(distance)), 3),
            "max_distance_m": round(float(distance.max()), 3),
            "fraction_within_reach": round(float((distance <= reach).mean()), 4),
        }
    return out


def main() -> None:
    started = time.time()
    # No horizon given: RoboTwin's own per-task step limit governs. An earlier
    # version passed a huge number here on the assumption the simulator would
    # stop the episode itself, which it only does when eval_mode is set -- so an
    # episode that was not going to succeed ran until something else stopped it.
    evaluator = RoboTwinEvaluator(TASK, action_type="ee")

    # The screen depends on the task and the seed range, not on the policy, and
    # costs one scripted rollout per seed tried. Both arms must be scored on the
    # *same* arrangements for the comparison to be paired at all, so it is
    # computed once and reused rather than recomputed and hoped to agree.
    cache = Path(f"/home/ubuntu/egorun/robotwin_seeds_{TASK}_{SCENES}.json")
    if cache.exists() and json.loads(cache.read_text())["seeds"]:
        payload = json.loads(cache.read_text())
        seeds = tuple(payload["seeds"])
        evaluator._screened = (payload["start"], payload["stop"], seeds)
        # The sentences come back with the seeds. They are produced by the
        # expert's rollout, which is exactly what reusing the cache skips, and
        # without them a language-conditioned policy refuses. Carrying them here
        # also makes the pairing exact rather than merely likely: both arms are
        # asked to do the same task in the same words, not two samples from the
        # same pool.
        evaluator._instructions = {int(k): v for k, v in payload["instructions"].items()}
        print(f"[{ARM}] reusing screened seeds {list(seeds)}", flush=True)
    else:
        print(f"[{ARM}] screening seeds with RoboTwin's own expert...", flush=True)
        seeds = evaluator.screen(SCENES, limit=SCENES * 6)
        start, stop, _ = evaluator._screened
        # Only a non-empty result is worth keeping. Caching an empty screen
        # makes the second arm reuse a failure instead of discovering it.
        if seeds:
            cache.write_text(
                json.dumps(
                    {
                        "start": start,
                        "stop": stop,
                        "seeds": list(seeds),
                        "instructions": {
                            str(seed): evaluator.instruction_for(seed) for seed in seeds
                        },
                    }
                )
            )
    print(f"[{ARM}] expert solved {len(seeds)}/{SCENES} wanted: {list(seeds)}", flush=True)
    if not seeds:
        raise SystemExit("the scripted expert solved nothing; the install is wrong")

    # The sentence every arm trained on. Held constant so the ablation is on
    # data rather than on wording; RoboTwin's per-scene phrasing would otherwise
    # test language generalisation none of these were trained for.
    served = Pi0Policy(
        layout=LAYOUT, port=PORT, variant="pi05", deterministic=False,
        instruction=PROMPT,
    )
    # Both directions through the adapter plane: the state arrives as
    # quaternions and the checkpoint reads Euler; the actions come back as Euler
    # and RoboTwin reads quaternions.
    # No conversion this time, and that is the point: the training set was built
    # in RoboTwin's own action space and frame, so the checkpoint already speaks
    # what the simulator reads. adapt_policy returns the policy untouched when
    # nothing needs converting, which is the check that it really does.
    converted = adapt_policy(
        served, evaluator.action(), reading=evaluator.provides(), name=f"pi05_{ARM}"
    )
    policy = converted
    chain = getattr(converted, "chain", None)
    print(f"[{ARM}] action chain: {chain or 'direct -- already in the right space'}", flush=True)

    evaluator._instructions = {int(seed): PROMPT for seed in seeds}
    task = evaluator.task_for(seeds=seeds)
    print(f"[{ARM}] {len(seeds)} scenes x {task.horizon} steps ({TASK})", flush=True)
    record = evaluator.run(policy, task, Protocol())

    commanded = np.concatenate([e.array("action") for e in record.episodes])
    successes = [bool(e.labels.success) for e in record.episodes]
    result = {
        "arm": ARM,
        "task": TASK,
        "seeds": list(seeds),
        "episodes": len(record.episodes),
        "successes": int(sum(successes)),
        "success_rate": round(float(np.mean(successes)), 4),
        "expert_solve_rate": record.episodes[0].labels.annotations.get("expert_solve_rate"),
        "steps_per_episode": [int(len(e)) for e in record.episodes],
        "instruction": record.episodes[0].labels.annotations.get("instruction_given"),
        "action_chain": [f"{s.name}@{s.version}" for s in (getattr(converted, "chain", None) or ())],
        # The number that says whether the success rate means anything.
        "reachability": reachability(commanded),
        "seconds": round(time.time() - started, 1),
    }
    write_run(record, RUN)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    evaluator.close()


if __name__ == "__main__":
    main()
