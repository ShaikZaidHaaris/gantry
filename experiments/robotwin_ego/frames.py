"""Where the policy pointed, against where the arms actually are.

A success rate of zero has at least two readings and they call for opposite
responses: the policy learned nothing, or it learned something and was asked to
perform it in a frame half a metre away from the one it trained in. The second
is a plumbing fact and is fixable; the first is about the data.

Both are recorded, so both are answerable. Every episode carries the commands
the policy issued (``action``) and where RoboTwin's arms actually were at each
step (``endpose.vector``), in the same layout and the same units. The gap
between those two clouds is the thing to look at.

This reports it per arm, because a two-armed command is two independent reaching
problems and averaging them hides the case where one arm is fine and the other
is nowhere near.

What it deliberately does not do
--------------------------------
It does not fit the transform. Estimating an alignment from the commands and
then scoring the policy under it would tune the correction on the very episodes
being scored, and the number that came out would be about the fit rather than
about the data. Measuring the offset and reporting it is honest; applying it
silently is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from gantry.store import read_run

TASK = sys.argv[1] if len(sys.argv) > 1 else "pick_dual_bottles"
HERE = Path("/home/ubuntu/egorun")
PER_ARM = 8  # xyz + wxyz quaternion + gripper


def blocks(values: np.ndarray) -> dict[str, np.ndarray]:
    """The position of each arm, from a 16-wide pose vector."""
    return {
        arm: values[:, index * PER_ARM : index * PER_ARM + 3]
        for index, arm in enumerate(("left", "right"))
    }


def summarise(arm: str, commanded: np.ndarray, actual: np.ndarray) -> dict:
    centre = actual.mean(axis=0)
    offset = commanded.mean(axis=0) - centre

    # Reach measured from where this arm actually operates, not from the world
    # origin. An arm working at z=0.94 is not "a metre from the origin" in any
    # sense that matters; what matters is how far the command is from it.
    distance = np.linalg.norm(commanded - centre, axis=1)
    spread = np.linalg.norm(actual - centre, axis=1)

    return {
        "arm": arm,
        "robotwin_workspace_centre_m": [round(float(v), 3) for v in centre],
        "robotwin_workspace_radius_m": round(float(np.percentile(spread, 95)), 3),
        "commanded_centre_m": [round(float(v), 3) for v in commanded.mean(axis=0)],
        # The headline: how far the two clouds sit apart, and in which direction.
        "offset_m": [round(float(v), 3) for v in offset],
        "offset_magnitude_m": round(float(np.linalg.norm(offset)), 3),
        "commanded_distance_from_centre_m": {
            "median": round(float(np.median(distance)), 3),
            "p95": round(float(np.percentile(distance, 95)), 3),
        },
        "within_0.5m_of_centre_pct": round(float((distance <= 0.5).mean() * 100), 1),
        "within_0.75m_of_centre_pct": round(float((distance <= 0.75).mean() * 100), 1),
    }


def main() -> None:
    out = {"task": TASK, "arms": {}}
    for name in ("ego", "shuffled"):
        path = HERE / f"robotwin_run_{name}_{TASK}.json"
        if not path.exists():
            continue
        record = read_run(path)
        commanded = np.concatenate([e.array("action") for e in record.episodes])
        actual = np.concatenate([e.array("endpose.vector") for e in record.episodes])
        out["arms"][name] = {
            "steps": int(len(commanded)),
            "per_arm": [
                summarise(arm, blocks(commanded)[arm], blocks(actual)[arm])
                for arm in ("left", "right")
            ],
        }

    print(json.dumps(out, indent=2))
    (HERE / f"robotwin_frames_{TASK}.json").write_text(json.dumps(out, indent=2))

    for name, body in out["arms"].items():
        print(f"\n{name}:")
        for entry in body["per_arm"]:
            print(
                f"  {entry['arm']:5s} commanded sits {entry['offset_magnitude_m']} m from "
                f"where the arm works (offset {entry['offset_m']}); "
                f"{entry['within_0.75m_of_centre_pct']}% of commands within 0.75 m of it"
            )


if __name__ == "__main__":
    main()
