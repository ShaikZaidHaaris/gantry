"""Two contributor cohorts from the same corpus, cut on a real property of the footage.

    two_handed   clips where both hands were tracked and moving
    one_handed   clips where one hand was mostly absent

Not a corruption invented and then discovered. An arm the tracker never found is
held at its last value by the connector, so it sits in the data as a frozen block
of identical numbers -- well formed, right dtype, right width, and motionless.
Two of the eighteen clips have a left arm that moves 0.66 units in three hundred
frames. That is real footage differing in a real way.

Why this pair and not two random halves
---------------------------------------
The evaluation task is bimanual. Footage where both hands were visible should
teach it better than footage where one was not, so the ordering is *predicted by
the task* rather than asserted. That makes this a validation of the benchmark
and not only a measurement: if it cannot rank these two, it cannot rank two real
contributors either, and any A-vs-B number it later produces is noise with a
decimal point on it.

What this pair is not
---------------------
A clean single-variable contrast. The corpus does not record who filmed each
clip -- the source names did not survive the first build -- so the two halves may
draw on the same participants, and `leakage_checkable` says so rather than
letting a vacuous check read as a passed one. What is being compared is
"footage the tracker could follow" against "footage it could not", which is the
axis contributors actually vary on, and it is stated that way rather than dressed
up as an isolate of two-handedness.
"""

from __future__ import annotations

import json
from pathlib import Path

from build_arms import FPS, OUT, camera_to_world, detach, ego_episodes, robotwin_episodes
from gantry_connector_lerobot import write_episodes
from gantry_curate_split import Split, match_frames, moving_fraction

THRESHOLD = 0.5


def main() -> None:
    robotwin = robotwin_episodes()
    ego = ego_episodes(camera_to_world())

    split = Split(
        measure=moving_fraction(width=8),
        threshold=THRESHOLD,
        names=("two_handed", "one_handed"),
        criterion="fraction of frames in which both hands were moving",
        metadata={"source": "epic_kitchens", "task_is_bimanual": True},
    )
    parts = match_frames(split.cut(ego))

    summary = {"criterion": split.criterion, "threshold": THRESHOLD, "parts": {}}
    for name, part in parts.items():
        summary["parts"][name] = part.summary()
        print(json.dumps(part.summary()), flush=True)

    # Each half gets the same RoboTwin base, so the arms differ only in which
    # ego footage was added -- and a shuffled control for each, so "this half
    # carried information" stays answerable separately from "this half beat the
    # other one".
    arms = {}
    for name, part in parts.items():
        arms[f"rt_{name}"] = robotwin + list(part.episodes)
        arms[f"rt_{name}_shuf"] = robotwin + detach(list(part.episodes))

    for name, episodes in arms.items():
        write_episodes(
            episodes, OUT / name, fps=FPS, robot_type="aloha-agilex",
            accept_loss=True, videos=True,
        )
        frames = sum(len(e) for e in episodes)
        summary["parts"].setdefault("arms", {})[name] = {
            "episodes": len(episodes), "frames": frames
        }
        print(f"wrote {name}: {len(episodes)} episodes, {frames} frames", flush=True)

    Path("/home/ubuntu/egorun/ab_split.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
