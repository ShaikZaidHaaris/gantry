"""What to go and film next, taken from what the policy just failed.

Every other curator here decides what to remove. This one decides what does not
exist yet, which is the harder and more valuable half: a dataset can only be
improved so far by deletion.

Why the failed scenes specifically
----------------------------------
Demonstrations collected in the situations that beat a policy are worth more
than the same number collected at random. That result is published and it is
intuitive — a policy that already succeeds from the middle of the table learns
nothing from a hundred more demonstrations of the middle of the table.

What has stopped people acting on it is that reproducing a failure situation is
usually impossible: the rollout happened, the arm moved, and nobody wrote down
the layout. Here it is written down. A scene is a seed, a seed regenerates the
scene exactly, and the task file's region is the same rectangle whether it is
being sampled in a simulator or taped onto a real table. So "the situations
that beat it" is a list this can emit and a person can act on.

Why the stage matters
---------------------
"Collect more lifting data" and "collect more grasping data" commission
different work. When a funnel has said where the attempts died, that goes in
the order, because a demonstrator who knows they are filming the grasp films
the grasp.

What it refuses to invent
-------------------------
If nothing failed, it orders nothing — a policy at ceiling on a task does not
need more of that task, and manufacturing an order would be the framework
spending someone's collection budget to look useful. If the runs contain no
seeds, the order is emitted with a note saying it is untargeted, because an
untargeted order is worth having and worth labelling as the weaker thing.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from gantry.contracts.curation import (
    CollectionOrder,
    CurationAction,
    CurationPlan,
    Curator,
    Prediction,
    curator_descriptor,
)
from gantry.spine import ComponentRef, Descriptor, Provenance

VERSION = "0.1.0.dev0"

#: How many demonstrations to order per failed scene, when the caller does not
#: say. One is the honest default: it makes the order proportional to the
#: evidence rather than to an appetite.
PER_FAILURE = 1


def failure_scenes(runs: Sequence[Any]) -> tuple[list[int], list[str], Counter]:
    """The seeds that failed, the tasks they belong to, and where they died.

    Reads episode labels rather than aggregate metrics, because an order is
    per-scene and a success rate has already thrown the scenes away.
    """
    seeds: list[int] = []
    tasks: list[str] = []
    stages: Counter = Counter()
    for run in runs:
        for episode in getattr(run, "episodes", ()):
            labels = getattr(episode, "labels", None)
            if getattr(labels, "success", None) is not False:
                continue
            annotations = getattr(labels, "annotations", {}) or {}
            seed = annotations.get("initial_state")
            if isinstance(seed, int):
                seeds.append(seed)
            task = getattr(getattr(episode, "meta", None), "task", None)
            if task:
                tasks.append(str(task))
            # Where it got to before it stopped. A funnel emits these; when
            # none are present the order simply does not name a stage.
            reached = [event.name for event in getattr(labels, "stage_events", ()) or ()]
            stages[reached[-1] if reached else "unknown"] += 1
    return seeds, tasks, stages


def cluster_note(runs: Sequence[Any], axis: str = "x") -> str:
    """Where in the workspace the failures sit, in a sentence.

    Deliberately a sentence and not a distribution. The consumer is a person
    setting objects on a table, and "failures are on the far side" is
    actionable in a way that a covariance matrix is not.
    """
    values = []
    for run in runs:
        for episode in getattr(run, "episodes", ()):
            labels = getattr(episode, "labels", None)
            if getattr(labels, "success", None) is not False:
                continue
            where = (getattr(labels, "annotations", {}) or {}).get("object_start")
            if isinstance(where, (list, tuple)) and len(where) >= 2:
                values.append(float(where[0] if axis == "x" else where[1]))
    if len(values) < 3:
        return ""
    values.sort()
    middle = values[len(values) // 2]
    span = values[-1] - values[0]
    if span <= 0:
        return ""
    side = "high" if middle > (values[0] + values[-1]) / 2 else "low"
    return (
        f"failed scenes sit toward the {side} end of {axis} "
        f"(median {middle:+.3f}, over {values[0]:+.3f}..{values[-1]:+.3f})"
    )


class FromFailures(Curator):
    """Order demonstrations in the situations the policy could not handle."""

    def __init__(self, *, per_failure: int = PER_FAILURE, cap: int = 200, expect: float = 0.08):
        self._per_failure = per_failure
        #: An upper bound, because an order is somebody's week. A signal that
        #: can ask for unbounded human work should not be able to.
        self._cap = cap
        self._expect = expect

    def descriptor(self) -> Descriptor:
        return curator_descriptor(
            name="collect",
            version=VERSION,
            rung="screening",
            per_episode=False,
            # It decides from rollouts, so it says so and refuses without them.
            needs_runs=True,
            per_failure=self._per_failure,
            cap=self._cap,
        )

    def propose(self, episodes: Sequence[Any], runs: Sequence[Any] = ()) -> CurationPlan:
        seeds, tasks, stages = failure_scenes(runs)
        task = Counter(tasks).most_common(1)[0][0] if tasks else "unknown"
        named = [stage for stage, _ in stages.most_common() if stage != "unknown"]
        stage = named[0] if named else None
        n = min(self._cap, max(1, len(seeds) * self._per_failure))

        if not seeds and not tasks:
            # Nothing failed, or the runs carry no scene identity. Either way
            # there is nothing to target, and ordering blind work would be the
            # framework spending a budget to appear useful.
            return CurationPlan(
                actions=(
                    CurationAction(
                        kind="keep",
                        episodes=tuple(
                            str(getattr(getattr(e, "meta", None), "uid", "")) for e in episodes
                        ),
                        detail={"reason": "no failed scene to collect against"},
                    ),
                ),
                signal="collect",
                rung="screening",
                predicted=Prediction(magnitude=0.0),
                evidence=Provenance(
                    components=(ComponentRef("curation", "collect", VERSION),),
                    protocol={"runs_read": len(runs), "failures": 0},
                ),
                metadata={"note": "nothing failed; no collection ordered"},
            )

        note = cluster_note(runs)
        order = CollectionOrder(
            task=task,
            n=n,
            seeds=tuple(sorted(set(seeds))),
            stage=stage,
            note=note,
        )
        return CurationPlan(
            actions=(
                CurationAction(
                    kind="collect",
                    order=order,
                    detail={
                        "failures": len(seeds),
                        "stages": dict(stages),
                        "capped": n < len(seeds) * self._per_failure,
                    },
                ),
            ),
            signal="collect",
            rung="screening",
            predicted=Prediction(
                metric="success_rate",
                direction="+",
                magnitude=self._expect,
                tasks=(task,) if task != "unknown" else (),
            ),
            evidence=Provenance(
                components=(ComponentRef("curation", "collect", VERSION),),
                protocol={
                    "runs_read": len(runs),
                    "failures": len(seeds),
                    "distinct_scenes": len(set(seeds)),
                },
            ),
            # The scenes this was derived from. Verification refuses to score
            # the result on them: demonstrations collected at these exact
            # layouts would otherwise be tested on the layouts they were
            # filmed at, which measures memorisation and calls it improvement.
            evidence_seeds=tuple(sorted(set(seeds))),
            metadata={
                "note": (
                    f"{len(set(seeds))} distinct scene(s) failed"
                    + (f", most often at the {stage} stage" if stage else "")
                ),
            },
        )
