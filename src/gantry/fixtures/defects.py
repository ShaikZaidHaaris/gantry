"""The catalogue of things that can be wrong with a demonstration.

Each entry declares what it does to a trajectory *and how to tell*: which
statistic should move, in which direction, by how much. That second half is
what lets a generated suite audit itself rather than merely assert it did
something.

Two flavours:

**Behavioural** defects mutate the trajectory (a detour, a chattering
actuator). They are verified by a statistic shifting between the defective and
clean halves of a suite.

**Schema** defects change how a channel is *described* rather than what it
contains -- the same motion recorded in millimetres, or in a different reference
frame. They are verified by :func:`gantry.spine.compatible` refusing with a
specific code, which is the only way to catch a class of bug that is invisible
to every statistic: numbers that look fine and mean something else.

The registry is open. A plugin with its own idea of what "bad data" means
registers a defect and inherits the whole generate-and-verify machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from ..spine import EpisodeRecord
from . import statistics

#: A mutation applied to a trajectory in place.
Mutator = Callable[["object", np.random.Generator], None]


@dataclass(frozen=True)
class Defect:
    """One planted flaw, with its own acceptance test."""

    name: str
    description: str
    mutate: Mutator | None = None
    #: Schema overrides, for defects that change description rather than data.
    schema: Mapping[str, str] = field(default_factory=dict)
    #: How a detector should see it.
    statistic: Callable[[EpisodeRecord], float] | None = None
    direction: str | None = None  # "higher" | "lower"
    margin: float = 0.0
    #: For schema defects: the compatibility code this should provoke.
    expect_code: str | None = None

    @property
    def is_schema(self) -> bool:
        return bool(self.schema)


_DEFECTS: dict[str, Defect] = {}


def register_defect(defect: Defect) -> Defect:
    existing = _DEFECTS.get(defect.name)
    if existing is not None and existing != defect:
        raise ValueError(f"defect {defect.name!r} already registered with a different definition")
    _DEFECTS[defect.name] = defect
    return defect


def get_defect(name: str) -> Defect:
    try:
        return _DEFECTS[name]
    except KeyError:
        raise KeyError(f"unknown defect {name!r}; known: {', '.join(known_defects())}") from None


def known_defects() -> tuple[str, ...]:
    return tuple(sorted(_DEFECTS))


# --------------------------------------------------------------------------
# behavioural mutators
# --------------------------------------------------------------------------


def _detour(trajectory, rng: np.random.Generator) -> None:
    """Push a lateral bulge into the first segment: same endpoints, longer path."""
    end = trajectory.stage_steps["approach"]
    if end < 3:
        return
    phase = np.linspace(0.0, np.pi, end)
    lateral = rng.normal(size=3)
    lateral /= np.linalg.norm(lateral) + 1e-12
    trajectory.position[:end] += np.outer(np.sin(phase), lateral) * 0.55
    trajectory.refresh_action()


def _chatter(trajectory, rng: np.random.Generator) -> None:
    """Make the actuator flip repeatedly around the moment it engages."""
    start = trajectory.stage_steps["engage"]
    stop = trajectory.stage_steps.get("transport", len(trajectory.engagement))
    span = np.arange(start, stop)
    if len(span) < 4:
        return
    trajectory.engagement[span] = ((span - start) // 2 % 2 == 0).astype("float32")
    trajectory.refresh_action()


def _jerk(trajectory, rng: np.random.Generator) -> None:
    """Add high-frequency noise to the commanded action only.

    Deliberately does not refresh: this defect lives in the command, not in
    the motion, and recomputing would erase it.
    """
    noise = rng.normal(0.0, 0.09, trajectory.action[:, :3].shape)
    noise[::2] *= -1.0  # alternate sign so it is jerk, not drift
    trajectory.action[:, :3] += noise


def _late_engagement(trajectory, rng: np.random.Generator) -> None:
    """Delay engaging until the episode is nearly over."""
    total = len(trajectory.engagement)
    late = int(total * 0.88)
    trajectory.engagement[:] = 0.0
    trajectory.engagement[late:] = 1.0
    trajectory.stage_steps["engage"] = late
    trajectory.refresh_action()


def _never_completes(trajectory, rng: np.random.Generator) -> None:
    """Stop before the task is done, and say so."""
    cut = trajectory.stage_steps["engage"]
    trajectory.truncate(max(cut, 3))
    trajectory.success = False


# --------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------

register_defect(
    Defect(
        name="path_detour",
        description="reaches the same place by a longer route",
        mutate=_detour,
        statistic=statistics.path_efficiency,
        direction="lower",
        margin=0.05,
    )
)

register_defect(
    Defect(
        name="actuation_chatter",
        description="the actuator flips on and off instead of committing",
        mutate=_chatter,
        statistic=statistics.engagement_transition_rate,
        direction="higher",
        margin=0.02,
    )
)

register_defect(
    Defect(
        name="actuation_jerk",
        description="commands change sharply from step to step",
        mutate=_jerk,
        statistic=statistics.action_jerk,
        direction="higher",
        margin=0.01,
    )
)

register_defect(
    Defect(
        name="late_engagement",
        description="engages far later in the episode than needed",
        mutate=_late_engagement,
        statistic=statistics.engagement_onset,
        direction="higher",
        margin=0.1,
    )
)

register_defect(
    Defect(
        name="never_completes",
        description="stops partway and is marked unsuccessful",
        mutate=_never_completes,
        statistic=statistics.completion,
        direction="lower",
        margin=0.5,
    )
)

register_defect(
    Defect(
        name="unit_drift",
        description="identical motion recorded in a different unit of the same dimension",
        schema={"units": "mm"},
        expect_code="units.scale",
    )
)

register_defect(
    Defect(
        name="frame_drift",
        description="identical motion recorded against a different reference frame",
        schema={"frame": "sensor"},
        expect_code="frame.mismatch",
    )
)


#: Properties a suite may carry that are **not** defects, and that a correct
#: analysis must therefore not report. Named separately from defects because a
#: detector is judged on both what it finds and what it leaves alone.
DECOYS = {
    "duration_spread": (
        "episodes vary widely in length because they cover proportionally more "
        "ground at the same speed, so per-step behaviour is identical; raw "
        "counts will correlate with duration, per-step rates will not"
    ),
}
