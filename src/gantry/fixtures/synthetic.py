"""Generated episodes with known flaws, for testing everything above the spine.

The trajectories are abstract on purpose. An agent moves through space, engages
with something, transports it, and lets go -- a shape shared by a gripper, a
winch, a suction tool, or a surgical instrument. Stage names default to
``approach / engage / transport / release`` and are overridable, because core
must not decide that any particular robot's vocabulary is the canonical one.

Everything is seeded and reproducible: the same seed yields byte-identical
arrays, so a fixture failure is always a real regression and never noise.

Costs nothing to run, needs no GPU, no simulator, no checkpoint. That is the
whole point -- the feedback stack stays testable in CI on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..spine import ChannelSpec, EpisodeLabels, EpisodeRecord, StageEvent, episode_from_arrays
from .defects import DECOYS, Defect, get_defect
from .truth import FixtureSuite, GroundTruth

#: Default milestone names. Abstract by design; pass your own.
STAGES = ("approach", "engage", "transport", "release")

#: Steps per segment, one per stage.
SEGMENT_STEPS = (12, 8, 14, 8)

RATE_HZ = 20.0


@dataclass
class Trajectory:
    """A mutable draft, before it becomes an immutable record."""

    position: np.ndarray  # (T, 3)
    engagement: np.ndarray  # (T,)
    action: np.ndarray  # (T, 4)
    stage_steps: dict[str, int]
    success: bool = True
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.position)

    def truncate(self, steps: int) -> None:
        steps = max(1, min(steps, len(self)))
        self.position = self.position[:steps]
        self.engagement = self.engagement[:steps]
        self.action = self.action[:steps]
        self.extra = {name: array[:steps] for name, array in self.extra.items()}
        self.stage_steps = {name: step for name, step in self.stage_steps.items() if step < steps}

    def refresh_action(self) -> None:
        """Recompute the action from the motion it would produce."""
        delta = np.zeros_like(self.position)
        delta[1:] = np.diff(self.position, axis=0)
        self.action = np.concatenate([delta, self.engagement[:, None]], axis=1).astype("float32")


def _segment(start: np.ndarray, stop: np.ndarray, steps: int, last: bool) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, steps, endpoint=last)[:, None]
    return start + (stop - start) * fractions


def _draft(
    rng: np.random.Generator,
    stages: Sequence[str],
    segment_steps: Sequence[int],
    noise: float,
    include_view: bool,
    reach: float = 1.0,
) -> Trajectory:
    """Draft one episode. ``reach`` scales how much ground it covers.

    Steps and reach are stretched together by the caller, so per-step speed
    stays constant. A longer episode is one that travels further, not one that
    dawdles -- otherwise "duration" would silently mean "worse signal-to-noise"
    and the decoy would be a defect after all.
    """
    start = rng.uniform([-0.30, -0.30, 0.20], [0.30, 0.30, 0.40]) * reach
    origin = rng.uniform([-0.20, -0.20, 0.00], [0.20, 0.20, 0.05]) * reach
    destination = rng.uniform([-0.20, -0.20, 0.00], [0.20, 0.20, 0.05]) * reach
    hover = np.array([0.0, 0.0, 0.15]) * reach

    waypoints = [start, origin + hover, origin, destination + hover, destination]
    pieces, stage_steps, cursor = [], {}, 0
    for index, steps in enumerate(segment_steps):
        last = index == len(segment_steps) - 1
        pieces.append(_segment(waypoints[index], waypoints[index + 1], steps, last))
        cursor += steps
        stage_steps[stages[index]] = cursor - 1

    position = np.concatenate(pieces).astype("float32")
    position += rng.normal(0.0, noise, position.shape).astype("float32")

    engagement = np.zeros(len(position), dtype="float32")
    engagement[stage_steps[stages[1]] : stage_steps[stages[3]]] = 1.0

    trajectory = Trajectory(
        position, engagement, np.zeros((len(position), 4), "float32"), stage_steps
    )
    trajectory.refresh_action()

    if include_view:
        trajectory.extra["view"] = rng.integers(
            0, 256, size=(len(position), 8, 8, 3), dtype=np.uint8
        )
    return trajectory


def _schema(
    units: str = "m",
    frame: str = "world",
    rate_hz: float | None = RATE_HZ,
    include_view: bool = False,
) -> tuple[ChannelSpec, ...]:
    specs = [
        ChannelSpec(
            "position",
            "vector",
            (3,),
            "float32",
            units=units,
            frame=frame,
            rate_hz=rate_hz,
            semantics="position",
        ),
        ChannelSpec(
            "engagement",
            "scalar",
            (),
            "float32",
            units="fraction",
            rate_hz=rate_hz,
            semantics="actuation",
        ),
        ChannelSpec("action", "vector", (4,), "float32", rate_hz=rate_hz, semantics="actuation"),
    ]
    if include_view:
        specs.append(
            ChannelSpec(
                "view", "image", (8, 8, 3), "uint8", rate_hz=rate_hz, semantics="observation"
            )
        )
    return tuple(specs)


def _record(
    trajectory: Trajectory,
    index: int,
    source: str,
    schema: tuple[ChannelSpec, ...],
    scale: float,
) -> EpisodeRecord:
    arrays: dict[str, np.ndarray] = {
        "position": (trajectory.position * scale).astype("float32"),
        "engagement": trajectory.engagement,
        "action": trajectory.action,
    }
    arrays.update(trajectory.extra)
    labels = EpisodeLabels(
        success=trajectory.success,
        stage_events=tuple(
            StageEvent(name, step)
            for name, step in sorted(trajectory.stage_steps.items(), key=lambda kv: kv[1])
        ),
    )
    return episode_from_arrays(
        arrays,
        schema,
        id=f"ep_{index:04d}",
        source=source,
        labels=labels,
        embodiment="synthetic-agent",
        task="engage-and-transport",
    )


def make_suite(
    n: int = 24,
    defect: str | Defect | None = None,
    fraction: float = 0.5,
    *,
    seed: int = 0,
    source: str = "fixture",
    stages: Sequence[str] = STAGES,
    segment_steps: Sequence[int] = SEGMENT_STEPS,
    noise: float = 0.004,
    include_view: bool = False,
    duration_spread: bool = False,
) -> FixtureSuite:
    """Generate ``n`` episodes, planting ``defect`` in ``fraction`` of them.

    ``duration_spread`` is a decoy rather than a defect: episode lengths vary
    while per-step behaviour stays identical, so any analysis working from raw
    counts will find a difference that is not there.
    """
    if len(stages) != len(segment_steps):
        raise ValueError("stages and segment_steps must have the same length")

    chosen = get_defect(defect) if isinstance(defect, str) else defect
    seeds = np.random.SeedSequence(seed).spawn(n)
    affected = set(range(int(n * fraction)))

    scale = 1000.0 if chosen and chosen.schema.get("units") == "mm" else 1.0
    defective_schema = _schema(
        units=chosen.schema.get("units", "m") if chosen else "m",
        frame=chosen.schema.get("frame", "world") if chosen else "world",
        include_view=include_view,
    )
    clean_schema = _schema(include_view=include_view)

    episodes, planted = [], {}
    for index, child in enumerate(seeds):
        rng = np.random.default_rng(child)
        steps, stretch = list(segment_steps), 1.0
        if duration_spread:
            stretch = 1.0 + 2.0 * (index / max(n - 1, 1))
            steps = [max(3, int(round(value * stretch))) for value in steps]

        # Noise stays fixed, not scaled: per-step displacement is already
        # constant across stretches, so scaling noise too would make long
        # episodes genuinely noisier and turn the decoy into a real defect.
        trajectory = _draft(rng, stages, steps, noise, include_view, reach=stretch)
        is_defective = chosen is not None and index in affected

        # Mutators own their own consistency: one that edits position or
        # engagement refreshes the action, one that edits the action directly
        # must not, or it would undo itself.
        if is_defective and chosen.mutate is not None:
            chosen.mutate(trajectory, rng)

        record = _record(
            trajectory,
            index,
            source,
            defective_schema if is_defective else clean_schema,
            scale if is_defective else 1.0,
        )
        episodes.append(record)
        planted[record.meta.uid] = (chosen.name,) if is_defective else ()

    decoys = tuple(name for name, flag in [("duration_spread", duration_spread)] if flag)
    truth = GroundTruth(
        planted=planted,
        decoys=decoys,
        seed=seed,
        notes=tuple(DECOYS[name] for name in decoys),
    )
    return FixtureSuite(tuple(episodes), truth)


def make_clean(n: int = 24, *, seed: int = 0, **kwargs: Any) -> FixtureSuite:
    """A suite with nothing wrong with it. Every detector must stay quiet."""
    return make_suite(n=n, defect=None, fraction=0.0, seed=seed, **kwargs)


def make_defective(
    defect: str, n: int = 24, fraction: float = 0.5, *, seed: int = 0, **kwargs: Any
) -> FixtureSuite:
    return make_suite(n=n, defect=defect, fraction=fraction, seed=seed, **kwargs)


def make_duration_confound(n: int = 24, *, seed: int = 0, **kwargs: Any) -> FixtureSuite:
    """Episodes of very different lengths that are otherwise identical in quality.

    The trap this reproduces is real and cost this project a day: a statistic
    computed as a raw count made long episodes look defective, and the finding
    survived review until it was normalised per step and vanished. Anything
    that reports a defect on this suite is measuring duration.
    """
    return make_suite(n=n, defect=None, fraction=0.0, seed=seed, duration_spread=True, **kwargs)
