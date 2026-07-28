"""A closed-loop world made of nothing but arithmetic.

An agent moves through space and must touch a sequence of waypoints in order.
Each waypoint reached is a milestone; reaching the last one is success. That is
the entire simulation, and it is enough — a funnel diagnosis needs conditional
stage rates, not physics.

Why this exists
---------------
Until now Gantry could *read* milestones from someone else's evaluation log but
could not produce one. That left its best diagnosis dependent on a scorer
somebody else had written. A world with no dependencies closes the loop: record
data, screen it, run a policy, diagnose where it broke, all on a laptop, in
seconds, with nothing installed.

It is deliberately abstract. Waypoints in order describe a pick-and-place, a
suture, an inspection route, or a drone survey equally badly and equally
usefully, which is the right amount of commitment for a reference world.

Both halves ship together — the world and a policy that speaks its action
contract — because a rig that arrives without something that can drive it is a
rig nobody can check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from gantry.contracts.evaluator import (
    Evaluator,
    Protocol,
    Scene,
    TaskSpec,
    evaluator_descriptor,
)
from gantry.contracts.policy import (
    EpisodeContext,
    Observation,
    Policy,
    policy_descriptor,
)
from gantry.errors import ComponentError
from gantry.resolve import Requirement, requires_channels
from gantry.spine import (
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeRecord,
    Measurement,
    Provenance,
    RunRecord,
    StageEvent,
    episode_from_arrays,
)

VERSION = "0.1.0.dev0"

#: Default milestone names. Abstract on purpose; pass your own.
STAGES = ("approach", "engage", "transport", "release")

#: How close counts as touching a waypoint.
TOLERANCE = 0.05

POSITION = ChannelSpec(
    "position", "vector", (3,), "float32", units="m", frame="world", semantics="position"
)
TARGET = ChannelSpec(
    "target", "vector", (3,), "float32", units="m", frame="world", semantics="position"
)
COMMAND = ChannelSpec(
    "command", "vector", (3,), "float32", units="m", frame="world", semantics="actuation"
)


@dataclass
class _State:
    position: np.ndarray
    waypoints: np.ndarray
    reached: int = 0
    trace: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)
    events: list[StageEvent] = field(default_factory=list)

    @property
    def target(self) -> np.ndarray:
        index = min(self.reached, len(self.waypoints) - 1)
        return self.waypoints[index]

    @property
    def done(self) -> bool:
        return self.reached >= len(self.waypoints)


class GreedyPolicy(Policy):
    """Moves toward the current waypoint, imperfectly by a stated amount.

    ``skill`` in [0, 1] is *aim accuracy*, not speed. Below 1 the policy aims at
    a point offset from the true waypoint and converges there, stalling just
    short — the realistic failure of getting close and never closing the gap. A
    policy that were merely slow would still succeed given a generous horizon,
    which makes speed useless as a dial.

    The offset is seeded per waypoint, so it is stable within an episode and a
    rerun lands in the same place. Combined with a world whose stages differ in
    how tight they are, dropping skill makes failures concentrate at the tightest
    stage rather than spreading evenly — which is exactly the shape a funnel
    exists to find.
    """

    def __init__(
        self,
        skill: float = 0.9,
        wobble: float = 0.0,
        *,
        seed: int = 0,
        chunk: int = 1,
        step_limit: float = 0.12,
        aim_error: float = 0.25,
    ):
        if not 0.0 <= skill <= 1.0:
            raise ValueError(f"skill must be within [0, 1], got {skill}")
        if wobble < 0:
            raise ValueError(f"wobble must not be negative, got {wobble}")
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        self.skill, self.wobble, self._seed = skill, wobble, seed
        self._chunk, self._step_limit = chunk, step_limit
        self._aim_error = aim_error
        self._context: EpisodeContext | None = None

    def _aim(self, target: np.ndarray) -> np.ndarray:
        """Where this policy thinks the waypoint is.

        Keyed on the waypoint itself, so the same target always draws the same
        offset — a policy that mis-aims differently every step would wander into
        the goal by luck, which measures nothing.
        """
        if self.skill >= 1.0:
            return target
        episode = self._context.episode_id if self._context else ""
        key = (self._seed, episode, tuple(np.round(target, 6)))
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        offset = rng.normal(size=3)
        offset /= np.linalg.norm(offset) + 1e-12
        return target + offset * self._aim_error * (1.0 - self.skill)

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            "waypoint-greedy",
            VERSION,
            chunk=self._chunk,
            deterministic=True,
            skill=self.skill,
            wobble=self.wobble,
        )

    def action_spec(self) -> ChannelSpec:
        return COMMAND

    def observes(self) -> Requirement:
        return requires_channels("waypoint-greedy", "policy", POSITION, TARGET)

    def reset(self, context: EpisodeContext) -> None:
        self._context = context

    def act(self, observation: Observation) -> np.ndarray:
        position = np.asarray(observation["position"], dtype=float)
        target = self._aim(np.asarray(observation["target"], dtype=float))
        episode = self._context.episode_id if self._context else ""
        rng = np.random.default_rng(
            abs(hash((self._seed, episode, observation.step))) % (2**32)
        )

        chunk = []
        for offset in range(self._chunk):
            remaining = target - position
            distance = float(np.linalg.norm(remaining))
            step = remaining * 0.6
            if distance > self._step_limit:
                step = step / distance * self._step_limit
            if self.wobble:
                step = step + rng.normal(0.0, self.wobble, 3)
            chunk.append(step)
            # A chunk is executed open-loop, so the policy must predict from
            # where it believes it will be, not from where it last was.
            position = position + step
        return np.asarray(chunk, dtype="float32")


class WaypointWorld(Evaluator):
    """Touch the waypoints in order. Milestones, outcomes, and a real trajectory."""

    def __init__(
        self,
        stages: Sequence[str] = STAGES,
        *,
        tolerance: float | Sequence[float] = TOLERANCE,
        spread: float = 0.5,
    ):
        """``tolerance`` may be one number or one per stage.

        Per-stage tolerances are what make this world worth diagnosing: a
        uniform world fails uniformly, and a funnel over uniform failure only
        ever says "it failed early". Give one stage a tighter tolerance and an
        imprecise policy fails *there*, which is a finding.
        """
        if len(stages) < 1:
            raise ValueError("a world needs at least one waypoint")
        self.stages = tuple(stages)
        if isinstance(tolerance, (int, float)):
            self.tolerances = (float(tolerance),) * len(self.stages)
        else:
            self.tolerances = tuple(float(value) for value in tolerance)
            if len(self.tolerances) != len(self.stages):
                raise ValueError(
                    f"{len(self.tolerances)} tolerance(s) for {len(self.stages)} stage(s)"
                )
        self.spread = spread

    @property
    def tolerance(self) -> float:
        return min(self.tolerances)

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name="waypoint",
            version=VERSION,
            stage_events=True,
            outcomes=True,
            seedable=True,
            closed_loop=True,
            stages=list(self.stages),
            tolerances=list(self.tolerances),
        )

    def requires(self) -> Requirement:
        return requires_channels(
            "waypoint", "evaluation", COMMAND, description="a policy commanding a position delta"
        )

    def task_for(self, name: str = "waypoints", scenes: int = 20, horizon: int = 120) -> TaskSpec:
        return TaskSpec(
            name=name,
            scenes=tuple(
                Scene(id=f"layout-{index:02d}", instruction="touch each waypoint in order")
                for index in range(scenes)
            ),
            horizon=horizon,
            stages=self.stages,
        )

    # -- the world ---------------------------------------------------------

    def _reset(self, seed: int) -> _State:
        rng = np.random.default_rng(seed)
        start = rng.uniform(-self.spread, self.spread, 3)
        waypoints = rng.uniform(-self.spread, self.spread, (len(self.stages), 3))
        return _State(position=start.astype(float), waypoints=waypoints.astype(float))

    def _step(self, state: _State, command: np.ndarray, step: int) -> None:
        state.trace.append(state.position.copy())
        state.targets.append(state.target.copy())
        state.commands.append(np.asarray(command, dtype=float))
        state.position = state.position + np.asarray(command, dtype=float)
        tolerance = self.tolerances[min(state.reached, len(self.tolerances) - 1)]
        if not state.done and float(np.linalg.norm(state.position - state.target)) <= tolerance:
            state.events.append(StageEvent(self.stages[state.reached], step))
            state.reached += 1

    def run(self, policy: Policy, task: TaskSpec, protocol: Protocol) -> RunRecord:
        episodes: list[EpisodeRecord] = []
        successes = 0
        failures = 0

        trials = [
            (index, scene, epoch)
            for index, scene in enumerate(task.scenes)
            for epoch in range(protocol.epochs)
        ]
        if protocol.max_trials is not None:
            trials = trials[: protocol.max_trials]

        for index, scene, epoch in trials:
            seed = scene.seed if scene.seed is not None else protocol.seed_for(index, epoch)
            state = self._reset(seed)
            policy.reset(EpisodeContext(scene.id, scene.instruction, seed))

            step = 0
            error: str | None = None
            while step < task.horizon and not state.done:
                observation = Observation(
                    step, {"position": state.position.copy(), "target": state.target.copy()}
                )
                try:
                    chunk = np.asarray(policy.act(observation), dtype=float)
                except Exception as raised:  # noqa: BLE001 - one trial, not the run
                    error = f"{type(raised).__name__}: {raised}"
                    break
                if chunk.ndim != 2 or len(chunk) < 1:
                    raise ComponentError(
                        f"{policy.name} returned a chunk of shape {chunk.shape}; "
                        "expected (horizon, 3)"
                    )
                played = len(chunk) if protocol.execute is None else min(protocol.execute, len(chunk))
                for command in chunk[: max(1, played)]:
                    if step >= task.horizon or state.done:
                        break
                    self._step(state, command, step)
                    step += 1

            if error is not None:
                failures += 1
            elif state.done:
                successes += 1
            episodes.append(self._record(scene, epoch, state, protocol, task, error))

        scored = len(episodes) - failures
        return RunRecord(
            provenance=self._provenance(policy, task, protocol, failures),
            episodes=tuple(episodes),
            metrics=self._metrics(successes, scored, episodes),
        )

    def _record(
        self,
        scene: Scene,
        epoch: int,
        state: _State,
        protocol: Protocol,
        task: TaskSpec,
        error: str | None,
    ) -> EpisodeRecord:
        if not state.trace:
            state.trace.append(state.position.copy())
            state.targets.append(state.target.copy())
            state.commands.append(np.zeros(3))
        return episode_from_arrays(
            {
                "position": np.asarray(state.trace, dtype="float32"),
                "target": np.asarray(state.targets, dtype="float32"),
                "command": np.asarray(state.commands, dtype="float32"),
            },
            (POSITION, TARGET, COMMAND),
            id=f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id,
            source=task.name,
            labels=EpisodeLabels(
                success=None if error else state.done,
                stage_events=tuple(state.events),
                annotations={
                    "epoch": epoch,
                    "reached": state.reached,
                    "of": len(self.stages),
                    **({"error": error} if error else {}),
                },
            ),
            task=task.name,
            embodiment="waypoint-agent",
        )

    def _provenance(
        self, policy: Policy, task: TaskSpec, protocol: Protocol, failures: int
    ) -> Provenance:
        return Provenance(
            components=(
                policy.descriptor().component_ref(),
                self.descriptor().component_ref(),
            ),
            protocol={
                "task": task.name,
                "horizon": task.horizon,
                # The milestones this run was *looking for*, not the ones that
                # happened. A policy that fails at the second one in every
                # episode never emits it, and a reader inferring the vocabulary
                # from events would conclude the first rung was the whole task
                # and that everything passed.
                "stages": list(task.stages or self.stages),
                **protocol.as_dict(),
            },
            notes=(f"{failures} trial(s) failed to complete",) if failures else (),
        )

    def _metrics(
        self, successes: int, scored: int, episodes: Sequence[EpisodeRecord]
    ) -> dict[str, Measurement]:
        if scored < 1:
            return {}
        reached = [float(e.labels.annotations.get("reached", 0)) for e in episodes]
        return {
            "success_rate": Measurement(
                successes / scored, n=scored, units="fraction", method="completed all waypoints"
            ),
            "stages_reached": Measurement(
                float(np.mean(reached)), n=len(reached), method="mean milestones per trial"
            ),
        }
