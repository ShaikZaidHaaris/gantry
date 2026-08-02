"""Evaluate a policy against recorded episodes, with no world at all.

For each held-out episode, the recorded observations are replayed to the policy
and its predicted actions are compared against what was actually done. That is
the whole method. It needs no simulator, no robot, no reset, and runs on a
laptop in seconds.

What it is for
--------------
Ranking checkpoints and catching regressions cheaply. It is the measure behind
"open-loop MSE on held-out episodes", which is what most published policy
comparisons actually report.

What it cannot tell you, stated plainly
---------------------------------------
Every prediction is made from a state a competent demonstrator reached. A policy
whose own small errors drift it somewhere the recording never went is never
asked what it would do there, and that drift is the usual way a policy that
looks fine offline fails on a robot. Low error here is necessary and nowhere
near sufficient.

So this evaluator declares ``closed_loop=False``, and it declares no outcomes
and no stage events, because it has none: there is no world to succeed in.
Anything asking for a funnel gets refused at planning time rather than handed an
empty table. The refusal is the honest part of the design, not a gap in it.

Chunking
--------
A policy predicting a burst of actions is scored on every action in the burst,
each against the recorded action at the step it would have been executed. A
chunk that is right at its first step and wrong by its eighth is a real and
common failure, and scoring only the first would hide exactly the thing the
protocol lever exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from gantry.contracts.evaluator import (
    Evaluator,
    Protocol,
    Scene,
    TaskSpec,
    evaluator_descriptor,
)
from gantry.contracts.policy import EpisodeContext, Observation, Policy
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
    episode_from_arrays,
    episode_from_labels,
)

VERSION = "0.1.0.dev0"


@dataclass(frozen=True)
class Errors:
    """Per-episode prediction error."""

    mse: float
    mae: float
    max_abs: float
    steps: int

    def as_dict(self) -> dict[str, float]:
        return {"mse": self.mse, "mae": self.mae, "max_abs": self.max_abs, "steps": self.steps}


def score_episode(
    policy: Policy,
    episode: EpisodeRecord,
    action_name: str,
    *,
    execute: int | None,
    horizon: int,
    seed: int | None = None,
) -> tuple[Errors, np.ndarray]:
    """Replay one episode past the policy, and return both the error and what
    the policy actually predicted.

    The predictions are returned, not discarded, because a run record should
    contain what the policy *did*. Emitting only a score leaves the feedback
    plane with nothing to read: every trajectory module refuses, and the only
    thing anyone can do with the run is compare one number. The behaviour is
    the interesting part -- a policy can score respectably and still be visibly
    jittery, and that is exactly what a screen would catch.
    """
    truth = np.asarray(episode.array(action_name), dtype=float)
    observed = episode.read()
    steps = min(len(truth), horizon)
    if steps < 1:
        raise ComponentError(f"{episode.meta.uid} has no steps to score")

    policy.reset(
        EpisodeContext(
            episode_id=episode.meta.uid,
            instruction=episode.meta.task,
            seed=seed,
        )
    )

    residuals: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    step = 0
    while step < steps:
        window = Observation(step, {name: values[step] for name, values in observed.items()})
        chunk = np.asarray(policy.act(window), dtype=float)
        if chunk.ndim != truth.ndim:
            raise ComponentError(
                f"{policy.name} returned rank {chunk.ndim}, expected {truth.ndim} "
                "(horizon + action shape)"
            )
        played = len(chunk) if execute is None else min(execute, len(chunk))
        played = max(1, min(played, steps - step))
        residuals.append(chunk[:played] - truth[step : step + played])
        predicted.append(chunk[:played])
        step += played

    stacked = np.concatenate(residuals, axis=0)
    predictions = np.concatenate(predicted, axis=0)
    return (
        Errors(
            mse=float(np.mean(stacked**2)),
            mae=float(np.mean(np.abs(stacked))),
            max_abs=float(np.max(np.abs(stacked))),
            steps=int(len(stacked)),
        ),
        predictions,
    )


class OfflineEvaluator(Evaluator):
    """Open-loop action error over a set of held-out episodes."""

    def __init__(
        self,
        episodes: Sequence[EpisodeRecord] = (),
        action_name: str = "action",
        *,
        action_spec: ChannelSpec | None = None,
    ):
        """``episodes`` may be empty at construction.

        A manifest builds every plane independently and cannot hand this one a
        dataset at that point, so an unbound evaluator is a legitimate object:
        it can be described, resolved and planned against. It refuses to *run*
        until :meth:`bind` gives it the cohort under test, which is what
        ``needs_dataset`` announces.
        """
        self._episodes = {episode.meta.id: episode for episode in episodes}
        self._action_name = action_name
        self._action_spec = action_spec or (episodes[0].channel(action_name) if episodes else None)

    @property
    def bound(self) -> bool:
        return bool(self._episodes)

    def bind(self, episodes: Sequence[EpisodeRecord]) -> "OfflineEvaluator":
        """A new evaluator whose answer key is this cohort."""
        if not episodes:
            raise ValueError("the offline evaluator needs at least one held-out episode")
        return OfflineEvaluator(episodes, self._action_name, action_spec=self._action_spec)

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        return evaluator_descriptor(
            name="offline",
            version=VERSION,
            # No world, so no success and no milestones. Saying so is what lets
            # the resolver refuse a funnel instead of returning an empty one.
            stage_events=False,
            outcomes=False,
            seedable=True,
            closed_loop=False,
            # The answer key is the recording, so this evaluator is only
            # meaningful once bound to the cohort under test.
            needs_dataset=True,
            held_out=len(self._episodes),
            action=self._action_name,
        )

    def requires(self) -> Requirement:
        """What a policy must emit.

        Before binding, the width is not known -- it is a property of the data
        that has not arrived yet -- so the requirement is declared with a
        wildcard shape rather than guessed at. Binding narrows it to the
        recorded channel's actual spec.
        """
        spec = self._action_spec or ChannelSpec(self._action_name, "vector", (None,), "float32")
        return requires_channels(
            "offline",
            "evaluation",
            spec,
            description="a policy emitting the recorded action channel",
        )

    def task_for(self, name: str = "held-out", horizon: int = 10_000) -> TaskSpec:
        """A task covering every held-out episode, one scene each."""
        return TaskSpec(
            name=name,
            scenes=tuple(
                Scene(id=episode_id, instruction=episode.meta.task)
                for episode_id, episode in self._episodes.items()
            ),
            horizon=horizon,
        )

    def run(self, policy: Policy, task: TaskSpec, protocol: Protocol) -> RunRecord:
        episodes: list[EpisodeRecord] = []
        errors: list[Errors] = []
        failures = 0

        trials = [
            (index, scene, epoch)
            for index, scene in enumerate(task.scenes)
            for epoch in range(protocol.epochs)
        ]
        if protocol.max_trials is not None:
            trials = trials[: protocol.max_trials]

        for index, scene, epoch in trials:
            recorded = self._episodes.get(scene.id)
            if recorded is None:
                raise ComponentError(
                    f"task {task.name!r} names scene {scene.id!r}, which is not among "
                    f"the held-out episodes ({len(self._episodes)} available)"
                )
            annotations: dict[str, object] = {"epoch": epoch, "scene": scene.id}
            predictions = None
            try:
                measured, predictions = score_episode(
                    policy,
                    recorded,
                    self._action_name,
                    execute=protocol.execute,
                    horizon=task.horizon,
                    seed=protocol.seed_for(index, epoch),
                )
            except ComponentError as error:
                # One failed trial, recorded and moved past. A world that
                # faulted would halt instead; this is a prediction that did not
                # come out, which is data.
                failures += 1
                annotations["error"] = str(error)
            else:
                errors.append(measured)
                annotations.update(measured.as_dict())

            episode_id = f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id
            labels = EpisodeLabels(success=None, annotations=annotations)
            if predictions is None:
                episodes.append(
                    episode_from_labels(
                        id=episode_id,
                        source=task.name,
                        labels=labels,
                        steps=len(recorded),
                        embodiment=recorded.meta.embodiment,
                        task=recorded.meta.task,
                    )
                )
            else:
                episodes.append(
                    episode_from_arrays(
                        {self._action_name: predictions.astype("float32")},
                        (replace(self._action_spec, name=self._action_name),),
                        id=episode_id,
                        source=task.name,
                        labels=labels,
                        embodiment=recorded.meta.embodiment,
                        task=recorded.meta.task,
                    )
                )

        return RunRecord(
            provenance=self._provenance(policy, task, protocol, failures),
            episodes=tuple(episodes),
            metrics=self._metrics(errors, failures),
        )

    # -- assembly ----------------------------------------------------------

    def _provenance(
        self, policy: Policy, task: TaskSpec, protocol: Protocol, failures: int
    ) -> Provenance:
        return Provenance(
            components=(
                policy.descriptor().component_ref(),
                self.descriptor().component_ref(),
            ),
            protocol={"task": task.name, "horizon": task.horizon, **protocol.as_dict()},
            notes=(
                "open-loop: every prediction is made from a state the "
                "demonstrator reached, so compounding error is not measured",
            )
            + ((f"{failures} trial(s) failed to score",) if failures else ()),
        )

    def _metrics(self, errors: Sequence[Errors], failures: int) -> Mapping[str, Measurement]:
        if not errors:
            return {}
        n = len(errors)

        def summarise(name: str, values: Sequence[float]) -> Measurement:
            array = np.asarray(values, dtype=float)
            spread = float(array.std(ddof=1)) if n > 1 else 0.0
            half = 1.959963984540054 * spread / np.sqrt(n) if n > 1 else 0.0
            return Measurement(
                float(array.mean()),
                n=n,
                ci=(float(array.mean() - half), float(array.mean() + half)),
                method=f"mean {name} over held-out episodes",
                detail={"failed_trials": failures} if failures else {},
            )

        return {
            "action_mse": summarise("mse", [e.mse for e in errors]),
            "action_mae": summarise("mae", [e.mae for e in errors]),
            "action_max_abs": summarise("max abs", [e.max_abs for e in errors]),
        }
