"""Read an Inspect Robots evaluation log as Gantry records.

`Inspect Robots <https://github.com/robocurve/inspect-robots>`_ (MIT) writes an
immutable, schema-versioned JSON log for every run: the resolved task, policy
and embodiment, the git revision, the seed and horizon, and per-scene scores
across epochs. This reads one and hands back records the feedback plane can
analyse -- which means diagnosing runs from any of their benchmarks and any of
their supported rigs, without a robot, a simulator, or a GPU.

What a log does and does not contain
------------------------------------
It contains outcomes, not trajectories. Each scene yields per-epoch score dicts,
an optional operator judgement, a termination reason, and whatever the policy
put in ``trial_metadata``. Camera frames, when recorded at all, live in a
side-car directory the log only points at.

So every episode this connector produces is **label-only**: real outcomes, real
milestones where a scorer bothered to record them, and no channels. That is not
a limitation being worked around, it is the shape of the source, and declaring
it honestly is what lets the resolver tell a caller that a trajectory-based
analysis cannot run here. The alternative -- inventing a channel so the types fit
 -- produces an analysis of numbers nobody measured.

Stage events
------------
Their scoring is success-or-not, which is the "the eval says it failed but not
where" problem that makes a funnel diagnosis impossible. Their ``Score.metadata``
is free-form, though, and flows into ``trial_metadata``, so a scorer that
records milestones there gets a funnel for free. :func:`stage_metadata` builds
that payload; :data:`STAGE_KEY` is the key it goes under.

Compatibility
-------------
Their ``from_dict`` rejects any schema version but its own. This is deliberately
more forgiving: it reads the versions it knows and refuses others by name, so a
log from a newer release fails with a sentence rather than a traceback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ChannelSpec,
    ComponentRef,
    Descriptor,
    EpisodeLabels,
    EpisodeRecord,
    Measurement,
    Provenance,
    RunRecord,
    StageEvent,
    episode_from_labels,
)

VERSION = "0.1.0.dev0"

#: Log schema versions this reader understands.
SUPPORTED_SCHEMAS = (1,)

#: Where a scorer should put milestones inside ``Score.metadata``.
STAGE_KEY = "gantry_stages"

#: Score keys treated as the run's outcome, in order of preference. A log may
#: carry several scores per scene and only one of them is "did it work".
DEFAULT_SUCCESS_KEYS = ("success", "success_at_end", "solved", "passed")


def stage_metadata(**steps: int) -> dict[str, dict[str, int]]:
    """Build the metadata payload a scorer attaches to record milestones.

    Used on the Inspect Robots side::

        Score(value=True, metadata=stage_metadata(reached=12, grasped=31))

    Milestones are recorded as the step each was first reached. A stage the
    trial never reached is simply absent, which is what makes the funnel's
    conditional rates meaningful: absence is the signal.
    """
    for name, step in steps.items():
        if step < 0:
            raise ValueError(f"stage {name!r} cannot be reached at step {step}")
    return {STAGE_KEY: dict(steps)}


def _stages_from(metadata: Mapping[str, Any] | None) -> tuple[StageEvent, ...]:
    if not isinstance(metadata, Mapping):
        return ()
    raw = metadata.get(STAGE_KEY)
    if not isinstance(raw, Mapping):
        return ()
    events = [
        StageEvent(str(name), int(step))
        for name, step in raw.items()
        if isinstance(step, (int, float)) and not isinstance(step, bool) and step >= 0
    ]
    return tuple(sorted(events, key=lambda event: event.step))


def _outcome(scores: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    """Read success out of a score dict, or ``None`` if it does not say."""
    for key in keys:
        if key in scores:
            value = scores[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value >= 0.5)
    return None


class EvalLogConnector(Connector):
    """One Inspect Robots eval log, seen as a set of trial records.

    One episode per *trial*, not per scene: a scene run over three epochs is
    three attempts with three outcomes, and collapsing them would throw away
    exactly the variation an evaluation exists to measure.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source: str | None = None,
        success_keys: Sequence[str] = DEFAULT_SUCCESS_KEYS,
    ):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._success_keys = tuple(success_keys)
        self._log = self._load()
        self._spec = self._log.get("eval", {})
        self._source = source or f"{self._spec.get('task', 'eval')}"
        self._episodes = self._build()

    # -- reading -----------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._path.read_text())
        except json.JSONDecodeError as error:
            raise ConfigError(f"{self._path}: not valid JSON ({error})") from error
        if not isinstance(payload, dict):
            raise ConfigError(f"{self._path}: expected a JSON object at the top level")
        version = payload.get("version")
        if version not in SUPPORTED_SCHEMAS:
            raise ConfigError(
                f"{self._path}: eval-log schema version {version!r} is not supported; "
                f"this reader understands {list(SUPPORTED_SCHEMAS)}"
            )
        for field in ("eval", "results", "stats"):
            if field not in payload:
                raise ConfigError(f"{self._path}: log is missing its {field!r} section")
        return payload

    def _build(self) -> dict[str, EpisodeRecord]:
        episodes: dict[str, EpisodeRecord] = {}
        for scene in self._log.get("samples", ()):
            scene_id = str(scene.get("scene_id", "?"))
            epochs = scene.get("epochs") or ()
            parallel = {
                name: scene.get(name) or ()
                for name in (
                    "operator_judgements",
                    "operator_notes",
                    "trial_metadata",
                    "termination_reasons",
                )
            }
            for index, scores in enumerate(epochs):
                scores = scores if isinstance(scores, Mapping) else {}
                metadata = _at(parallel["trial_metadata"], index) or {}
                stages = _stages_from(metadata)
                labels = EpisodeLabels(
                    success=_outcome(scores, self._success_keys),
                    stage_events=stages,
                    annotations={
                        "scores": dict(scores),
                        "scene_status": scene.get("status"),
                        "instruction": scene.get("instruction"),
                        "operator_judgement": _at(parallel["operator_judgements"], index),
                        "operator_note": _at(parallel["operator_notes"], index),
                        "termination_reason": _at(parallel["termination_reasons"], index),
                        "epoch": index,
                    },
                )
                # Steps are not recorded per trial, so the horizon is whatever
                # the milestones prove happened. Claiming the run's configured
                # max would be inventing a number.
                steps = max((event.step for event in stages), default=-1) + 1
                episode_id = f"{scene_id}#{index}" if len(epochs) > 1 else scene_id
                episodes[episode_id] = episode_from_labels(
                    id=episode_id,
                    source=self._source,
                    labels=labels,
                    steps=steps,
                    embodiment=self._spec.get("embodiment"),
                    task=self._spec.get("task"),
                    extra={"scene": scene_id, "log": str(self._path)},
                )
        return episodes

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        labels = [episode.labels for episode in self._episodes.values()]
        return connector_descriptor(
            name="evallog",
            version=VERSION,
            lazy=True,
            stage_events=any(label.stage_events for label in labels),
            outcomes=any(label.success is not None for label in labels),
            media=False,
            path=str(self._path),
            task=self._spec.get("task"),
            policy=self._spec.get("policy"),
            embodiment=self._spec.get("embodiment"),
            log_status=self._log.get("status"),
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._episodes)

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        if episode_id not in self._episodes:
            raise KeyError(f"{self._source}: no trial {episode_id!r}")
        return ()

    def open(self, episode_id: str) -> EpisodeRecord:
        try:
            return self._episodes[episode_id]
        except KeyError:
            raise KeyError(f"{self._source}: no trial {episode_id!r}") from None

    # -- the run behind the episodes ---------------------------------------

    def provenance(self) -> Provenance:
        """Everything the log knows about what produced these numbers.

        Their ``EvalSpec`` and Gantry's provenance were designed for the same
        job, so this is close to a rename: task, policy and embodiment become
        component refs, and the horizon and seed become the protocol -- which is
        where they belong, since changing either changes the result.
        """
        spec, stats = self._spec, self._log.get("stats", {})
        components = []
        for plane, key in (("policy", "policy"), ("evaluation", "embodiment")):
            name = spec.get(key)
            if name:
                components.append(
                    ComponentRef(
                        plane=plane,
                        name=str(name),
                        version=str(spec.get("inspect_robots_version", "unknown")),
                        detail={"role": key},
                    )
                )
        return Provenance(
            components=tuple(components),
            protocol={
                "task": spec.get("task"),
                "seed": spec.get("seed"),
                "max_steps": spec.get("max_steps"),
                "max_seconds": spec.get("max_seconds"),
                "policy_config": spec.get("policy_config", {}),
            },
            created_at=spec.get("created"),
            host=None,
            notes=tuple(
                note
                for note in (
                    f"inspect-robots {spec.get('inspect_robots_version')}",
                    f"git {spec.get('git_commit')}" if spec.get("git_commit") else None,
                    f"log status {self._log.get('status')!r}",
                    f"{stats.get('total_steps')} steps in {stats.get('duration_s')}s"
                    if stats.get("total_steps") is not None
                    else None,
                )
                if note
            ),
        )

    def run(self) -> RunRecord:
        """The whole log as one :class:`~gantry.spine.run.RunRecord`."""
        results = self._log.get("results", {})
        metrics = {
            str(name): Measurement(float(value), n=results.get("total_trials"), method="reported")
            for name, value in (results.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return RunRecord(
            provenance=self.provenance(),
            episodes=tuple(self._episodes.values()),
            metrics=metrics,
        )


def _at(sequence: Sequence[Any], index: int) -> Any:
    return sequence[index] if index < len(sequence) else None


def read_run(path: str | os.PathLike[str], **kwargs: Any) -> RunRecord:
    """Read one eval log straight into a run record."""
    return EvalLogConnector(path, **kwargs).run()
