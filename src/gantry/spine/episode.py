"""Spine A: what a demonstration or a rollout *is*, storage-agnostic.

Two decisions carry most of the weight:

**View, don't convert.** An EpisodeRecord holds a :class:`StepSource`, not
arrays. Connectors expose foreign formats lazily, so opening a thousand
episodes of a terabyte-scale dataset reads metadata and nothing else. There is
no import step and no second copy on disk.

**A rollout is an episode.** What a policy produced and what a human recorded
are the same type. That is what lets one feedback module screen a dataset and
diagnose an evaluation without knowing which it was handed.

Stage events are optional and declared. The funnel diagnosis needs them; plenty
of sources cannot supply them; the resolver is what tells you so, before a run
starts rather than after.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .channel import ChannelSpec
from .verdict import Verdict


@runtime_checkable
class StepSource(Protocol):
    """Lazy access to an episode's timesteps.

    Implementations live in connectors and may read from a columnar file, an
    archive, a log stream, a network share, anything at all. Core ships only
    the in-memory one.
    """

    @property
    def num_steps(self) -> int: ...

    @property
    def channels(self) -> tuple[str, ...]: ...

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return ``{channel: array}`` with a leading time axis."""
        ...


class ArraySource:
    """A :class:`StepSource` backed by arrays already in memory.

    For fixtures, tests, and small data. Anything large should be read lazily
    by its own source rather than loaded through this one.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        if not arrays:
            raise ValueError("an episode needs at least one channel")
        lengths = {name: len(array) for name, array in arrays.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"channels disagree on length: {lengths}")
        self._arrays = {name: np.asarray(array) for name, array in arrays.items()}
        self._num_steps = next(iter(lengths.values()))

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self._arrays)

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        wanted = tuple(channels) if channels is not None else self.channels
        missing = [name for name in wanted if name not in self._arrays]
        if missing:
            raise KeyError(f"no such channel(s): {missing}")
        return {name: self._arrays[name][start:stop] for name in wanted}


class EmptySource:
    """A :class:`StepSource` with no channels at all.

    For records that carry outcomes and milestones but no trajectory — which is
    what most evaluation logs actually contain. Such an episode is a legitimate
    record, not a broken one, and representing it honestly is what lets the
    resolver tell a caller that a trajectory-based analysis cannot run on it,
    instead of that analysis returning an empty result someone then reads
    meaning into.
    """

    def __init__(self, num_steps: int = 0):
        if num_steps < 0:
            raise ValueError(f"num_steps must not be negative, got {num_steps}")
        self._num_steps = num_steps

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def channels(self) -> tuple[str, ...]:
        return ()

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        if channels:
            raise KeyError(f"no such channel(s): {list(channels)}; this record has none")
        return {}


@dataclass(frozen=True)
class StageEvent:
    """A named milestone reached at a particular step.

    Stage names are free text on purpose. A pick-and-place funnel and a
    surgical-suture funnel have nothing in common but this shape, and core
    should not be adjudicating which stage names are legitimate.
    """

    name: str
    step: int
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeLabels:
    """Everything known *about* an episode rather than measured within it."""

    success: bool | None = None
    stage_events: tuple[StageEvent, ...] = ()
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def reached(self, stage: str) -> bool:
        return any(event.name == stage for event in self.stage_events)

    def step_of(self, stage: str) -> int | None:
        for event in self.stage_events:
            if event.name == stage:
                return event.step
        return None

    @property
    def stages(self) -> tuple[str, ...]:
        return tuple(event.name for event in self.stage_events)


@dataclass(frozen=True)
class EpisodeMeta:
    """Identity and origin. ``source`` namespaces ``id``.

    The namespacing is not decoration: merging two datasets that both number
    their episodes from zero silently collapses them otherwise, and a lost
    fifth of a training set does not announce itself.

    Identity has to survive a conversion
    ------------------------------------
    A format change renumbers everything. The same demonstration is
    ``mg/demo_1`` in one collection and ``lift_train_mg/episode_000000`` in the
    copy a trainer reads, and nothing connects them — so anything computed
    about the first (it failed; it is worth dropping; it caused this behaviour)
    cannot be acted on in the second.

    ``derived_from`` carries the names it had before. A converter appends the
    source's uid, so a claim made about a demonstration in the format where the
    evidence lives can be applied in the format the trainer reads. Without it
    the two are separate datasets that happen to contain the same motions.
    """

    id: str
    source: str
    embodiment: str | None = None
    task: str | None = None
    collected_by: str | None = None
    license: str | None = None
    #: Uids this episode had in earlier formats, oldest first.
    derived_from: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.source}/{self.id}"

    @property
    def lineage(self) -> tuple[str, ...]:
        """Every name this episode has been known by, newest last."""
        return tuple(self.derived_from) + (self.uid,)

    def known_as(self, name: str) -> bool:
        """Whether this episode answers to that name, now or in a past format."""
        return name in self.lineage

    def descended(self, source_uid: str) -> "EpisodeMeta":
        """The same episode, recording that it came from ``source_uid``.

        Used by whatever writes a converted copy. Appends rather than replaces,
        so a chain of conversions stays walkable end to end.
        """
        if source_uid in self.lineage:
            return self
        return replace(self, derived_from=tuple(self.derived_from) + (source_uid,))


@dataclass(frozen=True)
class EpisodeRecord:
    """One episode: what it is, what it contains, and how to read it."""

    meta: EpisodeMeta
    schema: tuple[ChannelSpec, ...]
    source: StepSource
    labels: EpisodeLabels = field(default_factory=EpisodeLabels)

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return self.source.num_steps

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.schema)

    def channel(self, name: str) -> ChannelSpec:
        for spec in self.schema:
            if spec.name == name:
                return spec
        raise KeyError(f"{self.meta.uid}: no channel {name!r}")

    def has(self, name: str) -> bool:
        return any(spec.name == name for spec in self.schema)

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        return self.source.read(channels, start, stop)

    def array(self, name: str) -> np.ndarray:
        return self.read([name])[name]

    def with_labels(self, labels: EpisodeLabels) -> "EpisodeRecord":
        return replace(self, labels=labels)

    def select(self, names: Iterable[str]) -> tuple[ChannelSpec, ...]:
        wanted = set(names)
        return tuple(spec for spec in self.schema if spec.name in wanted)

    # -- validation --------------------------------------------------------

    def validate(self, deep: bool = False) -> Verdict:
        """Check the record against itself.

        ``deep=False`` checks specs and the schema/source channel sets, which
        costs nothing. ``deep=True`` additionally reads every channel to
        confirm shapes and dtypes really match what was declared. Conformance
        suites run deep on a sample; production paths generally do not.
        """
        checks = [spec.validate() for spec in self.schema]

        names = [spec.name for spec in self.schema]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            checks.append(
                Verdict.no(
                    "schema.duplicate",
                    f"{self.meta.uid}: duplicate channel name(s) {sorted(duplicates)}",
                )
            )

        available = set(self.source.channels)
        for spec in self.schema:
            if spec.name not in available and not spec.optional:
                checks.append(
                    Verdict.no(
                        "schema.missing",
                        f"{self.meta.uid}: channel {spec.name!r} is declared but absent "
                        "from the source",
                        channel=spec.name,
                    )
                )

        undeclared = available - set(names)
        if undeclared:
            checks.append(
                Verdict.note(
                    "schema.undeclared",
                    f"{self.meta.uid}: source carries undeclared channel(s) {sorted(undeclared)}",
                    hint="undeclared channels are invisible to the resolver",
                )
            )

        checks.append(self._check_stage_events())

        if deep:
            checks.append(self._check_data())

        return Verdict.all(checks)

    def _check_stage_events(self) -> Verdict:
        checks = []
        for event in self.labels.stage_events:
            if not 0 <= event.step < max(len(self), 1):
                checks.append(
                    Verdict.no(
                        "stage.range",
                        f"{self.meta.uid}: stage {event.name!r} at step {event.step} "
                        f"is outside 0..{len(self) - 1}",
                        stage=event.name,
                    )
                )
        names = [event.name for event in self.labels.stage_events]
        repeated = {name for name in names if names.count(name) > 1}
        if repeated:
            checks.append(
                Verdict.no(
                    "stage.duplicate",
                    f"{self.meta.uid}: stage(s) {sorted(repeated)} recorded more than once",
                    hint="a stage is a first-arrival milestone; record it once",
                )
            )
        return Verdict.all(checks)

    def _check_data(self) -> Verdict:
        checks = []
        for spec in self.schema:
            if spec.name not in self.source.channels:
                continue
            array = self.read([spec.name])[spec.name]
            checks.append(spec.accepts(array))
        return Verdict.all(checks)


def episode_from_labels(
    *,
    id: str,
    source: str,
    labels: EpisodeLabels,
    steps: int = 0,
    **meta: Any,
) -> EpisodeRecord:
    """An episode that records what happened without recording how.

    The shape an evaluation log gives you: an outcome, perhaps some milestones,
    and no trajectory. Building one is deliberately as easy as building a full
    record, because the alternative is someone inventing a channel to make the
    types fit.
    """
    return EpisodeRecord(
        meta=EpisodeMeta(id=id, source=source, **meta),
        schema=(),
        source=EmptySource(steps),
        labels=labels,
    )


def episode_from_arrays(
    arrays: Mapping[str, np.ndarray],
    schema: Sequence[ChannelSpec],
    *,
    id: str,
    source: str,
    labels: EpisodeLabels | None = None,
    **meta: Any,
) -> EpisodeRecord:
    """Build an in-memory episode in one call.

    Deliberately terse. If constructing a record for a test takes more than a
    few lines nobody writes the tests, and the fixtures this spine is checked
    against are written entirely through this function.
    """
    return EpisodeRecord(
        meta=EpisodeMeta(id=id, source=source, **meta),
        schema=tuple(schema),
        source=ArraySource(arrays),
        labels=labels or EpisodeLabels(),
    )
