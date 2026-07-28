"""Spine B: what an evaluation produced.

A RunRecord is provenance plus episodes plus measurements. Its episodes are
:class:`~gantry.spine.episode.EpisodeRecord` values, identical in type to
recorded demonstrations, which is the whole trick: feedback modules take
records and never learn whether a human or a policy generated them.

Aggregation is deliberately awkward across mismatched provenance. Averaging a
simulated arm and a real humanoid into one number is a category error, so
:meth:`RunSet.grouped_by` exists and a bare ``combine`` does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .episode import EpisodeRecord
from .provenance import Measurement, Provenance
from .verdict import Verdict


@dataclass(frozen=True)
class RunRecord:
    """One evaluation run under one fixed provenance."""

    provenance: Provenance
    episodes: tuple[EpisodeRecord, ...] = ()
    metrics: Mapping[str, Measurement] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def digest(self) -> str:
        return self.provenance.digest

    # -- convenience over the episodes ------------------------------------

    @property
    def stages(self) -> tuple[str, ...]:
        """Every stage name any episode reported, in first-seen order.

        Stage vocabulary comes from the data, never from core. A funnel over
        reach/grasp/lift and a funnel over incise/suture/close are the same
        computation on different words.
        """
        seen: list[str] = []
        for episode in self.episodes:
            for stage in episode.labels.stages:
                if stage not in seen:
                    seen.append(stage)
        return tuple(seen)

    @property
    def has_stage_events(self) -> bool:
        return any(episode.labels.stage_events for episode in self.episodes)

    @property
    def has_outcomes(self) -> bool:
        return any(episode.labels.success is not None for episode in self.episodes)

    def outcomes(self) -> tuple[bool | None, ...]:
        return tuple(episode.labels.success for episode in self.episodes)

    def with_metrics(self, metrics: Mapping[str, Measurement]) -> "RunRecord":
        merged = {**self.metrics, **metrics}
        return replace(self, metrics=merged)

    def validate(self, deep: bool = False) -> Verdict:
        checks = [episode.validate(deep=deep) for episode in self.episodes]
        if not self.episodes:
            checks.append(
                Verdict.note("run.empty", f"run {self.digest} contains no episodes")
            )
        uids = [episode.meta.uid for episode in self.episodes]
        duplicates = {uid for uid in uids if uids.count(uid) > 1}
        if duplicates:
            checks.append(
                Verdict.no(
                    "run.duplicate_episode",
                    f"run {self.digest}: repeated episode uid(s) {sorted(duplicates)}",
                    hint="episode ids must be namespaced by source",
                )
            )
        return Verdict.all(checks)


@dataclass(frozen=True)
class RunSet:
    """Several runs held together, with aggregation kept honest.

    There is no ``mean()`` here. Callers state which planes they are holding
    fixed and get back groups that are legitimately comparable; anything else
    is refused rather than silently averaged.
    """

    runs: tuple[RunRecord, ...] = ()

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.runs)

    def add(self, run: RunRecord) -> "RunSet":
        return RunSet(self.runs + (run,))

    def grouped_by(self, *planes: str) -> dict[tuple[str, ...], tuple[RunRecord, ...]]:
        """Group runs by their component refs on the given planes."""
        groups: dict[tuple[str, ...], list[RunRecord]] = {}
        for run in self.runs:
            key = tuple(
                (component.ref if (component := run.provenance.component(plane)) else "∅")
                for plane in planes
            )
            groups.setdefault(key, []).append(run)
        return {key: tuple(value) for key, value in groups.items()}

    def comparable(self, holding: Sequence[str]) -> Verdict:
        """Are all runs in this set comparable while holding these planes?"""
        if len(self.runs) < 2:
            return Verdict.yes()
        first = self.runs[0].provenance
        for run in self.runs[1:]:
            if not first.comparable_to(run.provenance, holding):
                differing = [
                    plane
                    for plane in holding
                    if not first.comparable_to(run.provenance, [plane])
                ]
                return Verdict.no(
                    "runset.incomparable",
                    f"runs differ on {differing}, which this comparison holds fixed",
                    hint="group with RunSet.grouped_by before aggregating",
                    planes=differing,
                )
        return Verdict.yes()

    def select(self, predicate: Callable[[RunRecord], bool]) -> "RunSet":
        return RunSet(tuple(run for run in self.runs if predicate(run)))

    def episodes(self) -> Iterator[EpisodeRecord]:
        for run in self.runs:
            yield from run.episodes


def runset(runs: Iterable[RunRecord]) -> RunSet:
    return RunSet(tuple(runs))


def describe(run: RunRecord) -> dict[str, Any]:
    """A JSON-able summary. Used by the CLI and by report writers."""
    return {
        "digest": run.digest,
        "episodes": len(run),
        "stages": list(run.stages),
        "has_stage_events": run.has_stage_events,
        "provenance": run.provenance.as_dict(),
        "metrics": {name: value.as_dict() for name, value in run.metrics.items()},
    }
