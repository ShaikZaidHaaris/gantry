"""What was planted, and the audit that proves it.

A fixture whose ground truth is merely asserted is worse than no fixture: it
turns a silent generator bug into a green test suite. So a
:class:`FixtureSuite` can be asked to verify itself — for every planted defect,
the statistic that should have moved is checked against the clean half of the
same suite, and for schema defects the compatibility refusal is checked for the
exact code it should raise.

``verify()`` is called by the fixture's own tests and by conformance suites
before they trust a suite to judge anything else.
"""

from __future__ import annotations

import statistics as _stdlib_statistics
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..spine import EpisodeRecord, Verdict, compatible
from .defects import DECOYS, get_defect


@dataclass(frozen=True)
class GroundTruth:
    """The answer key for a generated suite."""

    #: episode uid -> the defects planted in it
    planted: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: properties present that a correct analysis must *not* report
    decoys: tuple[str, ...] = ()
    seed: int = 0
    notes: tuple[str, ...] = ()

    @property
    def defects(self) -> tuple[str, ...]:
        found: list[str] = []
        for names in self.planted.values():
            for name in names:
                if name not in found:
                    found.append(name)
        return tuple(sorted(found))

    def uids_with(self, defect: str) -> tuple[str, ...]:
        return tuple(uid for uid, names in self.planted.items() if defect in names)

    def uids_clean(self) -> tuple[str, ...]:
        return tuple(uid for uid, names in self.planted.items() if not names)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "defects": list(self.defects),
            "decoys": list(self.decoys),
            "planted": {uid: list(names) for uid, names in self.planted.items()},
            "notes": list(self.notes),
            "decoy_descriptions": {name: DECOYS[name] for name in self.decoys if name in DECOYS},
        }


@dataclass(frozen=True)
class FixtureSuite:
    """Episodes plus the answer key, plus the audit."""

    episodes: tuple[EpisodeRecord, ...]
    truth: GroundTruth

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def by_uid(self, uid: str) -> EpisodeRecord:
        for episode in self.episodes:
            if episode.meta.uid == uid:
                return episode
        raise KeyError(uid)

    def clean(self) -> tuple[EpisodeRecord, ...]:
        uids = set(self.truth.uids_clean())
        return tuple(e for e in self.episodes if e.meta.uid in uids)

    def with_defect(self, defect: str) -> tuple[EpisodeRecord, ...]:
        uids = set(self.truth.uids_with(defect))
        return tuple(e for e in self.episodes if e.meta.uid in uids)

    # -- audit -------------------------------------------------------------

    def validate(self, deep: bool = True) -> Verdict:
        """Every episode is a coherent record."""
        return Verdict.all(episode.validate(deep=deep) for episode in self.episodes)

    def verify(self) -> Verdict:
        """Does this suite actually contain what its ground truth claims?"""
        checks = [self.validate(deep=True)]
        clean = self.clean()
        for name in self.truth.defects:
            defect = get_defect(name)
            affected = self.with_defect(name)
            if not affected:
                checks.append(
                    Verdict.no(
                        "fixture.absent",
                        f"ground truth lists {name!r} but no episode carries it",
                    )
                )
                continue
            if defect.is_schema:
                checks.append(self._verify_schema(defect, affected, clean))
            else:
                checks.append(self._verify_statistic(defect, affected, clean))
        return Verdict.all(checks)

    def _verify_statistic(self, defect, affected, clean) -> Verdict:
        if not clean:
            return Verdict.note(
                "fixture.no_control",
                f"{defect.name!r} cannot be verified: the suite has no clean episodes",
            )
        assert defect.statistic is not None
        hurt = _stdlib_statistics.median(defect.statistic(e) for e in affected)
        well = _stdlib_statistics.median(defect.statistic(e) for e in clean)
        moved = (
            (hurt < well - defect.margin)
            if defect.direction == "lower"
            else (hurt > well + defect.margin)
        )
        if not moved:
            return Verdict.no(
                "fixture.not_planted",
                f"{defect.name!r} should read {defect.direction} than clean "
                f"by at least {defect.margin}, but got {hurt:.4f} vs {well:.4f}",
                hint="the generator changed without the catalogue keeping up",
                defect=defect.name,
                defective=hurt,
                clean=well,
            )
        return Verdict.yes()

    def _verify_schema(self, defect, affected, clean) -> Verdict:
        if not clean:
            return Verdict.note(
                "fixture.no_control",
                f"{defect.name!r} cannot be verified: the suite has no clean episodes",
            )
        channel = defect.schema.get("channel", "position")
        verdict = compatible(affected[0].channel(channel), clean[0].channel(channel))
        if defect.expect_code not in verdict.codes():
            return Verdict.no(
                "fixture.not_planted",
                f"{defect.name!r} should make channel {channel!r} incompatible with "
                f"{defect.expect_code!r}, but got {verdict.codes() or ('compatible',)}",
                defect=defect.name,
            )
        return Verdict.yes()

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "clean": len(self.clean()),
            "steps_total": sum(len(e) for e in self.episodes),
            **self.truth.as_dict(),
        }
