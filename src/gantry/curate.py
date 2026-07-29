"""Applying a plan: turning a proposal into the data a trainer will actually read.

Kept separate from every format on purpose. Selecting which episodes survive is
a property of the plan and the identifiers; writing them out is a property of
whoever's file layout you are using. Mixing the two would mean a plan could only
be applied to the formats core happens to know about, which is the coupling this
project exists to avoid.

So this half is pure: episodes in, surviving episodes out, plus a record of
exactly what happened. The writing is the caller's, through whatever writer
their connector ships.

Whitelist, not suggestion
-------------------------
``keep`` means *only* these. That is the shape the strongest signals produce —
rank every demonstration, keep the top third — and reading it as a hint would
quietly train on everything while reporting that a third was selected.

What it refuses
---------------
A plan that names episodes the dataset does not contain is refused rather than
applied to the ones it does. Identifiers drift during format conversions, and
a drop-list that matches nothing applies perfectly cleanly and does nothing at
all — the most expensive kind of silence, because the retrain still costs an
hour and the verdict still looks like a measurement.

A plan that would empty the dataset is refused too. So is one that changes
nothing, because verifying it would burn two retrains to compare a thing with
itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts.curation import CollectionOrder, CurationPlan
from .spine import Verdict


def uid_of(episode: Any) -> str:
    """What this episode is called now."""
    meta = getattr(episode, "meta", None)
    return str(getattr(meta, "uid", None) or getattr(episode, "uid", "") or "")


def names_of(episode: Any) -> tuple[str, ...]:
    """Every name this episode answers to, including ones from earlier formats.

    A plan is written against whichever format the evidence lived in — success
    labels usually survive only in a collection's native one — and applied to
    whichever format the trainer reads. Matching on the current name alone
    means a plan can only ever be applied where it was made, which is not
    where it is needed.
    """
    meta = getattr(episode, "meta", None)
    lineage = getattr(meta, "lineage", None)
    if lineage:
        return tuple(str(name) for name in lineage)
    uid = uid_of(episode)
    return (uid,) if uid else ()


@dataclass(frozen=True)
class Applied:
    """What applying a plan did, and what it declined to do.

    Recorded rather than returned as a bare list, because "we dropped 1256"
    and "we dropped 1256 and 4 more were named but absent" are different
    states of the world and only one of them is fine.
    """

    kept: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    #: Cohort sampling weights. Not applied here — they are configuration for
    #: a trainer, not a change to the files — but carried so the caller can
    #: hand them on without reading the plan again.
    weights: Mapping[str, float] = field(default_factory=dict)
    #: Collection orders pass through untouched. They describe data that does
    #: not exist, and no amount of file writing will create it.
    orders: tuple[CollectionOrder, ...] = ()
    #: Named by the plan, absent from the data.
    missing: tuple[str, ...] = ()
    #: Present in the data, named by nothing — the plan simply had no opinion.
    untouched: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.dropped or self.weights)

    def validate(self, *, source_size: int) -> Verdict:
        checks = []
        if self.missing:
            checks.append(
                Verdict.no(
                    "curation.unactionable",
                    f"the plan names {len(self.missing)} episode(s) this dataset does "
                    f"not contain, e.g. {self.missing[0]!r}",
                    hint="identifiers drift across format conversions; a plan that "
                    "matches nothing applies cleanly and does nothing, and the "
                    "retrain afterwards still looks like a measurement",
                )
            )
        if source_size and not self.kept:
            checks.append(
                Verdict.no(
                    "curation.empties_the_dataset",
                    f"applying this leaves 0 of {source_size} episodes",
                )
            )
        if not self.changed and not self.orders:
            checks.append(
                Verdict.note(
                    "curation.no_change",
                    "this plan changes nothing about the data",
                    hint="verifying it would spend two retrains comparing a dataset "
                    "with itself",
                )
            )
        return Verdict.all(checks)

    def summary(self, *, source_size: int) -> str:
        bits = [f"{len(self.kept)}/{source_size} episodes kept"]
        if self.dropped:
            bits.append(f"{len(self.dropped)} dropped")
        if self.weights:
            bits.append("weights " + ", ".join(f"{k}={v:g}" for k, v in self.weights.items()))
        if self.orders:
            bits.append(f"{len(self.orders)} collection order(s) not applied here")
        return "; ".join(bits)


def select(
    plan: CurationPlan, episodes: Sequence[Any]
) -> tuple[tuple[Any, ...], Applied]:
    """The episodes that survive this plan, and a record of what happened.

    Order is preserved: a curated dataset that shuffles its episodes relative
    to the source makes every later comparison harder to reason about for no
    benefit.
    """
    # Every name in the dataset, under any format it has been through.
    present = {name for episode in episodes for name in names_of(episode)}
    present.discard("")

    keeps = set(plan.keeps)
    drops = set(plan.drops)
    missing = sorted((keeps | drops) - present)

    survivors = []
    kept, dropped, untouched = [], [], []
    for episode in episodes:
        uid = uid_of(episode)
        names = set(names_of(episode))
        # A whitelist when one is given: "keep the top third" has to mean the
        # other two thirds go, or the selection was decorative.
        if keeps and not (names & keeps):
            dropped.append(uid)
            continue
        if names & drops:
            dropped.append(uid)
            continue
        if not (names & keeps):
            untouched.append(uid)
        kept.append(uid)
        survivors.append(episode)

    return tuple(survivors), Applied(
        kept=tuple(kept),
        dropped=tuple(dropped),
        weights=dict(plan.weights),
        orders=plan.orders,
        missing=tuple(missing),
        untouched=tuple(untouched),
    )


def apply(
    plan: CurationPlan, episodes: Sequence[Any], *, strict: bool = True
) -> tuple[tuple[Any, ...], Applied]:
    """Select, and refuse if the result would not be worth training on.

    ``strict=False`` downgrades the refusals to the record, for a caller that
    wants to see what a half-matching plan would do before deciding. The
    default is to stop, because the failure this guards is invisible
    afterwards.
    """
    survivors, applied = select(plan, episodes)
    verdict = applied.validate(source_size=len(episodes))
    if strict:
        verdict.raise_if_refused(f"cannot apply {plan.signal!r}")
    return survivors, applied


def mix_config(plan: CurationPlan) -> dict[str, Any]:
    """Cohort weights, in the shape a training config wants.

    Weighting is not a change to the files — the same episodes are on disk
    either way — so it is emitted rather than applied. Kept next to
    :func:`apply` because a caller acting on a plan has to handle both, and
    handling only the half that rewrites data is how a weighting plan silently
    becomes a no-op.
    """
    return {
        "datasets": [
            {"cohort": cohort, "mix_ratio": weight}
            for cohort, weight in sorted(plan.weights.items())
        ]
    }
