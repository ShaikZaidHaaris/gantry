"""Recovering identity a conversion threw away.

From now on a converter records where each episode came from, and the problem
does not arise. But the datasets that already exist were written before that,
and they are the ones with the training runs attached -- so there has to be a
way to reconnect a converted copy to the collection it came from without
re-doing the conversion.

The reconnection is by content. Two episodes are the same demonstration if
their actions are the same numbers in the same order, which is a fact about the
data rather than about anyone's naming scheme, and survives renumbering,
subsetting and reordering.

Why actions rather than observations
------------------------------------
A conversion commonly rebuilds observations -- assembling a state vector from
several channels, resampling, dropping what the trainer will not read. Actions
are the part that usually crosses unchanged, because they are what was
commanded and there is nothing to reassemble.

Why exact rather than approximate
---------------------------------
A near-match is a guess, and a wrong link is worse than a missing one: it
attaches a measurement to the wrong demonstration and everything downstream is
quietly about something else. So this matches exactly, reports what it could
not match, and leaves those alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Mapping, Sequence

import numpy as np

from .spine import Verdict

#: What the compared channel means. Found by meaning rather than by name,
#: because the name is exactly the thing a conversion changes -- the same column
#: is ``actions`` in one format and ``action`` in another, and matching on the
#: string reconnects nothing while looking like it tried.
ACTUATION = "actuation"

#: Names to fall back on when nothing declares its meaning. A guess, and only
#: reached when the alternative is not looking at all.
CONVENTIONAL = ("action", "actions")

#: Rounding applied before hashing. A conversion that writes float32 where the
#: source held float64 changes the last bits without changing the
#: demonstration, and refusing to match those would defeat the purpose.
DECIMALS = 5


def actuation_channel(episode: Any) -> str | None:
    """The channel carrying what was commanded, whatever this format calls it."""
    specs = list(getattr(episode, "schema", ()) or ())
    for spec in specs:
        if getattr(spec, "semantics", None) == ACTUATION:
            return spec.name
    names = {spec.name for spec in specs}
    for candidate in CONVENTIONAL:
        if candidate in names:
            return candidate
    return None


def fingerprint(episode: Any, channel: str | None = None) -> str | None:
    """A content hash of what this episode commanded, or None if unreadable."""
    try:
        channel = channel or actuation_channel(episode)
        if channel is None:
            return None
        values = np.asarray(episode.read([channel])[channel], dtype=np.float64)
    except Exception:  # noqa: BLE001 - an unreadable episode simply does not match
        return None
    if values.size == 0:
        return None
    rounded = np.round(values, DECIMALS) + 0.0  # +0.0 collapses -0.0 and 0.0
    digest = blake2b(rounded.tobytes(), digest_size=16)
    digest.update(str(rounded.shape).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Relinked:
    """What could be reconnected, and what could not."""

    #: Converted uid -> the source uid it was found to be.
    links: Mapping[str, str] = field(default_factory=dict)
    #: Converted episodes no source episode matched.
    unmatched: tuple[str, ...] = ()
    #: Fingerprints appearing more than once on one side, and so unusable --     #: two identical demonstrations cannot be told apart by their content.
    ambiguous: tuple[str, ...] = ()

    def validate(self, *, target_size: int) -> Verdict:
        checks = []
        if not self.links:
            checks.append(
                Verdict.no(
                    "lineage.nothing_matched",
                    f"none of the {target_size} episode(s) matched any source episode",
                    hint="the conversion may have rewritten the actions themselves, "
                    "in which case content cannot reconnect them",
                )
            )
        if self.ambiguous:
            checks.append(
                Verdict.note(
                    "lineage.ambiguous",
                    f"{len(self.ambiguous)} episode(s) share their content with "
                    "another and were left unlinked",
                    hint="identical demonstrations cannot be told apart by content; "
                    "a wrong link is worse than none",
                )
            )
        if self.unmatched:
            checks.append(
                Verdict.note(
                    "lineage.unmatched",
                    f"{len(self.unmatched)} of {target_size} episode(s) found no "
                    f"source, e.g. {self.unmatched[0]!r}",
                )
            )
        return Verdict.all(checks)


def relink(source: Sequence[Any], target: Sequence[Any], *, channel: str | None = None) -> Relinked:
    """Match converted episodes back to the ones they came from, by content.

    ``source`` is the collection the evidence lives in; ``target`` is the copy
    a trainer reads. Neither is modified -- the result is a mapping the caller
    applies, because rewriting somebody's dataset as a side effect of asking a
    question is not this function's business.
    """
    by_print: dict[str, list[str]] = {}
    for episode in source:
        print_ = fingerprint(episode, channel)
        if print_:
            by_print.setdefault(print_, []).append(episode.meta.uid)

    links: dict[str, str] = {}
    unmatched, ambiguous = [], []
    for episode in target:
        uid = episode.meta.uid
        print_ = fingerprint(episode, channel)
        candidates = by_print.get(print_ or "", [])
        if len(candidates) == 1:
            links[uid] = candidates[0]
        elif len(candidates) > 1:
            ambiguous.append(uid)
        else:
            unmatched.append(uid)
    return Relinked(links=links, unmatched=tuple(unmatched), ambiguous=tuple(ambiguous))


def rename(plan_names: Sequence[str], links: Mapping[str, str]) -> tuple[str, ...]:
    """Translate names from the source's vocabulary into the target's.

    The inverse direction of :func:`relink`'s mapping, which is what a plan
    needs: it was written about ``mg/demo_1`` and has to be applied where that
    demonstration is called something else.
    """
    backwards = {source: converted for converted, source in links.items()}
    return tuple(backwards[name] for name in plan_names if name in backwards)
