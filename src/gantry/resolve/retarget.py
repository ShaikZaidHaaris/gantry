"""Finding a transform between two machines, when a declared one exists.

Adapters and retargeters look similar and are not. An adapter closes a gap that
has one correct answer: millimetres to metres is a multiplication, and reordering
labelled dimensions is a permutation. A retargeter closes a gap that does not -- mapping seven joints onto six, or a full pose onto a position, throws information
away, and *which* information depends on the pair and on somebody's judgement.

So they are searched differently. Adapters are looked up by refusal code, because
the code fully determines the fix. Retargeters are looked up by *pair*, and each
one is asked whether it handles this specific source and target. Nothing is
composed automatically: chaining two lossy judgements produces a third judgement
nobody made.

A retargeter is only tried once adapters have failed, and only when the gap is a
structural one. That ordering matters: if a plain unit conversion would do, using
a lossy retargeter instead would discard data for no reason.
"""

from __future__ import annotations

from typing import Sequence

from ..contracts.embodiment import Retargeter
from ..spine import ChannelSpec, Verdict
from .transform import RETARGETER, Step

#: Codes a retargeter may be asked to close. Deliberately the structural ones:
#: everything else has an adapter answer, and reaching for a lossy transform when
#: a lossless one exists is a bug, not a fallback.
RETARGETABLE = frozenset({"shape.mismatch", "semantics.mismatch", "dim_labels.mismatch"})


def as_step(retargeter: Retargeter, source: ChannelSpec, target: ChannelSpec) -> Step:
    """The runnable form of a retargeter, with this pair's loss worked out."""
    return Step(
        name=retargeter.name,
        version=retargeter.version,
        kind=RETARGETER,
        apply=retargeter.apply,
        losses=retargeter.losses(source, target),
        preserves_length=True,
    )


class RetargeterRegistry:
    """Retargeters, searched by asking each whether it handles a pair."""

    def __init__(self, retargeters: Sequence[Retargeter] = ()):
        self._retargeters: list[Retargeter] = list(retargeters)

    def add(self, retargeter: Retargeter) -> Retargeter:
        self._retargeters.append(retargeter)
        return retargeter

    def all(self) -> tuple[Retargeter, ...]:
        return tuple(self._retargeters)

    def __len__(self) -> int:
        return len(self._retargeters)

    def find(self, source: ChannelSpec, target: ChannelSpec) -> tuple[Step | None, Verdict]:
        """The first retargeter that accepts this pair, and why the others did not.

        First rather than best, because there is no ordering to be best by: two
        retargeters between the same pair encode two different judgements about
        what to discard, and picking one by a heuristic would be the framework
        making that call. Install one, or name the one you mean.
        """
        declined = []
        for retargeter in self._retargeters:
            verdict = retargeter.check(source, target)
            if verdict.ok:
                return as_step(retargeter, source, target), verdict
            declined.append(verdict)

        if not self._retargeters:
            return None, Verdict.no(
                "retarget.none_installed",
                f"no retargeter is installed to map {source.name!r} onto {target.name!r}",
                hint="a structural gap needs a declared transform; adapters cannot invent one",
            )
        return None, Verdict.no(
            "retarget.no_match",
            f"no installed retargeter handles {source.name!r} -> {target.name!r}",
            hint=f"{len(self._retargeters)} retargeter(s) declined this pair",
        ) & Verdict.all(declined)
