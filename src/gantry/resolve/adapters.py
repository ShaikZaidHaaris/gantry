"""Transforms that close a specific incompatibility, and say what they cost.

An adapter registers against the refusal codes it can close. When
:func:`gantry.spine.compatible` says ``units.scale``, the resolver looks for an
adapter advertising ``units.scale`` and, if one exists, plans it in. If none
does, the run is refused with the code still named, so the answer to "why not"
is always a specific missing capability rather than a shrug.

An adapter is for gaps with one correct answer: millimetres to metres is a
multiplication, reordering labelled dimensions is a permutation. A gap needing a
judgement about what to discard is a *retargeter* — a different thing, searched
differently, in :mod:`gantry.resolve.retarget`.

Cost is mandatory, not decorative. A lossy adapter states its loss in words,
that sentence is copied into the run's provenance, and it travels with every
number the run produces. A resampler that quietly drops a third of the frames
is how a benchmark becomes a rumour.

Planning is separate from applying. This module answers "can this gap be closed,
and at what cost", which is all the resolver needs; the resulting
:class:`~gantry.resolve.transform.Step` is what actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

from ..spine import ChannelSpec, Descriptor, Verdict
from .transform import ADAPTER, Apply, Chain, Step

#: Version of the adapter contract.
ADAPTER_CONTRACT = "adapter@1.0"

Guard = Callable[[ChannelSpec, ChannelSpec], Verdict]
Cost = Callable[[ChannelSpec, ChannelSpec], Sequence[str]]


@runtime_checkable
class SupportsAdaptation(Protocol):
    """The minimum an adapter must answer during planning."""

    @property
    def closes(self) -> tuple[str, ...]: ...

    def applies(self, provider: ChannelSpec, consumer: ChannelSpec) -> Verdict: ...

    def losses(self, provider: ChannelSpec, consumer: ChannelSpec) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class Adapter:
    """A declared transform between two channel descriptions."""

    name: str
    version: str
    closes: tuple[str, ...]
    #: Called during planning to confirm this pair is really adaptable.
    guard: Guard | None = None
    #: Called during planning to state the cost, in words, for this pair.
    cost: Cost | None = None
    #: Applied at run time.
    transform: Apply | None = None
    #: Whether applying this leaves the episode the same number of steps long.
    #: False for anything that resamples: a step window means something
    #: different on each side of it, so such an episode cannot be read lazily.
    preserves_length: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def applies(self, provider: ChannelSpec, consumer: ChannelSpec) -> Verdict:
        return self.guard(provider, consumer) if self.guard else Verdict.yes()

    def losses(self, provider: ChannelSpec, consumer: ChannelSpec) -> tuple[str, ...]:
        return tuple(self.cost(provider, consumer)) if self.cost else ()

    def as_step(self, provider: ChannelSpec, consumer: ChannelSpec) -> Step:
        """The runnable form, with this pair's cost already worked out."""
        if self.transform is None:
            raise ValueError(f"adapter {self} has no transform and cannot be applied")
        return Step(
            name=self.name,
            version=self.version,
            kind=ADAPTER,
            apply=self.transform,
            losses=self.losses(provider, consumer),
            preserves_length=self.preserves_length,
        )

    def descriptor(self) -> Descriptor:
        return Descriptor(
            plane="adapter",
            name=self.name,
            version=self.version,
            contract=ADAPTER_CONTRACT,
            provides={"closes": list(self.closes)},
            metadata=dict(self.metadata),
        )

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


class AdapterRegistry:
    """Adapters indexed by the refusal codes they close."""

    def __init__(self, adapters: Sequence[Adapter] = ()):
        self._by_code: dict[str, list[Adapter]] = {}
        for adapter in adapters:
            self.add(adapter)

    def add(self, adapter: Adapter) -> Adapter:
        """Register an adapter. Refuses one that cannot actually be applied.

        A transform-less adapter would plan cleanly and then fail on the first
        read — after the run has started, and only for whichever channel touched
        it first. Catching it here makes the failure immediate and total, which
        is the right shape for a configuration error.
        """
        if adapter.transform is None:
            raise ValueError(
                f"adapter {adapter} has no transform, so it can close a gap on paper "
                "and not in the data; give it one or do not register it"
            )
        for code in adapter.closes:
            self._by_code.setdefault(code, []).append(adapter)
        return adapter

    def for_code(self, code: str) -> tuple[Adapter, ...]:
        return tuple(self._by_code.get(code, ()))

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_code))

    def all(self) -> tuple[Adapter, ...]:
        seen: dict[str, Adapter] = {}
        for adapters in self._by_code.values():
            for adapter in adapters:
                seen[str(adapter)] = adapter
        return tuple(seen.values())

    def __len__(self) -> int:
        return len(self.all())

    def close(
        self, provider: ChannelSpec, consumer: ChannelSpec, codes: Sequence[str]
    ) -> tuple[Chain | None, tuple[str, ...]]:
        """Find adapters covering every code.

        Returns ``(chain, unclosed)``. A chain is only returned when *every*
        code is covered: a partially adapted channel is still the wrong data,
        and half a bridge is not a bridge.
        """
        chosen: list[Adapter] = []
        steps: list[Step] = []
        unclosed: list[str] = []

        for code in dict.fromkeys(codes):  # de-duplicated, order preserved
            candidate = next(
                (
                    adapter
                    for adapter in self.for_code(code)
                    if adapter.applies(provider, consumer).ok
                ),
                None,
            )
            if candidate is None:
                unclosed.append(code)
                continue
            if candidate not in chosen:
                chosen.append(candidate)
                steps.append(candidate.as_step(provider, consumer))

        if unclosed:
            return None, tuple(unclosed)
        return Chain(tuple(steps)), ()
