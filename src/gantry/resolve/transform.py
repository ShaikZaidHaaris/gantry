"""What a binding actually does to the data on its way through.

Adapters and retargeters are different things and must stay different: one has a
single correct answer, the other encodes a judgement about what to discard. But
by the time a plan is being executed, the difference has already been made — the
decision is taken, and what remains is a list of transforms to run in order.

:class:`Step` is that residue. Both kinds produce one, and neither is disguised
as the other: a step records which kind it came from, so provenance and a reader
can still tell a rescaling from a reduction.

The alternative, wrapping a retargeter in an adapter so a chain could hold it,
would be the framework coercing one concept into another to make a type fit —
the precise move this design exists to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..spine import AdapterStep, ChannelSpec

#: Applied to an array of shape ``(steps, *source.shape)``.
Apply = Callable[[np.ndarray, ChannelSpec, ChannelSpec], np.ndarray]

ADAPTER, RETARGETER = "adapter", "retargeter"


@dataclass(frozen=True)
class Step:
    """One transform, already decided, ready to run."""

    name: str
    version: str
    kind: str
    apply: Apply
    losses: tuple[str, ...] = ()
    #: False for anything that changes how many steps an episode has.
    preserves_length: bool = True

    def __post_init__(self) -> None:
        if self.kind not in (ADAPTER, RETARGETER):
            raise ValueError(f"unknown transform kind {self.kind!r}")

    @property
    def lossy(self) -> bool:
        return bool(self.losses)

    def provenance(self) -> AdapterStep:
        """The record form. Losses travel; the taxonomy does not need to."""
        return AdapterStep(self.name, self.version, self.losses)

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class Chain:
    """Transforms to run in order, or nothing at all."""

    steps: tuple[Step, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.steps)

    def __bool__(self) -> bool:
        return bool(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __str__(self) -> str:
        return " -> ".join(str(step) for step in self.steps) or "direct"

    @property
    def losses(self) -> tuple[str, ...]:
        return tuple(loss for step in self.steps for loss in step.losses)

    @property
    def lossy(self) -> bool:
        return bool(self.losses)

    @property
    def preserves_length(self) -> bool:
        return all(step.preserves_length for step in self.steps)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(step.kind for step in self.steps)

    @property
    def provenance(self) -> tuple[AdapterStep, ...]:
        return tuple(step.provenance() for step in self.steps)

    def then(self, step: Step) -> "Chain":
        return Chain(self.steps + (step,))

    def run(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        """Apply every step in planned order."""
        for step in self.steps:
            values = step.apply(values, source, target)
        return np.asarray(values)


def chain_of(steps: Sequence[Step]) -> Chain:
    return Chain(tuple(steps))
