"""Carry out what a plan promised.

Planning decides that a provider's channel can feed a consumer's, possibly
through a chain of transforms. This is the other half: reading the data so the
consumer sees what it asked for, under the name it asked for, in the units it
asked for.

Without this the plan is a document. A chain that gets reported and never applied
is worse than no chain at all, because the run looks converted.

Laziness survives where it can. Most transforms are per-timestep and preserve
episode length, so they compose into a wrapper that transforms on read and never
materialises anything. A resampler does not -- it changes how many steps there
are, which makes a step window meaningless -- so an episode adapted through one is
read once and held. That branch is explicit rather than clever: silently
returning the right *count* of the wrong rows is exactly the kind of quiet
wrongness the framework exists to prevent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..spine import ArraySource, EpisodeRecord, StepSource
from .plan import ChannelBinding, Wiring


class AdaptedSource:
    """A :class:`~gantry.spine.episode.StepSource` that transforms on read.

    Only used when every planned step preserves episode length, so a step window
    still means the same thing on both sides of the transform.
    """

    def __init__(self, inner: StepSource, bindings: Sequence[ChannelBinding]):
        self._inner = inner
        self._bindings = {binding.want.name: binding for binding in bindings}

    @property
    def num_steps(self) -> int:
        return self._inner.num_steps

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self._bindings)

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        wanted = tuple(channels) if channels is not None else self.channels
        missing = [name for name in wanted if name not in self._bindings]
        if missing:
            raise KeyError(f"no such channel(s): {missing}")

        bindings = [self._bindings[name] for name in wanted]
        raw = self._inner.read([b.provided.name for b in bindings], start, stop)
        return {
            binding.want.name: binding.chain.run(
                raw[binding.provided.name], binding.provided, binding.want
            )
            for binding in bindings
        }


def changes_length(wiring: Wiring) -> bool:
    return not all(binding.chain.preserves_length for binding in wiring.bindings)


def adapt_episode(episode: EpisodeRecord, wiring: Wiring) -> EpisodeRecord:
    """Present one episode the way a consumer asked for it.

    Renaming counts. A consumer that matched a channel by meaning or by alias
    sees it under *its own* name, so the module never has to know what the
    recording happened to call things -- which is the whole reason binding by
    semantics exists.
    """
    if not wiring.bindings:
        return episode

    schema = tuple(binding.want for binding in wiring.bindings)

    if changes_length(wiring):
        # Read once and hold: after a resample the step count differs, so a
        # lazy window would slice against the wrong timeline.
        raw = episode.read([b.provided.name for b in wiring.bindings])
        arrays = {
            b.want.name: b.chain.run(raw[b.provided.name], b.provided, b.want)
            for b in wiring.bindings
        }
        lengths = {name: len(values) for name, values in arrays.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"{episode.meta.uid}: transforms left channels at differing lengths "
                f"({lengths}); channels resampled together must land on one timeline"
            )
        return replace(episode, schema=schema, source=ArraySource(arrays))

    return replace(episode, schema=schema, source=AdaptedSource(episode.source, wiring.bindings))


def adapt_all(episodes: Sequence[EpisodeRecord], wiring: Wiring) -> tuple[EpisodeRecord, ...]:
    return tuple(adapt_episode(episode, wiring) for episode in episodes)
