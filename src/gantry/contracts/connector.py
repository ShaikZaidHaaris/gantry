"""Plane 1: reading demonstrations out of whatever format they live in.

A connector's only job is faithful *description*. It exposes a foreign format
as :class:`~gantry.spine.episode.EpisodeRecord` values and stops there. It does
not retarget, resample, rename, or repair. Those are adapters, and adapters are
separate, declared, and lossy on the record.

That restraint is the point. Every silent-coercion bug this design exists to
prevent begins with a reader that decided it knew what the data meant.

Invariants
----------
Each sentence below is one check in :mod:`gantry.conformance.connector`. A
connector that satisfies them is usable by any part of Gantry, forever, without
Gantry knowing anything about its format.

1. ``episode_ids()`` returns the same ids in the same order every time it is
   called, and they are unique.
2. ``schema(id)`` equals ``open(id).schema`` exactly, and can be answered
   without reading step data.
3. Every returned :class:`~gantry.spine.channel.ChannelSpec` is internally
   valid, and the data agrees with it.
4. Reads are selective: asking for two channels and a step window returns
   exactly those channels over exactly that window.
5. Reading the same episode twice returns identical data.
6. Opening an unknown id raises :class:`KeyError`.
7. Episode ids are namespaced by a non-empty source.
8. Declared capabilities are true. A connector claiming ``stage_events`` has
   them; one denying them does not.

Capabilities
------------
The keys below are defined by *this contract*, not by the spine, so a future
plane can add its own vocabulary without touching core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Sequence

from ..errors import ConfigError
from ..spine import ChannelSpec, Descriptor, EpisodeRecord

#: Version of this contract. Plugins pin it; the host checks it.
#: 1.1 added ``write``. A connector that can read a format can usually write
#: it, and until now that ability existed only as a module-level function some
#: connectors happened to expose — so anything wanting to *produce* a dataset
#: had to import a specific one by name, which is exactly the coupling this
#: plane exists to prevent. A minor bump: the default refuses, so a connector
#: predating it is read-only and says so rather than failing later.
CONNECTOR_CONTRACT = "connector@1.1"

#: Selective reads are honoured, so opening an episode does not materialise it.
CAP_LAZY = "lazy"
#: At least some episodes carry stage events (a funnel diagnosis needs these).
CAP_STAGE_EVENTS = "stage_events"
#: At least some episodes carry a success label.
CAP_OUTCOMES = "outcomes"
#: At least some episodes carry an image, depth, or blob channel.
CAP_MEDIA = "media"
#: Can produce a dataset in this format, not only read one.
CAP_WRITES = "writes"

CAPABILITIES = (CAP_LAZY, CAP_STAGE_EVENTS, CAP_OUTCOMES, CAP_MEDIA, CAP_WRITES)


def connector_descriptor(
    name: str,
    version: str,
    *,
    lazy: bool,
    stage_events: bool,
    outcomes: bool,
    media: bool = False,
    writes: bool = False,
    isolation: str = "in-process",
    requires: dict[str, Any] | None = None,
    **metadata: Any,
) -> Descriptor:
    """Build a well-formed descriptor for a connector.

    Every capability is a required argument rather than an optional one with a
    convenient default. Guessing here would let a plugin quietly claim
    something it cannot do, and the resolver's whole value is that a claim is
    load-bearing.
    """
    return Descriptor(
        plane="dataset",
        name=name,
        version=version,
        contract=CONNECTOR_CONTRACT,
        requires=requires or {},
        provides={
            CAP_LAZY: lazy,
            CAP_STAGE_EVENTS: stage_events,
            CAP_OUTCOMES: outcomes,
            CAP_MEDIA: media,
            CAP_WRITES: writes,
        },
        isolation=isolation,
        metadata=metadata,
    )


class Connector(ABC):
    """Read-only access to a set of episodes in some storage format."""

    # -- required ----------------------------------------------------------

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """What this connector is and what it can honestly offer."""

    @abstractmethod
    def episode_ids(self) -> tuple[str, ...]:
        """Every episode id, in a stable order.

        Ids, not records: enumerating a dataset must stay cheap no matter how
        large it is.
        """

    @abstractmethod
    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        """The channels of one episode, without reading its steps."""

    @abstractmethod
    def open(self, episode_id: str) -> EpisodeRecord:
        """A record for one episode, with its steps still unread."""

    # -- provided ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.episode_ids())

    def __iter__(self) -> Iterator[EpisodeRecord]:
        for episode_id in self.episode_ids():
            yield self.open(episode_id)

    def episodes(self) -> Iterator[EpisodeRecord]:
        """Every episode, opened lazily one at a time."""
        return iter(self)

    @property
    def writes(self) -> bool:
        return bool(self.descriptor().provides.get(CAP_WRITES, False))

    @classmethod
    def write(cls, episodes: "Sequence[EpisodeRecord]", path: str, **options: Any) -> Any:
        """Produce a dataset in this format. The inverse of reading it.

        A classmethod, because writing is a property of the format and not of
        any open dataset: the thing being written does not exist yet, so there
        is nothing to have opened in order to ask.

        Here rather than as a module-level function each connector happens to
        expose, because otherwise anything that wants to *make* a dataset has
        to import a particular format by name — and a framework that imports
        one format by name has a favourite, which is the thing this plane is
        for not having.

        The default refuses. Most connectors read a format somebody else
        writes, and quietly doing nothing would leave a caller believing a
        dataset exists.
        """
        raise ConfigError(
            f"{cls.__name__} reads this format but does not write it; a connector "
            f"that can declares {CAP_WRITES!r}"
        )

    def sample(self, n: int = 3) -> tuple[EpisodeRecord, ...]:
        """The first ``n`` episodes. Used by conformance and by previews."""
        return tuple(self.open(i) for i in self.episode_ids()[:n])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        descriptor = self.descriptor()
        return f"<{type(self).__name__} {descriptor.ref} n={len(self)}>"
