"""What a consumer needs, stated so a machine can check it.

A policy, an evaluator or a feedback module declares a Requirement: the
channels it must be fed and the capabilities it depends on. The resolver
matches that against what a provider describes, *before* anything loads.

Two kinds of need, deliberately separate:

``channels``
    Specs the consumer wants, with wildcards allowed. Matched against the
    provider's schema, by name first and by meaning second.

``capabilities``
    Facts about the provider that no channel expresses -- whether it can emit
    stage events, whether reads are selective. This is how a funnel diagnosis
    says out loud that it cannot run on outcome-only data, instead of
    producing an empty table and letting someone read meaning into it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..spine import ChannelSpec


@dataclass(frozen=True)
class Requirement:
    """One consumer's needs.

    ``aliases`` maps a wanted channel name to other names it may arrive under,
    which is how a consumer accepts a provider's vocabulary without anyone
    renaming data behind the user's back.
    """

    name: str
    plane: str
    channels: tuple[ChannelSpec, ...] = ()
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    description: str = ""

    def alias_names(self, want: str) -> tuple[str, ...]:
        return tuple(self.aliases.get(want, ()))

    @property
    def required_channels(self) -> tuple[ChannelSpec, ...]:
        return tuple(spec for spec in self.channels if not spec.optional)

    def __str__(self) -> str:
        return f"{self.plane}:{self.name}"


def requires_channels(
    name: str,
    plane: str,
    *channels: ChannelSpec,
    capabilities: Mapping[str, Any] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    description: str = "",
) -> Requirement:
    """Terse constructor, so declaring a requirement stays a one-liner."""
    return Requirement(
        name=name,
        plane=plane,
        channels=tuple(channels),
        capabilities=dict(capabilities or {}),
        aliases={key: tuple(value) for key, value in (aliases or {}).items()},
        description=description,
    )
