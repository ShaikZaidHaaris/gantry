"""Which implementations exist, per plane.

Deliberately dumb. The registry maps ``(plane, name)`` to a factory and knows
nothing else — every judgement about whether a component *fits* belongs to the
resolver, working from descriptors.

Discovery is opt-in. ``register`` is explicit and enough for a script or a
test; ``discover`` reads packaging entry points so an installed plugin appears
without anyone editing a list. Nothing is imported until it is asked for, so a
broken third-party plugin cannot take down an unrelated run at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..spine import Descriptor, entry_point_groups, known_planes

def ENTRY_POINT_GROUPS() -> Mapping[str, str]:
    """Packaging entry-point group -> plane, derived from the plane registry.

    Derived rather than listed, so registering a plane makes its group
    discoverable with no edit here. A hand-maintained copy of a registry is a
    copy that eventually disagrees with it.
    """
    return entry_point_groups()


@dataclass(frozen=True)
class Registration:
    """One installed implementation, not yet constructed."""

    plane: str
    name: str
    factory: Callable[..., Any]
    #: Optional cheap description, so the resolver can plan without paying
    #: construction costs. A policy that would load gigabytes of weights should
    #: provide this; a connector generally need not.
    describe: Callable[..., Descriptor] | None = None
    origin: str = "explicit"

    @property
    def ref(self) -> str:
        return f"{self.plane}:{self.name}"

    def build(self, config: Mapping[str, Any] | None = None) -> Any:
        return self.factory(**dict(config or {}))

    def descriptor(self, config: Mapping[str, Any] | None = None) -> Descriptor:
        """Describe without committing to a full build, where possible."""
        if self.describe is not None:
            return self.describe(**dict(config or {}))
        return self.build(config).descriptor()


class DuplicateRegistration(ValueError):
    pass


class Registry:
    """Everything installed, addressable by plane and name."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Registration] = {}

    # -- population --------------------------------------------------------

    def register(
        self,
        plane: str,
        name: str,
        factory: Callable[..., Any],
        *,
        describe: Callable[..., Descriptor] | None = None,
        origin: str = "explicit",
        replace: bool = False,
    ) -> Registration:
        if plane not in known_planes():
            raise ValueError(
                f"unknown plane {plane!r}; expected one of {known_planes()}"
            )
        key = (plane, name)
        if key in self._entries and not replace:
            raise DuplicateRegistration(
                f"{plane}:{name} is already registered by {self._entries[key].origin}; "
                "pass replace=True to override it deliberately"
            )
        registration = Registration(plane, name, factory, describe, origin)
        self._entries[key] = registration
        return registration

    def discover(self, groups: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """Register everything installed that declares a Gantry entry point.

        Loading is lazy: the entry point is resolved on first use, not here, so
        one plugin with a broken import does not break discovery for the rest.
        """
        from importlib.metadata import entry_points

        found: list[str] = []
        for group, plane in (groups or ENTRY_POINT_GROUPS()).items():
            for entry in entry_points(group=group):
                if (plane, entry.name) in self._entries:
                    continue
                self.register(
                    plane,
                    entry.name,
                    _LazyEntryPoint(entry),
                    origin=f"entry-point:{group}",
                )
                found.append(f"{plane}:{entry.name}")
        return tuple(found)

    # -- lookup ------------------------------------------------------------

    def get(self, plane: str, name: str) -> Registration:
        try:
            return self._entries[(plane, name)]
        except KeyError:
            available = self.names(plane)
            hint = f"installed on the {plane} plane: {', '.join(available)}" if available else (
                f"nothing is installed on the {plane} plane"
            )
            raise KeyError(f"no {plane} component named {name!r}; {hint}") from None

    def has(self, plane: str, name: str) -> bool:
        return (plane, name) in self._entries

    def names(self, plane: str) -> tuple[str, ...]:
        return tuple(sorted(name for (p, name) in self._entries if p == plane))

    def planes(self) -> tuple[str, ...]:
        return tuple(sorted({plane for (plane, _) in self._entries}))

    def all(self) -> tuple[Registration, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._entries

    def describe_all(self) -> dict[str, list[str]]:
        """What ``gantry list`` prints."""
        return {plane: list(self.names(plane)) for plane in self.planes()}


class _LazyEntryPoint:
    """Defers importing a plugin until something actually builds it."""

    def __init__(self, entry: Any):
        self._entry = entry
        self._loaded: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._loaded is None:
            self._loaded = self._entry.load()
        return self._loaded(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<lazy {self._entry.value}>"
