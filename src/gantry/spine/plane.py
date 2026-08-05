"""What kinds of component exist, and what the framework knows about each.

Every other vocabulary in core is open: modalities, meaning tags, units,
statistics, planted defects. Planes were the exception, frozen in a tuple, which
made the one decomposition most likely to need extending the one thing a plugin
could not extend. This closes that inconsistency.

A plane is registered with everything the framework needs in order to reason
about it generically -- which contract to version-check, which entry-point group
to discover, whether more than one may appear in a run, and whether it can be
run out of process. Once registered, a plane is a first-class citizen: the
registry accepts it, descriptors validate against it, manifests carry it, the
resolver version-checks it, and the CLI lists it. None of those had to be told
about it.

The six core planes are still declared here rather than discovered, because they
are the framework's own decomposition and something has to be the base case.
What changed is that being one of them is no longer a privilege.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: How many components of a plane one run may name.
ONE, MANY = "one", "many"


@dataclass(frozen=True)
class Plane:
    """One kind of swappable component, described well enough to handle generically."""

    name: str
    description: str
    #: The published contract, if there is one. ``None`` means version checking
    #: is skipped and said to be skipped, rather than skipped silently.
    contract: str | None = None
    #: Packaging entry-point group that installs into this plane.
    entry_point_group: str | None = None
    #: Whether a run may name several of these.
    cardinality: str = ONE
    #: Whether a component here can be run in another interpreter. Isolation
    #: needs a narrow, data-shaped interface; a plane without one gets an honest
    #: refusal rather than a half-working proxy.
    proxyable: bool = False

    def __post_init__(self) -> None:
        if self.cardinality not in (ONE, MANY):
            raise ValueError(f"unknown cardinality {self.cardinality!r}")

    @property
    def many(self) -> bool:
        return self.cardinality == MANY


_PLANES: dict[str, Plane] = {}


def register_plane(plane: Plane) -> Plane:
    """Register a plane. Idempotent for identical definitions.

    Redefining an existing plane differently is refused: two packages
    disagreeing about what "policy" means would make every descriptor's plane
    field mean whichever was imported last.
    """
    existing = _PLANES.get(plane.name)
    if existing is not None and existing != plane:
        raise ValueError(
            f"plane {plane.name!r} is already registered as {existing}; "
            "a plane is a shared vocabulary and cannot be redefined"
        )
    _PLANES[plane.name] = plane
    return plane


def get_plane(name: str) -> Plane | None:
    return _PLANES.get(name)


def known_planes() -> tuple[str, ...]:
    """Every registered plane, core and extension alike, in registration order."""
    return tuple(_PLANES)


def planes() -> tuple[Plane, ...]:
    return tuple(_PLANES.values())


def contract_for(name: str) -> str | None:
    plane = _PLANES.get(name)
    return plane.contract if plane else None


def entry_point_groups() -> Mapping[str, str]:
    """Group -> plane, for discovery. Derived, never maintained by hand."""
    return {
        plane.entry_point_group: plane.name for plane in _PLANES.values() if plane.entry_point_group
    }


def proxyable_planes() -> tuple[str, ...]:
    return tuple(plane.name for plane in _PLANES.values() if plane.proxyable)


# --------------------------------------------------------------------------
# the core planes
# --------------------------------------------------------------------------

for _plane in (
    Plane(
        "dataset",
        "reads recorded episodes out of some storage format",
        contract="connector@1.0",
        entry_point_group="gantry.connectors",
        cardinality=MANY,
        proxyable=True,
    ),
    Plane(
        "task",
        "what is being attempted, described so that any world can stage it and "
        "any person can judge it",
        contract="task@1.0",
        entry_point_group="gantry.tasks",
        cardinality=MANY,
    ),
    Plane(
        "scorer",
        "who decides whether a trial succeeded, given what evidence",
        contract="scorer@1.0",
        entry_point_group="gantry.scorers",
        cardinality=MANY,
    ),
    Plane(
        "curation",
        "what to do to the data, said precisely enough to be applied and to be found wrong",
        contract="curation@1.0",
        entry_point_group="gantry.curators",
        cardinality=MANY,
    ),
    Plane(
        "embodiment",
        "describes a machine, and what it costs to speak to another one",
        contract="embodiment@1.0",
        entry_point_group="gantry.embodiments",
    ),
    Plane(
        "policy",
        "turns observations into actions",
        contract="policy@1.0",
        entry_point_group="gantry.policies",
    ),
    Plane(
        "evaluation",
        "runs a policy against something and records what happened",
        contract="evaluator@1.1",
        entry_point_group="gantry.evaluators",
    ),
    Plane(
        "feedback",
        "turns records into findings, and findings into prescriptions",
        contract="feedback@1.0",
        entry_point_group="gantry.feedback",
        cardinality=MANY,
    ),
    Plane(
        "adapter",
        "closes a specific gap between two channel descriptions",
        entry_point_group="gantry.adapters",
        cardinality=MANY,
    ),
    Plane(
        "retargeter",
        "maps one machine's channel onto another's, declaring what it discards",
        entry_point_group="gantry.retargeters",
        cardinality=MANY,
    ),
):
    register_plane(_plane)


#: The planes core itself declares. Pinned by the frozen-surface test, so adding
#: one here is a deliberate act; :func:`known_planes` is what everything else
#: should ask, because it includes whatever plugins have registered.
CORE_PLANES = known_planes()
