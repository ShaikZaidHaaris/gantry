"""Plane 2: what a machine is, and what it costs to speak to another one.

An embodiment is **data, not code**. It declares its state and action channels,
its control rate, and where its kinematics live; it does not execute anything.
That separation is what lets a machine nobody has built yet participate: writing
a descriptor is a config file, not a plugin.

Retargeting is the hard half, and it is deliberately not hidden. There is no
universal robot space to convert into — the mapping from a seven-jointed arm to
a six-jointed one, or from joint angles to an end-effector pose, throws
information away, and which information depends on the pair. So a
:class:`Retargeter` is a declared, named transform between two specific
descriptions that states its loss in words. The resolver composes them and
reports the total.

The consequence is that some pairs simply do not connect, and the framework says
so rather than producing motion nobody can account for. A refusal is a worse
outcome than a working run and a much better one than a silent approximation.

Invariants
----------
1. A descriptor's state and action channels each validate on their own.
2. ``control_hz`` is positive when declared, and absent rather than guessed.
3. A retargeter's ``losses`` is non-empty whenever it discards information.
4. A retargeter accepts exactly what it declares it accepts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..spine import ChannelSpec, Descriptor, Verdict

EMBODIMENT_CONTRACT = "embodiment@1.0"

#: The machine can be commanded to a repeatable starting configuration.
CAP_RESETTABLE = "resettable"
#: Its starting state can be pinned by a seed.
CAP_SEEDABLE = "seedable"
#: It is a simulation rather than a physical machine.
CAP_SIMULATED = "simulated"


@dataclass(frozen=True)
class EmbodimentDescriptor:
    """A machine, described well enough to check anything against it."""

    name: str
    version: str
    state: tuple[ChannelSpec, ...] = ()
    action: tuple[ChannelSpec, ...] = ()
    control_hz: float | None = None
    #: Where the kinematics live (a URDF, an MJCF, a vendor file). A reference,
    #: never a parsed model: core has no business knowing these formats.
    kinematics: str | None = None
    #: Free-form operating notes — joint layout, gripper polarity, workspace
    #: limits. Written for a human or a policy that reads text, and carried
    #: verbatim rather than parsed.
    notes: str | None = None
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"embodiment:{self.name}@{self.version}"

    def channel(self, name: str) -> ChannelSpec | None:
        for spec in (*self.state, *self.action):
            if spec.name == name:
                return spec
        return None

    def validate(self) -> Verdict:
        checks = [spec.validate() for spec in (*self.state, *self.action)]
        if not self.action:
            checks.append(
                Verdict.note(
                    "embodiment.no_action",
                    f"{self.ref} declares no action channels, so nothing can command it",
                )
            )
        if self.control_hz is not None and self.control_hz <= 0:
            checks.append(
                Verdict.no(
                    "embodiment.control_hz",
                    f"{self.ref} declares control_hz={self.control_hz}, which is not a rate",
                )
            )
        names = [spec.name for spec in (*self.state, *self.action)]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            checks.append(
                Verdict.no(
                    "embodiment.duplicate_channel",
                    f"{self.ref} declares channel name(s) {sorted(duplicates)} twice",
                )
            )
        return Verdict.all(checks)

    def descriptor(self) -> Descriptor:
        return Descriptor(
            plane="embodiment",
            name=self.name,
            version=self.version,
            contract=EMBODIMENT_CONTRACT,
            provides={
                "control_hz": self.control_hz,
                "action_width": sum(spec.width for spec in self.action),
                **{capability: True for capability in self.capabilities},
            },
            metadata={**dict(self.metadata), "kinematics": self.kinematics},
        )


class Retargeter(ABC):
    """A declared transform between two specific channel descriptions.

    Not a converter to some universal space, because there isn't one. Each
    retargeter knows one pair and what that pair costs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier, recorded in provenance."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of this retargeter, recorded alongside its losses."""

    @abstractmethod
    def accepts(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        """Whether this retargeter handles this exact pair."""

    @abstractmethod
    def losses(self, source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
        """What is discarded, in words. Empty only when nothing is.

        Written for whoever reads the result months later, so "drops the wrist
        rotation" rather than "lossy".
        """

    @abstractmethod
    def apply(self, values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
        """Transform an array of shape ``(steps, *source.shape)``."""

    def check(self, source: ChannelSpec, target: ChannelSpec) -> Verdict:
        """Accepts the pair, and is honest about the cost."""
        verdict = self.accepts(source, target)
        if not verdict.ok:
            return verdict
        if source.width != target.width and not self.losses(source, target):
            return Verdict.no(
                "retarget.undeclared_loss",
                f"{self.name} maps {source.width} values to {target.width} but declares no loss",
                hint="a width change discards or invents information; say which",
            )
        return verdict

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}@{self.version}"


def embodiment_from_channels(
    name: str,
    version: str,
    *,
    state: Sequence[ChannelSpec] = (),
    action: Sequence[ChannelSpec] = (),
    **kwargs: Any,
) -> EmbodimentDescriptor:
    return EmbodimentDescriptor(
        name=name, version=version, state=tuple(state), action=tuple(action), **kwargs
    )
