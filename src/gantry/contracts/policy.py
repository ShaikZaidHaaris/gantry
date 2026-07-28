"""Plane 3: something that turns observations into actions.

Anything that produces actions fits here — a learned model, a scripted routine,
a planner, a language model driving the machine through tool calls, a person at
a joystick. The contract asks only for the shape of that exchange, so nothing
about the plane presumes a neural network.

Policies emit a **chunk**: several future actions from one call. That is not a
convenience, it is how these systems are actually run — inference is slower than
the control rate, so a burst is predicted and executed open-loop. How much of a
chunk gets executed before re-planning is a protocol decision, and it is one of
the largest levers on a measured result. Making the chunk the return type keeps
that lever visible instead of buried in a wrapper.

Invariants
----------
1. ``act`` returns at least one action, shaped ``(horizon, *action_spec.shape)``.
2. Every returned action matches the declared action spec.
3. ``reset`` starts a fresh episode; a policy that keeps state across a reset
   it did not declare is not reproducible.
4. A policy declaring determinism returns identical actions for identical
   input, which is what makes a paired comparison meaningful.
5. ``observes()`` names everything ``act`` will read, so the resolver can
   refuse a missing observation before a run rather than during one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..resolve.requirement import Requirement
from ..spine import ChannelSpec, Descriptor

POLICY_CONTRACT = "policy@1.0"

#: Longest chunk this policy will ever return.
CAP_CHUNK = "chunk"
#: Identical input yields identical output.
CAP_DETERMINISTIC = "deterministic"
#: Carries state between steps within an episode.
CAP_STATEFUL = "stateful"


@dataclass(frozen=True)
class EpisodeContext:
    """What a policy is told before an episode starts.

    ``instruction`` is the natural-language goal where there is one. ``seed`` is
    for policies with their own randomness, so a run pinned by seed is pinned
    all the way down rather than only in the environment.
    """

    episode_id: str
    instruction: str | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """One timestep as the policy sees it.

    ``step`` is included because a policy may legitimately depend on how far
    into the episode it is, and reconstructing that from a counter somewhere
    else is how off-by-one errors get into results.
    """

    step: int
    channels: Mapping[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.channels[name]
        except KeyError:
            raise KeyError(
                f"observation has no channel {name!r}; has {sorted(self.channels)}"
            ) from None

    def get(self, name: str, default: Any = None) -> Any:
        return self.channels.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.channels


def policy_descriptor(
    name: str,
    version: str,
    *,
    chunk: int,
    deterministic: bool,
    stateful: bool = False,
    isolation: str = "in-process",
    **metadata: Any,
) -> Descriptor:
    """Every capability explicit. A default here would be a claim nobody made."""
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    return Descriptor(
        plane="policy",
        name=name,
        version=version,
        contract=POLICY_CONTRACT,
        provides={
            CAP_CHUNK: chunk,
            CAP_DETERMINISTIC: deterministic,
            CAP_STATEFUL: stateful,
        },
        isolation=isolation,
        metadata=metadata,
    )


class Policy(ABC):
    """Observations in, action chunks out."""

    @abstractmethod
    def descriptor(self) -> Descriptor:
        """Identity and capabilities, read before anything is asked of it."""

    @abstractmethod
    def action_spec(self) -> ChannelSpec:
        """What this policy emits, described as fully as any other channel."""

    @abstractmethod
    def observes(self) -> Requirement:
        """Everything ``act`` will read."""

    @abstractmethod
    def act(self, observation: Observation) -> np.ndarray:
        """A chunk of at least one action, shaped ``(horizon, *action shape)``."""

    # -- provided ----------------------------------------------------------

    def reset(self, context: EpisodeContext) -> None:
        """Start a new episode. Stateless policies need not override this."""

    def close(self) -> None:
        """Release anything held. Default: nothing."""

    @property
    def name(self) -> str:
        return self.descriptor().name

    @property
    def chunk(self) -> int:
        return int(self.descriptor().provides.get(CAP_CHUNK, 1))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.descriptor().ref}>"
