"""Planning: which components exist, whether they fit, and at what cost.

The resolver works from descriptors and schemas only. It imports no plugin to
decide whether a plugin fits, so an unrunnable combination is refused in
milliseconds rather than after a model finishes loading — and the refusal names
the seam, the code, and what would close it.

Two kinds of transform, kept apart on purpose. An **adapter** closes a gap with
one correct answer and is found by refusal code. A **retargeter** closes a
structural gap that needs a judgement about what to discard, and is found by
asking each one about the specific pair. Both produce a :class:`Step`, which is
what actually runs — and a step remembers which kind it came from, so nothing is
disguised as anything else.
"""

from .adapters import ADAPTER_CONTRACT, Adapter, AdapterRegistry, SupportsAdaptation
from .apply import AdaptedSource, adapt_all, adapt_episode, changes_length
from .plan import ChannelBinding, Plan, ResolvedComponent, Resolution, Wiring
from .registry import (
    ENTRY_POINT_GROUPS,
    DuplicateRegistration,
    Registration,
    Registry,
)
from .requirement import Requirement, requires_channels
from .resolver import (
    STRUCTURAL,
    UNADAPTABLE,
    Attempt,
    Candidate,
    bind,
    bind_channel,
    candidates,
    check_capabilities,
    resolve,
)
from .retarget import RETARGETABLE, RetargeterRegistry
from .transform import ADAPTER, RETARGETER, Chain, Step, chain_of

__all__ = [
    "ADAPTER",
    "ADAPTER_CONTRACT",
    "ENTRY_POINT_GROUPS",
    "RETARGETABLE",
    "RETARGETER",
    "STRUCTURAL",
    "UNADAPTABLE",
    "AdaptedSource",
    "Adapter",
    "AdapterRegistry",
    "Attempt",
    "Candidate",
    "Chain",
    "ChannelBinding",
    "DuplicateRegistration",
    "Plan",
    "Registration",
    "Registry",
    "Requirement",
    "ResolvedComponent",
    "Resolution",
    "RetargeterRegistry",
    "Step",
    "SupportsAdaptation",
    "Wiring",
    "adapt_all",
    "adapt_episode",
    "bind",
    "bind_channel",
    "candidates",
    "chain_of",
    "changes_length",
    "check_capabilities",
    "requires_channels",
    "resolve",
]
