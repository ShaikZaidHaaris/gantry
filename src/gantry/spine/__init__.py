"""The Gantry spine: the only vocabulary all five planes share.

Nothing here knows about a specific dataset format, robot, policy, simulator,
or metric. That is not an accident of the current implementation — it is the
invariant the whole design rests on, and ``tests/test_neutrality.py`` fails the
build if it is ever broken.

Spine A is :class:`EpisodeRecord` (data at rest, read lazily). Spine B is
:class:`RunRecord` (data in motion, provenance-stamped). A rollout is an
episode, which is why one feedback module can screen a dataset and diagnose an
evaluation without being told which it received.
"""

from .channel import (
    ChannelSpec,
    Kind,
    Semantics,
    compatible,
    get_kind,
    get_semantics,
    known_kinds,
    known_semantics,
    register_kind,
    register_semantics,
)
from .descriptor import ContractVersion, Descriptor
from .episode import (
    ArraySource,
    EmptySource,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
    StageEvent,
    StepSource,
    episode_from_arrays,
    episode_from_labels,
)
from .plane import (
    CORE_PLANES,
    MANY,
    ONE,
    Plane,
    contract_for,
    entry_point_groups,
    get_plane,
    known_planes,
    planes,
    proxyable_planes,
    register_plane,
)
from .provenance import (
    PLANES,
    AdapterStep,
    ComponentRef,
    Measurement,
    Provenance,
    digest_of,
    mcnemar,
    proportion,
    seed_from,
)
from .run import RunRecord, RunSet, describe, runset
from .verdict import IncompatibleError, Reason, Verdict

#: Version of the spine contract itself. Plugins pin against this.
#: 1.1 added ChannelSpec.discriminators: metadata keys a plugin declares to be
#: load-bearing, which core then enforces without knowing what they mean. Added
#: because a quaternion stored scalar-first and one stored scalar-last were
#: indistinguishable to every check the spine had.
SPINE_CONTRACT = "spine@1.2"

__all__ = [
    "SPINE_CONTRACT",
    "CORE_PLANES",
    "MANY",
    "ONE",
    "PLANES",
    "Plane",
    "contract_for",
    "entry_point_groups",
    "get_plane",
    "known_planes",
    "planes",
    "proxyable_planes",
    "register_plane",
    # channels
    "ChannelSpec",
    "Kind",
    "Semantics",
    "compatible",
    "get_kind",
    "get_semantics",
    "known_kinds",
    "known_semantics",
    "register_kind",
    "register_semantics",
    # episodes
    "ArraySource",
    "EmptySource",
    "EpisodeLabels",
    "EpisodeMeta",
    "EpisodeRecord",
    "StageEvent",
    "StepSource",
    "episode_from_arrays",
    "episode_from_labels",
    # runs
    "RunRecord",
    "RunSet",
    "describe",
    "runset",
    # provenance
    "AdapterStep",
    "ComponentRef",
    "Measurement",
    "Provenance",
    "digest_of",
    "mcnemar",
    "proportion",
    "seed_from",
    # descriptors
    "ContractVersion",
    "Descriptor",
    # verdicts
    "IncompatibleError",
    "Reason",
    "Verdict",
]
