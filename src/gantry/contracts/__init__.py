"""The plane interfaces.

Each plane gets one contract, semver'd independently, defining an abstract
class and the capability vocabulary that goes with it. Contracts own their
vocabulary so that adding a plane, or a key to a plane, never edits the spine.

A contract's docstring states its invariants in prose, one per sentence, and
the matching module in :mod:`gantry.conformance` enforces exactly those
sentences. If the two ever disagree, the prose is the specification and the
kit is the bug.
"""

from .embodiment import (
    CAP_RESETTABLE,
    CAP_SEEDABLE as CAP_EMBODIMENT_SEEDABLE,
    CAP_SIMULATED,
    EMBODIMENT_CONTRACT,
    EmbodimentDescriptor,
    Retargeter,
    embodiment_from_channels,
)
from .evaluator import (
    CAP_CLOSED_LOOP,
    EVALUATOR_CONTRACT,
    Evaluator,
    Protocol,
    Scene,
    TaskSpec,
    evaluator_descriptor,
)
from .policy import (
    CAP_CHUNK,
    CAP_DETERMINISTIC,
    CAP_STATEFUL,
    POLICY_CONTRACT,
    EpisodeContext,
    Observation,
    Policy,
    policy_descriptor,
)
from .feedback import (
    CAP_MIN_COHORTS,
    CAP_PRESCRIBES,
    FEEDBACK_CONTRACT,
    SEVERITIES,
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from .connector import (
    CAP_LAZY,
    CAP_MEDIA,
    CAP_OUTCOMES,
    CAP_STAGE_EVENTS,
    CAPABILITIES,
    CONNECTOR_CONTRACT,
    Connector,
    connector_descriptor,
)

__all__ = [
    "CAPABILITIES",
    "CAP_CHUNK",
    "CAP_CLOSED_LOOP",
    "CAP_DETERMINISTIC",
    "CAP_EMBODIMENT_SEEDABLE",
    "CAP_RESETTABLE",
    "CAP_SIMULATED",
    "CAP_STATEFUL",
    "EMBODIMENT_CONTRACT",
    "EVALUATOR_CONTRACT",
    "POLICY_CONTRACT",
    "EmbodimentDescriptor",
    "EpisodeContext",
    "Evaluator",
    "Observation",
    "Policy",
    "Protocol",
    "Retargeter",
    "Scene",
    "TaskSpec",
    "embodiment_from_channels",
    "evaluator_descriptor",
    "policy_descriptor",
    "CAP_MIN_COHORTS",
    "CAP_PRESCRIBES",
    "FEEDBACK_CONTRACT",
    "SEVERITIES",
    "Cohort",
    "FeedbackModule",
    "Finding",
    "Report",
    "feedback_descriptor",
    "CAP_LAZY",
    "CAP_MEDIA",
    "CAP_OUTCOMES",
    "CAP_STAGE_EVENTS",
    "CONNECTOR_CONTRACT",
    "Connector",
    "connector_descriptor",
]
