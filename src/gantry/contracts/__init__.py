"""The plane interfaces.

Each plane gets one contract, semver'd independently, defining an abstract
class and the capability vocabulary that goes with it. Contracts own their
vocabulary so that adding a plane, or a key to a plane, never edits the spine.

A contract's docstring states its invariants in prose, one per sentence, and
the matching module in :mod:`gantry.conformance` enforces exactly those
sentences. If the two ever disagree, the prose is the specification and the
kit is the bug.
"""

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
from .curation import (
    CAP_NEEDS_GRADIENTS,
    CAP_NEEDS_RUNS,
    CAP_PER_EPISODE,
    CAP_RUNG,
    CURATION_CONTRACT,
    RUNGS,
    CollectionOrder,
    CurationAction,
    CurationOutcome,
    CurationPlan,
    Curator,
    Prediction,
    curator_descriptor,
)
from .embodiment import (
    CAP_RESETTABLE,
    CAP_SIMULATED,
    EMBODIMENT_CONTRACT,
    EmbodimentDescriptor,
    Retargeter,
    embodiment_from_channels,
)
from .embodiment import (
    CAP_SEEDABLE as CAP_EMBODIMENT_SEEDABLE,
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
from .scorer import (
    CAP_ABSTAINS,
    CAP_COST,
    CAP_EVIDENCE,
    SCORER_CONTRACT,
    Evidence,
    Judgement,
    Scorer,
    scorer_descriptor,
)

__all__ = [
    "CAPABILITIES",
    "CAP_ABSTAINS",
    "CAP_CHUNK",
    "CAP_CLOSED_LOOP",
    "CAP_COST",
    "CAP_DETERMINISTIC",
    "CAP_EMBODIMENT_SEEDABLE",
    "CAP_EVIDENCE",
    "CAP_LAZY",
    "CAP_MEDIA",
    "CAP_MIN_COHORTS",
    "CAP_NEEDS_GRADIENTS",
    "CAP_NEEDS_RUNS",
    "CAP_OUTCOMES",
    "CAP_PER_EPISODE",
    "CAP_PRESCRIBES",
    "CAP_RESETTABLE",
    "CAP_RUNG",
    "CAP_SIMULATED",
    "CAP_STAGE_EVENTS",
    "CAP_STATEFUL",
    "CONNECTOR_CONTRACT",
    "CURATION_CONTRACT",
    "Cohort",
    "CollectionOrder",
    "Connector",
    "CurationAction",
    "CurationOutcome",
    "CurationPlan",
    "Curator",
    "EMBODIMENT_CONTRACT",
    "EVALUATOR_CONTRACT",
    "EmbodimentDescriptor",
    "EpisodeContext",
    "Evaluator",
    "Evidence",
    "FEEDBACK_CONTRACT",
    "FeedbackModule",
    "Finding",
    "Judgement",
    "Observation",
    "POLICY_CONTRACT",
    "Policy",
    "Prediction",
    "Protocol",
    "RUNGS",
    "Report",
    "Retargeter",
    "SCORER_CONTRACT",
    "SEVERITIES",
    "Scene",
    "Scorer",
    "TaskSpec",
    "connector_descriptor",
    "curator_descriptor",
    "embodiment_from_channels",
    "evaluator_descriptor",
    "feedback_descriptor",
    "policy_descriptor",
    "scorer_descriptor",
]
