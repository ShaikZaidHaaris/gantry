"""Contract enforcement, shipped inside core so plugins can hold themselves to it.

A third party writing a connector runs exactly the same kit the first-party ones
run, and inherits new checks when they upgrade. That is deliberate: the
alternative is a first-party tier that is trusted and a community tier that is
not, which is how plugin ecosystems rot.

Every plane has a kit. That completeness matters more than it sounds: an
unguarded plane is where a plugin can claim anything, and the claims are what
the resolver plans from.

Kits return a :class:`~gantry.spine.verdict.Verdict` and import no test
framework, so they run from pytest, from CI, or from the command line.
"""

from .adapter import adapter_checks, check_adapter
from .connector import Check, check_connector, connector_checks
from .curation import check_curator, curation_checks
from .embodiment import check_embodiment, check_retargeter, embodiment_checks
from .evaluator import check_evaluator, evaluator_checks
from .feedback import check_feedback, feedback_checks
from .policy import check_policy, policy_checks
from .scorer import check_scorer, scorer_checks

#: Every kit, by the name the CLI knows it as.
KITS = {
    "connector": connector_checks,
    "curation": curation_checks,
    "policy": policy_checks,
    "scorer": scorer_checks,
    "evaluator": evaluator_checks,
    "feedback": feedback_checks,
    "embodiment": embodiment_checks,
    "adapter": adapter_checks,
}

__all__ = [
    "KITS",
    "Check",
    "adapter_checks",
    "check_adapter",
    "check_connector",
    "check_curator",
    "check_embodiment",
    "check_evaluator",
    "check_feedback",
    "check_policy",
    "check_scorer",
    "check_retargeter",
    "connector_checks",
    "curation_checks",
    "embodiment_checks",
    "evaluator_checks",
    "feedback_checks",
    "policy_checks",
    "scorer_checks",
]
