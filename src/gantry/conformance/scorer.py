"""Does this scorer honour the scorer contract?

One check per invariant sentence in :mod:`gantry.contracts.scorer`.

Three of these carry the weight.

**Evidence honesty.** A scorer that accepts evidence it declared it needs and
did not get will produce labels indistinguishable from real ones. That is worse
than crashing, because a crash gets fixed and a plausible label gets averaged.

**Determinism where claimed.** A predicate that says it is deterministic and is
not makes every agreement number computed against it meaningless — you cannot
measure whether a person agrees with something that changes its mind.

**Abstention where claimed.** A scorer that declares it may abstain and then
never does is either lucky or converting uncertainty into verdicts, and the kit
cannot tell which from one trial. So it is checked structurally: given evidence
that settles nothing, a scorer with ``abstains`` must return ``None`` rather
than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..contracts.scorer import (
    CAP_ABSTAINS,
    CAP_COST,
    CAP_EVIDENCE,
    COSTS,
    SCORER_CONTRACT,
    Evidence,
    Judgement,
    Scorer,
)
from ..spine import ContractVersion, Verdict


@dataclass(frozen=True)
class Context:
    scorer: Scorer
    evidence: Evidence
    task: Any
    judgements: tuple[Judgement, ...]
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _plane(context: Context) -> Verdict:
    descriptor = context.scorer.descriptor()
    if descriptor.plane != "scorer":
        return Verdict.no(
            "conformance.plane",
            f"{descriptor.ref} declares plane {descriptor.plane!r}, expected 'scorer'",
        )
    return Verdict.yes()


def _contract(context: Context) -> Verdict:
    declared = context.scorer.descriptor().contract
    if not declared:
        return Verdict.no(
            "conformance.no_contract", f"{context.scorer.name} declares no contract"
        )
    try:
        return ContractVersion.parse(declared).satisfies(
            ContractVersion.parse(SCORER_CONTRACT)
        )
    except Exception as error:  # noqa: BLE001
        return Verdict.no("conformance.contract_unparseable", f"{declared!r}: {error}")


def _capabilities_well_formed(context: Context) -> Verdict:
    provides = context.scorer.descriptor().provides
    checks = []
    if not provides.get(CAP_EVIDENCE):
        checks.append(
            Verdict.no(
                "conformance.no_evidence_declared",
                f"{context.scorer.name} does not say what it needs to see",
                hint="a scorer that reads whatever it is handed can depend on "
                "evidence a real bench cannot supply",
            )
        )
    if provides.get(CAP_COST) not in COSTS:
        checks.append(
            Verdict.no(
                "conformance.cost",
                f"{context.scorer.name} declares cost {provides.get(CAP_COST)!r}, "
                f"expected one of {list(COSTS)}",
            )
        )
    return Verdict.all(checks)


def _one_judgement_per_criterion(context: Context) -> Verdict:
    """The task's criteria, in the task's order."""
    expected = tuple(c.check for c in getattr(context.task, "success", ()) or ())
    got = tuple(j.criterion for j in context.judgements)
    if got != expected:
        return Verdict.no(
            "conformance.criteria_mismatch",
            f"{context.scorer.name} judged {list(got)}; the task declares "
            f"{list(expected)}",
            hint="one judgement per criterion, in order, so a caller can line them "
            "up without matching on names",
        )
    return Verdict.yes()


def _refuses_evidence_it_cannot_read(context: Context) -> Verdict:
    """Handed nothing, it declines rather than judging."""
    bare = Evidence(trial="conformance-bare")
    verdict = context.scorer.check_evidence(bare)
    needs_something = bool(context.scorer.needs)
    if needs_something and verdict.ok:
        return Verdict.no(
            "conformance.accepts_missing_evidence",
            f"{context.scorer.name} needs {list(context.scorer.needs)} and accepted "
            "evidence carrying none of it",
        )
    return Verdict.yes()


def _deterministic_if_claimed(context: Context) -> Verdict:
    if not context.scorer.deterministic:
        return Verdict.yes()
    again = context.scorer.judge(context.evidence, context.task)
    before = [(j.criterion, j.passed) for j in context.judgements]
    after = [(j.criterion, j.passed) for j in again]
    if before != after:
        return Verdict.no(
            "conformance.nondeterministic",
            f"{context.scorer.name} declares determinism and judged the same "
            f"evidence differently: {before} then {after}",
            hint="agreement with another judge cannot be measured against "
            "something that changes its mind",
        )
    return Verdict.yes()


def _abstains_when_it_cannot_tell(context: Context) -> Verdict:
    """A scorer claiming abstention must actually abstain on empty evidence.

    Structural rather than statistical: the kit gives it a state that settles
    nothing and requires ``None`` back. A scorer that returns ``False`` there is
    converting "I do not know" into "it failed", which is how uncertainty
    becomes evidence.
    """
    provides = context.scorer.descriptor().provides
    if not provides.get(CAP_ABSTAINS, False):
        return Verdict.yes()
    empty = Evidence(trial="conformance-empty", final_state={}, video="/dev/null")
    try:
        judgements = context.scorer.judge(empty, context.task)
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.abstain_raised",
            f"{context.scorer.name} raised {type(error).__name__} on evidence that "
            "settles nothing, instead of abstaining",
        )
    decided = [j for j in judgements if not j.abstained]
    if decided:
        return Verdict.no(
            "conformance.does_not_abstain",
            f"{context.scorer.name} declares it may abstain but returned a verdict "
            f"for {[j.criterion for j in decided]} from evidence that settles nothing",
        )
    return Verdict.yes()


def _rationale_present(context: Context) -> Verdict:
    """Every judgement says why, so a disagreement is diagnosable.

    A note rather than a refusal: a scorer is entitled to be terse. But two
    judges disagreeing with no stated reason is a dead end, and that is the
    situation this warns about before it happens.
    """
    silent = [j.criterion for j in context.judgements if not j.rationale.strip()]
    if silent:
        return Verdict.note(
            "conformance.silent_judgement",
            f"{context.scorer.name} gave no reason for {silent}",
            hint="a disagreement between two judges is only diagnosable if both "
            "said what they saw",
        )
    return Verdict.yes()


def _overall_treats_abstention_as_unknown(context: Context) -> Verdict:
    """One abstention makes the trial unknown, not failed."""
    mixed = (
        Judgement(criterion="a", passed=True),
        Judgement(criterion="b", passed=None),
    )
    if context.scorer.overall(mixed) is not None:
        return Verdict.no(
            "conformance.abstention_became_a_verdict",
            f"{context.scorer.name} resolved a trial containing an abstention to a "
            "definite answer",
            hint="a scorer that could not tell has not established that the trial "
            "failed; recording it as failed converts uncertainty into evidence",
        )
    return Verdict.yes()


CHECKS = (
    Check("plane", "the descriptor declares the scorer plane", _plane),
    Check("contract", "the declared contract is compatible", _contract),
    Check("capabilities", "evidence and cost are declared and well-formed", _capabilities_well_formed),
    Check("criteria", "one judgement per criterion, in the task's order", _one_judgement_per_criterion),
    Check("evidence", "evidence it cannot read is refused", _refuses_evidence_it_cannot_read),
    Check("deterministic", "determinism, where claimed, holds", _deterministic_if_claimed),
    Check("abstains", "abstention, where claimed, happens", _abstains_when_it_cannot_tell),
    Check("rationale", "judgements say why", _rationale_present),
    Check("abstention_overall", "an abstention leaves the trial unknown", _overall_treats_abstention_as_unknown),
)


def scorer_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_scorer(
    scorer: Scorer,
    evidence: Evidence,
    task: Any,
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit by actually judging something with the scorer."""
    try:
        judgements = scorer.judge(evidence, task)
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.check_raised",
            f"judging raised {type(error).__name__}: {error}",
            hint="the kit judges a real trial; it must complete before anything is checked",
        )
    context = Context(scorer, evidence, task, tuple(judgements), strict)
    selected = [c for c in CHECKS if only is None or c.code in set(only)]
    results = []
    for check in selected:
        try:
            results.append(check.run(context))
        except Exception as error:  # noqa: BLE001
            results.append(
                Verdict.no(
                    "conformance.check_raised",
                    f"check {check.code!r} raised {type(error).__name__}: {error}",
                    check=check.code,
                )
            )
    return Verdict.all(results)
