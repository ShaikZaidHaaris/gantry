"""Does this curator honour the curation contract?

One check per invariant sentence in :mod:`gantry.contracts.curation`.

Three of these carry most of the weight.

**Determinism**, because a curator that proposes a different set of episodes on
a second reading of the same data cannot be reviewed and cannot be verified —
the plan that was tested is not the plan that would be applied.

**Actionability**, because a plan naming episodes the dataset does not contain
is not a plan. It is the most common way a curation pipeline silently does
nothing: the identifiers drift during a format conversion and every drop is a
no-op.

**Falsifiability**, because this is the one plane that tells you to spend money,
and the whole design rests on a plan being able to turn out wrong. A curator
that proposes without predicting has produced advice, and advice does not get
verified, which means it does not get better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..contracts.curation import (
    CAP_NEEDS_RUNS,
    CAP_RUNG,
    CURATION_CONTRACT,
    RUNGS,
    CurationPlan,
    Curator,
)
from ..spine import ContractVersion, Verdict


@dataclass(frozen=True)
class Context:
    curator: Curator
    episodes: tuple[Any, ...]
    runs: tuple[Any, ...]
    plan: CurationPlan
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _plane(context: Context) -> Verdict:
    descriptor = context.curator.descriptor()
    if descriptor.plane != "curation":
        return Verdict.no(
            "conformance.plane",
            f"{descriptor.ref} declares plane {descriptor.plane!r}, expected 'curation'",
        )
    return Verdict.yes()


def _contract(context: Context) -> Verdict:
    declared = context.curator.descriptor().contract
    if not declared:
        return Verdict.no(
            "conformance.no_contract",
            f"{context.curator.name} declares no contract version",
        )
    try:
        return ContractVersion.parse(declared).satisfies(ContractVersion.parse(CURATION_CONTRACT))
    except Exception as error:  # noqa: BLE001
        return Verdict.no("conformance.contract_unparseable", f"{declared!r}: {error}")


def _rung_declared(context: Context) -> Verdict:
    """The descriptor's rung and the plan's rung are the same claim."""
    declared = context.curator.descriptor().provides.get(CAP_RUNG)
    if declared not in RUNGS:
        return Verdict.no(
            "conformance.rung",
            f"{context.curator.name} declares rung {declared!r}, expected one of {list(RUNGS)}",
        )
    if context.plan.rung != declared:
        return Verdict.no(
            "conformance.rung_disagrees",
            f"{context.curator.name} declares rung {declared!r} but proposed a plan "
            f"at {context.plan.rung!r}",
            hint="the descriptor is what a resolver plans from; it has to be true",
        )
    return Verdict.yes()


def _plan_valid(context: Context) -> Verdict:
    return context.plan.validate()


def _falsifiable(context: Context) -> Verdict:
    """A plan predicts something a result could contradict."""
    prediction = context.plan.predicted
    if not context.plan.proposes_a_change:
        return Verdict.yes()  # nothing proposed, nothing to falsify
    if prediction.magnitude <= 0:
        return Verdict.no(
            "conformance.unfalsifiable",
            f"{context.curator.name} proposes {context.plan.summary()} without "
            "predicting how much it will help",
            hint="this plane tells people to spend money; the claim has to be "
            "capable of being wrong",
        )
    return Verdict.yes()


def _actionable(context: Context) -> Verdict:
    """Every episode a plan names exists in the data it was given."""
    known = set()
    for episode in context.episodes:
        meta = getattr(episode, "meta", None)
        uid = getattr(meta, "uid", None) or getattr(episode, "uid", None)
        if uid:
            known.add(str(uid))
    if not known:
        return Verdict.note(
            "conformance.no_identity",
            "the episodes given to the kit expose no uid, so actionability was not checked",
        )
    named = set(context.plan.drops) | set(context.plan.keeps)
    missing = sorted(named - known)
    if missing:
        return Verdict.no(
            "conformance.unactionable",
            f"{context.curator.name} names {len(missing)} episode(s) absent from the "
            f"data it read, e.g. {missing[0]!r}",
            hint="identifiers usually drift during a format conversion; a plan that "
            "names nothing real applies cleanly and does nothing",
        )
    return Verdict.yes()


def _deterministic(context: Context) -> Verdict:
    """Reading the same data twice proposes the same thing."""
    again = context.curator.propose(list(context.episodes), list(context.runs))
    if set(again.drops) != set(context.plan.drops):
        return Verdict.no(
            "conformance.nondeterministic",
            f"{context.curator.name} dropped {len(context.plan.drops)} episode(s) "
            f"then {len(again.drops)} on the same data",
            hint="a plan that changes between readings cannot be verified, because "
            "what was tested is not what would be applied",
        )
    if again.weights != context.plan.weights:
        return Verdict.no(
            "conformance.nondeterministic_weights",
            f"{context.curator.name} proposed different weights on the same data",
        )
    return Verdict.yes()


def _refuses_empty(context: Context) -> Verdict:
    """Given nothing, it refuses rather than proposing something."""
    verdict = context.curator.check_inputs([], list(context.runs))
    if verdict.ok:
        return Verdict.no(
            "conformance.accepts_empty",
            f"{context.curator.name} accepts an empty dataset",
            hint="a plan derived from no data is a plan about nothing",
        )
    return Verdict.yes()


def _runs_declared(context: Context) -> Verdict:
    """A curator needing rollouts says so, and refuses without them."""
    needs = bool(context.curator.descriptor().provides.get(CAP_NEEDS_RUNS, False))
    if not needs:
        return Verdict.yes()
    verdict = context.curator.check_inputs(list(context.episodes), [])
    if verdict.ok:
        return Verdict.no(
            "conformance.runs_not_enforced",
            f"{context.curator.name} declares it needs evaluation runs but accepts "
            "being given none",
        )
    return Verdict.yes()


def _evidence_named(context: Context) -> Verdict:
    """A curator reading rollouts names the scenes it read.

    This is what lets verification refuse to score a plan on the scenes that
    produced it. Nothing else in the pipeline can reconstruct the list.
    """
    from ..contracts.curation import ROLLOUT_RUNGS

    if context.plan.rung not in ROLLOUT_RUNGS:
        return Verdict.yes()
    if not context.plan.evidence_seeds:
        return Verdict.no(
            "conformance.evidence_unnamed",
            f"{context.curator.name} argues at the {context.plan.rung!r} rung but "
            "names none of the scenes it read",
        )
    return Verdict.yes()


CHECKS = (
    Check("plane", "the descriptor declares the curation plane", _plane),
    Check("contract", "the declared contract is compatible", _contract),
    Check("rung", "the declared rung is real and matches the plan", _rung_declared),
    Check("plan_valid", "the proposed plan validates", _plan_valid),
    Check("falsifiable", "the plan predicts something that could be wrong", _falsifiable),
    Check("actionable", "every episode named exists in the data read", _actionable),
    Check("deterministic", "the same data proposes the same plan", _deterministic),
    Check("refuses_empty", "an empty dataset is refused, not curated", _refuses_empty),
    Check("runs_declared", "a curator needing rollouts enforces it", _runs_declared),
    Check("evidence_named", "a rollout-reading curator names the scenes", _evidence_named),
)


def curation_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_curator(
    curator: Curator,
    episodes: Sequence[Any],
    runs: Sequence[Any] = (),
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit by actually proposing a plan with the curator."""
    try:
        plan = curator.propose(list(episodes), list(runs))
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.check_raised",
            f"proposing raised {type(error).__name__}: {error}",
            hint="the kit proposes a real plan; it must complete before anything is checked",
        )
    context = Context(curator, tuple(episodes), tuple(runs), plan, strict)
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
