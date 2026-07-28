"""Does this evaluator honour the evaluator contract?

The capability checks matter most here. An evaluator that overstates what it
can report causes an analysis to run on data that cannot support it, and the
output of that analysis looks exactly like a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..contracts.evaluator import (
    CAP_CLOSED_LOOP,
    CAP_OUTCOMES,
    CAP_SEEDABLE,
    CAP_STAGE_EVENTS,
    EVALUATOR_CONTRACT,
    Evaluator,
    Protocol,
    TaskSpec,
)
from ..spine import ContractVersion, RunRecord, Verdict


@dataclass(frozen=True)
class Context:
    evaluator: Evaluator
    policy: Any
    task: TaskSpec
    protocol: Protocol
    record: RunRecord
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _plane(context: Context) -> Verdict:
    descriptor = context.evaluator.descriptor()
    if descriptor.plane != "evaluation":
        return Verdict.no(
            "conformance.plane",
            f"{descriptor.ref} declares plane {descriptor.plane!r}, expected 'evaluation'",
        )
    return Verdict.yes()


def _contract(context: Context) -> Verdict:
    return context.evaluator.descriptor().contract_version.satisfies(
        ContractVersion.parse(EVALUATOR_CONTRACT)
    )


def _record_validates(context: Context) -> Verdict:
    return context.record.validate(deep=True)


def _one_episode_per_trial(context: Context) -> Verdict:
    expected = len(context.task) * context.protocol.epochs
    if context.protocol.max_trials is not None:
        expected = min(expected, context.protocol.max_trials)
    if len(context.record) != expected:
        return Verdict.no(
            "conformance.trial_count",
            f"{len(context.task)} scene(s) x {context.protocol.epochs} epoch(s) "
            f"should give {expected} episode(s), got {len(context.record)}",
            hint="one episode per attempt; collapsing epochs discards the variation "
            "an evaluation exists to measure",
        )
    return Verdict.yes()


def _provenance_is_complete(context: Context) -> Verdict:
    provenance = context.record.provenance
    checks = []
    for plane in ("policy", "evaluation"):
        if provenance.component(plane) is None:
            checks.append(
                Verdict.no(
                    "conformance.provenance",
                    f"the record names no {plane} component",
                    hint="a number that does not know what produced it is not a result",
                )
            )
    if not provenance.protocol:
        checks.append(
            Verdict.no(
                "conformance.protocol_missing",
                "the record carries no protocol",
                hint="execution settings change results and belong in the record",
            )
        )
    return Verdict.all(checks)


def _capability(cap: str, present: Callable[[RunRecord], bool]) -> Callable[[Context], Verdict]:
    def check(context: Context) -> Verdict:
        declared = bool(context.evaluator.descriptor().provides.get(cap))
        observed = present(context.record)
        if observed and not declared:
            return Verdict.no(
                "conformance.capability_understated",
                f"declares {cap}=False but the record has it",
                hint="the resolver will refuse analyses this evaluator could feed",
                capability=cap,
            )
        if declared and not observed:
            return Verdict.no(
                "conformance.capability_overstated",
                f"declares {cap}=True but the record does not have it",
                hint="an analysis will run on data that cannot support it, and the "
                "result will look like any other",
                capability=cap,
            )
        return Verdict.yes()

    return check


def _seeding_is_honest(context: Context) -> Verdict:
    if not context.evaluator.descriptor().provides.get(CAP_SEEDABLE):
        return Verdict.yes()
    again = context.evaluator.evaluate(context.policy, context.task, context.protocol)
    mine = [episode.labels.success for episode in context.record.episodes]
    theirs = [episode.labels.success for episode in again.episodes]
    if mine != theirs:
        return Verdict.no(
            "conformance.not_seedable",
            "declares seedable but two runs at the same protocol disagreed",
            hint="paired comparison rests on this",
        )
    return Verdict.yes()


def _describes_its_own_task(context: Context) -> Verdict:
    """It can build a task, and the task it builds actually runs.

    This is what separates conforming from usable. A runner has no other way to
    start a run, so an evaluator that cannot name its own scenes is a plugin
    nothing can drive.
    """
    try:
        task = context.evaluator.task_for("conformance")
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.task_for_raised",
            f"task_for() raised {type(error).__name__}: {error}",
        )
    checks = [task.validate()]
    if not task.scenes:
        checks.append(
            Verdict.no(
                "conformance.task_empty",
                "task_for() returned a task with no scenes, so nothing could be run",
            )
        )
        return Verdict.all(checks)
    try:
        record = context.evaluator.evaluate(context.policy, task, Protocol(max_trials=1))
    except Exception as error:  # noqa: BLE001
        checks.append(
            Verdict.no(
                "conformance.own_task_unrunnable",
                f"the evaluator's own task could not be run: {type(error).__name__}: {error}",
                hint="a task it describes and cannot execute is worse than none",
            )
        )
    else:
        if len(record) < 1:
            checks.append(
                Verdict.no(
                    "conformance.own_task_empty",
                    "running the evaluator's own task produced no episodes",
                )
            )
    return Verdict.all(checks)


def _refuses_a_broken_task(context: Context) -> Verdict:
    from ..spine import IncompatibleError

    empty = TaskSpec(name="conformance-empty", scenes=(), horizon=10)
    try:
        context.evaluator.evaluate(context.policy, empty, context.protocol)
    except IncompatibleError:
        return Verdict.yes()
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.bad_task_wrong_error",
            f"a task with no scenes raised {type(error).__name__}, expected a refusal",
        )
    return Verdict.no(
        "conformance.bad_task_accepted",
        "a task with no scenes produced a run instead of a refusal",
    )


CHECKS: tuple[Check, ...] = (
    Check("plane", "declares the evaluation plane", _plane),
    Check("contract", "implements a compatible evaluator contract", _contract),
    Check("record_valid", "the returned record validates", _record_validates),
    Check("trial_count", "one episode per attempt", _one_episode_per_trial),
    Check("provenance", "the record names its components and protocol", _provenance_is_complete),
    Check(
        "capability_stage_events",
        "the stage_events claim matches the record",
        _capability(CAP_STAGE_EVENTS, lambda r: r.has_stage_events),
    ),
    Check(
        "capability_outcomes",
        "the outcomes claim matches the record",
        _capability(CAP_OUTCOMES, lambda r: r.has_outcomes),
    ),
    Check("seedable", "a seedable claim is true", _seeding_is_honest),
    Check("task_for", "it can describe a task it can run", _describes_its_own_task),
    Check("bad_task", "a task with no scenes is refused", _refuses_a_broken_task),
)


def evaluator_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_evaluator(
    evaluator: Evaluator,
    policy: Any,
    task: TaskSpec,
    protocol: Protocol | None = None,
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit by actually evaluating something with it."""
    protocol = protocol or Protocol()
    try:
        record = evaluator.evaluate(policy, task, protocol)
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.check_raised",
            f"evaluating raised {type(error).__name__}: {error}",
            hint="the kit runs a real evaluation; it must complete before anything is checked",
        )
    context = Context(evaluator, policy, task, protocol, record, strict)
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
