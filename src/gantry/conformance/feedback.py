"""Does this feedback module honour the feedback contract?

One check per invariant sentence in :mod:`gantry.contracts.feedback`.

Two of these matter more than the rest. **Purity**, because an analysis that
returns something different the second time cannot be reviewed, argued with, or
reproduced -- and the drift is invisible unless someone runs it twice on purpose.
And **refusing below the cohorts it needs**, because the failure mode there is
not an error: it is a confident, well-formatted answer to a question nobody
asked.

The kit also insists that a prescription carries evidence. A module is entitled
to say "collect more of this"; it is not entitled to say it without a number
behind it, because that is the sentence someone spends a collection budget on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Sequence

from ..contracts.feedback import CAP_MIN_COHORTS, FEEDBACK_CONTRACT, Cohort, FeedbackModule, Report
from ..spine import ContractVersion, IncompatibleError, Measurement, Verdict


@dataclass(frozen=True)
class Context:
    module: FeedbackModule
    cohorts: tuple[Cohort, ...]
    report: Report
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _plane(context: Context) -> Verdict:
    descriptor = context.module.descriptor()
    if descriptor.plane != "feedback":
        return Verdict.no(
            "conformance.plane",
            f"{descriptor.ref} declares plane {descriptor.plane!r}, expected 'feedback'",
        )
    return Verdict.yes()


def _contract(context: Context) -> Verdict:
    return context.module.descriptor().contract_version.satisfies(
        ContractVersion.parse(FEEDBACK_CONTRACT)
    )


def _requirement_is_valid(context: Context) -> Verdict:
    requirement = context.module.requirement()
    checks = [spec.validate() for spec in requirement.channels]
    if requirement.plane != "feedback":
        checks.append(
            Verdict.no(
                "conformance.requirement_plane",
                f"{context.module.name} declares a requirement on plane {requirement.plane!r}",
            )
        )
    return Verdict.all(checks)


def _is_pure(context: Context) -> Verdict:
    """Same cohorts, same report. Twice."""
    again = context.module.run(list(context.cohorts))
    if json.dumps(again.as_dict(), sort_keys=True, default=str) != json.dumps(
        context.report.as_dict(), sort_keys=True, default=str
    ):
        return Verdict.no(
            "conformance.not_pure",
            f"{context.module.name} returned a different report for the same cohorts",
            hint="an analysis that drifts cannot be reviewed or reproduced; seed any "
            "resampling it does",
        )
    return Verdict.yes()


def _numbers_are_measurements(context: Context) -> Verdict:
    checks = []
    for name, value in context.report.measurements.items():
        if not isinstance(value, Measurement):
            checks.append(
                Verdict.no(
                    "conformance.bare_number",
                    f"{context.module.name}: measurement {name!r} is a "
                    f"{type(value).__name__}, not a Measurement",
                    hint="a number without its n and interval is not a result",
                )
            )
    for finding in context.report.findings:
        for name, value in finding.measurements.items():
            if not isinstance(value, Measurement):
                checks.append(
                    Verdict.no(
                        "conformance.bare_number",
                        f"{context.module.name}: finding {finding.code!r} carries a bare "
                        f"number for {name!r}",
                    )
                )
    return Verdict.all(checks)


def _measurements_carry_n(context: Context) -> Verdict:
    missing = [name for name, value in context.report.measurements.items() if value.n is None]
    if missing and context.strict:
        return Verdict.no(
            "conformance.missing_n",
            f"{context.module.name}: measurement(s) {missing} do not say how many "
            "observations they rest on",
        )
    if missing:
        return Verdict.note(
            "conformance.missing_n",
            f"{context.module.name}: measurement(s) {missing} carry no n",
        )
    return Verdict.yes()


def _prescriptions_have_evidence(context: Context) -> Verdict:
    checks = []
    for finding in context.report.findings:
        if finding.prescription and not (finding.measurements or finding.evidence):
            checks.append(
                Verdict.no(
                    "conformance.unevidenced_prescription",
                    f"{context.module.name}: finding {finding.code!r} prescribes an action "
                    "with no measurement or evidence behind it",
                    hint="this is the sentence someone spends a collection budget on",
                )
            )
    return Verdict.all(checks)


def _refuses_below_its_minimum(context: Context) -> Verdict:
    needed = int(context.module.descriptor().provides.get(CAP_MIN_COHORTS, 1))
    if needed <= 1:
        return Verdict.yes()
    too_few = list(context.cohorts[: needed - 1])
    try:
        context.module.run(too_few)
    except IncompatibleError:
        return Verdict.yes()
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.wrong_refusal",
            f"given {len(too_few)} cohort(s) it raised {type(error).__name__}, expected a refusal",
        )
    return Verdict.no(
        "conformance.answered_anyway",
        f"{context.module.name} declares min_cohorts={needed} but answered with {len(too_few)}",
        hint="a confident answer to a question that was not askable is worse than an error",
    )


def _refuses_an_empty_cohort(context: Context) -> Verdict:
    try:
        context.module.run([Cohort("conformance-empty", ())] * max(context.module.min_cohorts(), 1))
    except IncompatibleError:
        return Verdict.yes()
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.wrong_refusal",
            f"an empty cohort raised {type(error).__name__}, expected a refusal",
        )
    return Verdict.no(
        "conformance.empty_accepted",
        "an empty cohort produced a report instead of a refusal",
    )


def _report_names_itself(context: Context) -> Verdict:
    checks = []
    if context.report.module != context.module.name:
        checks.append(
            Verdict.no(
                "conformance.report_name",
                f"report says module={context.report.module!r} but the module is "
                f"{context.module.name!r}",
            )
        )
    given = {cohort.name for cohort in context.cohorts}
    if set(context.report.cohorts) != given:
        checks.append(
            Verdict.no(
                "conformance.report_cohorts",
                f"report names cohorts {sorted(context.report.cohorts)}, was given {sorted(given)}",
                hint="a comparative result is unreadable without knowing what was compared",
            )
        )
    return Verdict.all(checks)


CHECKS: tuple[Check, ...] = (
    Check("plane", "declares the feedback plane", _plane),
    Check("contract", "implements a compatible feedback contract", _contract),
    Check("requirement", "its declared requirement is well-formed", _requirement_is_valid),
    Check("pure", "the same cohorts give the same report", _is_pure),
    Check("measurements", "every number is a Measurement", _numbers_are_measurements),
    Check("n", "measurements say how many observations they rest on", _measurements_carry_n),
    Check("evidence", "every prescription carries evidence", _prescriptions_have_evidence),
    Check("min_cohorts", "it refuses below the cohorts it needs", _refuses_below_its_minimum),
    Check("empty_cohort", "an empty cohort is refused", _refuses_an_empty_cohort),
    Check("report_identity", "the report names its module and cohorts", _report_names_itself),
)


def feedback_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_feedback(
    module: FeedbackModule,
    cohorts: Sequence[Cohort],
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit by actually analysing something with the module."""
    needed = module.min_cohorts()
    if len(cohorts) < needed:
        return Verdict.no(
            "conformance.too_few_for_the_kit",
            f"{module.name} needs {needed} cohort(s) to be exercised, got {len(cohorts)}",
        )
    try:
        report = module.run(list(cohorts))
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.check_raised",
            f"analysing raised {type(error).__name__}: {error}",
            hint="the kit runs a real analysis; it must complete before anything is checked",
        )
    context = Context(module, tuple(cohorts), report, strict)
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
