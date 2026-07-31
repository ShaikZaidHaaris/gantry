"""Does this embodiment description hold together, and is its retargeter honest?

An embodiment is data, so most of its conformance is whether the description is
coherent enough to check anything against. That sounds weak and is not: a
descriptor with undeclared units or unlabelled dimensions makes every downstream
check vacuous, and a check that cannot fail is worse than no check because it
reads as a pass.

The retargeter half is where the real risk is. A retargeter that discards a
dimension and declares no loss is the single most dangerous plugin shape in the
framework: it produces plausible motion, reports nothing, and its provenance
says the run was clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..contracts.embodiment import EMBODIMENT_CONTRACT, EmbodimentDescriptor, Retargeter
from ..spine import ChannelSpec, ContractVersion, Verdict


@dataclass(frozen=True)
class Context:
    embodiment: EmbodimentDescriptor
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _describes_itself(context: Context) -> Verdict:
    return context.embodiment.validate()


def _contract(context: Context) -> Verdict:
    return context.embodiment.descriptor().contract_version.satisfies(
        ContractVersion.parse(EMBODIMENT_CONTRACT)
    )


def _channels_are_described(context: Context) -> Verdict:
    """Units and frames on everything physical, or checking downstream is theatre."""
    checks = []
    for spec in (*context.embodiment.state, *context.embodiment.action):
        if spec.kind not in {"scalar", "vector", "matrix"}:
            continue
        if spec.units is None and spec.semantics is None:
            message = (
                f"{context.embodiment.ref}: channel {spec.name!r} declares neither "
                "units nor meaning"
            )
            hint = "a channel matched only by name and width is matched by coincidence"
            checks.append(
                Verdict.no("conformance.channel_undescribed", message, hint)
                if context.strict
                else Verdict.note("conformance.channel_undescribed", message, hint)
            )
    return Verdict.all(checks)


def _actions_are_labelled(context: Context) -> Verdict:
    """Actuators want names, so a limit violation can say which actuator."""
    checks = []
    for spec in context.embodiment.action:
        if spec.width > 1 and spec.dim_labels is None:
            message = (
                f"{context.embodiment.ref}: action channel {spec.name!r} is "
                f"{spec.width} wide with no dimension labels"
            )
            hint = "without labels, a fault can only be reported as an index"
            checks.append(
                Verdict.no("conformance.unlabelled_action", message, hint)
                if context.strict
                else Verdict.note("conformance.unlabelled_action", message, hint)
            )
    return Verdict.all(checks)


def _rate_is_declared(context: Context) -> Verdict:
    if context.embodiment.control_hz is None:
        return Verdict.note(
            "conformance.no_control_rate",
            f"{context.embodiment.ref} declares no control rate, so anything reasoning "
            "about time has to guess",
        )
    return Verdict.yes()


CHECKS: tuple[Check, ...] = (
    Check("describes_itself", "the descriptor is internally coherent", _describes_itself),
    Check("contract", "implements a compatible embodiment contract", _contract),
    Check(
        "channels_described", "physical channels declare units or meaning", _channels_are_described
    ),
    Check("actions_labelled", "wide action channels name their dimensions", _actions_are_labelled),
    Check("control_rate", "a control rate is declared", _rate_is_declared),
)


def embodiment_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_embodiment(
    embodiment: EmbodimentDescriptor,
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    context = Context(embodiment, strict)
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


# --------------------------------------------------------------------------
# retargeters
# --------------------------------------------------------------------------


def check_retargeter(
    retargeter: Retargeter,
    source: ChannelSpec,
    target: ChannelSpec,
    *,
    samples: int = 4,
) -> Verdict:
    """Does this retargeter do what it says, and admit what it costs?

    Run against a pair it claims to handle. The important check is the last one:
    a width reduction that declares no loss is a plugin quietly deciding that
    throwing data away does not need mentioning.
    """
    checks: list[Verdict] = []

    accepted = retargeter.check(source, target)
    if not accepted.ok:
        return (
            Verdict.no(
                "conformance.declined",
                f"{retargeter} declined the pair it was given, so nothing could be checked",
            )
            & accepted
        )

    values = np.arange(samples * source.width, dtype=float).reshape(samples, source.width)
    try:
        out = np.asarray(retargeter.apply(values, source, target))
    except Exception as error:  # noqa: BLE001
        return Verdict.no(
            "conformance.apply_raised",
            f"{retargeter} accepted the pair then raised on apply: {type(error).__name__}: {error}",
        )

    if out.ndim != 2 or out.shape[0] != samples:
        checks.append(
            Verdict.no(
                "conformance.shape",
                f"{retargeter} returned shape {out.shape} for {samples} steps",
                hint="a retargeter maps channels, never the time axis",
            )
        )
    elif out.shape[1] != target.width:
        checks.append(
            Verdict.no(
                "conformance.width",
                f"{retargeter} produced width {out.shape[1]} but the target is {target.width}",
            )
        )

    losses = retargeter.losses(source, target)
    if source.width != target.width and not losses:
        checks.append(
            Verdict.no(
                "conformance.silent_loss",
                f"{retargeter} maps {source.width} values to {target.width} and declares no loss",
                hint="this is the shape that produces plausible motion and a clean "
                "provenance; say what goes",
            )
        )
    if losses and any(not str(loss).strip() for loss in losses):
        checks.append(Verdict.no("conformance.empty_loss", f"{retargeter} declares a blank loss"))

    second = np.asarray(retargeter.apply(values, source, target))
    if not np.array_equal(out, second):
        checks.append(
            Verdict.no(
                "conformance.not_deterministic",
                f"{retargeter} gave different output for the same input",
            )
        )
    return Verdict.all(checks)
