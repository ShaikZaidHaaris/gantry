"""Does this adapter close what it claims, and only what it claims?

An adapter is trusted twice over: the resolver believes its ``closes`` list when
planning, and the runner believes its transform when reading. Both beliefs are
load-bearing and neither is checked anywhere else.

The check that matters most is that a *lossless* claim is true. An adapter
declaring no cost is telling every reader downstream that nothing was given up,
and provenance repeats that claim to whoever reads the result months later. If
the transform is not actually reversible, that sentence is false in a way nobody
can detect from the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..resolve.adapters import Adapter
from ..spine import ChannelSpec, Verdict


@dataclass(frozen=True)
class Context:
    adapter: Adapter
    source: ChannelSpec
    target: ChannelSpec
    values: np.ndarray
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _declares_codes(context: Context) -> Verdict:
    if not context.adapter.closes:
        return Verdict.no(
            "conformance.closes_nothing",
            f"{context.adapter} declares no refusal codes, so the resolver can never find it",
        )
    undotted = [code for code in context.adapter.closes if "." not in code]
    if undotted:
        return Verdict.no(
            "conformance.code_shape",
            f"{context.adapter} declares undotted code(s) {undotted}",
            hint="codes are namespaced so their origin is readable",
        )
    return Verdict.yes()


def _is_applicable(context: Context) -> Verdict:
    if context.adapter.transform is None:
        return Verdict.no(
            "conformance.no_transform",
            f"{context.adapter} has no transform and can only ever close a gap on paper",
        )
    return Verdict.yes()


def _accepts_its_pair(context: Context) -> Verdict:
    verdict = context.adapter.applies(context.source, context.target)
    if not verdict.ok:
        return (
            Verdict.no(
                "conformance.declined",
                f"{context.adapter} declined the pair it was given, so nothing else "
                "could be checked",
            )
            & verdict
        )
    return Verdict.yes()


def _transforms_soundly(context: Context) -> Verdict:
    out = np.asarray(context.adapter.transform(context.values, context.source, context.target))
    checks = []
    if out.ndim != context.values.ndim:
        checks.append(
            Verdict.no(
                "conformance.rank",
                f"{context.adapter} changed the rank from {context.values.ndim} to {out.ndim}",
            )
        )
    if context.adapter.preserves_length and len(out) != len(context.values):
        checks.append(
            Verdict.no(
                "conformance.length",
                f"{context.adapter} claims to preserve length but returned "
                f"{len(out)} steps for {len(context.values)}",
                hint="an episode read lazily through this would slice the wrong timeline",
            )
        )
    if not context.adapter.preserves_length and len(out) == len(context.values):
        checks.append(
            Verdict.note(
                "conformance.length_unchanged",
                f"{context.adapter} says it may change length and did not on this input",
            )
        )
    return Verdict.all(checks)


def _is_deterministic(context: Context) -> Verdict:
    first = np.asarray(context.adapter.transform(context.values, context.source, context.target))
    second = np.asarray(context.adapter.transform(context.values, context.source, context.target))
    if first.shape != second.shape or not np.allclose(first, second):
        return Verdict.no(
            "conformance.not_deterministic",
            f"{context.adapter} gave different output for the same input",
        )
    return Verdict.yes()


def _lossless_claim_is_true(context: Context) -> Verdict:
    """If it says it costs nothing, going back must return what went in.

    Only checkable when the adapter is reversible in principle -- same width and
    a declared inverse pair. Where it is checkable, it is the difference between
    a truthful provenance and a comforting one.
    """
    if context.adapter.losses(context.source, context.target):
        return Verdict.yes()
    if not context.adapter.preserves_length:
        return Verdict.note(
            "conformance.unverifiable_lossless",
            f"{context.adapter} claims no loss but changes length, so the claim was not checked",
        )
    if context.source.width != context.target.width:
        return Verdict.no(
            "conformance.silent_loss",
            f"{context.adapter} maps width {context.source.width} to "
            f"{context.target.width} and declares no loss",
            hint="a width change discards or invents values; say which",
        )
    forward = np.asarray(context.adapter.transform(context.values, context.source, context.target))
    back = np.asarray(context.adapter.transform(forward, context.target, context.source))
    if not np.allclose(back, context.values, rtol=1e-6, atol=1e-9):
        return Verdict.no(
            "conformance.not_lossless",
            f"{context.adapter} declares no loss, but applying it and its reverse did "
            "not return the original values",
            hint="provenance will tell every later reader that nothing was given up",
        )
    return Verdict.yes()


CHECKS: tuple[Check, ...] = (
    Check("closes", "it declares dotted refusal codes", _declares_codes),
    Check("applicable", "it has a transform", _is_applicable),
    Check("accepts", "it accepts the pair under test", _accepts_its_pair),
    Check(
        "transform", "the transform keeps rank and honours its length claim", _transforms_soundly
    ),
    Check("deterministic", "the same input gives the same output", _is_deterministic),
    Check("lossless", "a no-loss claim survives a round trip", _lossless_claim_is_true),
)


def adapter_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_adapter(
    adapter: Adapter,
    source: ChannelSpec,
    target: ChannelSpec,
    values: np.ndarray | None = None,
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit against one adapter and a pair it claims to handle."""
    if values is None:
        # Fractional and irregular on purpose. Whole numbers survive rounding,
        # truncation and integer casts unchanged, so probe data made of them
        # cannot catch an adapter that quietly loses precision -- the round trip
        # comes back clean and the lossless claim goes unchallenged.
        steps = 8
        base = np.arange(steps * source.width, dtype=float).reshape(steps, source.width)
        values = base * 0.37 + 0.618
    context = Context(adapter, source, target, np.asarray(values, dtype=float), strict)
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
