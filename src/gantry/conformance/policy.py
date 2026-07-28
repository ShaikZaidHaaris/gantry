"""Does this policy honour the policy contract?

One check per invariant sentence in :mod:`gantry.contracts.policy`. Same shape
as every other kit: a verdict, no test framework, runnable from CI or a CLI.

Determinism gets a real check rather than a trusted flag. A policy that claims
it and does not have it silently destroys every paired comparison built on top
of it, and the failure shows up as an unexplained variance months later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..contracts.policy import (
    CAP_CHUNK,
    CAP_DETERMINISTIC,
    POLICY_CONTRACT,
    EpisodeContext,
    Observation,
    Policy,
)
from ..spine import ContractVersion, Verdict


@dataclass(frozen=True)
class Context:
    policy: Policy
    observations: tuple[Observation, ...]
    strict: bool


@dataclass(frozen=True)
class Check:
    code: str
    description: str
    run: Callable[[Context], Verdict]


def _plane(context: Context) -> Verdict:
    descriptor = context.policy.descriptor()
    if descriptor.plane != "policy":
        return Verdict.no(
            "conformance.plane",
            f"{descriptor.ref} declares plane {descriptor.plane!r}, expected 'policy'",
        )
    return Verdict.yes()


def _contract(context: Context) -> Verdict:
    return context.policy.descriptor().contract_version.satisfies(
        ContractVersion.parse(POLICY_CONTRACT)
    )


def _action_spec_valid(context: Context) -> Verdict:
    return context.policy.action_spec().validate()


def _emits_a_chunk(context: Context) -> Verdict:
    spec = context.policy.action_spec()
    checks = []
    for observation in context.observations:
        chunk = np.asarray(context.policy.act(observation))
        if chunk.ndim != len(spec.shape) + 1:
            checks.append(
                Verdict.no(
                    "conformance.chunk_rank",
                    f"act() returned rank {chunk.ndim}, expected "
                    f"{len(spec.shape) + 1} (horizon + action)",
                    hint="a policy returns a chunk of actions, not a single action",
                )
            )
            continue
        if len(chunk) < 1:
            checks.append(
                Verdict.no("conformance.chunk_empty", "act() returned an empty chunk")
            )
            continue
        checks.append(spec.accepts(chunk))
    return Verdict.all(checks)


def _chunk_within_declared(context: Context) -> Verdict:
    declared = int(context.policy.descriptor().provides.get(CAP_CHUNK, 1))
    checks = []
    for observation in context.observations:
        chunk = np.asarray(context.policy.act(observation))
        if len(chunk) > declared:
            checks.append(
                Verdict.no(
                    "conformance.chunk_overrun",
                    f"declares a chunk of {declared} but returned {len(chunk)}",
                    hint="a runner sizes its buffers from the declaration",
                )
            )
    return Verdict.all(checks)


def _determinism_is_honest(context: Context) -> Verdict:
    if not context.policy.descriptor().provides.get(CAP_DETERMINISTIC):
        return Verdict.yes()
    checks = []
    for observation in context.observations:
        context.policy.reset(EpisodeContext("conformance", seed=0))
        first = np.asarray(context.policy.act(observation))
        context.policy.reset(EpisodeContext("conformance", seed=0))
        second = np.asarray(context.policy.act(observation))
        if first.shape != second.shape or not np.array_equal(first, second):
            checks.append(
                Verdict.no(
                    "conformance.not_deterministic",
                    "declares determinism but returned different actions for the "
                    "same observation after the same reset",
                    hint="every paired comparison downstream depends on this",
                )
            )
            break
    return Verdict.all(checks)


def _reset_is_accepted(context: Context) -> Verdict:
    try:
        context.policy.reset(EpisodeContext("conformance", instruction="do the thing", seed=1))
    except Exception as error:  # noqa: BLE001 - classifying the failure is the check
        return Verdict.no(
            "conformance.reset_failed",
            f"reset() raised {type(error).__name__}: {error}",
        )
    return Verdict.yes()


def _requirements_are_declared(context: Context) -> Verdict:
    """Whatever it declares must be well-formed. Declaring nothing is allowed.

    An empty requirement is a *declaration* — "I read nothing" — and a constant
    baseline is exactly that. Failing it would punish a correct answer. The
    thing worth catching is a policy that reads a channel it never declared,
    and no inspection of the declaration alone can see that.
    """
    requirement = context.policy.observes()
    checks = [spec.validate() for spec in requirement.channels]
    if not requirement.channels:
        checks.append(
            Verdict.note(
                "conformance.observes_nothing",
                f"{context.policy.name} reads nothing, so it cannot respond to what it sees",
                hint="expected for a fixed baseline, surprising for anything else",
            )
        )
    return Verdict.all(checks)


CHECKS: tuple[Check, ...] = (
    Check("plane", "declares the policy plane", _plane),
    Check("contract", "implements a compatible policy contract", _contract),
    Check("action_spec", "its action channel is internally valid", _action_spec_valid),
    Check("observes", "it declares what it reads", _requirements_are_declared),
    Check("reset", "reset() accepts an episode context", _reset_is_accepted),
    Check("chunk", "act() returns a non-empty, well-shaped chunk", _emits_a_chunk),
    Check("chunk_declared", "chunks stay within the declared size", _chunk_within_declared),
    Check("determinism", "a determinism claim is true", _determinism_is_honest),
)


def policy_checks() -> tuple[tuple[str, str], ...]:
    return tuple((check.code, check.description) for check in CHECKS)


def check_policy(
    policy: Policy,
    observations: Sequence[Observation],
    *,
    strict: bool = False,
    only: Sequence[str] | None = None,
) -> Verdict:
    """Run the kit against a policy and some observations it should accept."""
    if not observations:
        return Verdict.no(
            "conformance.no_observations",
            "the kit needs at least one observation to exercise the policy",
        )
    context = Context(policy, tuple(observations), strict)
    selected = [c for c in CHECKS if only is None or c.code in set(only)]
    results = []
    for check in selected:
        try:
            results.append(check.run(context))
        except Exception as error:  # noqa: BLE001 - a raising check is a failure
            results.append(
                Verdict.no(
                    "conformance.check_raised",
                    f"check {check.code!r} raised {type(error).__name__}: {error}",
                    check=check.code,
                )
            )
    return Verdict.all(results)
