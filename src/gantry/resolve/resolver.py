"""Plan a run from descriptors alone, or refuse and say exactly why.

The resolver never imports a plugin to decide whether it fits. It reads
descriptors and schemas, checks every seam, and either returns a
:class:`~gantry.resolve.plan.Plan` or a refusal naming the seam. That is why an
unrunnable combination fails in milliseconds instead of after a model finishes
loading, and why the answer is always specific.

Finding a candidate happens in two passes, in this order:

1. **By name**, including any alias the consumer declared.
2. **By meaning** — same modality and same semantics — but only when the
   consumer stated a semantics to match on.

There is no third pass. Matching on shape alone would pair a wrist pose with a
base pose because both happen to be three numbers wide, and a framework that
guesses is the thing this design exists to replace.

Once a candidate is found, three strategies are tried in a fixed order, and the
order is the whole design:

1. **Direct** — the specs already agree.
2. **Adapt** — every gap has a single correct answer, so close them all.
3. **Retarget** — the gap is structural and needs a declared judgement about
   what to discard.

Cheapest and most faithful first. Reaching for a lossy retargeter when a unit
conversion would have done is not a fallback, it is data thrown away for
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..spine import ChannelSpec, ContractVersion, Descriptor, Verdict, compatible
from .adapters import AdapterRegistry
from .plan import ChannelBinding, Plan, Resolution, ResolvedComponent, Wiring
from .registry import Registry
from .requirement import Requirement
from .retarget import RETARGETABLE, RetargeterRegistry
from .transform import Chain

#: Nothing closes these, ever. An image is not a vector, and no transform makes
#: it one.
IMMUTABLE = frozenset({"kind.mismatch"})

#: Gaps in the shape of a channel rather than its description. Most need a
#: retargeter, which states what it discards — but not all: re-encoding a
#: quaternion as an axis-angle is exact *and* three numbers narrower. So an
#: adapter that explicitly declares ``shape.mismatch`` gets first refusal,
#: because a lossless answer should always beat a lossy one. Assuming lossless
#: implied same-width is what sent every width change to a retargeter.
STRUCTURAL = frozenset({"kind.mismatch", "shape.mismatch"})

#: Kept as the historical name for the same set.
UNADAPTABLE = STRUCTURAL


# --------------------------------------------------------------------------
# finding candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A provider channel worth trying, and how it was found."""

    spec: ChannelSpec
    matched_by: str


def candidates(
    want: ChannelSpec, provided: Sequence[ChannelSpec], requirement: Requirement
) -> tuple[Candidate, ...]:
    """Provider channels worth trying for this want, best first."""
    by_name = {spec.name: spec for spec in provided}
    found: list[Candidate] = []

    def already(spec: ChannelSpec) -> bool:
        return any(spec.name == chosen.spec.name for chosen in found)

    if want.name in by_name:
        found.append(Candidate(by_name[want.name], "name"))
    for alias in requirement.alias_names(want.name):
        if alias in by_name and not already(by_name[alias]):
            found.append(Candidate(by_name[alias], "alias"))
    if want.semantics is not None:
        for spec in provided:
            if spec.semantics == want.semantics and spec.kind == want.kind and not already(spec):
                found.append(Candidate(spec, "semantics"))
    return tuple(found)


# --------------------------------------------------------------------------
# strategies for one candidate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """The outcome of trying one candidate: a chain, or the reason there is none."""

    chain: Chain | None
    verdict: Verdict

    @property
    def ok(self) -> bool:
        return self.chain is not None


def _direct(gap: Verdict) -> Attempt | None:
    return Attempt(Chain(), gap) if gap.ok else None


def _adapt(
    candidate: Candidate,
    want: ChannelSpec,
    gap: Verdict,
    adapters: AdapterRegistry,
    requirement: Requirement,
) -> Attempt:
    if any(code in IMMUTABLE for code in gap.codes()):
        return Attempt(None, Verdict.yes())  # nothing closes a modality change
    chain, unclosed = adapters.close(candidate.spec, want, gap.codes())
    if chain is not None:
        return Attempt(chain, gap)
    return Attempt(
        None,
        Verdict.no(
            "resolve.no_adapter",
            f"{requirement}: {candidate.spec.name!r} could feed {want.name!r} but for "
            f"{', '.join(unclosed)}",
            hint=f"install an adapter declaring {', '.join(repr(c) for c in unclosed)}",
            consumer=requirement.name,
            channel=want.name,
            unclosed=list(unclosed),
        )
        & gap,
    )


def _retarget(
    candidate: Candidate,
    want: ChannelSpec,
    gap: Verdict,
    retargeters: RetargeterRegistry,
    requirement: Requirement,
) -> Attempt:
    blocking = [code for code in gap.codes() if code in STRUCTURAL]
    retargetable = [code for code in gap.codes() if code in RETARGETABLE]
    if not blocking:
        return Attempt(None, Verdict.yes())
    if not retargetable:
        return Attempt(
            None,
            Verdict.no(
                "resolve.channel_incompatible",
                f"{requirement}: {candidate.spec.name!r} cannot feed {want.name!r} "
                f"({', '.join(blocking)})",
                hint="nothing converts one modality into another",
                consumer=requirement.name,
                channel=want.name,
            )
            & gap,
        )

    step, verdict = retargeters.find(candidate.spec, want)
    if step is not None:
        return Attempt(Chain((step,)), verdict)
    return Attempt(
        None,
        Verdict.no(
            "resolve.channel_incompatible",
            f"{requirement}: {candidate.spec.name!r} cannot feed {want.name!r} "
            f"({', '.join(blocking)})",
            hint="no adapter can change a channel's width; that needs a retargeter, "
            "and none installed handles this pair",
            consumer=requirement.name,
            channel=want.name,
        )
        & gap
        & verdict,
    )


# --------------------------------------------------------------------------
# binding
# --------------------------------------------------------------------------


def bind_channel(
    want: ChannelSpec,
    provided: Sequence[ChannelSpec],
    requirement: Requirement,
    adapters: AdapterRegistry | None = None,
    retargeters: RetargeterRegistry | None = None,
) -> tuple[ChannelBinding | None, Verdict]:
    """Wire one wanted channel, trying each strategy in order."""
    adapters = adapters or AdapterRegistry()
    retargeters = retargeters or RetargeterRegistry()

    found = candidates(want, provided, requirement)
    if not found:
        available = ", ".join(spec.name for spec in provided) or "nothing"
        return None, Verdict.no(
            "resolve.no_candidate",
            f"{requirement}: nothing provides channel {want.name!r}",
            hint=(
                f"available: {available}. Declare a matching name, an alias, or "
                "semantics on both sides"
            ),
            consumer=requirement.name,
            channel=want.name,
        )

    failures: list[Verdict] = []
    for candidate in found:
        gap = compatible(candidate.spec, want)
        attempts = [
            _direct(gap),
            _adapt(candidate, want, gap, adapters, requirement),
            _retarget(candidate, want, gap, retargeters, requirement),
        ]
        for attempt in attempts:
            if attempt is None or not attempt.ok:
                continue
            notes = tuple(r for r in gap.reasons if r.code.endswith(".undeclared"))
            binding = ChannelBinding(want, candidate.spec, attempt.chain, candidate.matched_by, notes)
            return binding, Verdict.yes(*gap.reasons)
        failures.extend(a.verdict for a in attempts if a is not None and not a.verdict.ok)

    return None, Verdict.all(failures)


def bind(
    requirement: Requirement,
    provided: Sequence[ChannelSpec],
    adapters: AdapterRegistry | None = None,
    retargeters: RetargeterRegistry | None = None,
) -> tuple[Wiring | None, Verdict]:
    """Wire every channel one consumer wants."""
    bindings: list[ChannelBinding] = []
    skipped: list[str] = []
    problems: list[Verdict] = []

    for want in requirement.channels:
        binding, verdict = bind_channel(want, provided, requirement, adapters, retargeters)
        if binding is not None:
            bindings.append(binding)
            if verdict.reasons:
                problems.append(Verdict.yes(*verdict.reasons))
        elif want.optional:
            skipped.append(want.name)
        else:
            problems.append(verdict)

    outcome = Verdict.all(problems)
    if not outcome.ok:
        return None, outcome
    return Wiring(requirement.name, tuple(bindings), tuple(skipped)), outcome


# --------------------------------------------------------------------------
# capabilities
# --------------------------------------------------------------------------


def check_capabilities(requirement: Requirement, provider: Descriptor) -> Verdict:
    """Does the provider offer what this consumer depends on?"""
    if not requirement.capabilities:
        return Verdict.yes()
    verdict = provider.provides_all(requirement.capabilities)
    if verdict.ok:
        return verdict
    missing = [
        reason.detail.get("capability")
        for reason in verdict.reasons
        if reason.detail.get("capability")
    ]
    return Verdict.no(
        "resolve.capability",
        f"{requirement} needs {', '.join(str(name) for name in missing)}, "
        f"which {provider.ref} does not provide",
        hint="choose a provider that does, or a consumer that does not need it",
        consumer=requirement.name,
        capabilities=missing,
    ) & verdict


# --------------------------------------------------------------------------
# whole-run resolution
# --------------------------------------------------------------------------


def _resolve_component(
    registry: Registry, plane: str, name: str, config: Mapping[str, Any]
) -> tuple[ResolvedComponent | None, Verdict]:
    try:
        registration = registry.get(plane, name)
    except KeyError as error:
        return None, Verdict.no("resolve.not_installed", str(error), plane=plane, name=name)
    try:
        descriptor = registration.descriptor(config)
    except Exception as error:  # noqa: BLE001 - a component that cannot describe itself
        return None, Verdict.no(
            "resolve.describe_failed",
            f"{plane}:{name} could not describe itself: {type(error).__name__}: {error}",
            plane=plane,
            name=name,
        )

    checks = [descriptor.validate()]
    if descriptor.plane != plane:
        checks.append(
            Verdict.no(
                "resolve.plane_mismatch",
                f"{name} was asked for as a {plane} component but declares "
                f"plane {descriptor.plane!r}",
            )
        )
    verdict = Verdict.all(checks)
    if not verdict.ok:
        return None, verdict
    return ResolvedComponent(plane, name, descriptor, dict(config)), verdict


def _expected_contract(plane: str, descriptor: Descriptor) -> Verdict:
    """The host speaks the same contract major the component implements.

    The expected contract comes from the plane registry, so a plane a plugin
    registered is version-checked exactly like a core one. A table here would be
    a second place to remember, and the plane nobody remembered would be the one
    that silently skipped its check.
    """
    from ..spine import contract_for

    expected = contract_for(plane)
    if expected is None:
        return Verdict.note(
            "resolve.contract_unpublished",
            f"no contract is published for the {plane} plane, so "
            f"{descriptor.ref} was not version-checked",
        )
    return descriptor.contract_version.satisfies(ContractVersion.parse(expected))


def resolve(
    registry: Registry,
    *,
    components: Mapping[str, Mapping[str, Any]],
    consumers: Iterable[Requirement] = (),
    provider_plane: str = "dataset",
    provided_channels: Sequence[ChannelSpec] | None = None,
    adapters: AdapterRegistry | None = None,
    retargeters: RetargeterRegistry | None = None,
    protocol: Mapping[str, Any] | None = None,
) -> Resolution:
    """Plan a run.

    ``components`` maps plane -> ``{"name": ..., "config": {...}}``. Consumers
    are resolved independently so that, when one cannot run, the refusal can
    name the ones that could.
    """
    resolved: list[ResolvedComponent] = []
    checks: list[Verdict] = []

    for plane, spec in components.items():
        name = spec.get("name")
        if not name:
            checks.append(
                Verdict.no("resolve.unnamed", f"the {plane} entry has no 'name'", plane=plane)
            )
            continue
        component, verdict = _resolve_component(registry, plane, name, spec.get("config", {}))
        checks.append(verdict)
        if component is not None:
            resolved.append(component)
            checks.append(_expected_contract(plane, component.descriptor))

    provider = next((c for c in resolved if c.plane == provider_plane), None)
    if consumers and provider is None:
        checks.append(
            Verdict.no(
                "resolve.no_provider",
                f"consumers were requested but no {provider_plane} component resolved",
            )
        )

    wirings: list[Wiring] = []
    satisfied: list[str] = []
    unsatisfied: list[Verdict] = []

    if provider is not None:
        channels = tuple(provided_channels or ())
        for requirement in consumers:
            wiring, binding = bind(requirement, channels, adapters, retargeters)
            combined = Verdict.all(
                [check_capabilities(requirement, provider.descriptor), binding]
            )
            if combined.ok and wiring is not None:
                wirings.append(wiring)
                satisfied.append(requirement.name)
                if combined.reasons:
                    checks.append(Verdict.yes(*combined.reasons))
            else:
                unsatisfied.append(combined)

    verdict = Verdict.all(checks + unsatisfied)
    if not verdict.ok:
        return Resolution(verdict, None, tuple(satisfied))

    wired = tuple(wirings)
    # Every declared loss becomes a note on the plan, so it is copied into
    # provenance and travels with the run's numbers rather than being
    # discovered later in someone's changelog.
    notes = tuple(
        f"{step.name} is lossy: {'; '.join(step.losses)}"
        for wiring in wired
        for step in wiring.adapters
        if step.losses
    )
    plan = Plan(tuple(resolved), wired, dict(protocol or {}), notes)
    return Resolution(verdict, plan, ())
