"""The output of resolution: an executable plan, or a refusal you can act on.

A refusal is not an error string. It names the seam that failed, the code that
failed there, and what would close it -- and when several consumers were asked
for, it names the ones that *would* have run. "Incompatible" ends a
conversation; "funnel needs stage events, which this evaluator does not emit;
screen and attribution would run" continues it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..spine import (
    AdapterStep,
    ChannelSpec,
    ComponentRef,
    Descriptor,
    Provenance,
    Reason,
    Verdict,
)
from .transform import Chain


@dataclass(frozen=True)
class ChannelBinding:
    """One consumer channel, wired to one provider channel."""

    want: ChannelSpec
    provided: ChannelSpec
    chain: Chain = field(default_factory=Chain)
    matched_by: str = "name"
    #: Anything true but unverifiable about this pairing, most often a tag
    #: declared on one side only. Carried so a plan can show what it could not
    #: check, rather than presenting an unchecked match as a checked one.
    notes: tuple[Reason, ...] = ()

    @property
    def direct(self) -> bool:
        return not self.chain

    def __str__(self) -> str:
        arrow = "->" if self.direct else f"-[{self.chain}]->"
        return f"{self.provided.name} {arrow} {self.want.name}"


@dataclass(frozen=True)
class Wiring:
    """Every binding for one consumer."""

    consumer: str
    bindings: tuple[ChannelBinding, ...] = ()
    skipped: tuple[str, ...] = ()  # optional wants with no provider

    @property
    def adapters(self) -> tuple[AdapterStep, ...]:
        """Every transform this wiring will run, in provenance form.

        Named for adapters because that is what provenance calls them, and a
        reader needs the name and the loss rather than the framework's taxonomy
        of why the loss happened. :attr:`transforms` keeps the distinction for
        anyone who does care.
        """
        return tuple(step for binding in self.bindings for step in binding.chain.provenance)

    @property
    def transforms(self) -> tuple[tuple[str, str], ...]:
        """``(name, kind)`` for each step, so an adapter reads apart from a retargeter."""
        return tuple((step.name, step.kind) for binding in self.bindings for step in binding.chain)

    @property
    def lossy(self) -> bool:
        return any(binding.chain.lossy for binding in self.bindings)


@dataclass(frozen=True)
class ResolvedComponent:
    """A component the plan will build, pinned to the config it was planned with."""

    plane: str
    name: str
    descriptor: Descriptor
    config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return self.descriptor.ref

    def component_ref(self) -> ComponentRef:
        return self.descriptor.component_ref(config=self.config)


@dataclass(frozen=True)
class Plan:
    """A run that can be executed, with its provenance already determined."""

    components: tuple[ResolvedComponent, ...] = ()
    wirings: tuple[Wiring, ...] = ()
    protocol: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def component(self, plane: str) -> ResolvedComponent | None:
        return next((c for c in self.components if c.plane == plane), None)

    def wiring(self, consumer: str) -> Wiring | None:
        return next((w for w in self.wirings if w.consumer == consumer), None)

    @property
    def adapters(self) -> tuple[AdapterStep, ...]:
        seen: dict[tuple[str, str], AdapterStep] = {}
        for wiring in self.wirings:
            for step in wiring.adapters:
                seen[(step.name, step.version)] = step
        return tuple(seen.values())

    @property
    def lossy(self) -> bool:
        return any(step.losses for step in self.adapters)

    def provenance(self, **extra: Any) -> Provenance:
        """The stamp every record from this run will carry.

        Determined at planning time, not after the fact, so a result cannot
        exist without knowing what produced it.
        """
        return Provenance(
            components=tuple(c.component_ref() for c in self.components),
            protocol=dict(self.protocol),
            adapters=self.adapters,
            notes=self.notes,
            **extra,
        )

    def explain(self) -> str:
        lines = ["plan"]
        for component in self.components:
            lines.append(f"  {component.plane:11} {component.ref}")
        for wiring in self.wirings:
            lines.append(f"  wiring {wiring.consumer}")
            for binding in wiring.bindings:
                lines.append(f"    {binding}")
                for note in binding.notes:
                    lines.append(f"      unchecked: {note.message}")
            for name in wiring.skipped:
                lines.append(f"    {name}: not provided (optional)")
        for step in self.adapters:
            if step.losses:
                lines.append(f"  loss via {step.name}: {'; '.join(step.losses)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Resolution:
    """Either a plan, or the reasons there isn't one."""

    verdict: Verdict
    plan: Plan | None = None
    #: Consumers that would resolve on their own, named when others did not.
    alternatives: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.verdict.ok and self.plan is not None

    def __bool__(self) -> bool:
        return self.ok

    def require(self) -> Plan:
        """The plan, or an exception carrying the full explanation."""
        self.verdict.raise_if_refused("cannot plan this run")
        assert self.plan is not None
        return self.plan

    def explain(self) -> str:
        if self.ok:
            assert self.plan is not None
            return self.plan.explain()
        text = self.verdict.explain()
        if self.alternatives:
            text += f"\n  runnable as requested: {', '.join(self.alternatives)}"
        return text
