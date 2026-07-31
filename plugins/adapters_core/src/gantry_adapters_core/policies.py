"""A policy whose actions are converted into the space the evaluator reads.

Gantry already binds a policy's *observations* through the adapter plane, and it
already *checks* that the actions a policy emits can be accepted. It does not
convert the action stream at run time — so a policy trained in one pose encoding
and an evaluator reading another either matched exactly or did not run at all,
with no third option.

That is the missing third option, and it is deliberately not a special case for
any one pair. Give it a policy and the channel an evaluator wants; it plans a
chain from the registered adapters, refuses now if no chain closes the gap, and
applies the same chain to every action.

Both directions, or neither
---------------------------------
The same gap exists on the way in. ``ClosedLoop`` hands a policy the world's
observation dict untouched, so a world publishing poses as quaternions and a
policy reading Euler angles fail in the same way the actions did — except
quietly, because a state of the wrong width is usually a shape error and a state
of the right width in the wrong encoding is not an error at all.

Pass ``reading`` — the channels the world actually publishes — and the
observations are wired through the same plane, using the resolver's own
``bind``. Leave it out and only the action is converted, which is honest when
the state already matches and a trap when it does not.

Refuse at construction, not on the first step
---------------------------------------------
Planning happens once, in the constructor. An unplannable pair raises there,
before a simulator is built and before a checkpoint is loaded, rather than a
thousand steps into a run that has already booked the GPU.

What it will not do
-------------------
It will not invent a conversion. If the registry holds nothing that closes the
gap, the refusal carries the same codes it would have carried at plan time and
names the encodings on both sides. The point of the whole plane is that "these
two are not the same thing" stays sayable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from gantry.errors import ConfigError
from gantry.resolve import AdapterRegistry, Chain, bind
from gantry.spine import ChannelSpec, compatible


class AdaptedPolicy:
    """Wraps a policy, converting its actions into ``target``.

    Everything other than the action passes through untouched: the wrapped
    policy still sees the observations it asked for, still declares its own
    prompt handling, and its descriptor is reported with the chain recorded
    alongside rather than in place of it.
    """

    def __init__(
        self,
        policy: Any,
        target: ChannelSpec,
        *,
        adapters: AdapterRegistry | None = None,
        reading: Sequence[ChannelSpec] = (),
        name: str | None = None,
    ):
        self._policy = policy
        self._target = target
        self._source = policy.action_spec()
        self._registry = adapters if adapters is not None else installed_adapters()

        gap = compatible(self._source, target)
        # An `.undeclared` reason is one side saying something the other did not
        # say -- the absence of a claim, not a conflict. The resolver carries
        # these as notes rather than failures, and demanding an adapter "close"
        # one would refuse every pair where an evaluator tags something extra.
        self._notes = tuple(r for r in gap.reasons if r.code.endswith(".undeclared"))
        real = tuple(r for r in gap.reasons if not r.code.endswith(".undeclared"))
        if not real:
            # Nothing to do, and saying so is better than planning an empty
            # chain that reads as a conversion in the record.
            self._chain: Chain | None = None
            self._losses: tuple[str, ...] = ()
        else:
            codes = tuple(reason.code for reason in real)
            chain, unclosed = self._registry.close(self._source, target, codes)
            if chain is None:
                raise ConfigError(
                    f"cannot convert {self._source.name!r} into {target.name!r}: "
                    f"nothing registered closes {list(unclosed)}. "
                    f"{self._source.name!r} is {self._source.width} wide "
                    f"({dict(self._source.metadata)}) and {target.name!r} is "
                    f"{target.width} wide ({dict(target.metadata)}). These are not the "
                    "same thing, and a run that proceeded would send numbers that are "
                    "accepted and mean something else"
                )
            self._chain = chain
            self._losses = chain.losses
        self._name = name or f"{getattr(policy, '_name', 'policy')}+adapted"
        self._inputs: dict[str, tuple[str, Chain]] = {}
        if reading:
            self._wire(tuple(reading))

    def _wire(self, provided: Sequence[ChannelSpec]) -> None:
        """Plan the observation side through the resolver's own binding."""
        wiring, verdict = bind(self._policy.observes(), provided, self._registry)
        if wiring is None:
            raise ConfigError(
                f"{self._policy_name()}: cannot read what this world publishes. {verdict.explain()}"
            )
        for binding in wiring.bindings:
            if binding.chain:
                self._inputs[binding.want.name] = (binding.provided.name, binding.chain)
            elif binding.provided.name != binding.want.name:
                # A rename is still a wiring decision, and the policy looks the
                # channel up by the name it asked for.
                self._inputs[binding.want.name] = (binding.provided.name, Chain())
        self._wiring = wiring

    def _policy_name(self) -> str:
        return str(getattr(self._policy, "_name", type(self._policy).__name__))

    # -- what changed ------------------------------------------------------

    @property
    def chain(self) -> Chain | None:
        return self._chain

    @property
    def notes(self) -> tuple[Any, ...]:
        """What could not be checked -- a tag declared on one side only. Carried
        so a plan can show what it did not verify, rather than presenting an
        unchecked match as a checked one."""
        return self._notes

    @property
    def losses(self) -> tuple[str, ...]:
        """What the conversion costs, in words. Empty is a claim, and it is the
        adapter's claim rather than this wrapper's."""
        return self._losses

    def action_spec(self) -> ChannelSpec:
        return self._target

    def descriptor(self) -> Any:
        inner = self._policy.descriptor()
        steps = [f"{step.name}@{step.version}" for step in (self._chain or ())]
        # Recorded rather than hidden: a report that says a policy scored 12%
        # should be able to say what was done to its output on the way.
        return replace(
            inner,
            metadata={
                **dict(inner.metadata),
                "action_adapted_from": self._source.name,
                "action_adapted_to": self._target.name,
                "action_adapter_chain": steps,
                "action_adapter_losses": list(self._losses),
                "action_adapter_unchecked": [r.code for r in self._notes],
            },
        )

    # -- everything else is the wrapped policy -----------------------------

    def observes(self) -> Any:
        return self._policy.observes()

    def reset(self, context: Any) -> None:
        self._policy.reset(context)

    def act(self, observation: Any) -> np.ndarray:
        values = np.asarray(self._policy.act(self._read(observation)))
        if self._chain is None:
            return values
        # A chunk is (horizon, width) and a single action is (width,). The
        # adapters take (steps, width), so a bare action is lifted and put back
        # rather than silently read as a one-wide chunk.
        flat = values.ndim == 1
        block = values[None, :] if flat else values
        out = self._chain.run(block, self._source, self._target)
        return out[0] if flat else out

    def _read(self, observation: Any) -> Any:
        """The observation, with each wired channel under the name the policy
        asked for and in the encoding it asked for."""
        if not self._inputs:
            return observation
        channels = dict(getattr(observation, "channels", observation) or {})
        for wanted, (source, chain) in self._inputs.items():
            if source not in channels:
                continue
            values = np.asarray(channels[source])
            if chain:
                flat = values.ndim == 1
                block = values[None, :] if flat else values
                out = chain.run(block, self._by_name(source), self._by_name(wanted, want=True))
                values = out[0] if flat else out
            channels[wanted] = values
        rebuilt = getattr(observation, "channels", None)
        if rebuilt is None:
            return channels
        return type(observation)(observation.step, channels)

    def _by_name(self, name: str, want: bool = False) -> ChannelSpec:
        source = self._wiring.bindings
        for binding in source:
            if want and binding.want.name == name:
                return binding.want
            if not want and binding.provided.name == name:
                return binding.provided
        raise KeyError(name)  # pragma: no cover - _inputs is built from these

    def __getattr__(self, item: str) -> Any:
        # Anything this wrapper does not model belongs to the policy underneath.
        return getattr(self._policy, item)


def adapt_policy(
    policy: Any,
    to: ChannelSpec,
    *,
    adapters: AdapterRegistry | None = None,
    reading: Sequence[ChannelSpec] = (),
    name: str | None = None,
) -> Any:
    """``policy``, emitting ``to`` and reading what ``reading`` publishes.

    Returns the policy unchanged only when nothing at all needs converting —
    including the observation side, which is why ``reading`` has to be passed
    here rather than checked later.
    """
    gap = compatible(policy.action_spec(), to)
    unresolved = [r for r in gap.reasons if not r.code.endswith(".undeclared")]
    if not reading and not unresolved:
        return policy
    return AdaptedPolicy(policy, to, adapters=adapters, reading=reading, name=name)


def installed_adapters() -> AdapterRegistry:
    """Every adapter installed, found the way the runner finds them.

    Distinct from ``adapters.default_registry``, which is the three core
    transforms only. This one is whatever is installed -- loaded from entry
    points rather than imported, so the module names no particular adapter and a
    new one is available the moment it is installed.
    """
    from importlib.metadata import entry_points

    registry = AdapterRegistry()
    for entry in entry_points(group="gantry.adapters"):
        try:
            registry.add(entry.load())
        except Exception:  # pragma: no cover - a broken plugin is not this one's problem
            continue
    return registry
