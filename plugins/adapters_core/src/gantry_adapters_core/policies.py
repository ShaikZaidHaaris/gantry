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
from typing import Any

import numpy as np

from gantry.errors import ConfigError
from gantry.resolve import AdapterRegistry, Chain
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
        name: str | None = None,
    ):
        self._policy = policy
        self._target = target
        self._source = policy.action_spec()

        gap = compatible(self._source, target)
        if gap.ok:
            # Nothing to do, and saying so is better than planning an empty
            # chain that reads as a conversion in the record.
            self._chain: Chain | None = None
            self._losses: tuple[str, ...] = ()
        else:
            registry = adapters if adapters is not None else installed_adapters()
            codes = tuple(reason.code for reason in gap.reasons)
            chain, unclosed = registry.close(self._source, target, codes)
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

    # -- what changed ------------------------------------------------------

    @property
    def chain(self) -> Chain | None:
        return self._chain

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
            },
        )

    # -- everything else is the wrapped policy -----------------------------

    def observes(self) -> Any:
        return self._policy.observes()

    def reset(self, context: Any) -> None:
        self._policy.reset(context)

    def act(self, observation: Any) -> np.ndarray:
        values = np.asarray(self._policy.act(observation))
        if self._chain is None:
            return values
        # A chunk is (horizon, width) and a single action is (width,). The
        # adapters take (steps, width), so a bare action is lifted and put back
        # rather than silently read as a one-wide chunk.
        flat = values.ndim == 1
        block = values[None, :] if flat else values
        out = self._chain.run(block, self._source, self._target)
        return out[0] if flat else out

    def __getattr__(self, item: str) -> Any:
        # Anything this wrapper does not model belongs to the policy underneath.
        return getattr(self._policy, item)


def adapt_policy(
    policy: Any,
    to: ChannelSpec,
    *,
    adapters: AdapterRegistry | None = None,
    name: str | None = None,
) -> Any:
    """``policy``, emitting ``to``. Returns the policy unchanged if it already does."""
    source = policy.action_spec()
    if compatible(source, to).ok:
        return policy
    return AdaptedPolicy(policy, to, adapters=adapters, name=name)


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
