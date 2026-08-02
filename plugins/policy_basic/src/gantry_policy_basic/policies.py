"""Reference policies: the ceiling, the floor, and a dial between them.

None of these is useful on a robot. All three are useful for knowing whether
anything else is, because a measurement with no reference points is a number
without a scale.

**Replay** reproduces what was recorded. On an open-loop evaluation it scores a
perfect zero, which is the ceiling: no policy can do better on this measure, so
a harness that does not put replay at zero is broken.

**Constant** emits the same action forever. That is the floor -- the score of
learning nothing -- and a policy that fails to beat it has not learned anything
either. This project has watched a real evaluation sit below its own random
baseline for a week because nobody had computed the floor.

**Noisy replay** is the ceiling plus a known amount of error, which turns a
harness into something testable: raise the noise, the score must worsen
monotonically, and if it does not the measurement is not measuring.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.policy import (
    EpisodeContext,
    Observation,
    Policy,
    policy_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import ChannelSpec, Descriptor, seed_from

VERSION = "0.1.0.dev0"


def channel(spec: ChannelSpec | Mapping[str, Any]) -> ChannelSpec:
    """A channel spec, or the JSON object a manifest can carry one as.

    Without this these policies are constructible only from Python, which makes
    the reference floor and ceiling -- the two things every comparison needs --     the only components a manifest cannot name.
    """
    if isinstance(spec, ChannelSpec):
        return spec
    from gantry.serial import spec_from_dict

    return spec_from_dict(spec)


class _Base(Policy):
    """Shared plumbing: an action spec, an observation requirement, a chunk."""

    def __init__(
        self,
        action: ChannelSpec | Mapping[str, Any],
        observes: Sequence[ChannelSpec | Mapping[str, Any]],
        chunk: int = 1,
    ):
        if chunk < 1:
            raise ValueError(f"chunk must be at least 1, got {chunk}")
        self._action = channel(action)
        self._observes = tuple(channel(spec) for spec in observes)
        self._chunk = chunk
        self._context: EpisodeContext | None = None

    def action_spec(self) -> ChannelSpec:
        return self._action

    def observes(self) -> Requirement:
        return requires_channels(self.name, "policy", *self._observes)

    def reset(self, context: EpisodeContext) -> None:
        self._context = context

    def _repeat(self, action: np.ndarray) -> np.ndarray:
        """One action broadcast across the declared chunk."""
        return np.repeat(np.asarray(action, dtype="float32")[None, ...], self._chunk, axis=0)


class ReplayPolicy(_Base):
    """Emits the action that was actually recorded at this step.

    Reads its answer from the observation, which is only legitimate because the
    point is to establish the ceiling. Named so nobody mistakes it for a model.
    """

    def __init__(
        self, action: ChannelSpec, *, observes: Sequence[ChannelSpec] = (), chunk: int = 1
    ):
        action = channel(action)
        super().__init__(action, tuple(observes) + (action,), chunk)

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            "replay",
            VERSION,
            chunk=self._chunk,
            deterministic=True,
            reads_the_answer=True,
        )

    def act(self, observation: Observation) -> np.ndarray:
        return self._repeat(observation[self._action.name])


class ConstantPolicy(_Base):
    """Emits the same action at every step. The floor."""

    def __init__(
        self,
        action: ChannelSpec,
        value: float | Sequence[float] = 0.0,
        *,
        observes: Sequence[ChannelSpec] = (),
        chunk: int = 1,
    ):
        super().__init__(action, observes, chunk)
        action = self._action
        self._value = (
            np.broadcast_to(np.asarray(value, dtype="float32"), action.shape or (1,))
            .reshape(action.shape)
            .copy()
        )

    def descriptor(self) -> Descriptor:
        return policy_descriptor("constant", VERSION, chunk=self._chunk, deterministic=True)

    def act(self, observation: Observation) -> np.ndarray:
        return self._repeat(self._value)


class NoisyReplayPolicy(_Base):
    """Replay plus Gaussian noise of a known scale. A dial between the two.

    Seeded from the episode context and the step, so the same episode under the
    same seed produces the same error. Without that, a paired comparison between
    two settings would be measuring the noise.
    """

    def __init__(
        self,
        action: ChannelSpec,
        sigma: float = 0.1,
        *,
        seed: int = 0,
        observes: Sequence[ChannelSpec] = (),
        chunk: int = 1,
    ):
        action = channel(action)
        super().__init__(action, tuple(observes) + (action,), chunk)
        if sigma < 0:
            raise ValueError(f"sigma must not be negative, got {sigma}")
        self._sigma = sigma
        self._seed = seed

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            "noisy-replay",
            VERSION,
            chunk=self._chunk,
            deterministic=True,
            sigma=self._sigma,
            reads_the_answer=True,
        )

    def act(self, observation: Observation) -> np.ndarray:
        truth = np.asarray(observation[self._action.name], dtype="float32")
        episode = self._context.episode_id if self._context else ""
        rng = np.random.default_rng(seed_from(self._seed, episode, observation.step))
        noise = rng.normal(0.0, self._sigma, truth.shape).astype("float32")
        return self._repeat(truth + noise)
