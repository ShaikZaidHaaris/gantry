"""A cheap fitted policy, for asking whether data carries signal at all.

What this is for, and what it is not
------------------------------------
This is not a robot policy and will never drive anything. It exists to answer
one narrow question, offline, in about a minute, for roughly the price of
nothing: **is there a learnable relationship between what the camera saw and
what the hands did?**

That question is worth its own cheap answer because the expensive one costs a
day. In this project a full fine-tune and a closed-loop evaluation were run on
data whose relationship nobody had checked, and eight GPU-hours produced two
zeros. A probe that costs a minute would have said so first.

Why ridge regression on downsampled pixels
------------------------------------------
Because it cannot flatter the data. It has no capacity to memorise, one
hyperparameter, no initialisation, no schedule, no early stopping, and it fits
in closed form -- so the same episodes give the same answer every time, and the
treatment arm and its shuffled control differ in *nothing* except which frames
sit beside which actions. A deep probe would introduce a dozen ways for the two
fits to differ by accident, and every one of them would be indistinguishable
from the signal being measured.

Its weakness is the honest one: a linear map on coarse pixels finds only coarse
structure. Data it calls empty is very likely empty; data it calls learnable has
cleared a low bar, not a high one. That is the correct shape for a screening
gate -- cheap, and wrong only in the direction of letting something through.

Which channels it is allowed to read
------------------------------------
Whatever the caller says, and the caller should think about it. Proprioceptive
state predicts the next action almost perfectly in *every* dataset, because the
next action is mostly where the arm already is -- so a probe allowed to read it
finds signal in footage whose pixels say nothing at all, and finds it against
the shuffled control too, since the control's state comes from the donor
episode. Both arms are affected, but not equally: the real arm keeps a trivially
predictive channel and the control loses it, so the comparison reports a large
effect that is entirely about proprioception and not about the contribution.

That is not a hypothetical. A fixture built with random actions and untouched
state passed the gate cleanly before ``kinds`` existed.

So a caller screening *video* passes ``kinds=("image",)`` and gets an answer
about the imagery. The default reads everything, which is right for a caller who
knows what its channels are and wrong for a gate that does not.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from gantry.contracts.policy import Observation, Policy, policy_descriptor
from gantry.resolve import Requirement, requires_channels
from gantry.spine import ChannelSpec, Descriptor

VERSION = "0.1.0.dev0"

#: Side of the square each image channel is reduced to before being flattened.
#: Small on purpose. The probe is not meant to see fine detail -- it is meant to
#: notice whether coarse scene structure moves with the actions -- and a larger
#: grid buys resolution the model cannot use while costing time the gate does
#: not have.
GRID = 8

#: Ridge penalty. Large enough that a short episode cannot be memorised through
#: a wide feature vector, which is the one way this probe could report signal
#: that is not there.
PENALTY = 10.0

#: Steps sampled per episode when fitting. A cap rather than everything: a
#: thousand-frame clip contributes a thousand nearly identical rows otherwise,
#: and the long clips would decide the answer for the whole upload.
PER_EPISODE = 200


def _image_features(frames: np.ndarray, grid: int) -> np.ndarray:
    """Area-average each frame down to ``grid`` x ``grid``, in grey."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim == 4 and values.shape[-1] in (1, 3, 4):
        values = values[..., :3].mean(axis=-1)
    if values.ndim != 3:
        return values.reshape(len(values), -1)
    steps, height, width = values.shape
    rows = np.array_split(np.arange(height), min(grid, height))
    cols = np.array_split(np.arange(width), min(grid, width))
    out = np.empty((steps, len(rows), len(cols)), dtype=np.float32)
    for r, row in enumerate(rows):
        band = values[:, row, :].mean(axis=1)
        for c, col in enumerate(cols):
            out[:, r, c] = band[:, col].mean(axis=1)
    return out.reshape(steps, -1) / 255.0


def features(episode: Any, names: Sequence[str], grid: int = GRID) -> np.ndarray:
    """One row per step: every observation channel, flattened and concatenated.

    Channels are read in the order given rather than in whatever order a reader
    returns them, so the feature vector means the same thing for every episode
    in a cohort. Two episodes whose columns disagree would make the fit
    meaningless in a way nothing downstream could detect.
    """
    read = episode.read(list(names))
    blocks = []
    for name in names:
        values = np.asarray(read[name])
        if values.ndim >= 3:
            blocks.append(_image_features(values, grid))
        else:
            blocks.append(values.reshape(len(values), -1).astype(np.float32))
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(episode), 0), np.float32)


class FittedProbe(Policy):
    """The result of a fit: a linear map from observations to one action."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: np.ndarray,
        names: tuple[str, ...],
        spec: ChannelSpec,
        *,
        grid: int = GRID,
        trained_on: int = 0,
    ):
        self._weights = weights
        self._bias = bias
        self._names = names
        self._spec = spec
        self._grid = grid
        self._trained_on = trained_on

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            name="probe",
            version=VERSION,
            chunk=1,
            deterministic=True,
            stateful=False,
            fitted_on_steps=self._trained_on,
            reads=list(self._names),
        )

    def action_spec(self) -> ChannelSpec:
        return self._spec

    def observes(self) -> Requirement:
        return requires_channels("probe", "policy", *self._names)

    def act(self, observation: Observation) -> np.ndarray:
        blocks = []
        for name in self._names:
            values = np.asarray(observation[name])
            if values.ndim >= 2:
                blocks.append(_image_features(values[None, ...], self._grid)[0])
            else:
                blocks.append(values.reshape(-1).astype(np.float32))
        row = np.concatenate(blocks) if blocks else np.zeros(0, np.float32)
        return (row @ self._weights + self._bias).reshape(1, -1)


class ProbeLearner:
    """Fits a :class:`FittedProbe`. Satisfies ``gantry.contracts.policy.Learns``."""

    def __init__(
        self,
        *,
        grid: int = GRID,
        penalty: float = PENALTY,
        per_episode: int = PER_EPISODE,
        action: str = "action",
        kinds: Sequence[str] | None = None,
    ):
        self._grid = int(grid)
        self._penalty = float(penalty)
        self._per_episode = int(per_episode)
        self._action = action
        #: Channel kinds the probe may read. ``None`` means all of them, which
        #: is the right default for a caller that knows its own channels and the
        #: wrong one for a gate screening somebody else's upload.
        self._kinds = None if kinds is None else tuple(kinds)

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            name="probe-learner",
            version=VERSION,
            chunk=1,
            deterministic=True,
            stateful=False,
            grid=self._grid,
            penalty=self._penalty,
            kinds=None if self._kinds is None else list(self._kinds),
        )

    def observation_names(self, episodes: Sequence[Any]) -> tuple[str, ...]:
        """Channels every episode has, minus the one being predicted.

        The intersection, not the union. A channel two thirds of the upload
        carries would make the feature vector mean different things for
        different episodes, and there is nothing downstream that could notice.
        """
        common: set[str] | None = None
        for episode in episodes:
            names = {
                spec.name
                for spec in episode.schema
                if spec.name != self._action
                and (self._kinds is None or spec.kind in self._kinds)
            }
            common = names if common is None else (common & names)
        return tuple(sorted(common or ()))

    def fit(self, episodes: Sequence[Any], *, seed: int | None = 0) -> FittedProbe:
        """A probe fitted to these episodes. Never mutates the learner.

        Not a nicety: the treatment arm and its shuffled control are two fits
        from one learner, and state carried between them would quietly make the
        control resemble the treatment -- which is the comparison itself.
        """
        if not episodes:
            raise ValueError("nothing to fit")
        names = self.observation_names(episodes)
        if not names:
            raise ValueError("the episodes share no observation channel to read")

        rng = np.random.default_rng(0 if seed is None else seed)
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for episode in episodes:
            steps = len(episode)
            if steps == 0:
                continue
            take = np.arange(steps)
            if steps > self._per_episode:
                take = np.sort(rng.choice(steps, self._per_episode, replace=False))
            rows.append(features(episode, names, self._grid)[take])
            targets.append(np.asarray(episode.array(self._action), np.float32).reshape(steps, -1)[take])

        if not rows:
            raise ValueError("every episode was empty")
        x = np.concatenate(rows).astype(np.float64)
        y = np.concatenate(targets).astype(np.float64)

        # Centre both, so the penalty applies to the relationship and not to the
        # means. An uncentred ridge shrinks the intercept too, which makes every
        # prediction drift toward zero and both arms look equally bad.
        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - x_mean, y - y_mean
        gram = xc.T @ xc + self._penalty * np.eye(xc.shape[1])
        weights = np.linalg.solve(gram, xc.T @ yc)
        bias = y_mean - x_mean @ weights

        spec = next(
            (s for s in episodes[0].schema if s.name == self._action),
            ChannelSpec(name=self._action, kind="control", shape=(y.shape[1],)),
        )
        return FittedProbe(
            weights, bias, names, spec, grid=self._grid, trained_on=int(len(x))
        )
