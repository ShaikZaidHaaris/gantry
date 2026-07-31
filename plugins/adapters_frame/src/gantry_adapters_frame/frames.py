"""Move a pose channel from the frame it was measured in to the frame it is executed in.

A pose is three numbers and a rotation *relative to something*, and the
something is not in the array. Two channels can agree on width, on encoding, on
units and on which arm is which, and still disagree about where the origin is —
at which point every command is well formed, smooth, and aimed half a metre from
where it was meant.

That is not hypothetical here. The ego pipeline recovers hand poses by solving
against the camera's own intrinsics, so its poses are in the **camera** frame,
and ``Mount.aligned()`` — the identity mount — passes them through unchanged.
A simulator executes end-effector poses in its **world** frame. Running one as
the other put the commanded positions 0.30 m (left) and 0.65 m (right) from
where the arms actually work.

The transform is known, not fitted
----------------------------------
This does not estimate an alignment. Estimating one from the commands and then
scoring the policy under it would tune the correction on the very episodes being
scored, and whatever came out would be a statement about the fit.

Instead the transform is *read*: a world that renders a camera knows where that
camera is, and publishes it. RoboTwin puts its head camera's extrinsics in every
observation. So the mapping from ego-camera coordinates to world coordinates is
the same rigid transform the renderer itself used, taken from the same place.

What this assumes, and where it would be wrong
-----------------------------------------------
That the camera the data was captured from and the camera the robot looks
through play the same role: both see the workspace from roughly where a head
would be. A head-mounted camera on a person and one on a robot are not the same
instrument, and this does not pretend they are. It claims only that "in front of
and below the viewpoint" means the same thing to both, which is the claim the
ego pipeline was already making implicitly by calling the identity mount aligned.
It is now written down, applied explicitly, and recorded.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from gantry_adapters_rotation.rotation import KEY, OFFSET_KEY, WIDTHS, from_matrix, to_matrix

from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

VERSION = "0.1.0.dev0"


def rigid(matrix: Any) -> np.ndarray:
    """A 4x4 rigid transform from a 4x4, a 3x4 or a (rotation, translation) pair.

    Accepts the shapes simulators actually publish. A 3x4 is the usual
    ``[R | t]`` and gets its bottom row; anything else is refused rather than
    reshaped into something plausible.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape == (3, 4):
        values = np.vstack([values, [0.0, 0.0, 0.0, 1.0]])
    if values.shape != (4, 4):
        raise ConfigError(
            f"a rigid transform is 4x4 or 3x4; got {values.shape}. Reshaping this "
            "would produce a transform that applies cleanly and means nothing"
        )
    return values


def invert(transform: np.ndarray) -> np.ndarray:
    """The inverse of a rigid transform, by transpose rather than by solve.

    A rigid transform's rotation is orthonormal, so its inverse is exact. Going
    through a general inverse would introduce error into a quantity that has
    none, and would quietly succeed on a matrix that is not rigid at all.
    """
    transform = rigid(transform)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ConfigError(
            "this transform's rotation is not orthonormal, so it is not a rigid "
            "motion and inverting it by transpose would be wrong. A scaled or "
            "skewed transform between frames is a different claim and needs saying"
        )
    out = np.eye(4)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ translation
    return out


def blocks_of(spec: ChannelSpec) -> tuple[tuple[int, int, str], ...]:
    """``(position start, rotation start, encoding)`` for each pose in a channel.

    Read from what the channel declares rather than inferred from its width. The
    convention is ``[xyz][rotation][gripper?]`` per arm, so the position is the
    three numbers immediately before each rotation block — which is exactly what
    ``rotation_offset`` already names, and why this needs no vocabulary of its own.
    """
    encoding = spec.metadata.get(KEY)
    if encoding is None:
        raise ConfigError(
            f"{spec.name!r} declares no rotation encoding, so there is no way to know "
            "which of its numbers are a rotation and which are a position. A frame "
            "shift applied to the wrong three numbers is undetectable downstream"
        )
    declared = spec.metadata.get(OFFSET_KEY)
    if declared is None:
        raise ConfigError(
            f"{spec.name!r} does not say where its rotations start, and a width alone "
            f"does not say. Set {OFFSET_KEY!r}"
        )
    if isinstance(declared, (int, np.integer)) and not isinstance(declared, bool):
        declared = (int(declared),)
    out = []
    for start in (int(value) for value in declared):
        if start < 3:
            raise ConfigError(
                f"{spec.name!r} puts a rotation at {start}, leaving no room for the "
                "position that a frame shift has to move with it"
            )
        out.append((start - 3, start, str(encoding)))
    return tuple(out)


def shift(values: np.ndarray, spec: ChannelSpec, transform: np.ndarray) -> np.ndarray:
    """Every pose in ``values``, expressed in the frame ``transform`` maps into.

    Positions are rotated *and* translated; orientations are only rotated. Doing
    the same thing to both is the classic version of this mistake and produces
    orientations that drift with how far the origin moved.
    """
    transform = rigid(transform)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    out = np.array(values, dtype=float, copy=True)
    if out.ndim == 1:
        out = out[None, :]
        flat = True
    else:
        flat = False

    for position, start, encoding in blocks_of(spec):
        width = WIDTHS[encoding]
        out[:, position : position + 3] = out[:, position : position + 3] @ rotation.T + translation
        turned = rotation @ to_matrix(out[:, start : start + width], encoding)
        out[:, start : start + width] = from_matrix(turned, encoding)

    return out[0] if flat else out


class PoseInFrame:
    """A policy whose poses are read and written in a different frame than the world's.

    Wraps a policy that thinks in one frame and is being run in another. The
    observation is brought *into* the policy's frame on the way in, and its
    commands are taken back *out* to the world's frame on the way out — one
    transform and its exact inverse, so nothing is applied twice and nothing is
    left half-converted.

    The transform is fetched from the observation each step, by name. A world
    that renders a camera publishes where that camera is; a world that does not
    cannot use this, and says so rather than assuming an identity that would be
    silently wrong whenever the camera moved.
    """

    def __init__(
        self,
        policy: Any,
        *,
        extrinsics: str,
        state_channels: Sequence[str] = (),
        world_frame: str = "world",
        policy_frame: str = "camera",
        name: str | None = None,
    ):
        self._policy = policy
        self._extrinsics = str(extrinsics)
        self._state = tuple(state_channels)
        self._world_frame = str(world_frame)
        self._policy_frame = str(policy_frame)
        self._name = name or f"{getattr(policy, '_name', 'policy')}@{policy_frame}"
        self._spec = policy.action_spec()
        # Fail now if the layout cannot be read, rather than on the first step.
        blocks_of(self._spec)
        self._seen: np.ndarray | None = None

    # -- contract ----------------------------------------------------------

    def action_spec(self) -> ChannelSpec:
        """The wrapped policy's channel, in the world's frame.

        The frame is the whole point of this component, so it is stated. A pose
        channel that does not say what it is relative to can be bound to
        anything of the same width.
        """
        from dataclasses import replace

        return replace(self._spec, frame=self._world_frame)

    def descriptor(self) -> Any:
        from dataclasses import replace

        inner = self._policy.descriptor()
        return replace(
            inner,
            metadata={
                **dict(inner.metadata),
                "pose_frame_from": self._policy_frame,
                "pose_frame_to": self._world_frame,
                "pose_frame_source": self._extrinsics,
            },
        )

    def observes(self) -> Any:
        return self._policy.observes()

    def reset(self, context: Any) -> None:
        self._seen = None
        self._policy.reset(context)

    def act(self, observation: Any) -> np.ndarray:
        channels = dict(getattr(observation, "channels", observation) or {})
        to_world = rigid(self._transform(channels))
        to_policy = invert(to_world)

        for name in self._state:
            if name in channels:
                channels[name] = shift(np.asarray(channels[name]), self._spec, to_policy)

        rebuilt = (
            channels
            if getattr(observation, "channels", None) is None
            else type(observation)(observation.step, channels)
        )
        commanded = np.asarray(self._policy.act(rebuilt))
        return shift(commanded, self._spec, to_world)

    # -- internals ---------------------------------------------------------

    def _transform(self, channels: dict) -> np.ndarray:
        """Camera-to-world, from what the world published.

        ``extrinsic_cv`` is the *world-to-camera* direction, which is the one
        renderers publish and the opposite of the one needed here. Named
        explicitly rather than guessed from the shape, because both directions
        are 3x4 and applying the wrong one is a plausible trajectory pointed
        backwards through the origin.
        """
        if self._extrinsics not in channels:
            raise ConfigError(
                f"{self._name}: this world publishes no {self._extrinsics!r}, so the "
                f"transform from {self._policy_frame!r} to {self._world_frame!r} is not "
                "knowable. Running anyway would execute camera-frame poses as world-frame "
                "ones, which is well formed and wrong by wherever the camera happens to be"
            )
        published = rigid(channels[self._extrinsics])
        # A world-to-camera matrix inverted once is camera-to-world.
        transform = invert(published) if self._extrinsics.endswith("_cv") else published
        self._seen = transform
        return transform

    def __getattr__(self, item: str) -> Any:
        return getattr(self._policy, item)
