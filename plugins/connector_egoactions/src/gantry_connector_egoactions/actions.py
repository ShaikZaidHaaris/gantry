"""Estimated hands as robot state and action, which is what a policy trains on.

The last link. Upstream there are episodes with a camera and a pair of estimated
hands; a training set needs ``observation.state`` and ``action`` at a declared
width, one row per step, with the language attached. This makes that conversion a
connector rather than a script, so the whole path from an uploaded folder to an
openpi-ready dataset is connectors composed over connectors and nothing else:

    egovideo -> handpose -> egoactions -> lerobot.write()

Why a connector and not a function
----------------------------------
Because everything downstream -- the writer, the resolver, the conformance kit,
the provenance -- already speaks connector. A function would have needed each of
those taught about it separately, and the one that got missed would be the one
that silently dropped a channel.

Actions are the next state, and that is a choice
------------------------------------------------
A demonstration records where the hand *was*. A policy is trained to predict what
to *do*. The convention here is that the action at step ``t`` is the state at
``t+1`` -- the pose the arm should move to next -- which is what the ALOHA-family
configs expect and what makes an absolute-pose action space meaningful.

The last step of an episode has no successor, so it is dropped rather than
repeated. Repeating it teaches a policy that the correct action in the final
state is to stay there, which is a lesson about the end of a recording rather
than about the task.

Unsolved frames are dropped, not interpolated
---------------------------------------------
A frame the estimator could not solve has no pose. Interpolating across it
invents a hand position nobody observed and puts it in the training set
indistinguishable from a real one. So those steps are cut, the episode is
reported as shorter than the footage, and how much was cut is on the record -- which is also the number that tells somebody their filming needs to change.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
from gantry_retargeter_hands import HandToArm, arm_command, assemble, hand_command

from gantry.contracts.connector import Connector, connector_descriptor
from gantry.errors import ConfigError
from gantry.spine import (
    ArraySource,
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeMeta,
    EpisodeRecord,
)

VERSION = "0.1.0.dev0"

#: What a LeRobot dataset calls the two channels a policy reads and writes.
STATE = "observation.state"
ACTION = "action"

#: Shortest episode worth keeping once unsolved frames are cut. Below this there
#: is no motion to learn from, only a pose.
MIN_STEPS = 8


def _resize(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Frames at the size a policy will actually see.

    Nearest-neighbour by index rather than an interpolating resize, so this needs
    no image library and cannot introduce a dependency into a connector. The
    training images are resized again by the model's own pipeline anyway; what
    matters here is that they stop being 1080p.
    """
    array = np.asarray(frames)
    if array.ndim != 4:
        return array
    height, width = int(size[0]), int(size[1])
    rows = (np.arange(height) * array.shape[1] // height).clip(0, array.shape[1] - 1)
    cols = (np.arange(width) * array.shape[2] // width).clip(0, array.shape[2] - 1)
    return array[:, rows][:, :, cols]


def _hold(arm: np.ndarray, solved: np.ndarray) -> tuple[np.ndarray, float]:
    """Carry the last solved pose forward through gaps.

    Forward fill and not interpolation: a held arm is a claim that it stayed
    where it was, which is true of an idle arm and is checkable against the
    footage. Interpolating between two solved poses across a gap invents motion
    that may not have happened, and puts it in the training set looking exactly
    like motion that did.
    """
    out = np.array(arm, dtype="float32", copy=True)
    filled = 0
    last: np.ndarray | None = None
    for index in range(len(out)):
        if solved[index]:
            last = out[index].copy()
        elif last is not None:
            out[index] = last
            filled += 1
    return out, filled / max(1, len(out))


def retargeter_from(spec: Any) -> Any:
    """A retargeter from plain data, or one already built.

    Same rule as everywhere else: an argument that has to cross a manifest
    accepts plain data. The measured quantities stay required -- a hand with no
    travel and a mount that was never established are still refused, and being
    reachable from JSON does not make them optional.
    """
    if spec is None or hasattr(spec, "accepts"):
        return spec
    if isinstance(spec, Mapping) and spec and all(hasattr(v, "accepts") for v in spec.values()):
        return dict(spec)

    from gantry_retargeter_hands import REACHES, Hand, HandToArm, Mount

    config = dict(spec)
    raw_mount = config.pop("mount", None)
    if raw_mount in (None, "aligned"):
        mount = Mount.aligned()
    elif isinstance(raw_mount, Mapping):
        mount = Mount(**dict(raw_mount))
    else:
        raise ConfigError(f"a mount is 'aligned' or an object, got {raw_mount!r}")

    raw_reach = config.pop("reach", None)
    if raw_reach is None:
        reach = None
    elif isinstance(raw_reach, str):
        try:
            reach = REACHES[raw_reach]
        except KeyError:
            raise ConfigError(
                f"unknown reach {raw_reach!r}; known: {sorted(REACHES)}. An arm not "
                "listed is described as an object with radius/inner/floor/ceiling"
            ) from None
    else:
        from gantry_retargeter_hands import Reach

        reach = Reach(**dict(raw_reach))

    raw_hand = config.pop("hand", None)
    if raw_hand is None:
        raise ConfigError(
            "a retargeter needs the person's hand measured: thumb-to-index closed "
            "and open, and the wrist-to-fingertip span if the source is unscaled. "
            "These are per-person and cannot be defaulted, an adult's open hand "
            "and a child's differ by a factor that lands in the gripper signal"
        )
    # One per hand, because people are not symmetric.
    if isinstance(raw_hand, Mapping) and {"left", "right"} & set(raw_hand):
        return {
            side: HandToArm(
                mount=mount, hand=Hand(**dict(values)), reach=reach, name=f"hands.{side}", **config
            )
            for side, values in raw_hand.items()
        }
    return HandToArm(mount=mount, hand=Hand(**dict(raw_hand)), reach=reach, **config)


class EgoActionConnector(Connector):
    """Hands from an upstream connector, as one arm's or two arms' state and action."""

    def __init__(
        self,
        source: Connector,
        *,
        retargeter: HandToArm | Mapping[str, HandToArm],
        hands: Sequence[str] = ("left", "right"),
        name: str = "egoactions",
        video: str = "ego_rgb",
        rotation_repr: str = "euler_xyz",
        control_mode: str = "eef_abs_pose",
        min_steps: int = MIN_STEPS,
        keep_video: bool = True,
        hold_missing: bool = False,
        size: tuple[int, int] | None = None,
        cache: bool = True,
    ):
        """``retargeter`` is one, or one per hand.

        One per hand is the honest default for a bimanual set, because each hand
        is calibrated separately -- people are not symmetric, and a single
        calibration applied to both puts the difference straight into the gripper
        signal of whichever was not measured.
        """
        self._source = source
        self._hands = tuple(hands)
        if not self._hands:
            raise ConfigError(f"{name}: needs at least one hand to retarget")
        retargeter = retargeter_from(retargeter)
        self._retargeters = (
            dict(retargeter)
            if isinstance(retargeter, Mapping)
            else {hand: retargeter for hand in self._hands}
        )
        missing = [hand for hand in self._hands if hand not in self._retargeters]
        if missing:
            raise ConfigError(f"{name}: no retargeter for {missing}")
        self._name = name
        self._video = video
        self._rotation = rotation_repr
        self._control_mode = control_mode
        self._min_steps = int(min_steps)
        self._keep_video = bool(keep_video)
        self._hold_missing = bool(hold_missing)
        # Stored frame size, after estimation. Estimation wants every pixel; a
        # policy does not -- openpi resizes to 224 internally regardless, so
        # keeping 1080p in the training set costs disk, encode time and, the way
        # it was found, 44 GB of RAM across a two-dozen-clip build.
        self._size = tuple(size) if size else None
        self._cache_on = bool(cache)
        self._per_arm = arm_command("arm", rotation_repr=rotation_repr, control_mode=control_mode)
        self._cache: dict[str, EpisodeRecord] = {}
        self._dropped: dict[str, dict[str, Any]] = {}

    # -- contract ----------------------------------------------------------

    @property
    def width(self) -> int:
        return self._per_arm.width * len(self._hands)

    @property
    def labels(self) -> tuple[str, ...]:
        """Per-dimension names, arms in the order given.

        The only written record of which half of the vector is which arm. A
        fourteen-wide float32 is the same object either way round, and a swap
        produces valid actions sent to the wrong arm.
        """
        parts = ("x", "y", "z", "rx", "ry", "rz", "grip")
        if self._per_arm.width != len(parts):  # pragma: no cover - other encodings
            parts = tuple(f"d{index}" for index in range(self._per_arm.width))
        return tuple(f"{hand}_{part}" for hand in self._hands for part in parts)

    def _channel(self, name: str, rate: float | None) -> ChannelSpec:
        return ChannelSpec(
            name,
            "vector",
            (self.width,),
            "float32",
            frame="base",
            rate_hz=rate,
            semantics="actuation" if name == ACTION else f"action.{self._control_mode}",
            dim_labels=self.labels,
            discriminators=("arms",),
            metadata={"arms": len(self._hands), "rotation_repr": self._rotation},
        )

    def descriptor(self) -> Descriptor:
        upstream = self._source.descriptor()
        return connector_descriptor(
            name=self._name,
            version=VERSION,
            lazy=False,
            stage_events=False,
            outcomes=bool(upstream.provides.get("outcomes", False)),
            media=self._keep_video,
            writes=False,
            derived_from=f"{upstream.name}@{upstream.version}",
            arms=len(self._hands),
            hands=list(self._hands),
            width=self.width,
            control_mode=self._control_mode,
            action_convention="action[t] is state[t+1]",
            hold_missing=self._hold_missing,
            # Carried forward, because a training set built through
            # non-commercial weights is itself encumbered.
            estimator_licence=upstream.metadata.get("licence", "unknown"),
            retargeters={
                hand: rt.provenance()["retargeter"] for hand, rt in self._retargeters.items()
            },
        )

    def episode_ids(self) -> tuple[str, ...]:
        return tuple(f"{self._name}/{i}" for i in self._source.episode_ids())

    def schema(self, episode_id: str) -> tuple[ChannelSpec, ...]:
        return self.open(episode_id).schema

    def open(self, episode_id: str) -> EpisodeRecord:
        if episode_id in self._cache:
            return self._cache[episode_id]
        derived = self._derive(episode_id)
        if self._cache_on:
            self._cache[episode_id] = derived
        return derived

    # -- the conversion ----------------------------------------------------

    def _derive(self, episode_id: str) -> EpisodeRecord:
        upstream_id = self._upstream(episode_id)
        original = self._source.open(upstream_id)
        rate = original.channel(self._video).rate_hz if original.has(self._video) else None

        per_arm: dict[str, np.ndarray] = {}
        per_hand_solved: dict[str, np.ndarray] = {}
        held: dict[str, float] = {}
        for hand in self._hands:
            wrist, aperture = f"{hand}_wrist", f"{hand}_aperture"
            for needed in (wrist, aperture):
                if not original.has(needed):
                    raise ConfigError(
                        f"{episode_id}: no {needed!r} channel; this reads hands "
                        f"estimated upstream and {upstream_id} has "
                        f"{list(original.channel_names)}"
                    )
            spec = original.channel(wrist)
            source = hand_command(
                wrist,
                hand=hand,
                scale=spec.metadata.get("scale", "normalized"),
                rotation_repr=spec.metadata.get("rotation_repr", "quat_wxyz"),
                frame="world",
            )
            values = np.concatenate(
                [original.array(wrist), original.array(aperture)[:, None]], axis=1
            )
            verdict = self._retargeters[hand].accepts(source, self._per_arm)
            verdict.raise_if_refused(f"{episode_id}: {hand} hand cannot be retargeted")
            arm = self._retargeters[hand].apply(values, source, self._per_arm)
            # A frame with no pose is at the origin.
            ok = np.any(original.array(wrist)[:, :3] != 0.0, axis=1)
            per_hand_solved[hand] = ok
            if self._hold_missing:
                arm, filled = _hold(arm, ok)
                held[hand] = filled
            per_arm[hand] = arm

        # Requiring every hand simultaneously is the strict reading and it is
        # brutal: measured on real ego footage, each hand solves in roughly half
        # the frames and the intersection was 8%. Holding an idle arm at its last
        # pose is the alternative, and it is a claim about the world rather than a
        # convenience -- a person working one-handed leaves the other still, so a
        # bimanual robot imitating them should too. Declared, recorded, and off by
        # default so nobody gets it without asking.
        any_hand = np.any(np.stack(list(per_hand_solved.values())), axis=0)
        every_hand = np.all(np.stack(list(per_hand_solved.values())), axis=0)
        if self._hold_missing:
            # A frame before *any* hand was ever seen cannot be filled from
            # anything, so it goes.
            seen = np.maximum.accumulate(any_hand.astype(int)).astype(bool)
            solved = any_hand | (seen & np.roll(seen, 1))
            solved[0] = any_hand[0]
        else:
            solved = every_hand

        merged = assemble(per_arm, self._channel(STATE, rate))
        kept = merged[solved]
        if len(kept) < self._min_steps + 1:
            raise ConfigError(
                f"{episode_id}: only {len(kept)} of {len(merged)} steps are usable "
                f"({', '.join(f'{h} {int(v.sum())}' for h, v in per_hand_solved.items())} "
                f"solved), which is below the {self._min_steps} needed for an episode "
                "with motion in it. This is a filming problem rather than a bug, the "
                "hands were not in frame"
            )

        # action[t] = state[t+1]. The final state has no successor, so it goes
        # rather than being repeated: repeating teaches a policy that the right
        # move in the last state is to stay there.
        state, action = kept[:-1], kept[1:]
        arrays: dict[str, np.ndarray] = {STATE: state, ACTION: action}
        schema = [self._channel(STATE, rate), self._channel(ACTION, rate)]
        if self._keep_video and original.has(self._video):
            frames = original.array(self._video)[solved][:-1]
            spec = original.channel(self._video)
            if self._size:
                frames = _resize(frames, self._size)
                spec = replace(spec, shape=(self._size[0], self._size[1], spec.shape[2]))
            arrays[self._video] = frames
            schema.insert(0, spec)

        self._dropped[episode_id] = {
            "steps_in": int(len(merged)),
            "steps_out": int(len(state)),
            "dropped_unsolved": round(1.0 - float(solved.mean()), 4),
            "both_hands_solved": round(float(every_hand.mean()), 4),
            **({"held_idle_arm": {h: round(v, 4) for h, v in held.items()}} if held else {}),
        }
        annotations = dict(original.labels.annotations)
        return EpisodeRecord(
            meta=EpisodeMeta(
                id=episode_id,
                source=self._name,
                task=original.meta.task or annotations.get("instruction"),
                # The body the actions are *for*, not the one that produced them.
                # The human is upstream and reachable through derived_from.
                embodiment=f"bimanual_{len(self._hands)}arm",
                derived_from=(upstream_id,),
                license=annotations.get("estimator_licence"),
                extra={
                    **dict(original.meta.extra),
                    **self._dropped[episode_id],
                    "action_convention": "action[t] is state[t+1]",
                    "arms": len(self._hands),
                },
            ),
            schema=tuple(schema),
            source=ArraySource(arrays),
            labels=EpisodeLabels(
                success=original.labels.success,
                annotations={**annotations, **self._dropped[episode_id]},
            ),
        )

    def yield_rate(self) -> dict[str, Any]:
        """How much of the footage survived, per episode and overall.

        The number that decides whether an upload was worth processing, and the
        one that turns into filming advice. Empty until episodes have been
        opened, because it is measured rather than predicted.
        """
        if not self._dropped:
            return {"measured": False}
        inn = sum(v["steps_in"] for v in self._dropped.values())
        out = sum(v["steps_out"] for v in self._dropped.values())
        return {
            "measured": True,
            "episodes": len(self._dropped),
            "steps_in": inn,
            "steps_out": out,
            "kept": round(out / inn, 4) if inn else 0.0,
            "per_episode": dict(self._dropped),
        }

    def _upstream(self, episode_id: str) -> str:
        prefix = f"{self._name}/"
        if not episode_id.startswith(prefix):
            raise KeyError(f"no episode {episode_id!r}")
        upstream = episode_id[len(prefix) :]
        if upstream not in self._source.episode_ids():
            raise KeyError(f"no episode {episode_id!r}")
        return upstream
