"""Which Gantry channel becomes which key in an openpi observation.

A served policy is a socket, and the thing on the far end has already decided
what it reads: π₀.₅ trained on an ALOHA data config wants ``cam_high`` and a
fourteen-wide ``state``; the same weights served under the DROID config want
``observation/exterior_image_1_left`` and a joint vector plus a separate gripper.
Those are different wire contracts and neither is guessable from the checkpoint
name.

So the mapping is declared, and a layout is plain data. That has two consequences
worth stating:

**A preset is not a special case.** ``ALOHA`` below is a dict somebody could have
typed into a manifest. Adding a new robot config is adding an entry, not editing
this plugin, and nothing in the policy class knows the names of any of them.

**A wrong mapping is caught before anything moves.** The layout declares widths
and the requirement is built from it, so a fourteen-wide bimanual state wired to
a seven-wide single-arm server is refused by the resolver with both names, rather
than sending half an arm's worth of numbers and reading the result as poor
performance.

On the bimanual dimension labels
--------------------------------
Fourteen numbers with left and right concatenated is the most dangerous shape in
this file, because swapping the halves produces a perfectly valid array that
drives the wrong arm. Nothing about ``(14,) float32`` says which seven come
first. The labels are the only place that is written down, so they are written
down here and travel on the channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from gantry.errors import ConfigError

#: One ALOHA arm: six joints and a gripper, in the order the leader-follower
#: teleop rig reports them.
ARM = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")


def bimanual_labels(prefix_left: str = "left", prefix_right: str = "right") -> tuple[str, ...]:
    """Fourteen labels, left arm then right.

    The order is the claim. Swap it and every action is valid, well-shaped, and
    sent to the wrong arm.
    """
    return tuple(
        [f"{prefix_left}_{joint}" for joint in ARM] + [f"{prefix_right}_{joint}" for joint in ARM]
    )


@dataclass(frozen=True)
class Layout:
    """What the server on the other end reads and returns.

    ``images`` maps a Gantry channel name to the key openpi expects. ``state``
    and ``action`` are widths. ``prompt_key`` is the field the language
    instruction goes in — separate because a server that takes no prompt is a
    different kind of policy and should not be handed one silently.
    """

    name: str
    images: Mapping[str, str]
    state: int
    action: int
    state_key: str = "state"
    prompt_key: str | None = "prompt"
    #: Per-dimension names for the action and state vectors. Optional in general
    #: and close to mandatory for anything bimanual.
    labels: tuple[str, ...] = ()
    #: Whether the two halves of the vectors are two arms. Recorded because it
    #: changes what a retargeter may bind to this policy.
    arms: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.images:
            raise ConfigError(
                f"layout {self.name!r} names no cameras; a vision-language-action "
                "policy served with no image key would be reading nothing"
            )
        for width, what in ((self.state, "state"), (self.action, "action")):
            if int(width) < 1:
                raise ConfigError(f"layout {self.name!r}: {what} width must be positive")
        if self.labels and len(self.labels) != self.action:
            raise ConfigError(
                f"layout {self.name!r}: {len(self.labels)} labels for an action width "
                f"of {self.action}. The labels are the only written record of which "
                "half of a bimanual vector is which arm, so they cannot be approximate"
            )
        if self.arms > 1 and not self.labels:
            raise ConfigError(
                f"layout {self.name!r} is {self.arms}-armed and declares no dimension "
                "labels. Nothing about a concatenated vector says which arm comes "
                "first, and swapping them produces a valid array that drives the "
                "wrong arm"
            )

    @property
    def channels(self) -> tuple[str, ...]:
        """Gantry-side channel names this layout reads, cameras then state."""
        return (*self.images, self.state_key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": self.name,
            "images": dict(self.images),
            "state": self.state,
            "action": self.action,
            "arms": self.arms,
            "prompt_key": self.prompt_key,
            **dict(self.metadata),
        }


#: Bimanual ALOHA — two 6-DoF arms with grippers, three cameras. The
#: configuration this project is aimed at.
ALOHA = Layout(
    name="aloha",
    images={
        "cam_high": "cam_high",
        "cam_left_wrist": "cam_left_wrist",
        "cam_right_wrist": "cam_right_wrist",
    },
    state=14,
    action=14,
    labels=bimanual_labels(),
    arms=2,
    metadata={"rig": "ALOHA / ViperX bimanual", "control": "absolute joint positions"},
)

#: Single Franka, two cameras, joint position plus a separate gripper scalar.
DROID = Layout(
    name="droid",
    images={
        "exterior_image": "observation/exterior_image_1_left",
        "wrist_image": "observation/wrist_image_left",
    },
    state=8,
    action=8,
    state_key="observation/joint_position",
    labels=tuple([f"joint_{index}" for index in range(7)] + ["gripper"]),
    arms=1,
    metadata={"rig": "DROID / Franka", "control": "joint velocity"},
)

#: The LIBERO config, for comparing against a suite this project already runs.
LIBERO = Layout(
    name="libero",
    images={"image": "image", "wrist_image": "wrist_image"},
    state=8,
    action=7,
    labels=("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"),
    arms=1,
    metadata={"rig": "LIBERO / Panda", "control": "OSC pose deltas"},
)

#: By name, for a manifest. Not a registry the policy consults — a lookup table
#: a caller may use, so that an unlisted rig is a dict rather than a fork.
LAYOUTS: dict[str, Layout] = {layout.name: layout for layout in (ALOHA, DROID, LIBERO)}


def layout_for(raw: Layout | Mapping[str, Any] | str) -> Layout:
    """A layout from a name, a dict, or one already built.

    The dict path is what makes this modular in practice: a rig nobody here has
    heard of is described in JSON in a manifest and works, without this file
    knowing it exists.
    """
    if isinstance(raw, Layout):
        return raw
    if isinstance(raw, str):
        try:
            return LAYOUTS[raw]
        except KeyError:
            raise ConfigError(
                f"unknown layout {raw!r}; built in are {sorted(LAYOUTS)}. A rig not "
                "listed is described as an object with images/state/action rather "
                "than by adding a case here"
            ) from None
    data = dict(raw)
    labels = data.pop("labels", ())
    return Layout(
        name=str(data.pop("name", "custom")),
        images=dict(data.pop("images", {})),
        state=int(data.pop("state", 0)),
        action=int(data.pop("action", 0)),
        state_key=str(data.pop("state_key", "state")),
        prompt_key=data.pop("prompt_key", "prompt"),
        labels=tuple(str(label) for label in labels),
        arms=int(data.pop("arms", 1)),
        metadata=data,
    )


def check_labels(labels: Sequence[str], arms: int) -> None:
    """Both arms present and neither duplicated, for a multi-arm layout."""
    if arms < 2:
        return
    if len(set(labels)) != len(labels):
        raise ConfigError(f"duplicate dimension labels: {sorted(labels)}")
