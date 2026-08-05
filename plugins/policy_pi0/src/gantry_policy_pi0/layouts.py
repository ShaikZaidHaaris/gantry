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
    instruction goes in -- separate because a server that takes no prompt is a
    different kind of policy and should not be handed one silently.
    """

    name: str
    images: Mapping[str, str]
    state: int
    action: int
    #: The key openpi expects the state under, on the wire.
    state_key: str = "state"
    #: The key the cameras nest under, or ``None`` for flat top-level keys.
    #:
    #: Both shapes are real and neither is guessable from the config name. The
    #: Aloha-family transforms read ``data["images"]["cam_high"]``; the DROID one
    #: reads ``data["observation/exterior_image_1_left"]`` at the top level. Send
    #: the wrong shape and the server raises KeyError deep inside its own
    #: transform stack, which is a long way from anything that names the cause.
    images_key: str | None = "images"
    #: Whether the cameras go on the wire channel-first, ``(3, H, W)``.
    #:
    #: The Aloha-family transform does ``rearrange(img, "c h w -> h w c")``, so
    #: it wants channel-first and silently mangles anything else -- a (224,224,3)
    #: frame arrives as a 3-pixel-tall image of 224 channels and dies inside PIL,
    #: several layers below anything that names the cause. Nothing in the config
    #: name says which way round it is.
    channels_first: bool = True
    #: The Gantry channel it is read from. Separate because these are two
    #: different namespaces and conflating them was a real bug: a dataset whose
    #: channel is ``observation.state`` served to a config whose wire key is
    #: ``state`` failed with "the observation is missing ['state']", which reads
    #: as a missing channel rather than as a mapping that was never expressed.
    #: Images always had this mapping; state did not, and nothing noticed until a
    #: real server was on the other end.
    state_from: str | None = None
    prompt_key: str | None = "prompt"
    #: Per-dimension names for the action and state vectors. Optional in general
    #: and close to mandatory for anything bimanual.
    labels: tuple[str, ...] = ()
    #: Whether the two halves of the vectors are two arms. Recorded because it
    #: changes what a retargeter may bind to this policy.
    arms: int = 1
    #: Declared on both the action and the state channel. This is where a pose
    #: encoding goes: ``rotation_repr`` and ``rotation_offset`` are invisible in
    #: the width and change what every number means, so a layout that does not
    #: say cannot be converted to anything and binds only to a channel that got
    #: lucky.
    metadata: Mapping[str, Any] = field(default_factory=dict)
    #: Which of those keys are load-bearing rather than descriptive. Nominated
    #: rather than assumed: ``rig: "ALOHA / ViperX bimanual"`` is prose for a
    #: reader and would refuse every channel that did not repeat it word for
    #: word, while ``rotation_repr`` genuinely must match or be converted.
    discriminators: tuple[str, ...] = ()
    #: What the state channel is, semantically. Separate from the action's
    #: because a policy may read one thing and command another.
    state_semantics: str | None = None

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
        unknown = [key for key in self.discriminators if key not in self.metadata]
        if unknown:
            raise ConfigError(
                f"layout {self.name!r} nominates {unknown} as discriminating and "
                "declares no value for them, so every channel would be refused for "
                "disagreeing with nothing"
            )
        if self.arms > 1 and not self.labels:
            raise ConfigError(
                f"layout {self.name!r} is {self.arms}-armed and declares no dimension "
                "labels. Nothing about a concatenated vector says which arm comes "
                "first, and swapping them produces a valid array that drives the "
                "wrong arm"
            )

    @property
    def reads(self) -> str:
        """The Gantry channel the state comes from."""
        return self.state_from or self.state_key

    @property
    def channels(self) -> tuple[str, ...]:
        """Gantry-side channel names this layout reads, cameras then state."""
        return (*self.images, self.reads)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": self.name,
            "images": dict(self.images),
            "state": self.state,
            "action": self.action,
            "arms": self.arms,
            "state_key": self.state_key,
            "images_key": self.images_key,
            "channels_first": self.channels_first,
            "state_from": self.reads,
            "prompt_key": self.prompt_key,
            **dict(self.metadata),
        }


#: Bimanual ALOHA -- two 6-DoF arms with grippers, three cameras. The
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
    # DROID's transform reads flat top-level keys rather than a nested block,
    # and takes frames the way a camera hands them over.
    images_key=None,
    channels_first=False,
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

#: By name, for a manifest. Not a registry the policy consults -- a lookup table
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
        state_from=data.pop("state_from", None),
        images_key=data.pop("images_key", "images"),
        channels_first=bool(data.pop("channels_first", True)),
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
