"""Which SO-101 machine a recorded array of floats belongs to, and the guard.

A LeRobot dataset holds ``action`` as float32 of some width. That is all it
holds. Nothing in the format says whether those numbers are one arm's six
joints or two arms' twelve, whether they are percent of calibrated travel or
radians, or which of the two published calibration conventions they are
relative to. ``docs/writing-a-plugin.md`` names the consequence: a channel with
no units and no meaning "can only be matched by name and width, which is
matching by coincidence".

This module is where the coincidence is removed. The caller *declares* which
embodiment the corpus was recorded on, and this module checks the declaration
against the file before anything downstream believes it:

* width must equal the embodiment's degrees of freedom, both directions. A
  12-wide action opened as ``embodiment:so101_follower@1.0`` is refused, and so
  is a 6-wide action opened as ``embodiment:so101_bimanual@1.0``;
* per-dimension names, when the dataset carries them, must equal the
  embodiment's joint names *in that order*. A rename is where an order swap
  hides, and on the bimanual ref a swapped pair of halves is a robot performing
  the task mirrored with every structural check green.

Width alone is not enough and is not claimed to be. Two single-arm corpora, one
recorded on the left arm of a bimanual bench and one on the right, are both six
wide, both percent-of-travel, both in the LeRobot joint order, and are different
machines. Only the declared ref distinguishes them, which is why the ref is
recorded on every episode rather than inferred from the array.

Everything about the embodiments themselves is imported from
``gantry_evaluator_so101``. This module owns no facts about SO-101 hardware; it
owns the checking of a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from gantry_evaluator_so101.bimanual import (
    BIMANUAL_DOF,
    BIMANUAL_JOINT_FRAME,
    BIMANUAL_JOINT_NAMES,
    DEFAULT_MOUNTING,
    MOUNTING_DISCRIMINATOR,
    MountingGeometry,
    bimanual_joint_channel,
)
from gantry_evaluator_so101.embodiment import (
    ACTION,
    CALIBRATION,
    CONTROL_HZ,
    DOF,
    FOLLOWER_JOINT_FRAME,
    JOINT_NAMES,
    NORMALIZATION,
    NORMALIZATION_RANGES,
    SEMANTICS_ACTION,
    SEMANTICS_STATE,
    STATE,
    joint_channel,
)

from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

#: The two refs this connector will bind a corpus to. Spelled as full refs
#: (``embodiment:<name>@<version>``) because the version is part of the
#: identity: if an ``@2.0`` SO-101 ever redefines the joint order, a corpus
#: bound to the bare name would silently follow it.
FOLLOWER_REF = "embodiment:so101_follower@1.0"
BIMANUAL_REF = "embodiment:so101_bimanual@1.0"

#: The extra discriminator this connector adds to the embodiment's own.
#: ``calibration_variant`` says which *convention* the numbers are relative to;
#: this says which calibration *files* produced them. Two corpora recorded on
#: the same rig under the same convention, either side of a recalibration, agree
#: on every other declared key and describe different mappings from these
#: numbers to poses. See :mod:`gantry_connector_so101.sidecar`.
CALIBRATION_DIGEST = "calibration_digest"


@dataclass(frozen=True)
class Binding:
    """One embodiment a corpus may be declared to be a recording of."""

    ref: str
    name: str
    version: str
    dof: int
    joint_names: tuple[str, ...]
    frame: str
    #: Builds one fully described joint channel. Delegated to the embodiment
    #: package so units, dtype, meaning tags and the discriminator keys have one
    #: owner and cannot drift from the descriptor a policy is resolved against.
    build: Callable[..., ChannelSpec]

    @property
    def joint_channels(self) -> tuple[str, ...]:
        """The channel names this binding governs. Others pass through."""
        return (STATE, ACTION)

    def semantics_for(self, channel: str) -> str:
        return SEMANTICS_STATE if channel == STATE else SEMANTICS_ACTION

    # -- the guard ---------------------------------------------------------

    def check(self, channel: str, width: int, dim_labels: tuple[str, ...] | None) -> None:
        """Refuse a recording that is not this machine's. Raises, never warns.

        Both directions are checked with one comparison, which is the point: a
        6-wide corpus opened as bimanual and a 12-wide corpus opened as a single
        follower are the same mistake seen from either end, and either one
        produces an action space that is half or twice the machine's.
        """
        if width != self.dof:
            raise ConfigError(
                f"{channel!r} is {width} wide but {self.ref} has {self.dof} degrees of "
                f"freedom ({', '.join(self.joint_names[:3])}...). "
                f"{_width_hint(width, self.ref)} "
                "Refusing rather than reshaping: downstream nothing can tell a bimanual "
                "corpus from a single-arm one except by width, so a width that disagrees "
                "with the declared embodiment is the last place the disagreement is "
                "visible."
            )
        if dim_labels is not None and tuple(dim_labels) != self.joint_names:
            same_set = set(dim_labels) == set(self.joint_names)
            raise ConfigError(
                f"{channel!r} labels its dimensions {list(dim_labels)} but {self.ref} is "
                f"{list(self.joint_names)}. "
                + (
                    "The same names in a different order is a permutation, not a rename: "
                    "executed as-is the arm moves every joint to another joint's target, "
                    "and on the bimanual ref a swapped pair of halves is the task "
                    "performed mirrored with every structural check green."
                    if same_set
                    else "These are different dimensions, so this corpus was not recorded "
                    "on the declared machine."
                )
            )


def _width_hint(width: int, ref: str) -> str:
    """Name the other binding when the width is exactly the other machine's."""
    for other in BINDINGS.values():
        if other.dof == width and other.ref != ref:
            return f"A {width}-wide recording is {other.ref}; declare that instead."
    return "Nothing installed here describes a machine of that width."


def _follower_channel(
    name: str,
    *,
    semantics: str,
    normalization: str,
    calibration_variant: str,
    rate_hz: float,
    extra: Mapping[str, Any] | None = None,
    mounting: MountingGeometry | None = None,
) -> ChannelSpec:
    if mounting is not None:
        raise ConfigError(
            f"a mounting geometry was supplied for {FOLLOWER_REF}, which is one arm. "
            "Base separation is a fact about a two-arm bench and describes nothing here; "
            "carrying it would put a discriminator on the channel that no single-arm "
            "corpus can honour."
        )
    return joint_channel(
        name,
        frame=FOLLOWER_JOINT_FRAME,
        semantics=semantics,
        normalization=normalization,
        calibration_variant=calibration_variant,
        rate_hz=rate_hz,
        extra=extra,
    )


def _bimanual_channel(
    name: str,
    *,
    semantics: str,
    normalization: str,
    calibration_variant: str,
    rate_hz: float,
    extra: Mapping[str, Any] | None = None,
    mounting: MountingGeometry | None = None,
) -> ChannelSpec:
    return bimanual_joint_channel(
        name,
        semantics=semantics,
        mounting=mounting if mounting is not None else DEFAULT_MOUNTING,
        normalization=normalization,
        calibration_variant=calibration_variant,
        rate_hz=rate_hz,
        extra=extra,
    )


BINDINGS: dict[str, Binding] = {
    FOLLOWER_REF: Binding(
        ref=FOLLOWER_REF,
        name="so101_follower",
        version="1.0",
        dof=DOF,
        joint_names=JOINT_NAMES,
        frame=FOLLOWER_JOINT_FRAME,
        build=_follower_channel,
    ),
    BIMANUAL_REF: Binding(
        ref=BIMANUAL_REF,
        name="so101_bimanual",
        version="1.0",
        dof=BIMANUAL_DOF,
        joint_names=BIMANUAL_JOINT_NAMES,
        frame=BIMANUAL_JOINT_FRAME,
        build=_bimanual_channel,
    ),
}


def get_binding(ref: str) -> Binding:
    """The declared embodiment, or a refusal naming the ones that exist.

    Nothing is inferred from the dataset's ``robot_type``. LeRobot's robot type
    is a free string chosen by whoever ran the recorder; treating it as an
    embodiment ref would make the binding a property of a typo.
    """
    try:
        return BINDINGS[ref]
    except KeyError:
        raise ConfigError(
            f"unknown embodiment ref {ref!r}; this connector binds {sorted(BINDINGS)}. "
            "The ref is required and is not guessed from the dataset: the width of an "
            "action array cannot say which arm of a two-arm bench recorded it, and "
            "LeRobot's robot_type is a free string."
        ) from None


__all__ = [
    "ACTION",
    "BIMANUAL_REF",
    "BINDINGS",
    "CALIBRATION",
    "CALIBRATION_DIGEST",
    "CONTROL_HZ",
    "FOLLOWER_REF",
    "MOUNTING_DISCRIMINATOR",
    "NORMALIZATION",
    "NORMALIZATION_RANGES",
    "STATE",
    "Binding",
    "MountingGeometry",
    "get_binding",
]
