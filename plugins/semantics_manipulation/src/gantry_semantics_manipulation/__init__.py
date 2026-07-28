"""Manipulation channel semantics, aligned with the Inspect Robots vocabulary.

Core ships only generic meaning tags, because a robot-specific vocabulary in
core would privilege one kind of machine over every other. This package is that
vocabulary, as a plugin: install it and manipulation channels get full checking;
don't, and nothing in Gantry notices.

The action taxonomy is taken from `Inspect Robots
<https://github.com/robocurve/inspect-robots>`_ (MIT), where it is carried on
``ActionSemantics`` and has been exercised across seven real robot families. It
is adopted rather than invented for two reasons: it is field-tested in a way a
fresh design would not be, and using the same words means a channel described
by one framework resolves in the other without a translation layer.

The distinctions it makes are the ones that actually bite:

**Absolute versus delta.** ``joint_pos`` and ``joint_delta`` are the same width,
the same units, and the same dimension. Executing one as the other sends an arm
to a position it read as a displacement. No shape check catches this; only a
declared control mode does.

**Rotation representation.** A quaternion in ``wxyz`` and one in ``xyzw`` are
four floats each and differ by a permutation. Widths agree, so nothing else
notices.

**Gripper kind.** Binary and continuous grippers accept the same number and mean
different things by 0.5.

**Frame.** A pose in the base frame and the same pose in the world frame are
both three numbers.

Every one of these is a silent failure between two components that pass every
structural check, which is exactly the class of bug the descriptor discipline
exists to make loud.
"""

from .vocabulary import (
    CONTROL_MODES,
    DELTA_MODES,
    FRAMES,
    GRIPPER_KINDS,
    ROTATION_REPRS,
    ROTATION_WIDTHS,
    action_channel,
    control_mode_of,
    is_absolute,
    register_all,
    state_channel,
)

register_all()

__all__ = [
    "CONTROL_MODES",
    "DELTA_MODES",
    "FRAMES",
    "GRIPPER_KINDS",
    "ROTATION_REPRS",
    "ROTATION_WIDTHS",
    "action_channel",
    "control_mode_of",
    "is_absolute",
    "register_all",
    "state_channel",
]
