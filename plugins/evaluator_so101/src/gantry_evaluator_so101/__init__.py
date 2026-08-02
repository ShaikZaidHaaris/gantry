"""A leader-follower pair of SO-ARM101 arms, as an embodiment and an evaluator.

Two jobs, one rig. The follower runs a policy and the run comes back as
RunRecords like any simulator's (ARCHITECTURE.md plane 4, backend family 2).
The same loop, driven from the leader instead of from a policy, records
demonstration corpora — the corpus factory of CORPUS-PROTOCOL.md. Recording is
not a second code path: :class:`LeaderTeleop` wears the policy contract, so the
clamp, the calibration stamp and the tracking measurement are the same ones the
evaluation path uses.

Three declarations are load-bearing, because the resolver plans analyses from
them and a wrong claim produces a run that looks fine:

``seedable = False``
    A physical rig cannot return identical records for identical seeds. Object
    placement is a human act with its own error. Declaring True would let the
    resolver plan a paired comparison that is not valid, so the paired-seed
    power calculation in CORPUS-PROTOCOL.md applies to the sim tier only. The
    sentence to quote is :data:`SEEDABLE_IS_FALSE`.

``stage_events = False`` unless a pose source is configured
    A funnel needs object pose, which on hardware means a perception rig that
    does not exist yet. :func:`gripper_stall` infers contact from a servo that
    stopped following its command; that is a GRASP candidate and nothing else,
    and it is reported as an annotation rather than a stage event, because one
    populated rung beside several empty ones reads as a finished analysis. See
    :data:`STAGE_EVENTS_NEED_A_POSE_SOURCE`.

``closed_loop = True``, ``outcomes = True``
    The world responds to what the policy did, and a person watching can always
    say what happened.

The seam
--------
Four modules: ``transport`` is the servo bus and its mock, ``arm`` is one
six-servo arm plus the pair and their health signals, ``embodiment`` is the
plane-2 data and the retargeters, ``evaluator`` is the operator station. Where
two of them define the same word this file picks one and exports only that, so
the ambiguity is resolved once, here, instead of at each call site:

* joint order is exported once, as :data:`JOINTS` (``arm.JOINT_KEYS`` and
  ``embodiment.JOINT_NAMES`` are the same tuple under other names);
* :func:`so101_embodiment` is ``embodiment``'s, the one that also describes the
  cameras, the referee stream and the leader;
* ``Calibration``, ``JointCalibration`` and ``SafetyLimits`` exist in both
  ``transport`` and ``arm`` as genuinely different types — a bus-level register
  map versus an arm-level verified file — so neither is re-exported here.
  Import them from the module you mean.

:func:`_agree` runs at import and refuses the package outright if the modules
disagree about joint order, normalisation or calibration variant. That is not
defensive decoration: order is the entire content of a six-float action vector,
and a package whose modules disagree about it will drive the wrong joint while
every structural check passes.
"""

from gantry.errors import ConfigError

from .arm import CALIBRATION_VARIANT as _ARM_CALIBRATION
from .arm import (
    CONTROL_HZ,
    DOF,
    MOTOR_NAMES,
    SEEDABLE_IS_FALSE,
    STAGE_EVENTS_NEED_A_POSE_SOURCE,
    GripperStallDetector,
    SO101Arm,
    SO101Pair,
    analyse_drift,
    load_calibration,
    so101_channels,
)
from .arm import JOINT_KEYS as _ARM_JOINTS
from .arm import NORMALIZATION as _ARM_NORMALIZATION
from .embodiment import (
    CALIBRATION,
    FRONT_CAMERA,
    NORMALIZATION,
    REFEREE_CAMERA,
    WRIST_CAMERA,
    JointAnglesToNormalized,
    LeaderToFollower,
    NormalizedToJointAngles,
    policy_observation_channels,
    referee_channels,
    register_all,
    so101_embodiment,
    so101_leader,
)
from .embodiment import JOINT_NAMES as _EMBODIMENT_JOINTS
from .evaluator import (
    ACTION,
    GRIPPER,
    JOINTS,
    QUANTISATION,
    RANGE,
    REQUESTED,
    STATE,
    VERSION,
    Console,
    LeaderTeleop,
    PoseSource,
    Sample,
    ScriptedConsole,
    SO101Evaluator,
    TerminalConsole,
    VirtualClock,
    WallClock,
    action_spec,
    gripper_stall,
    mock_pair,
    state_spec,
)
from .evaluator import CALIBRATION_VARIANT as _EVALUATOR_CALIBRATION
from .transport import (
    MockTransport,
    SerialTransport,
    Transport,
    list_serial_ports,
    open_serial,
)


def _agree() -> None:
    """Refuse to publish a package whose modules describe different machines.

    Each of these is a value that every structural check downstream would
    happily accept while being silently wrong about which physical thing it
    names. Joint order is the worst of them: a permutation is not a rename, and
    an action vector assembled in one order and written in another drives the
    elbow with a wrist command.
    """
    for what, values in (
        ("joint order", {"arm": _ARM_JOINTS, "embodiment": _EMBODIMENT_JOINTS, "evaluator": JOINTS}),
        ("normalisation", {"arm": _ARM_NORMALIZATION, "embodiment": NORMALIZATION}),
        (
            "calibration variant",
            {"arm": _ARM_CALIBRATION, "embodiment": CALIBRATION, "evaluator": _EVALUATOR_CALIBRATION},
        ),
    ):
        distinct = {tuple(value) if isinstance(value, tuple) else value for value in values.values()}
        if len(distinct) > 1:
            raise ConfigError(
                f"gantry-evaluator-so101 is internally inconsistent about {what}: "
                + ", ".join(f"{module}={value!r}" for module, value in values.items())
                + ". These modules describe one physical arm and must agree."
            )


_agree()

__all__ = [
    "ACTION",
    "CALIBRATION",
    "CONTROL_HZ",
    "DOF",
    "FRONT_CAMERA",
    "GRIPPER",
    "JOINTS",
    "MOTOR_NAMES",
    "NORMALIZATION",
    "QUANTISATION",
    "RANGE",
    "REFEREE_CAMERA",
    "REQUESTED",
    "SEEDABLE_IS_FALSE",
    "STAGE_EVENTS_NEED_A_POSE_SOURCE",
    "STATE",
    "VERSION",
    "WRIST_CAMERA",
    "Console",
    "GripperStallDetector",
    "JointAnglesToNormalized",
    "LeaderTeleop",
    "LeaderToFollower",
    "MockTransport",
    "NormalizedToJointAngles",
    "PoseSource",
    "SO101Arm",
    "SO101Evaluator",
    "SO101Pair",
    "Sample",
    "ScriptedConsole",
    "SerialTransport",
    "TerminalConsole",
    "Transport",
    "VirtualClock",
    "WallClock",
    "action_spec",
    "analyse_drift",
    "gripper_stall",
    "list_serial_ports",
    "load_calibration",
    "mock_pair",
    "open_serial",
    "policy_observation_channels",
    "referee_channels",
    "register_all",
    "so101_channels",
    "so101_embodiment",
    "so101_leader",
    "state_spec",
]
