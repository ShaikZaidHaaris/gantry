"""RoboTwin 2.0 as an evaluator — dual-arm, and it accepts end-effector actions."""

from .robotwin import (
    ACTION_TYPES,
    ARMS,
    CONTROL_HZ,
    TASKS,
    VERSION,
    DualArm,
    RoboTwinEvaluator,
    flatten,
    for_ego,
    labels_for,
    make_env,
    width_of,
)

__all__ = [
    "ACTION_TYPES",
    "ARMS",
    "CONTROL_HZ",
    "TASKS",
    "VERSION",
    "DualArm",
    "RoboTwinEvaluator",
    "flatten",
    "for_ego",
    "labels_for",
    "make_env",
    "width_of",
]
