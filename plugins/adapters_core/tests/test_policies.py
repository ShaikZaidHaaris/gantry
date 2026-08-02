"""A policy's actions, converted into the space the evaluator reads.

Gantry binds a policy's observations through the adapter plane and checks that
its actions can be accepted, but never converted the action stream at run time.
So a policy trained in one pose encoding and an evaluator reading another either
matched exactly or did not run -- with no third option. This is that option, and
the tests are mostly about what it still refuses to do.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_adapters_core import AdaptedPolicy, adapt_policy, installed_adapters

from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import AdapterRegistry, requires_channels
from gantry.spine import ChannelSpec

EULER = ChannelSpec(
    "action",
    "vector",
    (14,),
    "float32",
    semantics="action.eef_abs_pose",
    metadata={"rotation_repr": "euler_xyz", "rotation_offset": (3, 10), "arms": 2},
)
QUAT = ChannelSpec(
    "action",
    "vector",
    (16,),
    "float32",
    semantics="action.eef_abs_pose",
    metadata={"rotation_repr": "quat_wxyz", "rotation_offset": (3, 11), "arms": 2},
)


class Fixed(Policy):
    """Emits whatever it was given, so the conversion is visible in the output."""

    def __init__(self, spec=EULER, values=None, chunked=True):
        self.spec = spec
        self.values = values
        self.chunked = chunked

    def descriptor(self):
        return policy_descriptor(
            name="fixed", version="0.1", horizon=4, chunk=self.chunked, deterministic=True
        )

    def action_spec(self):
        return self.spec

    def observes(self):
        return requires_channels("fixed", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        if self.values is not None:
            return np.asarray(self.values, dtype=float)
        width = self.spec.width
        return np.zeros((4, width)) if self.chunked else np.zeros(width)


def test_a_chunk_of_euler_actions_comes_back_as_quaternions():
    values = np.zeros((4, 14))
    values[:, 5] = np.pi / 2  # left arm, a quarter turn in z
    values[:, 6] = 0.04  # left gripper
    values[:, 12] = np.pi / 2  # right arm, a quarter turn in z
    values[:, 13] = 0.08  # right gripper

    made = adapt_policy(Fixed(values=values), QUAT)
    out = made.act({})

    assert out.shape == (4, 16)
    half = np.sqrt(0.5)
    assert np.allclose(out[:, 3:7], [half, 0, 0, half])
    assert np.allclose(out[:, 7], 0.04)  # left gripper, moved but unchanged
    assert np.allclose(out[:, 11:15], [half, 0, 0, half])
    assert np.allclose(out[:, 15], 0.08)


def test_a_single_action_is_lifted_and_put_back_rather_than_read_as_a_chunk():
    """A bare (14,) is one action. Handed to an adapter as-is it would be read
    as fourteen steps of a one-wide channel."""
    made = adapt_policy(Fixed(chunked=False), QUAT)
    out = made.act({})
    assert out.shape == (16,)


def test_the_declared_action_is_the_one_the_evaluator_asked_for():
    made = adapt_policy(Fixed(), QUAT)
    assert made.action_spec().width == 16
    assert made.action_spec().metadata["rotation_repr"] == "quat_wxyz"


def test_a_policy_that_already_speaks_the_target_is_returned_untouched():
    """Not wrapped in an empty chain that would read as a conversion in the
    record."""
    policy = Fixed(spec=QUAT)
    assert adapt_policy(policy, QUAT) is policy


def test_an_unclosable_gap_is_refused_in_the_constructor():
    """Before a simulator is built and before a checkpoint is loaded, rather
    than a thousand steps into a run that has already booked the GPU."""
    joints = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="action.joint_pos",
        metadata={"arms": 2},
    )
    with pytest.raises(ConfigError, match="mean something else"):
        AdaptedPolicy(Fixed(), joints, adapters=AdapterRegistry())


def test_the_refusal_names_the_encodings_on_both_sides():
    with pytest.raises(ConfigError) as caught:
        AdaptedPolicy(Fixed(), QUAT, adapters=AdapterRegistry())  # empty registry
    message = str(caught.value)
    assert "euler_xyz" in message and "quat_wxyz" in message
    assert "14 wide" in message and "16 wide" in message


def test_the_conversion_is_recorded_on_the_descriptor():
    """A report that says a policy scored 12% should be able to say what was
    done to its output on the way."""
    made = adapt_policy(Fixed(), QUAT)
    metadata = made.descriptor().metadata
    assert metadata["action_adapted_from"] == "action"
    assert metadata["action_adapted_to"] == "action"
    assert any("rotation" in step for step in metadata["action_adapter_chain"])


def test_the_losses_are_the_adapters_claim_not_this_wrappers():
    """The rotation adapter claims its conversion is exact. That claim travels
    rather than being restated here."""
    made = adapt_policy(Fixed(), QUAT)
    assert made.losses == ()


def test_everything_that_is_not_the_action_is_the_policy_underneath():
    policy = Fixed()
    made = adapt_policy(policy, QUAT)
    made.reset("ctx")
    assert policy.context == "ctx"
    assert made.observes() is not None
    assert made.chunked is True  # reached through to the wrapped policy


def test_it_finds_installed_adapters_without_naming_one():
    registry = installed_adapters()
    assert len(registry) >= 1
    # the module itself names no adapter; the rotation one is found by entry point
    assert "metadata.mismatch" in registry.codes()


# -- the same gap, on the way in ------------------------------------------------
#
# ClosedLoop hands a policy the world's observation dict untouched. A world
# publishing poses as quaternions and a policy reading Euler angles fail the
# same way the actions did -- except quietly, because a state of the wrong width
# is a shape error and a state of the right width in the wrong encoding is not
# an error at all.

STATE_QUAT = ChannelSpec(
    "endpose.vector",
    "vector",
    (16,),
    "float32",
    semantics="observation.eef_abs_pose",
    metadata={"rotation_repr": "quat_wxyz", "rotation_offset": (3, 11), "arms": 2},
)
STATE_EULER = ChannelSpec(
    "endpose.vector",
    "vector",
    (14,),
    "float32",
    semantics="observation.eef_abs_pose",
    metadata={"rotation_repr": "euler_xyz", "rotation_offset": (3, 10), "arms": 2},
)


class Reader(Policy):
    """Records exactly what it was handed."""

    def __init__(self, wants=STATE_EULER):
        self.wants = wants
        self.saw = None

    def descriptor(self):
        return policy_descriptor(
            name="reader", version="0.1", horizon=1, chunk=False, deterministic=True
        )

    def action_spec(self):
        return QUAT

    def observes(self):
        return requires_channels("reader", "policy", self.wants)

    def reset(self, context):
        pass

    def act(self, observation):
        self.saw = dict(getattr(observation, "channels", observation))
        return np.zeros(16)


def test_the_state_is_converted_into_the_encoding_the_policy_reads():
    policy = Reader()
    made = adapt_policy(policy, QUAT, reading=(STATE_QUAT,))

    values = np.zeros(16)
    values[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]  # left, quarter turn in z
    made.act({"endpose.vector": values})

    state = policy.saw["endpose.vector"]
    assert state.shape == (14,)
    assert np.allclose(state[3:6], [0, 0, np.pi / 2])


def test_a_state_that_already_matches_is_passed_straight_through():
    policy = Reader(wants=STATE_QUAT)
    made = adapt_policy(policy, QUAT, reading=(STATE_QUAT,))
    values = np.arange(16, dtype=float)
    made.act({"endpose.vector": values})
    assert np.allclose(policy.saw["endpose.vector"], values)


def test_a_world_that_publishes_nothing_the_policy_can_read_is_refused_now():
    """Before the simulator is built, not on the first step."""
    unrelated = ChannelSpec("joint_action.vector", "vector", (14,), "float32")
    with pytest.raises(ConfigError, match="cannot read what this world publishes"):
        adapt_policy(Reader(), QUAT, reading=(unrelated,))


def test_channels_the_policy_did_not_ask_about_are_left_alone():
    policy = Reader()
    made = adapt_policy(policy, QUAT, reading=(STATE_QUAT,))
    image = np.zeros((4, 4, 3), dtype="uint8")
    made.act({"endpose.vector": np.zeros(16), "observation.head_camera.rgb": image})
    assert policy.saw["observation.head_camera.rgb"].shape == (4, 4, 3)


def test_both_directions_are_converted_in_one_run():
    """The point of doing it here rather than in two places."""
    policy = Reader()
    made = adapt_policy(policy, EULER, reading=(STATE_QUAT,))
    out = made.act({"endpose.vector": np.zeros(16)})
    assert policy.saw["endpose.vector"].shape == (14,)  # in
    assert out.shape == (14,)  # out
