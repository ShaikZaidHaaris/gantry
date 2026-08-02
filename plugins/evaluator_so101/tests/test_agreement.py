"""The import-time guard, watched failing.

``_agree`` runs when this package is imported and refuses to publish an
evaluator whose channels describe a different machine from the descriptor the
entry point provides. It is the only thing standing between the two halves of
this package and a corpus that cannot be bound to the arm that recorded it, and
until this file existed nobody had ever seen it refuse anything.

That distinction is not pedantic here. The guard's predecessor in ``__init__``
compared joint order, the normalization value and the calibration variant, and
it passed happily while the evaluator stamped ``embodiment:so101-follower@1.0``
against an entry point providing ``embodiment:so101_follower@1.0``, put its
joints in frame ``base`` instead of ``so101_follower_joints``, clamped at 5.0
while advertising 8.0, and spelled a load-bearing discriminator
``normalisation`` where the descriptor spelled it ``normalization``. Every one
of those was invisible to it. So each is perturbed below, one at a time, and the
guard has to refuse.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from gantry_evaluator_so101 import so101_embodiment
from gantry_evaluator_so101.evaluator import (
    EMBODIMENT_REF,
    _agree,
    action_spec,
    state_spec,
)

from gantry.errors import ConfigError
from gantry.spine import compatible


def test_the_package_as_shipped_agrees_with_itself():
    """The baseline. Without it, every refusal below could be a broken guard."""
    _agree()


def test_the_evaluators_action_binds_to_the_descriptors():
    """The question the resolver asks, asked here so a drift cannot ship.

    ``ok`` alone is not enough and the assertion on the reasons is the point:
    a verdict carrying no reasons is a real agreement, whereas an ``ok`` verdict
    full of ``metadata.undeclared`` notes is the spine saying it had nothing to
    compare.
    """
    verdict = compatible(action_spec(), so101_embodiment().action[0])
    assert verdict.ok, verdict.explain()
    assert verdict.reasons == (), verdict.explain()


def test_a_renamed_machine_is_refused():
    """The ref is what binds a record to the arm that produced it.

    A record stamped with a ref no installed plugin provides cannot be resolved
    to any description of any machine, so a rename in plane 2 is a decision
    about every corpus recorded before it and has to be made deliberately.
    """
    assert so101_embodiment().ref == EMBODIMENT_REF
    with pytest.raises(ConfigError, match="embodiment ref"):
        _agree(machine=replace(so101_embodiment(), name="so101-follower"))


def test_a_channel_in_another_frame_is_refused():
    """Two arms with identical numbers and different frames are two arms."""
    with pytest.raises(ConfigError, match="frame"):
        _agree(action=replace(action_spec(), frame="base"))
    with pytest.raises(ConfigError, match="frame"):
        _agree(state=replace(state_spec(), frame="base"))


def test_a_channel_meaning_something_else_is_refused():
    """Percent of calibrated travel is not the manipulation vocabulary's angle.

    ``action.joint_pos`` is pinned to the angle dimension. Declaring these
    numbers against it passes every structural check and is wrong by whatever
    the calibration file says, which is LeRobot #3758 written into the schema.
    """
    with pytest.raises(ConfigError, match="semantics"):
        _agree(action=replace(action_spec(), semantics="action.joint_pos"))
    with pytest.raises(ConfigError, match="semantics"):
        _agree(state=replace(state_spec(), semantics="state.joint_range"))


def test_a_different_set_of_load_bearing_keys_is_refused():
    """Dropping a discriminator drops a refusal, and nothing looks different."""
    thinner = replace(action_spec(), discriminators=("calibration_variant",))
    with pytest.raises(ConfigError, match="discriminator keys"):
        _agree(action=thinner)


def test_a_permuted_joint_order_is_refused():
    """Order is the whole content of a six-float action vector."""
    labels = action_spec().dim_labels
    swapped = (*labels[:3], labels[4], labels[3], labels[5])
    with pytest.raises(ConfigError, match="joint order"):
        _agree(machine=_with_action_labels(swapped))


def test_clamping_at_one_number_and_advertising_another_is_refused():
    """The exact defect: 5.0 applied at the rig, 8.0 in the record.

    Neither number is wrong on its own. The record is wrong, because it
    describes motion limits the follower never ran under, and a later reader
    comparing two campaigns has no way to see it.
    """
    with pytest.raises(ConfigError, match="per-tick clamp"):
        _agree(clamp=5.0)


def test_a_discriminator_spelled_differently_on_each_side_is_refused():
    """The subtle half, and the reason ``ok`` is not the bar.

    ``compatible`` can only compare a key both sides declare. A key present on
    one side alone comes back as a ``metadata.undeclared`` **note** on a verdict
    that is otherwise ``ok`` — so spelling it ``normalisation`` here and
    ``normalization`` there does not weaken the normalization check, it removes
    it. The one mechanism built to make a normalization mismatch a hard refusal
    was inert for exactly this reason, and it was inert silently.
    """
    action = action_spec()
    british = replace(
        action,
        discriminators=tuple(
            "normalisation" if key == "normalization" else key
            for key in action.discriminators
        ),
        metadata={
            **{key: value for key, value in action.metadata.items() if key != "normalization"},
            "normalisation": action.metadata["normalization"],
        },
    )

    verdict = compatible(british, so101_embodiment().action[0])
    assert verdict.ok, "this is the failure that passes; if it refuses, rewrite the test"
    assert verdict.because("metadata.undeclared"), (
        "the spine is expected to downgrade a one-sided key to a note"
    )

    with pytest.raises(ConfigError, match="normalis"):
        _agree(action=british)


def _with_action_labels(labels):
    """A descriptor whose action channel is labelled ``labels``."""
    machine = so101_embodiment()
    return replace(
        machine, action=(replace(machine.action[0], dim_labels=labels),)
    )
