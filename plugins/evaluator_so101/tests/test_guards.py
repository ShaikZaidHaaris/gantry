"""The guards of this package, held against the one value that is not a number.

Every threshold check here is a comparison, and a comparison against NaN is
False. So the same guard can be written two ways that look equivalent and are
not::

    if not low <= x <= high:  raise      # NaN -> `not False` -> refused
    if x <= 0:                raise      # NaN -> False        -> accepted

The package uses both idioms, and every defect this file covers was in the
second. They share a shape: a guard that reads as a bound, accepts a value that
is not a bound, and then never fires — because the guard's own arithmetic is a
comparison too. A NaN clamp is not a loose clamp, it is no clamp, and nothing
downstream says so; a NaN travel limit converts in-range numbers into NaN
degrees and declares them converted; a NaN stall epsilon reports no contact
rather than no reading.

None of these are hardware-reachable in normal use. That is the argument for
checking them rather than against it: a limit nobody can trip is a limit nobody
notices is gone, and the number involved rides into provenance as the value
every episode was recorded under.

Where a non-finite value is *legitimate*, it stays legitimate, and that is
asserted too — ``SafetyLimits.unlimited`` encodes "no limit" as ``inf`` on
purpose, and a fix that broke it would have replaced a silent hole with a loud
one.

Everything here runs on a laptop with no arms and no serial library.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from gantry_evaluator_so101.arm import SafetyLimits as ArmLimits
from gantry_evaluator_so101.bimanual import so101_bimanual
from gantry_evaluator_so101.embodiment import (
    DOF,
    JOINT_NAMES,
    JointAnglesToNormalized,
    NormalizedToJointAngles,
    camera_channel,
    referee_camera_channel,
    so101_embodiment,
)
from gantry_evaluator_so101.transport import SafetyLimits as BusLimits

from gantry.conformance import check_embodiment
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

NAN = float("nan")
INF = float("inf")

#: A plausible calibrated travel, in degrees. Nobody's arm — see the note in
#: test_retargeters.py: this table is an argument precisely so it can never be a
#: default somebody inherits.
TRAVEL_DEG = {name: (-90.0, 90.0) for name in JOINT_NAMES}


def _camera(**kwargs):
    return camera_channel("observation.images.probe", frame="f", mount="m", **kwargs)


# -- 1. the relative-target guard -------------------------------------------


def test_a_nan_per_tick_limit_is_not_a_per_tick_limit():
    """The guard that stops the follower being driven into the table.

    ``abs(delta) > nan`` is False on every joint, so with a NaN limit
    ``_apply_relative_limit`` never fires: a command that moves a joint the
    whole range in one 33 ms tick is neither refused nor clamped, and
    ``clamped_ticks`` stays 0 so the episode records a clean run.
    """
    with pytest.raises(ConfigError, match="NaN") as raised:
        ArmLimits(max_relative_target=(NAN,) * DOF)
    # It names the joints and points at the supported way to do this on purpose.
    assert "shoulder_pan.pos" in str(raised.value)
    assert "unlimited" in str(raised.value)

    # One NaN among five real limits is the version that would actually ship.
    with pytest.raises(ConfigError, match="NaN"):
        ArmLimits(max_relative_target=(8.0, 8.0, NAN, 8.0, 8.0, 15.0))


def test_infinity_stays_legal_because_it_is_how_no_limit_is_spelled():
    """The check that must *not* have become ``isfinite``.

    ``unlimited()`` encodes no-limit as ``inf`` and demands a written reason for
    it, which travels into provenance. Refusing ``inf`` in ``__post_init__``
    would have broken the one supported way to run without the guard — and the
    reason requirement is exactly what a NaN was slipping past.
    """
    assert ArmLimits(max_relative_target=(INF,) * DOF).max_relative_target == (INF,) * DOF

    unlimited = ArmLimits.unlimited("arm is off the table on the bench")
    # Serialised as None, not inf: a reader sees "no limit", not a number.
    assert unlimited.as_dict()["max_relative_target"] == [None] * DOF
    assert unlimited.as_dict()["note"] == "arm is off the table on the bench"
    with pytest.raises(ConfigError, match="written reason"):
        ArmLimits.unlimited("   ")

    # And the ordinary presets are untouched.
    assert ArmLimits.teleop().max_relative_target == (8.0, 8.0, 8.0, 8.0, 8.0, 15.0)
    with pytest.raises(ConfigError, match="permits no motion"):
        ArmLimits(max_relative_target=(0.0,) * DOF)


# -- 2. the calibrated travel table -----------------------------------------


@pytest.mark.parametrize(
    "travel, why",
    [
        ((NAN, 90.0), "one bound unmeasured"),
        ((NAN, NAN), "the most degenerate entry there is"),
        ((-90.0, INF), "a joint whose travel ends nowhere"),
    ],
)
def test_a_travel_limit_that_is_not_a_number_is_refused(travel, why):
    """``nan == nan`` is False, so the span test could not see its worst case.

    The check below it exists to catch a zero-span joint, and ``(nan, nan)`` —
    a joint with no measured travel at all — walked straight through it.
    """
    bad = {**TRAVEL_DEG, "elbow_flex.pos": travel}
    for direction in (NormalizedToJointAngles, JointAnglesToNormalized):
        with pytest.raises(ConfigError, match="not finite") as raised:
            direction(bad)
        assert "elbow_flex.pos" in str(raised.value), why


def test_the_refusal_is_at_construction_because_the_input_check_cannot_see_this():
    """Why this one is not covered by the non-finite check in ``_in_range``.

    ``_in_range`` validates what arrives. A NaN travel limit produces a NaN from
    input that is clean and in range — the transform manufactures it — so the
    only place it can be caught is where the table is accepted. Asserting the
    good table converts to finite degrees is what makes the refusal meaningful
    rather than a blanket one.
    """
    with pytest.raises(ConfigError, match="not finite"):
        NormalizedToJointAngles({**TRAVEL_DEG, "elbow_flex.pos": (NAN, 90.0)})

    degrees = NormalizedToJointAngles(TRAVEL_DEG).apply(
        np.full((2, DOF), 50.0), so101_embodiment().state[0], _angles_spec()
    )
    assert np.all(np.isfinite(degrees))
    assert degrees == pytest.approx(0.0, abs=1e-4), "50 percent of -90..90 is 0 deg"

    # The zero-span refusal it sits in front of is still live.
    with pytest.raises(ConfigError, match="zero span"):
        NormalizedToJointAngles({**TRAVEL_DEG, "elbow_flex.pos": (10.0, 10.0)})


def _angles_spec() -> ChannelSpec:
    """The degree channel the calibrated conversions write into."""
    return ChannelSpec(
        name="observation.joint_angles",
        kind="vector",
        shape=(DOF,),
        dtype="float32",
        units="deg",
        frame="so101_follower_joints",
        dim_labels=JOINT_NAMES,
    )


# -- 3. the clamp a descriptor advertises -----------------------------------


@pytest.mark.parametrize("factory", [so101_embodiment, so101_bimanual], ids=["single", "bimanual"])
@pytest.mark.parametrize("value", [NAN, INF, 0.0, -1.0], ids=["nan", "inf", "zero", "negative"])
def test_a_descriptor_may_not_publish_a_clamp_that_bounds_nothing(factory, value):
    """The number in the record that says what the arm ran under.

    This is ``test_clamping_at_one_number_and_advertising_another_is_refused``
    reached from the other side: there the record claims 8.0 where the rig ran
    at 5.0, here it claims a limit that is not a limit at all. A NaN or infinite
    clamp passed ``<= 0`` and then passed ``check_embodiment(strict=True)``, so
    nothing between the factory and the corpus would have mentioned it.
    """
    with pytest.raises(ConfigError, match="does not bound anything"):
        factory(max_relative_target=value)


@pytest.mark.parametrize("factory", [so101_embodiment, so101_bimanual], ids=["single", "bimanual"])
def test_the_published_descriptors_are_unchanged_and_still_conform(factory):
    """The positive control. A guard that refused everything would pass above."""
    machine = factory()
    assert machine.action[0].metadata["max_relative_target"] == 8.0
    assert check_embodiment(machine, strict=True).ok
    assert factory(max_relative_target=5.0).action[0].metadata["max_relative_target"] == 5.0


def test_the_single_arms_ref_and_joint_order_survived_all_of_this():
    """The identity every recorded corpus is bound to. Nothing here may move it."""
    machine = so101_embodiment()
    assert machine.ref == "embodiment:so101_follower@1.0"
    assert machine.action[0].dim_labels == JOINT_NAMES
    assert machine.state[0].dim_labels == JOINT_NAMES
    assert machine.action[0].frame == "so101_follower_joints"


# -- 4. the stall proxy's thresholds ----------------------------------------


def test_a_nan_stall_threshold_reports_no_contact_rather_than_no_reading():
    """The gripper-stall proxy is the rig's only contact signal.

    A threshold that is never crossed produces an annotation saying the gripper
    never stalled, which reads as "no contact" and is really "no measurement".
    """
    with pytest.raises(ConfigError, match="NaN"):
        BusLimits.normalized(5.0, stall_epsilon_ticks=NAN)
    with pytest.raises(ConfigError, match="NaN"):
        BusLimits.normalized(5.0, stall_steps=NAN)


def test_the_legal_stall_thresholds_are_still_legal():
    """Zero epsilon is meaningful — it means any movement at all counts."""
    assert BusLimits.normalized(5.0, stall_epsilon_ticks=0).stall_epsilon_ticks == 0
    assert BusLimits.normalized(5.0).stall_epsilon_ticks == 3
    assert BusLimits.normalized(5.0, stall_steps=1).stall_steps == 1
    with pytest.raises(ConfigError):
        BusLimits.normalized(5.0, stall_steps=0)
    # The clamp check next to it, which was already right, stays right.
    for bad in (NAN, INF, 0.0, -1.0):
        with pytest.raises(ConfigError, match="not a clamp"):
            BusLimits.normalized(bad)


# -- 5. the frame size that goes into shape ---------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"height": NAN}, {"width": NAN}, {"height": 0}, {"width": -5}, {"height": 480.0}]
)
def test_a_frame_size_that_is_not_a_count_of_pixels_is_refused(kwargs):
    """``shape`` is what every consumer allocates against.

    ``height=nan`` produced the shape ``(nan, 640, 3)`` and passed strict
    conformance: a channel describing an image with no number of rows, declared
    as though it had been measured. A float is refused rather than truncated,
    because 480.0 and 480 are the same picture only if somebody rounds, and the
    rounding is the kind of decision this package makes explicit.
    """
    with pytest.raises(ConfigError, match="positive whole number of pixels"):
        _camera(**kwargs)


def test_the_referee_camera_inherits_the_check_and_the_defaults_still_build():
    assert _camera().shape == (480, 640, 3)
    assert referee_camera_channel("referee.images.overhead").shape == (480, 640, 3)
    with pytest.raises(ConfigError, match="positive whole number of pixels"):
        referee_camera_channel("referee.images.overhead", height=NAN)
    # The namespace refusal that was already there is undisturbed.
    with pytest.raises(ConfigError, match="observation namespace"):
        referee_camera_channel("observation.images.cheat")


# -- the family, asserted as a family ---------------------------------------


def test_no_threshold_in_this_package_accepts_a_nan():
    """One statement of the rule the five fixes above are instances of.

    A new guard written in the ``if x <= 0`` idiom will pass its own tests and
    fail this one, which is the only kind of test that catches the next
    occurrence rather than the last one.
    """
    constructors = {
        "arm per-tick clamp": lambda: ArmLimits(max_relative_target=(NAN,) * DOF),
        "bus clamp": lambda: BusLimits.normalized(NAN),
        "stall epsilon": lambda: BusLimits.normalized(5.0, stall_epsilon_ticks=NAN),
        "reference age": lambda: BusLimits.normalized(5.0, max_reference_age=NAN),
        "single-arm descriptor clamp": lambda: so101_embodiment(max_relative_target=NAN),
        "bimanual descriptor clamp": lambda: so101_bimanual(max_relative_target=NAN),
        "calibrated travel": lambda: NormalizedToJointAngles(
            {**TRAVEL_DEG, "gripper.pos": (0.0, NAN)}
        ),
        "camera height": lambda: _camera(height=NAN),
    }
    accepted = []
    for what, build in constructors.items():
        try:
            build()
        except ConfigError:
            continue
        accepted.append(what)
    assert not accepted, f"these accepted a NaN threshold: {accepted}"


def test_the_nan_safe_idiom_is_what_the_rest_of_the_package_already_used():
    """Proof the idiom works, so the fixes above are consistent with it.

    ``not low <= x <= high`` refuses NaN by construction. The guards written
    that way needed no change, and this is why.
    """
    low, high = 0.0, 100.0
    assert not (low <= NAN <= high)  # the comparison is False ...
    assert not low <= NAN <= high  # ... so the negated form refuses it
    assert not (NAN <= 0)  # while the positive form accepts it
    assert math.isnan(NAN)
