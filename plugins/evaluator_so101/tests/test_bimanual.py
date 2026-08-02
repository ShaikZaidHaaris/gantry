"""The bimanual SO-101 held to the same kits as everything else, with no rig.

Three things are under test here and they are not equally interesting.

The dull one is that the descriptor conforms. It has to, and it would.

The load-bearing one is **order**. A twelve-vector whose halves can be swapped is
a robot doing the task mirrored: same width, same dtype, same units, same
normalization, same joint names, every structural check green. So the block
layout, the labels and the two grippers are asserted as data, and a permuted
channel is asserted to be *refused* — not merely different.

The one the plugin doc calls the most dangerous thing you can publish is
**declared loss**. Both retargeters here throw information away — one deletes six
degrees of freedom, the other invents six — and a transform that does either and
declares nothing produces plausible motion and a clean provenance. So the losses
are asserted to exist, to be non-blank, and to *name the specific thing that
went*: the dropped arm's labels, and the exact pose the idle arm is being held
at. A test that only asserted ``losses() != ()`` would pass against the word
"lossy", which is the failure it is supposed to catch.

Everything runs on a laptop with no arms, no driver board and no serial library.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
from dataclasses import replace

import numpy as np
import pytest
from gantry_evaluator_so101 import bimanual as bi
from gantry_evaluator_so101.bimanual import (
    BIMANUAL_DOF,
    BIMANUAL_JOINT_FRAME,
    BIMANUAL_JOINT_NAMES,
    GRIPPER_INDICES,
    MOUNTING_DISCRIMINATOR,
    SIDE_BLOCKS,
    SIDES,
    BimanualToSingle,
    MountingGeometry,
    SingleToBimanual,
    bimanual_joint_channel,
    default_cameras,
    dim_index,
    halves,
    join,
    side_labels,
    side_slice,
    so101_bimanual,
)
from gantry_evaluator_so101.embodiment import (
    ACTION,
    CALIBRATION,
    DOF,
    JOINT_DISCRIMINATORS,
    JOINT_NAMES,
    LEADER_JOINT_FRAME,
    NORMALIZATION,
    SEMANTICS_ACTION,
    SEMANTICS_STATE,
    STATE,
    joint_channel,
    policy_observation_channels,
    referee_camera_channel,
    referee_channels,
    so101_embodiment,
)

from gantry.conformance import check_embodiment, check_retargeter
from gantry.contracts.embodiment import CAP_SEEDABLE, CAP_SIMULATED
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec, Verdict, compatible

BIMANUAL_REF = "embodiment:so101_bimanual@1.0"
SINGLE_REF = "embodiment:so101_follower@1.0"


def _single(name: str = ACTION) -> ChannelSpec:
    """The single-arm follower channel, as the installed descriptor publishes it."""
    machine = so101_embodiment()
    return machine.action[0] if name == ACTION else machine.state[0]


def _bi(name: str = ACTION) -> ChannelSpec:
    machine = so101_bimanual()
    return machine.action[0] if name == ACTION else machine.state[0]


def _mid(steps: int = 5, width: int = BIMANUAL_DOF) -> np.ndarray:
    """Values inside every joint's travel, all distinct, so a slice is checkable."""
    return np.arange(steps * width, dtype=float).reshape(steps, width) % 90.0 + 5.0


# -- the kit ----------------------------------------------------------------


def test_the_bimanual_embodiment_conforms():
    verdict = check_embodiment(so101_bimanual(), strict=True)
    assert verdict.ok, verdict.explain()
    # No reasons at all, not merely ok. A pass carrying `channel_undescribed` or
    # `unlabelled_action` notes is the kit saying it had nothing to check.
    assert verdict.reasons == (), verdict.explain()


def test_the_single_arm_descriptor_is_untouched():
    """The regression this file must not cause.

    Adding a second machine to a package must not renegotiate the first one's
    identity: every corpus already recorded is stamped with that ref.
    """
    single = so101_embodiment()
    assert single.ref == SINGLE_REF
    assert single.action[0].dim_labels == JOINT_NAMES
    assert single.action[0].frame == "so101_follower_joints"
    assert check_embodiment(single, strict=True).ok


def test_the_ref_is_the_one_the_entry_point_publishes():
    """A record is bound to a machine by its ref, so the two must be the same object.

    Checked through ``importlib.metadata`` rather than by importing the factory,
    because what a resolver reaches is the entry point, and an entry point
    pointing at the wrong callable is invisible from inside the module.
    """
    assert so101_bimanual().ref == BIMANUAL_REF
    published = {
        entry.name: entry for entry in metadata.entry_points(group="gantry.embodiments")
    }
    assert "so101_bimanual" in published, f"installed: {sorted(published)}"
    assert "so101" in published, "the single arm must still be registered"
    factory = published["so101_bimanual"].load()
    assert factory().ref == BIMANUAL_REF
    assert published["so101"].load()().ref == SINGLE_REF


def test_no_bimanual_evaluator_is_registered():
    """Plane 2 only, deliberately.

    SO101Evaluator drives a leader-follower pair. There is no bimanual hardware
    here to test a runner against, and a runner in ``gantry list`` reads as a rig
    that runs. The descriptor says so in words as well, so a refusal can quote it.
    """
    evaluators = {entry.name for entry in metadata.entry_points(group="gantry.evaluators")}
    assert "so101_bimanual" not in evaluators
    declared = so101_bimanual().metadata["evaluator"]
    assert declared["registered"] is False
    assert "leader-follower" in declared["why"]


def test_it_claims_neither_seeding_nor_simulation():
    capabilities = so101_bimanual().capabilities
    assert CAP_SEEDABLE not in capabilities
    assert CAP_SIMULATED not in capabilities


# -- order is the contract --------------------------------------------------


def test_the_twelve_dimensions_are_labelled_left_then_right():
    assert BIMANUAL_DOF == 12
    assert SIDES == ("left", "right")
    assert BIMANUAL_JOINT_NAMES == (
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "right_shoulder_pan.pos",
        "right_shoulder_lift.pos",
        "right_elbow_flex.pos",
        "right_wrist_flex.pos",
        "right_wrist_roll.pos",
        "right_gripper.pos",
    )
    for channel in (_bi(ACTION), _bi(STATE)):
        assert channel.dim_labels == BIMANUAL_JOINT_NAMES
        assert channel.shape == (BIMANUAL_DOF,)


def test_the_split_is_machine_readable_and_not_left_to_arithmetic():
    """An off-by-six is a mirrored robot and is invisible in shape, dtype or units."""
    assert SIDE_BLOCKS == {"left": (0, DOF), "right": (DOF, 2 * DOF)}
    assert side_slice("left") == slice(0, 6)
    assert side_slice("right") == slice(6, 12)
    assert side_labels("left") == tuple(f"left_{j}" for j in JOINT_NAMES)
    assert side_labels("right") == tuple(f"right_{j}" for j in JOINT_NAMES)
    assert dim_index("right_gripper.pos") == 11
    assert bi.other_side("left") == "right" and bi.other_side("right") == "left"


def test_there_are_two_grippers_and_the_singular_index_is_gone():
    """Inheriting ``gripper_index=5`` would name the left jaw and hide the right one."""
    assert GRIPPER_INDICES == {"left": 5, "right": 11}
    for channel in (_bi(ACTION), _bi(STATE)):
        assert "gripper_index" not in channel.metadata
        assert "gripper_dim" not in channel.metadata
        assert channel.metadata["gripper_indices"] == GRIPPER_INDICES
        assert channel.metadata["gripper_dims"] == {
            "left": "left_gripper.pos",
            "right": "right_gripper.pos",
        }


def test_the_order_contract_is_carried_as_data_not_only_as_prose():
    order = so101_bimanual().metadata["order"]
    assert order["dim_labels"] == list(BIMANUAL_JOINT_NAMES)
    assert order["side_blocks"] == {"left": [0, 6], "right": [6, 12]}
    assert order["gripper_indices"] == GRIPPER_INDICES
    assert order["per_arm_joint_names"] == list(JOINT_NAMES)
    assert "mirrored" in order["contract"]


def test_halves_and_join_are_inverses_and_the_halves_are_not_interchangeable():
    values = _mid()
    split = halves(values)
    assert split["left"].shape == (5, DOF) and split["right"].shape == (5, DOF)
    assert np.array_equal(join(split["left"], split["right"]), values)
    # The whole reason the order is a contract: assembling them the other way
    # round is a legal twelve-vector describing a different machine's motion.
    mirrored = join(split["right"], split["left"])
    assert not np.array_equal(mirrored, values)


def test_halves_and_join_refuse_a_wrong_width_rather_than_reshaping():
    with pytest.raises(ConfigError, match="steps, 12"):
        halves(np.zeros((4, 6)))
    with pytest.raises(ConfigError, match="steps, 6"):
        join(np.zeros((4, 12)), np.zeros((4, 6)))
    with pytest.raises(ConfigError, match="timeline"):
        join(np.zeros((4, 6)), np.zeros((3, 6)))


def test_a_permuted_pair_of_halves_is_refused_and_not_merely_different():
    """The defect stated as a refusal code.

    Swapping the two halves keeps every label, so the spine reports
    ``dim_labels.order`` — "needs a permutation, not a rename" — rather than
    letting two channels that describe mirrored machines bind.
    """
    channel = _bi(ACTION)
    swapped = replace(
        channel,
        dim_labels=(*BIMANUAL_JOINT_NAMES[DOF:], *BIMANUAL_JOINT_NAMES[:DOF]),
    )
    verdict = compatible(swapped, channel)
    assert not verdict.ok
    assert "dim_labels.order" in verdict.codes(), verdict.explain()


def test_a_bimanual_channel_does_not_bind_to_a_single_arm_channel():
    """Half a machine is not the machine, and the resolver has to be told so."""
    verdict = compatible(_bi(ACTION), _single(ACTION))
    assert not verdict.ok
    assert "shape.mismatch" in verdict.codes()
    assert "dim_labels.mismatch" in verdict.codes()
    assert "frame.mismatch" in verdict.codes()


# -- one source of truth for what the numbers mean --------------------------


def test_the_channel_is_built_from_the_single_arm_builder():
    """The defect this package already had once, asserted shut.

    Units, dtype, kind, meaning tags, rate and the discriminator *keys* have to
    come from one place. Two modules disagreeing about what ``normalization``
    means does not weaken a check — it deletes it, because the spine downgrades a
    one-sided discriminator to a note.
    """
    single = joint_channel(
        ACTION,
        frame=BIMANUAL_JOINT_FRAME,
        semantics=SEMANTICS_ACTION,
    )
    wide = bimanual_joint_channel(ACTION, semantics=SEMANTICS_ACTION)
    for attribute in ("kind", "dtype", "units", "frame", "rate_hz", "semantics", "optional"):
        assert getattr(wide, attribute) == getattr(single, attribute), attribute
    for key in ("normalization", "range_min", "range_max", "calibration_variant",
                "control_mode", "gripper"):
        assert wide.metadata[key] == single.metadata[key], key
    assert set(JOINT_DISCRIMINATORS) <= set(wide.discriminators)


def test_both_descriptors_agree_about_units_semantics_and_normalization():
    single_action, wide_action = _single(ACTION), _bi(ACTION)
    single_state, wide_state = _single(STATE), _bi(STATE)
    assert wide_action.units == single_action.units == "percent"
    assert wide_action.semantics == single_action.semantics == SEMANTICS_ACTION
    assert wide_state.semantics == single_state.semantics == SEMANTICS_STATE
    assert wide_action.metadata["normalization"] == single_action.metadata["normalization"]
    assert wide_action.metadata["calibration_variant"] == CALIBRATION
    assert wide_action.rate_hz == single_action.rate_hz == 30.0
    assert wide_action.metadata["max_relative_target"] > 0


def test_an_unknown_normalization_is_refused_rather_than_defaulted():
    with pytest.raises(ConfigError, match="normalization"):
        bimanual_joint_channel(ACTION, semantics=SEMANTICS_ACTION, normalization="RANGE_0_1")


# -- cameras ----------------------------------------------------------------


def test_the_policy_sees_two_scene_views_and_one_wrist_per_arm():
    machine = so101_bimanual()
    names = [spec.name for spec in machine.state if spec.kind == "image"]
    assert names == [
        "observation.images.top",
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    for spec in default_cameras():
        assert spec.metadata["policy_visible"] is True
        assert spec.metadata["intrinsics_ref"] is None
        assert spec.metadata["calibrated"] is False
    assert {spec.name for spec in policy_observation_channels(machine)} == {
        STATE, *names
    }


def test_the_referee_stream_is_not_a_policy_observation():
    """The separation already exists in this package; this honours it.

    Ground truth lives outside ``state`` and outside the ``observation.``
    namespace, so a pipeline globbing ``observation.images.*`` into the policy
    cannot reach it. A channel the policy can see is a channel it can exploit.
    """
    referee = referee_camera_channel()
    machine = so101_bimanual(referee_camera=referee)
    assert referee.name not in [spec.name for spec in machine.state]
    assert referee.name not in [spec.name for spec in policy_observation_channels(machine)]
    recovered = referee_channels(machine)
    assert [spec.name for spec in recovered] == [referee.name]
    assert recovered[0].metadata["policy_visible"] is False
    assert referee_channels(so101_bimanual()) == ()


def test_a_referee_stream_in_the_observation_namespace_is_refused():
    with pytest.raises(ConfigError, match="observation namespace"):
        referee_camera_channel("observation.images.overhead")
    smuggled = replace(referee_camera_channel(), name="observation.images.truth")
    with pytest.raises(ConfigError, match="observation namespace"):
        so101_bimanual(referee_camera=smuggled)


def test_a_visible_referee_and_an_invisible_policy_camera_are_both_refused():
    visible = replace(
        referee_camera_channel(),
        metadata={**referee_camera_channel().metadata, "policy_visible": True},
    )
    with pytest.raises(ConfigError, match="policy_visible=False"):
        so101_bimanual(referee_camera=visible)

    hidden = replace(
        default_cameras()[0],
        metadata={**default_cameras()[0].metadata, "policy_visible": False},
    )
    with pytest.raises(ConfigError, match="cannot be described as both"):
        so101_bimanual(cameras=(hidden,))


# -- base separation is part of the machine ---------------------------------


def test_the_mounting_geometry_is_a_declared_field_on_the_descriptor():
    mounting = so101_bimanual().metadata["mounting"]
    assert mounting["base_separation_m"] > 0
    assert mounting["sides"] == list(SIDES)
    assert mounting["measured"] is False, "the shipped default has never been measured"
    assert mounting["method"].strip()
    assert mounting["reach_overlap_m"] is None, "not derivable from the defective URDF"


def test_the_mounting_rides_on_the_channels_as_a_discriminator():
    for channel in (_bi(ACTION), _bi(STATE)):
        assert MOUNTING_DISCRIMINATOR in channel.discriminators
        assert channel.metadata[MOUNTING_DISCRIMINATOR]


def test_two_rigs_at_different_separations_refuse_to_bind():
    """Twenty centimetres apart and forty centimetres apart are different machines.

    Same width, same dtype, same units, same normalization, same joint order —
    and a policy trained on one does not transfer. Nothing structural sees it, so
    the mounting id is a discriminator and the mismatch is a hard refusal.
    """
    near = so101_bimanual(mounting=MountingGeometry.nominal(0.200)).action[0]
    far = so101_bimanual(mounting=MountingGeometry.nominal(0.400)).action[0]
    assert near.metadata[MOUNTING_DISCRIMINATOR] != far.metadata[MOUNTING_DISCRIMINATOR]
    verdict = compatible(near, far)
    assert not verdict.ok
    assert "metadata.mismatch" in verdict.codes(), verdict.explain()
    assert compatible(far, so101_bimanual(mounting=MountingGeometry.nominal(0.400)).action[0]).ok


def test_a_measured_rig_is_not_the_same_claim_as_a_nominal_one():
    """The same number, two different provenances, and one of them is a guess."""
    nominal = MountingGeometry.nominal(0.400)
    measured = MountingGeometry.from_measurement(
        base_separation_m=0.400, method="callipers across the base plates, 3 readings"
    )
    assert nominal.base_separation_m == measured.base_separation_m
    assert nominal.mounting_id != measured.mounting_id
    assert measured.measured is True
    assert nominal.digest != measured.digest


def test_the_discriminator_covers_less_than_the_declaration_and_says_so():
    """A discriminator is a refusal, so what it does *not* refuse on has to be stated.

    ``mounting_id`` covers separation, yaw and measured-or-not. It deliberately
    leaves out the free text and the two fields that do not change the arm-to-arm
    geometry, because prose in a discriminator makes rewording a note a refusal.
    That is a defensible line; a docstring claiming the id covers the whole
    declaration would not be. So the excluded fields must still be *visible* in
    the record, which means the digest has to ride there too rather than being
    recomputable only by importing this module.
    """
    a = MountingGeometry.from_measurement(
        base_separation_m=0.400, method="callipers", table_height_m=0.75, reach_overlap_m=0.10
    )
    b = MountingGeometry.from_measurement(
        base_separation_m=0.400, method="tape", table_height_m=0.95, reach_overlap_m=0.30
    )
    assert a.mounting_id == b.mounting_id, "the id covers geometry, not prose or table height"
    assert a.digest != b.digest, "the digest covers everything the id does not"

    left, right = so101_bimanual(mounting=a).action[0], so101_bimanual(mounting=b).action[0]
    # They bind, on purpose -- and the record still shows the difference.
    assert compatible(left, right).ok
    assert left.metadata["mounting"]["digest"] != right.metadata["mounting"]["digest"]
    assert left.metadata["mounting"]["table_height_m"] == 0.75
    assert so101_bimanual().metadata["mounting"]["what_the_id_does_not_cover"]

    # A geometric difference is still a hard refusal.
    far = so101_bimanual(
        mounting=MountingGeometry.from_measurement(base_separation_m=0.200, method="callipers")
    ).action[0]
    assert "metadata.mismatch" in compatible(left, far).codes()


def test_an_impossible_or_undocumented_mounting_is_refused():
    with pytest.raises(ConfigError, match="not a separation"):
        MountingGeometry.nominal(0.0)
    with pytest.raises(ConfigError, match="not a separation"):
        MountingGeometry.nominal(-0.3)
    with pytest.raises(ConfigError, match="written method"):
        MountingGeometry.from_measurement(base_separation_m=0.4, method="  ")
    with pytest.raises(ConfigError, match="one arm assumed"):
        MountingGeometry(
            base_separation_m=0.4,
            base_yaw_deg=(0.0,),
            layout="x",
            measured=True,
            method="x",
        )
    with pytest.raises(ConfigError, match="nobody has established it"):
        MountingGeometry.from_measurement(
            base_separation_m=0.4, method="tape", reach_overlap_m=-1.0
        )


# -- the honest constraint --------------------------------------------------


def test_the_descriptor_says_this_rig_cannot_be_teleoperated_on_two_arms():
    """Two arms is a leader-follower pair; a bimanual follower pair leaves no leader.

    Recording bimanual demonstrations needs four arms. That is declared here so a
    reader finds it in the artifact rather than by buying two more arms.
    """
    teleop = so101_bimanual().metadata["teleoperation"]
    assert teleop["policy_evaluable"] is True
    assert teleop["teleop_recordable"] is False
    assert teleop["followers_required"] == 2
    assert teleop["leaders_required"] == 2
    assert teleop["arms_required_total"] == 4
    assert "leader-follower pair" in teleop["why"]


def test_the_corpus_factory_cannot_feed_this_embodiment():
    """A bimanual policy needs bimanual demonstrations, and the factory records pairs."""
    consequence = so101_bimanual().metadata["teleoperation"]["corpus_consequence"]
    assert "CORPUS-PROTOCOL.md" in consequence
    assert "cannot" in consequence


def test_the_constraint_is_in_the_notes_and_in_the_module_docstring():
    """Both places a human actually reads, not only the metadata dict."""
    notes = so101_bimanual().notes
    assert "NOT TELEOP-RECORDABLE" in notes
    assert "four arms" in notes
    assert "ORDER IS THE CONTRACT" in notes
    assert "NO EVALUATOR" in notes

    doc = bi.__doc__
    assert "NOT teleop-recordable" in doc
    assert "four" in doc
    assert "CORPUS-PROTOCOL.md" in doc


def test_everything_the_descriptor_emits_survives_a_cp1252_console():
    """Refusals and losses are a product surface, and they get printed on Windows.

    A loss statement carrying an em dash raises UnicodeEncodeError on a cp1252
    stdout, which turns "this transform dropped an arm" into a traceback about
    encoding. The sibling modules keep their runtime strings ASCII; this keeps
    that true for the second machine rather than relying on it staying true.
    Docstrings and comments are exempt: nobody prints them.
    """
    machine = so101_bimanual(referee_camera=referee_camera_channel())
    emitted = [json.dumps(dict(machine.metadata), default=str), machine.notes or ""]
    for side in SIDES:
        emitted.append(" ".join(BimanualToSingle(side).losses(_bi(ACTION), _single(ACTION))))
        emitted.append(
            " ".join(
                SingleToBimanual.at_mid_travel(side).losses(_single(ACTION), _bi(ACTION))
            )
        )
    for text in emitted:
        offending = sorted({character for character in text if ord(character) > 127})
        assert not offending, offending


def test_the_two_buses_are_declared_not_simultaneous():
    """A skew silently assumed to be zero is a corpus where each arm reacts early."""
    skew = so101_bimanual().metadata["bus_skew"]
    assert skew["buses"] == 2
    assert skew["simultaneous"] is False
    assert skew["measured"] is False


def test_no_force_or_torque_channel_is_declared_anywhere():
    machine = so101_bimanual()
    for spec in (*machine.state, *machine.action):
        assert spec.semantics not in {"force", "torque", "wrench"}
    assert machine.metadata["sensors"]["force_sensing"] is False
    assert machine.metadata["sensors"]["torque_sensing"] is False


# -- the retargeter kit -----------------------------------------------------


@pytest.mark.parametrize("side", SIDES)
def test_bimanual_to_single_passes_the_kit(side):
    verdict = check_retargeter(BimanualToSingle(side), _bi(ACTION), _single(ACTION))
    assert verdict.ok, verdict.explain()


@pytest.mark.parametrize("side", SIDES)
def test_single_to_bimanual_passes_the_kit(side):
    verdict = check_retargeter(
        SingleToBimanual.at_mid_travel(side), _single(ACTION), _bi(ACTION)
    )
    assert verdict.ok, verdict.explain()


@pytest.mark.parametrize("side", SIDES)
def test_both_directions_pass_the_kit_on_the_state_channels_too(side):
    assert check_retargeter(BimanualToSingle(side), _bi(STATE), _single(STATE)).ok
    assert check_retargeter(
        SingleToBimanual.at_mid_travel(side), _single(STATE), _bi(STATE)
    ).ok


# -- the losses, which are the point ----------------------------------------


@pytest.mark.parametrize("side", SIDES)
def test_dropping_an_arm_declares_which_arm_and_what_becomes_unrepresentable(side):
    retargeter = BimanualToSingle(side)
    losses = retargeter.losses(_bi(ACTION), _single(ACTION))
    assert losses and all(loss.strip() for loss in losses)
    text = " ".join(losses)
    dropped = bi.other_side(side)
    # Names the arm that went, by every one of its labels, not "lossy".
    for label in side_labels(dropped):
        assert label in text, label
    assert "unrepresentable" in text
    assert "handover" in text
    # The second arm does not vanish from the table just because it left the vector.
    assert "obstacle" in text
    assert MOUNTING_DISCRIMINATOR in text


@pytest.mark.parametrize("side", SIDES)
def test_padding_an_arm_declares_the_exact_pose_it_invented(side):
    retargeter = SingleToBimanual.at_mid_travel(side)
    losses = retargeter.losses(_single(ACTION), _bi(ACTION))
    assert losses and all(loss.strip() for loss in losses)
    text = " ".join(losses)
    idle = bi.other_side(side)
    assert idle in text
    assert "invented" in text
    # The pose itself, joint by joint, so a reader can tell where the idle arm was.
    for label in JOINT_NAMES:
        assert f"{label}=50" in text, label
    assert NORMALIZATION in text
    assert "mid-travel" in text, "the justification travels with the pose"
    assert "not a bimanual demonstration" in text


def test_the_zero_is_not_do_nothing_argument_is_written_into_the_loss():
    """The distinction that would otherwise slam an arm into a stop.

    Zero in RANGE_0_100 is a commanded absolute position at the end of travel.
    An implicit zero fill is six servos driven into their stops on the first
    tick, and every episode of it looks like data.
    """
    losses = SingleToBimanual.at_mid_travel("left").losses(_single(ACTION), _bi(ACTION))
    text = " ".join(losses)
    assert "0 in RANGE_0_100 is a commanded absolute position" in text
    assert "stops" in text


def test_the_zero_argument_stays_true_under_the_other_normalization():
    """A loss statement that is false under a legal configuration is the defect.

    ``RANGE_M100_100`` is a normalization this package declares and accepts, and
    on it zero is mid-travel, not an end of travel. "A zero-filled half drives
    six servos into their stops" is true on RANGE_0_100 and false here, and a
    loss lands in the provenance where nobody can check it against anything. So
    the sentence is derived from where zero actually sits on the target channel's
    declared range rather than from the bottom of that range, which only happened
    to be zero on the default rig.
    """
    wide = so101_bimanual(normalization="RANGE_M100_100").action[0]
    narrow = replace(
        _single(ACTION),
        metadata={**_single(ACTION).metadata, "normalization": "RANGE_M100_100"},
    )
    retargeter = SingleToBimanual(
        "left",
        rest_pose=(0.0,) * DOF,
        rest_pose_source="mid-travel in RANGE_M100_100",
        rest_pose_normalization="RANGE_M100_100",
    )
    # It genuinely is a legal pair: it passes accepts, so its losses ship.
    assert retargeter.accepts(narrow, wide).ok, retargeter.accepts(narrow, wide).explain()

    text = " ".join(retargeter.losses(narrow, wide))
    assert "0 in RANGE_M100_100 is mid-travel" in text, text
    assert "into their stops" not in text, "zero is not an end of travel on this range"
    # The universal half of the statement survives both conventions, because it
    # is the part that is actually about the transform.
    assert "no value means 'leave that arm where it is'" in text
    # And the idle half really is the declared pose, not a zero that means idle.
    out = retargeter.apply(np.full((3, DOF), 10.0), narrow, wide)
    assert np.allclose(out[:, side_slice("right")], 0.0)


@pytest.mark.parametrize("side", SIDES)
def test_a_non_finite_command_is_refused_by_both_directions(side):
    """NaN is the one bad value a range test cannot see.

    It compares False against both bounds, so ``< low | > high`` catches infinity
    and lets NaN through. On this ref these two transforms are the last thing
    between a diverged policy and a recorded action: there is no bimanual arm
    layer downstream to catch it, and the same class already refuses a non-finite
    rest pose.
    """
    wide = _mid()
    wide[1, GRIPPER_INDICES["right"]] = np.nan
    with pytest.raises(ConfigError, match="not a finite position"):
        BimanualToSingle(side).apply(wide, _bi(ACTION), _single(ACTION))

    narrow = _mid(width=DOF)
    narrow[0, 2] = np.nan
    with pytest.raises(ConfigError, match="not a finite position"):
        SingleToBimanual.at_mid_travel(side).apply(narrow, _single(ACTION), _bi(ACTION))

    # The refusal still names the joint, not an index.
    with pytest.raises(ConfigError, match="right_gripper.pos=nan"):
        BimanualToSingle(side).apply(wide, _bi(ACTION), _single(ACTION))

    # Infinity was already refused and stays refused, by the travel check.
    spoiled = _mid()
    spoiled[2, 0] = np.inf
    with pytest.raises(ConfigError, match="not a finite position"):
        BimanualToSingle(side).apply(spoiled, _bi(ACTION), _single(ACTION))


def test_a_retargeter_that_declared_no_loss_would_be_refused():
    """Proof the check is live rather than merely satisfied.

    ``Retargeter.check`` turns a width change with no declared loss into
    ``retarget.undeclared_loss``. Asserting that our losses are non-empty proves
    nothing unless something would fail if they were.
    """

    class Silent(BimanualToSingle):
        def losses(self, source, target):
            return ()

    verdict = Silent("left").check(_bi(ACTION), _single(ACTION))
    assert not verdict.ok
    assert "retarget.undeclared_loss" in verdict.codes(), verdict.explain()
    assert not check_retargeter(Silent("left"), _bi(ACTION), _single(ACTION)).ok


# -- what the transforms actually compute -----------------------------------


@pytest.mark.parametrize("side", SIDES)
def test_dropping_an_arm_takes_the_half_it_says_it_takes(side):
    values = _mid()
    out = BimanualToSingle(side).apply(values, _bi(ACTION), _single(ACTION))
    assert out.shape == (5, DOF)
    assert np.allclose(out, values[:, side_slice(side)])
    # A copy, not a view: an in-place edit afterwards must not rewrite history.
    out[0, 0] = 99.0
    assert values[0, side_slice(side)][0] != 99.0


@pytest.mark.parametrize("side", SIDES)
def test_padding_holds_the_idle_arm_at_the_declared_pose_and_never_at_zero(side):
    values = _mid(width=DOF)
    retargeter = SingleToBimanual.at_mid_travel(side)
    out = retargeter.apply(values, _single(ACTION), _bi(ACTION))
    assert out.shape == (5, BIMANUAL_DOF)
    assert np.allclose(out[:, side_slice(side)], values)
    idle = out[:, side_slice(bi.other_side(side))]
    assert np.allclose(idle, 50.0)
    assert not np.any(idle == 0.0), "zero is a real commanded position, not 'do nothing'"


@pytest.mark.parametrize("side", SIDES)
def test_the_placed_arm_survives_a_round_trip(side):
    values = _mid(width=DOF)
    wide = SingleToBimanual.at_mid_travel(side).apply(values, _single(ACTION), _bi(ACTION))
    back = BimanualToSingle(side).apply(wide, _bi(ACTION), _single(ACTION))
    assert np.allclose(back, values)


def test_a_custom_rest_pose_is_carried_through_verbatim():
    pose = (10.0, 20.0, 30.0, 40.0, 60.0, 70.0)
    retargeter = SingleToBimanual(
        "left", rest_pose=pose, rest_pose_source="measured stow pose, tape on the bench"
    )
    out = retargeter.apply(_mid(width=DOF), _single(ACTION), _bi(ACTION))
    assert np.allclose(out[:, side_slice("right")], np.asarray(pose))
    text = " ".join(retargeter.losses(_single(ACTION), _bi(ACTION)))
    assert "measured stow pose" in text
    assert "left_shoulder_pan.pos" not in text, "the rest pose is one arm's six joints"
    assert "shoulder_pan.pos=10" in text


# -- refusals ---------------------------------------------------------------


def test_out_of_travel_values_are_refused_and_the_refusal_names_the_arm():
    """Clamping turns a diverged policy into motion that looks deliberate.

    And on two arms the message has to say *which* gripper, or an operator reads
    it and walks to the wrong side of the bench.
    """
    values = _mid()
    values[2, 11] = 150.0
    with pytest.raises(ConfigError) as raised:
        BimanualToSingle("right").apply(values, _bi(ACTION), _single(ACTION))
    assert "right_gripper.pos=150" in str(raised.value)
    assert "refusing rather than clamping" in str(raised.value)

    narrow = _mid(width=DOF)
    narrow[1, 5] = -20.0
    with pytest.raises(ConfigError, match="gripper.pos=-20"):
        SingleToBimanual.at_mid_travel("left").apply(narrow, _single(ACTION), _bi(ACTION))


def test_a_wrong_width_is_refused_rather_than_padded():
    with pytest.raises(ConfigError, match="steps, 12"):
        BimanualToSingle("left").apply(_mid(width=DOF), _bi(ACTION), _single(ACTION))
    with pytest.raises(ConfigError, match="steps, 6"):
        SingleToBimanual.at_mid_travel("left").apply(_mid(), _single(ACTION), _bi(ACTION))


def test_an_unknown_side_is_refused():
    for constructor in (
        lambda: BimanualToSingle("centre"),
        lambda: SingleToBimanual.at_mid_travel("third"),
        lambda: side_slice("middle"),
    ):
        with pytest.raises(ConfigError, match="unknown side"):
            constructor()


def test_an_unknown_dimension_is_refused_rather_than_returning_minus_one():
    with pytest.raises(ConfigError, match="not a dimension"):
        dim_index("gripper.pos")


def test_a_rest_pose_that_is_not_six_reachable_numbers_is_refused():
    with pytest.raises(ConfigError, match="rest_pose must be 6"):
        SingleToBimanual("left", rest_pose=(50.0,) * 5, rest_pose_source="x")
    with pytest.raises(ConfigError, match="non-finite"):
        SingleToBimanual("left", rest_pose=(50.0,) * 5 + (float("nan"),), rest_pose_source="x")
    with pytest.raises(ConfigError, match="outside RANGE_0_100"):
        SingleToBimanual("left", rest_pose=(50.0,) * 5 + (140.0,), rest_pose_source="x")


def test_the_rest_pose_must_say_where_it_came_from():
    """An invented pose with no provenance is a number somebody chose once."""
    with pytest.raises(ConfigError, match="rest_pose_source is required"):
        SingleToBimanual("left", rest_pose=(50.0,) * DOF, rest_pose_source="   ")


def test_a_rest_pose_written_in_another_convention_is_declined():
    """50 is mid-travel in RANGE_0_100 and three quarters of the way in RANGE_M100_100.

    Same number, different pose, no structural difference — so the convention the
    pose was written in is checked against the channel it will be executed on.
    """
    other = so101_bimanual(normalization="RANGE_M100_100").action[0]
    source = replace(
        _single(ACTION),
        metadata={**_single(ACTION).metadata, "normalization": "RANGE_M100_100"},
    )
    verdict = SingleToBimanual.at_mid_travel("left").accepts(source, other)
    assert not verdict.ok
    assert "retarget.rest_pose_normalization" in verdict.codes(), verdict.explain()


def test_mismatched_normalization_or_calibration_is_declined():
    target = _single(ACTION)
    for key, value in (
        ("normalization", "RANGE_M100_100"),
        ("calibration_variant", "so101_old_calib"),
    ):
        drifted = replace(target, metadata={**target.metadata, key: value})
        verdict = BimanualToSingle("left").accepts(_bi(ACTION), drifted)
        assert not verdict.ok
        assert f"retarget.{key}" in verdict.codes(), verdict.explain()


def test_an_undeclared_discriminator_is_an_unanswered_question_not_a_match():
    target = _single(ACTION)
    silent = replace(
        target,
        metadata={k: v for k, v in target.metadata.items() if k != "calibration_variant"},
    )
    verdict = BimanualToSingle("left").accepts(_bi(ACTION), silent)
    assert not verdict.ok
    assert "retarget.undeclared" in verdict.codes(), verdict.explain()


def test_a_state_channel_may_not_feed_an_action_channel():
    """Where the arm is is not where it is told to go.

    Both tags are dimensionless percent-of-travel, so only the meaning tag
    separates them; letting one feed the other would command the idle arm to
    wherever it was last seen — motion, out of a transform asked only to reshape.
    """
    verdict = BimanualToSingle("left").accepts(_bi(STATE), _single(ACTION))
    assert not verdict.ok
    assert "semantics.mismatch" in verdict.codes(), verdict.explain()


def test_a_leader_arm_is_not_a_retarget_target():
    """The leader declares no action channels; nothing may plan a rollout onto it."""
    leader = replace(_single(ACTION), frame=LEADER_JOINT_FRAME)
    verdict = BimanualToSingle("left").accepts(_bi(ACTION), leader)
    assert not verdict.ok
    assert "frame.mismatch" in verdict.codes(), verdict.explain()


def test_a_source_that_is_not_this_machine_is_declined_by_both_directions():
    single, wide = _single(ACTION), _bi(ACTION)
    assert not BimanualToSingle("left").accepts(single, single).ok
    assert not SingleToBimanual.at_mid_travel("left").accepts(wide, wide).ok

    unlabelled = replace(wide, dim_labels=None)
    verdict = BimanualToSingle("left").accepts(unlabelled, single)
    assert not verdict.ok
    assert "retarget.not_so101_bimanual" in verdict.codes(), verdict.explain()


def test_a_declined_pair_makes_the_kit_report_a_decline_rather_than_a_pass():
    """The kit must not silently pass a retargeter that never ran."""
    verdict = check_retargeter(BimanualToSingle("left"), _single(ACTION), _single(ACTION))
    assert not verdict.ok
    assert "conformance.declined" in verdict.codes(), verdict.explain()


def test_a_clamp_of_zero_is_refused():
    with pytest.raises(ConfigError, match="positive per-tick limit"):
        so101_bimanual(max_relative_target=0.0)


def test_the_retargeters_are_deterministic_and_named_for_provenance():
    for retargeter in (BimanualToSingle("right"), SingleToBimanual.at_mid_travel("right")):
        assert retargeter.name.startswith("so101-")
        assert "right" in retargeter.name
        assert retargeter.version
    assert isinstance(
        BimanualToSingle("left").accepts(_bi(ACTION), _single(ACTION)), Verdict
    )
