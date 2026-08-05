"""Isolation that actually isolates, and the two kits that were missing."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from gantry.conformance import (
    KITS,
    check_adapter,
    check_connector,
    check_embodiment,
    check_retargeter,
)
from gantry.contracts.embodiment import EmbodimentDescriptor, Retargeter
from gantry.fixtures import make_clean
from gantry.isolate import RemoteConnector, isolated_or_refuse, wants_isolation
from gantry.spine import ChannelSpec, Descriptor, Verdict

# ==========================================================================
# isolation
# ==========================================================================


@pytest.fixture
def csv_path(tmp_path):
    from gantry_connector_csv import write_episodes

    return write_episodes(make_clean(n=6, seed=1).episodes, tmp_path / "d.csv")


def test_a_connector_in_another_interpreter_behaves_identically(csv_path):
    """The point of isolation: the host never imports what it was avoiding."""
    remote = RemoteConnector("csv", {"path": str(csv_path)}, python=sys.executable)
    try:
        assert len(remote.episode_ids()) == 6
        episode = remote.open("ep_0000")
        assert episode.channel_names == ("position", "engagement", "action")
        assert len(episode) == 42
    finally:
        remote.close()


def test_it_passes_the_full_conformance_kit_over_the_pipe(csv_path):
    remote = RemoteConnector("csv", {"path": str(csv_path)}, python=sys.executable)
    try:
        verdict = check_connector(remote, strict=True)
        assert verdict.ok, verdict.explain()
    finally:
        remote.close()


def test_reads_stay_lazy_across_the_boundary(csv_path):
    """A window is forwarded, not materialised on the far side and shipped whole."""
    remote = RemoteConnector("csv", {"path": str(csv_path)}, python=sys.executable)
    try:
        episode = remote.open("ep_0000")
        window = episode.read(["position"], start=3, stop=7)
        assert set(window) == {"position"}
        assert window["position"].shape == (4, 3)
        assert np.allclose(window["position"], episode.array("position")[3:7])
    finally:
        remote.close()


def test_the_same_numbers_come_back_as_in_process(csv_path):
    from gantry_connector_csv import CsvConnector

    local = CsvConnector(csv_path)
    remote = RemoteConnector("csv", {"path": str(csv_path)}, python=sys.executable)
    try:
        for episode_id in local.episode_ids():
            for name in ("position", "action"):
                assert np.allclose(
                    local.open(episode_id).array(name), remote.open(episode_id).array(name)
                )
    finally:
        remote.close()


def test_an_unknown_episode_raises_across_the_boundary(csv_path):
    remote = RemoteConnector("csv", {"path": str(csv_path)}, python=sys.executable)
    try:
        with pytest.raises(KeyError):
            remote.open("no-such-episode")
    finally:
        remote.close()


def test_a_plugin_missing_from_the_other_environment_is_reported(tmp_path):
    from gantry.errors import ComponentError

    with pytest.raises(ComponentError, match="not installed"):
        RemoteConnector("no-such-connector", {}, python=sys.executable)


def test_isolation_is_read_from_the_descriptor():
    assert not wants_isolation(Descriptor("dataset", "a", "1.0", "connector@1.0"))
    assert wants_isolation(
        Descriptor("dataset", "a", "1.0", "connector@1.0", isolation="container")
    )


def test_a_plane_that_cannot_be_proxied_is_refused_not_run_anyway():
    """Declaring isolation and being run in-process anyway defeats the purpose."""
    descriptor = Descriptor("policy", "big", "1.0", "policy@1.0", isolation="container")
    component, verdict = isolated_or_refuse(descriptor, "policy", "big")
    assert component is None
    assert "isolate.unsupported_plane" in verdict.codes()
    assert "run the whole manifest inside the environment" in verdict.reasons[0].hint


# ==========================================================================
# the embodiment kit
# ==========================================================================


def arm(**over) -> EmbodimentDescriptor:
    base = dict(
        name="arm",
        version="1.0",
        state=(
            ChannelSpec(
                "joints",
                "vector",
                (6,),
                "float32",
                units="rad",
                semantics="joint_position",
                dim_labels=tuple(f"j{i}" for i in range(6)),
            ),
        ),
        action=(
            ChannelSpec(
                "command",
                "vector",
                (6,),
                "float32",
                units="rad",
                semantics="joint_position",
                dim_labels=tuple(f"j{i}" for i in range(6)),
            ),
        ),
        control_hz=20.0,
    )
    return EmbodimentDescriptor(**{**base, **over})


def test_a_well_described_embodiment_conforms_strictly():
    verdict = check_embodiment(arm(), strict=True)
    assert verdict.ok, verdict.explain()


def test_an_undescribed_channel_is_a_note_and_a_strict_failure():
    bare = arm(action=(ChannelSpec("command", "vector", (6,), "float32"),))
    assert check_embodiment(bare).ok
    assert "conformance.channel_undescribed" in check_embodiment(bare, strict=True).codes()


def test_an_unlabelled_actuator_is_caught():
    unlabelled = arm(
        action=(
            ChannelSpec(
                "command", "vector", (6,), "float32", units="rad", semantics="joint_position"
            ),
        )
    )
    verdict = check_embodiment(unlabelled, strict=True)
    assert "conformance.unlabelled_action" in verdict.codes()
    assert "index" in verdict.because("conformance.unlabelled_action")[0].hint


def test_a_missing_control_rate_is_reported():
    assert "conformance.no_control_rate" in check_embodiment(arm(control_hz=None)).codes()


def test_an_embodiment_that_cannot_be_commanded_says_so():
    assert "embodiment.no_action" in check_embodiment(arm(action=())).codes()


def test_the_kit_is_discoverable():
    assert "embodiment" in KITS and "adapter" in KITS


# ==========================================================================
# retargeters
# ==========================================================================


SEVEN = ChannelSpec(
    "a",
    "vector",
    (7,),
    "float32",
    units="rad",
    semantics="joint_position",
    dim_labels=tuple(f"j{i}" for i in range(7)),
)
SIX = ChannelSpec(
    "a",
    "vector",
    (6,),
    "float32",
    units="rad",
    semantics="joint_position",
    dim_labels=tuple(f"j{i}" for i in range(6)),
)


def test_a_real_retargeter_conforms():
    from gantry_retargeters_core import DropDimensions

    verdict = check_retargeter(DropDimensions(["j6"]), SEVEN, SIX)
    assert verdict.ok, verdict.explain()


class _SilentlyDrops(Retargeter):
    """Discards a dimension and mentions nothing. The dangerous shape."""

    @property
    def name(self) -> str:
        return "silent"

    @property
    def version(self) -> str:
        return "1.0"

    def accepts(self, source, target) -> Verdict:
        return Verdict.yes()

    def losses(self, source, target) -> tuple[str, ...]:
        return ()

    def apply(self, values, source, target):
        return np.asarray(values)[:, : target.width]


def test_a_retargeter_that_discards_silently_is_caught():
    """It produces plausible motion and a clean provenance. Nothing else finds it."""
    verdict = check_retargeter(_SilentlyDrops(), SEVEN, SIX)
    assert not verdict.ok
    # the contract's own check fires first, and the kit reports it
    assert any("loss" in code for code in verdict.codes()), verdict.explain()


# ==========================================================================
# the adapter kit
# ==========================================================================


def metres(**over) -> ChannelSpec:
    base = dict(
        name="position",
        kind="vector",
        shape=(3,),
        dtype="float32",
        units="m",
        frame="world",
        rate_hz=20.0,
        semantics="position",
    )
    return ChannelSpec(**{**base, **over})


def test_the_real_adapters_conform():
    from gantry_adapters_core import PERMUTE, RESAMPLE, UNIT_CONVERT

    assert check_adapter(UNIT_CONVERT, metres(units="mm"), metres()).ok
    assert check_adapter(RESAMPLE, metres(rate_hz=30.0), metres(rate_hz=20.0)).ok

    labelled = metres(dim_labels=("a", "b", "c"))
    reordered = metres(dim_labels=("c", "a", "b"))
    assert check_adapter(PERMUTE, labelled, reordered).ok


def test_an_adapter_with_no_codes_can_never_be_found():
    from gantry.resolve import Adapter

    orphan = Adapter("orphan", "1.0", closes=(), transform=lambda v, s, t: v)
    assert "conformance.closes_nothing" in check_adapter(orphan, metres(), metres()).codes()


def test_an_undotted_code_is_refused():
    from gantry.resolve import Adapter

    sloppy = Adapter("sloppy", "1.0", closes=("units",), transform=lambda v, s, t: v)
    assert "conformance.code_shape" in check_adapter(sloppy, metres(), metres()).codes()


def test_a_false_lossless_claim_is_caught():
    """The check that matters: provenance would tell every reader nothing was lost."""
    from gantry.resolve import Adapter

    lossy_but_quiet = Adapter(
        "quiet",
        "1.0",
        closes=("units.scale",),
        transform=lambda values, source, target: np.round(np.asarray(values), 0),
        cost=lambda source, target: (),
    )
    verdict = check_adapter(lossy_but_quiet, metres(units="mm"), metres())
    assert "conformance.not_lossless" in verdict.codes()
    assert "nothing was given up" in verdict.because("conformance.not_lossless")[0].hint


def test_a_false_length_claim_is_caught():
    from gantry.resolve import Adapter

    liar = Adapter(
        "liar",
        "1.0",
        closes=("rate.mismatch",),
        transform=lambda values, source, target: np.asarray(values)[::2],
        cost=lambda source, target: ("halved",),
        preserves_length=True,
    )
    verdict = check_adapter(liar, metres(rate_hz=30.0), metres(rate_hz=20.0))
    assert "conformance.length" in verdict.codes()
    assert "wrong timeline" in verdict.because("conformance.length")[0].hint


def test_a_nondeterministic_adapter_is_caught():
    from gantry.resolve import Adapter

    jittery = Adapter(
        "jittery",
        "1.0",
        closes=("units.scale",),
        transform=lambda v, s, t: np.asarray(v) + np.random.default_rng().normal(0, 1, np.shape(v)),
        cost=lambda s, t: ("noise",),
    )
    assert (
        "conformance.not_deterministic"
        in check_adapter(jittery, metres(units="mm"), metres()).codes()
    )


# ==========================================================================
# an embodiment is now checked, not merely validated and dropped
# ==========================================================================


def test_a_policy_that_cannot_command_the_machine_is_refused(tmp_path):
    """Before this, a manifest could name an embodiment and be silently ignored."""
    from gantry_connector_csv import write_episodes
    from gantry_evaluator_waypoint import GreedyPolicy, WaypointWorld

    from gantry.manifest import Manifest
    from gantry.resolve import Registry
    from gantry.runner import run_manifest

    path = write_episodes(make_clean(n=3, seed=1).episodes, tmp_path / "s.csv")
    six_axis = EmbodimentDescriptor(
        name="six-axis",
        version="1.0",
        action=(
            ChannelSpec(
                "command",
                "vector",
                (6,),
                "float32",
                units="rad",
                semantics="joint_position",
                dim_labels=tuple(f"j{i}" for i in range(6)),
            ),
        ),
        control_hz=20.0,
    )
    registry = Registry()
    registry.discover()
    registry.register("embodiment", "six-axis", lambda **_: six_axis, replace=True)
    registry.register("policy", "greedy", lambda **_: GreedyPolicy(skill=0.9), replace=True)
    registry.register("evaluation", "waypoint", lambda **_: WaypointWorld(), replace=True)

    manifest = Manifest.from_dict(
        {
            "name": "mismatch",
            "cohorts": {"s": {"name": "csv", "config": {"path": str(path)}}},
            "embodiment": "six-axis",
            "policy": "greedy",
            "evaluation": "waypoint",
            "feedback": ["funnel"],
        }
    )
    outcome = run_manifest(manifest, registry)
    assert not outcome.ok
    assert any("cannot command this embodiment" in r for r in outcome.refusals)


def test_a_width_alone_never_establishes_that_something_is_a_pose():
    """A six-jointed arm is not a position plus an axis-angle rotation.

    Inferring the encoding from the width would make every six-wide channel
    look like a pose, and this retargeter would hand back three joint angles as
    a position in metres.
    """
    from gantry_retargeters_core import PoseToPosition

    joints = ChannelSpec("a", "vector", (6,), "float32", units="rad", semantics="joint_position")
    position = ChannelSpec("a", "vector", (3,), "float32", units="m", semantics="position")
    verdict = PoseToPosition().check(joints, position)
    assert "retarget.not_a_pose" in verdict.codes()
    assert "a width alone never establishes" in verdict.reasons[0].hint
