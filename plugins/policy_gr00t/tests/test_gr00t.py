"""The policy: wiring discovered, checked, and driven against a fake server.

The layout used here is the one shipped with the RoboMimic-to-LeRobot lift
dataset in this project — a seven-wide action of x/y/z/roll/pitch/yaw/gripper
over an eight-wide state — so the shapes are the real ones even though the model
behind the socket is four lines of arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gantry_policy_gr00t import (
    Client,
    Endpoint,
    Gr00tPolicy,
    Layout,
    Wants,
    check,
    observation_specs,
)
from server import FakeServer, modality_payload

from gantry.conformance import check_policy
from gantry.contracts.policy import EpisodeContext, Observation
from gantry.errors import ComponentError, ConfigError
from gantry.spine import IncompatibleError

HORIZON = 16

MODALITY_JSON = {
    "state": {
        "x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2}, "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4}, "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6}, "gripper": {"start": 6, "end": 8},
    },
    "action": {
        "x": {"start": 0, "end": 1}, "y": {"start": 1, "end": 2}, "z": {"start": 2, "end": 3},
        "roll": {"start": 3, "end": 4}, "pitch": {"start": 4, "end": 5},
        "yaw": {"start": 5, "end": 6}, "gripper": {"start": 6, "end": 7},
    },
    "video": {
        "image": {"original_key": "observation.images.image"},
        "wrist_image": {"original_key": "observation.images.wrist_image"},
    },
    "annotation": {"human.action.task_description": {"original_key": "task_index"}},
}

FIELDS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def modality(state_deltas=(0,), video_deltas=(0,)) -> dict:
    return {
        "video": modality_payload(list(video_deltas), ["image", "wrist_image"]),
        "state": modality_payload(list(state_deltas), FIELDS),
        "action": modality_payload(list(range(HORIZON)), FIELDS),
        "language": modality_payload([0], ["annotation.human.action.task_description"]),
    }


def responder(observation, options):
    """Returns the state's x broadcast across the chunk, so a test can trace it."""
    x = float(np.asarray(observation["state"]["x"]).reshape(-1)[-1])
    return {key: np.full((1, HORIZON, 1), x, dtype="float32") for key in FIELDS}


@pytest.fixture
def layout_path(tmp_path) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "modality.json").write_text(json.dumps(MODALITY_JSON))
    return tmp_path


@pytest.fixture
def server():
    fake = FakeServer(modality(), responder, identity={"model_id": "gr00t-n1d7"})
    yield fake
    fake.stop()


def build(server, layout_path, **kwargs) -> Gr00tPolicy:
    return Gr00tPolicy(
        layout_path, Endpoint(port=server.port, timeout_ms=3000), **kwargs
    )


@pytest.fixture
def policy(server, layout_path):
    p = build(server, layout_path)
    yield p
    p.close()


def observation(step: int = 0, value: float = 1.0) -> Observation:
    return Observation(
        step,
        {
            "observation.state": np.full(8, value, dtype="float32"),
            "observation.images.image": np.zeros((8, 8, 3), dtype=np.uint8),
            "observation.images.wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    )


# -- the contract ----------------------------------------------------------


def test_conforms(policy):
    verdict = check_policy(policy, [observation(0), observation(1, 0.5)], strict=True)
    assert verdict.ok, verdict.explain()


def test_the_chunk_it_declares_is_the_horizon_the_server_asked_for(policy):
    assert policy.descriptor().provides["chunk"] == HORIZON
    assert policy.act(observation()).shape == (HORIZON, 7)


def test_the_action_channel_names_every_dimension(policy):
    spec = policy.action_spec()
    assert spec.shape == (7,)
    assert spec.dim_labels == tuple(FIELDS)
    assert spec.dtype == "float32"


def test_it_asks_for_the_channels_the_dataset_actually_stores(policy):
    """The model says 'image'; the dataset says 'observation.images.image'.

    modality.json is where those two names are written down together, so the
    usual case needs no mapping from anybody.
    """
    names = [spec.name for spec in policy.observes().channels]
    assert names == [
        "observation.state",
        "observation.images.image",
        "observation.images.wrist_image",
    ]


def test_the_state_channel_is_labelled_by_the_layout(policy):
    state = policy.observes().channels[0]
    assert state.shape == (8,)
    # gripper spans two elements, so it is numbered rather than guessed at.
    assert state.dim_labels == ("x", "y", "z", "roll", "pitch", "yaw", "gripper.0", "gripper.1")


def test_the_server_identity_is_read_once_and_carried(policy):
    assert policy.descriptor().metadata["identity"]["model_id"] == "gr00t-n1d7"


# -- what actually goes over the wire --------------------------------------


def test_the_state_is_split_into_the_fields_the_model_reads(policy, server):
    policy.reset(EpisodeContext("scene-0", instruction="lift the cube"))
    policy.act(observation(0, value=0.25))
    sent = server.seen[-1]
    assert set(sent["state"]) == set(FIELDS)
    # (batch, time, width); gripper is the two-wide one.
    assert sent["state"]["x"].shape == (1, 1, 1)
    assert sent["state"]["gripper"].shape == (1, 1, 2)
    assert sent["state"]["x"].dtype == np.float32


def test_the_video_goes_as_uint8_with_batch_and_time_axes(policy, server):
    policy.act(observation())
    sent = server.seen[-1]
    assert sent["video"]["image"].shape == (1, 1, 8, 8, 3)
    assert sent["video"]["image"].dtype == np.uint8


def test_the_instruction_comes_from_the_episode(policy, server):
    policy.reset(EpisodeContext("scene-0", instruction="lift the cube"))
    policy.act(observation())
    language = server.seen[-1]["language"]
    assert language["annotation.human.action.task_description"] == [["lift the cube"]]


def test_a_configured_instruction_wins_over_the_episode(server, layout_path):
    policy = build(server, layout_path, instruction="always this")
    policy.reset(EpisodeContext("scene-0", instruction="something else"))
    policy.act(observation())
    assert server.seen[-1]["language"]["annotation.human.action.task_description"] == [
        ["always this"]
    ]
    policy.close()


def test_reset_reaches_the_server_with_the_seed(policy, server):
    policy.reset(EpisodeContext("scene-3", seed=11))
    assert server.resets[-1] == {"episode_id": "scene-3", "seed": 11}


def test_the_chunk_is_reassembled_in_layout_order(policy):
    """Every field returns the same number here, so order is checked by shape
    and by the join round-tripping through the layout."""
    chunk = policy.act(observation(value=0.75))
    assert chunk.shape == (HORIZON, 7)
    assert np.allclose(chunk, 0.75)
    assert chunk.dtype == np.float32


# -- history ---------------------------------------------------------------


def test_a_model_that_reads_the_past_gets_the_past(layout_path):
    fake = FakeServer(modality(state_deltas=(-2, 0), video_deltas=(-2, 0)), responder)
    policy = build(fake, layout_path)
    try:
        policy.reset(EpisodeContext("scene-0"))
        for step, value in enumerate([0.1, 0.2, 0.3]):
            policy.act(observation(step, value))
        sent = fake.seen[-1]["state"]["x"]
        assert sent.shape == (1, 2, 1)
        assert np.allclose(sent[0, 0], 0.1), "two steps back"
        assert np.allclose(sent[0, 1], 0.3), "now"
    finally:
        policy.close()
        fake.stop()


def test_before_there_is_a_past_the_earliest_frame_is_repeated(layout_path):
    """The padding GR00T's own loader applies at an episode boundary, said out loud."""
    fake = FakeServer(modality(state_deltas=(-2, 0), video_deltas=(-2, 0)), responder)
    policy = build(fake, layout_path)
    try:
        assert policy.descriptor().metadata["pads_history"] is True
        policy.reset(EpisodeContext("scene-0"))
        policy.act(observation(0, 0.4))
        sent = fake.seen[-1]["state"]["x"]
        assert np.allclose(sent[0, 0], 0.4) and np.allclose(sent[0, 1], 0.4)
    finally:
        policy.close()
        fake.stop()


def test_history_does_not_leak_across_episodes(layout_path):
    fake = FakeServer(modality(state_deltas=(-2, 0), video_deltas=(-2, 0)), responder)
    policy = build(fake, layout_path)
    try:
        policy.reset(EpisodeContext("scene-0"))
        policy.act(observation(0, 0.1))
        policy.act(observation(1, 0.2))
        policy.reset(EpisodeContext("scene-1"))
        policy.act(observation(0, 0.9))
        assert np.allclose(fake.seen[-1]["state"]["x"], 0.9)
    finally:
        policy.close()
        fake.stop()


def test_a_model_that_reads_only_now_keeps_no_history(policy):
    assert policy.descriptor().metadata["pads_history"] is False


# -- refusing a wiring that would produce numbers anyway --------------------


def test_a_checkpoint_that_wants_a_field_this_dataset_lacks_is_refused(layout_path):
    fake = FakeServer(
        {
            "video": modality_payload([0], ["ego_view"]),
            "state": modality_payload([0], ["left_arm", "right_arm"]),
            "action": modality_payload([0], ["left_arm"]),
            "language": modality_payload([0], ["annotation.human.task"]),
        },
        responder,
    )
    try:
        with pytest.raises(IncompatibleError) as raised:
            build(fake, layout_path)
        codes = raised.value.verdict.codes()
        assert codes.count("gr00t.missing_key") == 3, "state, action and video all disagree"
        assert "left_arm" in str(raised.value)
    finally:
        fake.stop()


def test_a_wrong_width_coming_back_is_refused_rather_than_reshaped(layout_path):
    def too_wide(observation, options):
        return {key: np.zeros((1, HORIZON, 3), dtype="float32") for key in FIELDS}

    fake = FakeServer(modality(), too_wide)
    policy = build(fake, layout_path)
    try:
        with pytest.raises(ComponentError, match="3 wide and the layout says 1"):
            policy.act(observation())
    finally:
        policy.close()
        fake.stop()


def test_a_missing_action_key_says_what_came_back(layout_path):
    def partial(observation, options):
        return {"x": np.zeros((1, HORIZON, 1), dtype="float32")}

    fake = FakeServer(modality(), partial)
    policy = build(fake, layout_path)
    try:
        with pytest.raises(ComponentError, match="needs 'y'"):
            policy.act(observation())
    finally:
        policy.close()
        fake.stop()


def test_a_float_image_is_refused_because_the_scale_is_nobody_s_guess(policy):
    floats = Observation(
        0,
        {
            "observation.state": np.zeros(8, dtype="float32"),
            "observation.images.image": np.zeros((8, 8, 3), dtype="float32"),
            "observation.images.wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        },
    )
    with pytest.raises(ComponentError, match="uint8"):
        policy.act(floats)


def test_a_server_that_is_not_there_fails_before_the_run(layout_path):
    with pytest.raises(ConfigError, match="is the GR00T server running"):
        Gr00tPolicy(layout_path, Endpoint(port=1, timeout_ms=200))


def test_a_client_may_be_supplied_directly(server, layout_path):
    """So a caller can share one socket, or hand in something else entirely."""
    client = Client(Endpoint(port=server.port, timeout_ms=3000))
    policy = Gr00tPolicy(layout_path, client=client)
    assert policy.act(observation()).shape == (HORIZON, 7)
    policy.close()


# -- the layout on its own -------------------------------------------------


def test_a_layout_splits_and_rejoins_a_vector(layout_path):
    layout = Layout.from_json(layout_path)
    vector = np.arange(7, dtype="float32")
    parts = layout.split("action", vector)
    assert [key for key in parts] == FIELDS
    assert np.array_equal(layout.join("action", parts), vector)


def test_a_layout_with_a_hole_in_it_is_refused(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "modality.json").write_text(
        json.dumps({"action": {"a": {"start": 0, "end": 2}, "b": {"start": 3, "end": 4}}})
    )
    with pytest.raises(ConfigError, match="leaves a gap"):
        Layout.from_json(tmp_path)


def test_overlapping_fields_are_refused(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "modality.json").write_text(
        json.dumps({"action": {"a": {"start": 0, "end": 3}, "b": {"start": 2, "end": 4}}})
    )
    with pytest.raises(ConfigError, match="overlaps"):
        Layout.from_json(tmp_path)


def test_a_missing_modality_json_says_what_it_is_for(tmp_path):
    with pytest.raises(ConfigError, match="no modality.json"):
        Layout.from_json(tmp_path)


def test_fields_are_ordered_by_position_not_by_file_order(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "modality.json").write_text(
        json.dumps({"action": {"late": {"start": 2, "end": 4}, "early": {"start": 0, "end": 2}}})
    )
    layout = Layout.from_json(tmp_path)
    assert [f.key for f in layout.action] == ["early", "late"]


def test_what_a_dataset_must_provide_can_be_asked_before_anything_runs(layout_path):
    layout = Layout.from_json(layout_path)
    wants = Wants.from_server(
        {"video": {"modality_keys": ["image"], "delta_indices": [0]},
         "state": {"modality_keys": FIELDS, "delta_indices": [0]}}
    )
    specs = observation_specs(layout, wants)
    assert [spec.name for spec in specs] == ["observation.state", "observation.images.image"]
    assert check(layout, wants).ok


# -- against the dataset actually on this machine --------------------------

REAL = Path("/tmp/gantry-real/lift_lerobot/ph")
real_only = pytest.mark.skipif(not REAL.exists(), reason="no real LeRobot dataset present")


@real_only
def test_it_reads_the_real_datasets_modality_json():
    layout = Layout.from_json(REAL)
    assert layout.width("action") == 7
    assert layout.width("state") == 8
    assert layout.video["image"] == "observation.images.image"
    assert layout.labels("action")[:3] == ("x", "y", "z")


# -- staying on its own plane ----------------------------------------------


def test_a_layout_can_be_built_from_any_connectors_schema(layout_path):
    """The preferred path: read the schema, not the dataset's sidecar.

    A policy that parses modality.json knows a dataset format, which is a leak
    across planes. Dimension labels are the framework's own vocabulary for the
    same fact, and any connector that describes its channels provides them.
    """
    from gantry.spine import ChannelSpec

    schema = (
        ChannelSpec("observation.state", "vector", (8,), "float32",
                    dim_labels=("x", "y", "z", "roll", "pitch", "yaw", "gripper.0", "gripper.1")),
        ChannelSpec("action", "vector", (7,), "float32", dim_labels=tuple(FIELDS)),
    )
    from_schema = Layout.from_schema(schema, video=["image", "wrist_image"])
    from_file = Layout.from_json(layout_path)

    assert from_schema.labels("action") == from_file.labels("action")
    assert from_schema.labels("state") == from_file.labels("state")
    assert from_schema.width("state") == 8
    assert [f.key for f in from_schema.state] == [f.key for f in from_file.state]


def test_an_unlabelled_channel_says_why_it_cannot_be_used():
    from gantry.spine import ChannelSpec

    with pytest.raises(ConfigError, match="no dimension labels"):
        Layout.from_schema((ChannelSpec("action", "vector", (7,), "float32"),))


def test_a_policy_can_be_built_from_a_schema_derived_layout(server, layout_path):
    """Without the sidecar, the camera wiring is the caller's to declare.

    from_schema deliberately does not invent it: which column holds the model's
    'image' is a fact about the dataset's layout, and a policy that guessed at
    it would be back to knowing a format.
    """
    layout = Layout.from_schema(
        (
            __import__("gantry.spine", fromlist=["x"]).ChannelSpec(
                "observation.state", "vector", (8,), "float32",
                dim_labels=("x", "y", "z", "roll", "pitch", "yaw", "gripper.0", "gripper.1"),
            ),
            __import__("gantry.spine", fromlist=["x"]).ChannelSpec(
                "action", "vector", (7,), "float32", dim_labels=tuple(FIELDS),
            ),
        ),
        video=["image", "wrist_image"],
    )
    policy = Gr00tPolicy(
        layout,
        Endpoint(port=server.port, timeout_ms=3000),
        video_channels={
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
    )
    assert policy.act(observation()).shape == (HORIZON, 7)
    policy.close()
