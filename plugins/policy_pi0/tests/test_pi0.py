"""A pi0 server, checked without a pi0 server.

The fake client is the openpi wire and nothing more: a dict in, ``{"actions":
(H, D)}`` out. Everything worth testing here is on this side of that socket —
the prompt discipline, the bimanual widths, and what happens when the server
changes its mind.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_policy_pi0 import (
    ALOHA,
    DROID,
    LAYOUTS,
    Layout,
    Pi0Policy,
    bimanual,
    bimanual_labels,
    layout_for,
    prompts_of,
)

from gantry.conformance import check_policy
from gantry.contracts.policy import EpisodeContext, Observation
from gantry.errors import ComponentError, ConfigError
from gantry.spine import ChannelSpec, compatible


class FakeServer:
    """openpi's websocket client, as far as this plugin can tell."""

    def __init__(self, horizon=50, width=14, raises=None, actions=None):
        self.horizon = horizon
        self.width = width
        self.raises = raises
        self.actions = actions
        self.seen: list[dict] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, observation):
        self.seen.append(observation)
        if self.raises:
            raise self.raises
        if self.actions is not None:
            return {"actions": self.actions}
        return {"actions": np.zeros((self.horizon, self.width), dtype="float32")}


def observation(layout=ALOHA, *, state=None, drop=(), step=0):
    channels = {
        name: np.zeros((8, 8, 3), dtype="uint8") for name in layout.images if name not in drop
    }
    if layout.state_key not in drop:
        channels[layout.state_key] = (
            np.zeros(layout.state, dtype="float32") if state is None else np.asarray(state)
        )
    return Observation(step, channels)


def policy(server=None, **kwargs):
    # The fake answers at whatever width the layout declares, so a layout swap in
    # a test is a layout swap and not an accidental width mismatch.
    width = layout_for(kwargs.get("layout", "aloha")).action
    server = server or FakeServer(width=width)
    made = Pi0Policy(client=server, **kwargs)
    made.server = server
    return made


# -- the prompt, which is the expensive silent failure ----------------------


def test_an_episode_with_no_instruction_is_refused():
    """A language-conditioned model handed no prompt does not fail — it becomes
    unconditioned, scores badly, and is indistinguishable in every log from a
    checkpoint that did not train."""
    made = policy()
    with pytest.raises(ConfigError) as caught:
        made.reset(EpisodeContext("ep-0", instruction=None))
    assert "runs unconditioned" in str(caught.value)

    with pytest.raises(ConfigError):
        made.reset(EpisodeContext("ep-0", instruction="   "))


def test_the_prompt_is_sent_with_every_single_call():
    """The server is stateless per inference. Sending the prompt once would work
    for exactly one step and then silently stop conditioning."""
    made = policy()
    made.reset(EpisodeContext("ep-0", instruction="put the mug in the sink"))
    for step in range(3):
        made.act(observation(step=step))
    assert len(made.server.seen) == 3
    assert all(call["prompt"] == "put the mug in the sink" for call in made.server.seen)


def test_a_configured_fallback_is_used_and_recorded_as_a_fallback():
    """A single hard-wired prompt is fine for a single-task benchmark and quietly
    wrong for anything varying the task; the two look identical afterwards."""
    made = policy(instruction="pick up the cube")
    made.reset(EpisodeContext("ep-0", instruction=None))
    assert made.prompt == "pick up the cube"
    assert made.used_fallback() is True

    made.reset(EpisodeContext("ep-1", instruction="open the drawer"))
    assert made.prompt == "open the drawer"
    assert made.used_fallback() is False


def test_a_layout_that_takes_no_prompt_does_not_get_one():
    """A server that takes no prompt is a different kind of policy and should not
    be handed one silently."""
    silent = Layout(name="mute", images={"image": "image"}, state=7, action=7, prompt_key=None)
    made = policy(FakeServer(width=7), layout=silent)
    made.reset(EpisodeContext("ep-0", instruction=None))  # no refusal
    made.act(observation(silent))
    assert "prompt" not in made.server.seen[0]


def test_prompts_of_audits_a_finished_run():
    """One line for the failure this module worries about most: a run whose
    prompts are all one string was not language-conditioned."""

    class Episode:
        def __init__(self, text):
            self.labels = type("L", (), {"annotations": {"instruction": text}})()

    assert prompts_of([Episode("a"), Episode("a"), Episode("b")]) == {"a": 2, "b": 1}


# -- bimanual widths ---------------------------------------------------------


def test_the_arm_order_is_written_down_on_the_action_spec():
    """Fourteen numbers, left then right, and nothing about the array says which.
    Swap them and every action is valid and sent to the wrong arm."""
    spec = bimanual().action_spec()
    assert spec.shape == (14,)
    assert spec.dim_labels[0] == "left_waist"
    assert spec.dim_labels[7] == "right_waist"
    assert spec.metadata["arms"] == 2


def test_a_two_armed_layout_must_declare_its_labels():
    with pytest.raises(ConfigError, match="drives the wrong arm"):
        Layout(name="bad", images={"cam": "cam"}, state=14, action=14, arms=2)


def test_labels_that_do_not_match_the_width_are_refused():
    with pytest.raises(ConfigError, match="cannot be approximate"):
        Layout(name="bad", images={"cam": "cam"}, state=14, action=14, labels=("a", "b"), arms=2)


def test_a_single_arm_action_does_not_satisfy_a_bimanual_policy():
    """The resolver catches it by name, rather than by an arm moving somewhere
    unexpected."""
    two = bimanual().action_spec()
    one = ChannelSpec(
        "action",
        "vector",
        (14,),
        "float32",
        semantics="actuation",
        discriminators=("arms",),
        metadata={"arms": 1},
    )
    assert not compatible(one, two).ok
    assert "arms" in compatible(one, two).explain()


def test_a_state_of_the_wrong_width_says_what_it_probably_is():
    made = bimanual(client=FakeServer())
    made.reset(EpisodeContext("ep-0", instruction="do it"))
    with pytest.raises(ComponentError, match="one arm's worth"):
        made.act(observation(ALOHA, state=np.zeros(7, dtype="float32")))


# -- the chunk horizon belongs to the server --------------------------------


def test_the_horizon_is_read_from_the_server_and_pinned():
    made = policy(FakeServer(horizon=50))
    assert made.descriptor().metadata["chunk_source"] == "server"
    made.reset(EpisodeContext("ep-0", instruction="x"))
    chunk = made.act(observation())
    assert chunk.shape == (50, 14)
    assert made.descriptor().provides["chunk"] == 50


def test_a_horizon_that_changes_mid_run_is_a_refusal():
    """A server whose horizon changed is a server that was swapped, and the
    trials either side of it are not one measurement."""
    server = FakeServer(horizon=50)
    made = policy(server)
    made.reset(EpisodeContext("ep-0", instruction="x"))
    made.act(observation())
    server.horizon = 10
    with pytest.raises(ComponentError, match="not one measurement"):
        made.act(observation())


def test_a_single_action_is_accepted_as_a_chunk_of_one():
    made = policy(FakeServer(actions=np.zeros(14, dtype="float32")))
    made.reset(EpisodeContext("ep-0", instruction="x"))
    assert made.act(observation()).shape == (1, 14)


def test_a_chunk_of_the_wrong_width_is_refused():
    made = policy(FakeServer(actions=np.zeros((50, 7), dtype="float32")))
    made.reset(EpisodeContext("ep-0", instruction="x"))
    with pytest.raises(ComponentError, match=r"expects \(horizon, 14\)"):
        made.act(observation())


def test_a_reply_with_no_actions_key_is_refused():
    class Odd(FakeServer):
        def infer(self, observation):
            return {"something_else": []}

    made = policy(Odd())
    made.reset(EpisodeContext("ep-0", instruction="x"))
    with pytest.raises(ComponentError, match="no 'actions' key"):
        made.act(observation())


def test_a_server_that_dies_names_the_address_and_the_step():
    made = policy(FakeServer(raises=RuntimeError("connection reset")))
    made.reset(EpisodeContext("ep-0", instruction="x"))
    with pytest.raises(ComponentError, match="localhost:8000 failed on step 3"):
        made.act(observation(step=3))


# -- the observation payload -------------------------------------------------


def test_a_missing_camera_is_refused_rather_than_zero_filled():
    """Blank frames would produce a confident action chunk from a model that saw
    nothing, and nothing anywhere would say so."""
    made = policy()
    made.reset(EpisodeContext("ep-0", instruction="x"))
    with pytest.raises(ComponentError, match="saw nothing"):
        made.act(observation(drop=("cam_left_wrist",)))


def test_float_images_are_converted_rather_than_sent_near_black():
    """Common from a pipeline that normalised early. Handing [0,1] floats to a
    server expecting bytes gives a near-black frame rather than an error."""
    made = policy()
    made.reset(EpisodeContext("ep-0", instruction="x"))
    channels = {name: np.full((8, 8, 3), 0.5, dtype="float32") for name in ALOHA.images}
    channels["state"] = np.zeros(14, dtype="float32")
    made.act(Observation(0, channels))
    sent = made.server.seen[0]["images"]["cam_high"]
    assert sent.dtype == np.uint8
    assert int(sent.max()) == 127
    # channel-first for the aloha family, which its transform requires
    assert sent.shape == (3, 8, 8)


def test_uint8_images_pass_through_untouched():
    made = policy()
    made.reset(EpisodeContext("ep-0", instruction="x"))
    made.act(observation())
    assert made.server.seen[0]["images"]["cam_high"].dtype == np.uint8


def test_the_channels_are_named_by_the_layout_not_by_this_plugin():
    made = policy(layout="droid")
    made.reset(EpisodeContext("ep-0", instruction="x"))
    made.act(observation(DROID))
    keys = set(made.server.seen[0])
    assert "observation/exterior_image_1_left" in keys
    assert "observation/joint_position" in keys


# -- layouts are data --------------------------------------------------------


def test_a_rig_nobody_here_has_heard_of_works_from_a_dict():
    """The modularity claim, checked: adding a robot config is adding an entry in
    a manifest, not editing this plugin."""
    made = policy(
        layout={
            "name": "some_new_rig",
            "images": {"head": "head_cam", "hand": "hand_cam"},
            "state": 9,
            "action": 9,
            "rig": "whatever",
        }
    )
    assert made.layout.name == "some_new_rig"
    assert made.action_spec().shape == (9,)
    assert made.descriptor().metadata["rig"] == "whatever"


def test_the_built_in_layouts_are_a_lookup_not_a_gate():
    assert set(LAYOUTS) == {"aloha", "droid", "libero"}
    assert layout_for("aloha") is ALOHA
    with pytest.raises(ConfigError, match="rather than by adding a case here"):
        layout_for("nope")


def test_a_layout_with_no_cameras_is_refused():
    with pytest.raises(ConfigError, match="reading nothing"):
        Layout(name="blind", images={}, state=7, action=7)


def test_bimanual_labels_are_left_then_right_and_unique():
    labels = bimanual_labels()
    assert len(labels) == 14
    assert len(set(labels)) == 14
    assert labels[:7] == tuple(
        f"left_{joint}"
        for joint in (
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
        )
    )


# -- the rest ----------------------------------------------------------------


def test_it_does_not_claim_determinism_it_does_not_have():
    """Flow matching samples. Claiming otherwise would let a paired comparison
    attribute sampling noise to whatever else changed."""
    assert policy().descriptor().provides["deterministic"] is False
    assert policy(deterministic=True).descriptor().provides["deterministic"] is True


def test_the_variant_is_recorded_so_a_report_can_say_which_was_measured():
    assert policy().descriptor().metadata["variant"] == "pi05"
    assert policy(variant="pi0_fast").descriptor().metadata["variant"] == "pi0_fast"


def test_the_socket_is_not_opened_until_it_is_needed():
    """So this plugin installs, declares itself and plans a run on a laptop with
    no JAX and no weights present."""
    opened = []

    def connect(host, port):
        opened.append((host, port))
        return FakeServer()

    made = Pi0Policy(connect=connect, host="gpu-box", port=9000)
    made.descriptor()
    made.observes()
    made.action_spec()
    assert opened == []

    made.reset(EpisodeContext("ep-0", instruction="x"))
    assert opened == [("gpu-box", 9000)]


def test_the_server_is_reset_between_episodes():
    made = policy()
    made.reset(EpisodeContext("ep-0", instruction="x"))
    made.reset(EpisodeContext("ep-1", instruction="y"))
    assert made.server.resets == 2


def test_a_pi0_policy_conforms():
    made = policy()
    made.reset(EpisodeContext("ep-0", instruction="x"))
    verdict = check_policy(made, [observation(step=step) for step in range(3)])
    assert verdict.ok, verdict.explain()


@pytest.mark.skip(reason="needs a running openpi server with pi0.5 weights")
def test_against_a_real_server():  # pragma: no cover
    made = bimanual(host="localhost", port=8000)
    made.reset(EpisodeContext("ep-0", instruction="pick up the mug"))
    assert made.act(observation()).shape[1] == 14


def test_the_wire_key_and_the_channel_it_reads_are_separate():
    """Two namespaces, and conflating them was a real bug found only when a real
    server was on the other end: a dataset whose channel is `observation.state`
    served to a config whose wire key is `state` failed with "missing ['state']",
    which reads as a missing channel rather than an unexpressed mapping."""
    layout = Layout(
        name="ego",
        images={"ego_rgb": "cam_high"},
        state=14,
        action=14,
        state_key="state",
        state_from="observation.state",
        labels=bimanual_labels(),
        arms=2,
    )
    made = policy(FakeServer(width=14), layout=layout)
    assert layout.reads == "observation.state"
    assert [c.name for c in made.observes().channels][-1] == "observation.state"

    made.reset(EpisodeContext("ep-0", instruction="x"))
    made.act(
        Observation(
            0,
            {
                "ego_rgb": np.zeros((8, 8, 3), dtype="uint8"),
                "observation.state": np.zeros(14, dtype="float32"),
            },
        )
    )
    sent = made.server.seen[0]
    assert "state" in sent and "observation.state" not in sent
    assert "cam_high" in sent["images"]


def test_state_from_defaults_to_the_wire_key():
    assert ALOHA.reads == ALOHA.state_key == "state"


def test_cameras_nest_or_stay_flat_as_the_config_requires():
    """Both shapes are real and neither is guessable from the config name. Send
    the wrong one and the server raises KeyError deep inside its own transform
    stack, a long way from anything that names the cause."""
    nested = policy()  # aloha: images_key="images"
    nested.reset(EpisodeContext("ep-0", instruction="x"))
    nested.act(observation())
    sent = nested.server.seen[0]
    assert "images" in sent and "cam_high" in sent["images"]
    assert "cam_high" not in sent

    flat = policy(FakeServer(width=8), layout="droid")
    flat.reset(EpisodeContext("ep-0", instruction="x"))
    flat.act(observation(DROID))
    sent = flat.server.seen[0]
    assert "observation/exterior_image_1_left" in sent
    assert "images" not in sent


def test_channel_order_follows_the_config_and_not_a_guess():
    """The aloha transform rearranges "c h w -> h w c", so it wants channel
    first and mangles anything else — a (224,224,3) frame arrives as a
    3-pixel-tall image of 224 channels and dies inside PIL, several layers below
    anything that names the cause."""
    aloha = policy()
    aloha.reset(EpisodeContext("ep-0", instruction="x"))
    aloha.act(observation())
    assert aloha.server.seen[0]["images"]["cam_high"].shape == (3, 8, 8)

    droid = policy(FakeServer(width=8), layout="droid")
    droid.reset(EpisodeContext("ep-0", instruction="x"))
    droid.act(observation(DROID))
    assert droid.server.seen[0]["observation/exterior_image_1_left"].shape == (8, 8, 3)
