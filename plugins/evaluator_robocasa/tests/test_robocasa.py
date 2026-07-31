"""RoboCasa's shape, checked without RoboCasa or its several gigabytes of assets.

The fake copies the robosuite API RoboCasa inherits — ``reset`` returning an
observation dict, ``step`` returning four values, ``_check_success``,
``action_dim`` — and records which (layout, style) it was built with, since the
scene identity is the thing this plugin exists to keep hold of.
"""

from __future__ import annotations

import numpy as np
import pytest
from gantry_evaluator_robocasa import CAMERAS, RANDOM_SENTINELS, RobocasaEvaluator

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec

SIZE = 8


class FakeKitchen:
    def __init__(self, task, robot, layout, style, width=12, win_at=3):
        self.task = task
        self.robot = robot
        self.layout = layout
        self.style = style
        self.action_dim = width
        self.win_at = win_at
        self.step_count = 0
        self.resets = 0
        self.closed = False

    def _observation(self):
        observation = {name: np.zeros((SIZE, SIZE, 3), dtype="uint8") for name in CAMERAS}
        observation["robot0_eef_pos"] = np.zeros(3, dtype="float32")
        observation["robot0_base_pos"] = np.zeros(3, dtype="float32")
        observation["object"] = np.zeros(14, dtype="float32")  # not in `observes`
        return observation

    def reset(self):
        self.resets += 1
        self.step_count = 0
        return self._observation()

    def step(self, action):
        self.step_count += 1
        return self._observation(), 0.0, False, {}

    def _check_success(self):
        return self.step_count >= self.win_at

    def close(self):
        self.closed = True


class Chunker(Policy):
    def __init__(self, width=12):
        self.width = width
        self.spec = ChannelSpec("action", "vector", (width,), "float32", semantics="actuation")

    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=4, chunk=True, deterministic=True
        )

    def action_spec(self):
        return self.spec

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((4, self.width), dtype="float32")


def evaluator(width=12, win_at=3, task="PickPlaceCounterToCabinet", **kwargs):
    built: list[FakeKitchen] = []

    def factory(task, *, robot, layout, style, **_):
        kitchen = FakeKitchen(task, robot, layout, style, width=width, win_at=win_at)
        built.append(kitchen)
        return kitchen

    made = RobocasaEvaluator(task, factory=factory, camera_size=SIZE, horizon=20, **kwargs)
    made.built = built
    return made


def test_a_random_layout_is_refused_because_it_records_no_scene():
    """RoboCasa's -1 sentinel is convenient and produces a run nobody can
    reproduce: the record would name the task and not the kitchen. Worse, a held
    scene and a varied one become indistinguishable afterwards."""
    for sentinel in RANDOM_SENTINELS:
        with pytest.raises(ConfigError, match="records no scene"):
            evaluator(layouts=(sentinel,))
        with pytest.raises(ConfigError, match="records no scene"):
            evaluator(styles=(sentinel,))


def test_an_empty_layout_list_is_refused():
    with pytest.raises(ConfigError, match="at least one layout"):
        evaluator(layouts=())


def test_every_scene_names_its_layout_style_and_draw():
    """The whole point: a run whose provenance says 'robocasa, 50 trials' cannot
    be re-created, and one whose scenes say (layout 3, style 7, draw 2) can."""
    made = evaluator(layouts=(0, 3), styles=(7,))
    task = made.task_for(per_scene=2)
    assert len(task.scenes) == 4
    assert [scene.id for scene in task.scenes] == [
        "PickPlaceCounterToCabinet/l0s7#00",
        "PickPlaceCounterToCabinet/l0s7#01",
        "PickPlaceCounterToCabinet/l3s7#00",
        "PickPlaceCounterToCabinet/l3s7#01",
    ]
    assert task.scenes[2].metadata == {"layout": 3, "style": 7, "draw": 0}


def test_one_environment_is_built_per_kitchen_and_reused_across_draws():
    """Composing a kitchen from asset files costs seconds. Rebuilding per trial
    would cost more than the trials."""
    made = evaluator(layouts=(0, 1), styles=(2,))
    made.run(Chunker(), made.task_for(per_scene=3), Protocol())
    assert len(made.built) == 2
    assert {(k.layout, k.style) for k in made.built} == {(0, 2), (1, 2)}
    # Six trials over two kitchens: each reset three times.
    assert [k.resets for k in made.built] == [3, 3]


def test_the_held_and_varied_scene_conditions_are_both_expressible():
    """This is the experiment the plugin is for — the same comparison this
    project ran by hand as 'lift narrow' against 'lift wide', except the wide
    condition is a set of kitchens rather than a bigger rectangle on one table."""
    held = evaluator(layouts=(0,), styles=(0,)).task_for(per_scene=10)
    varied = evaluator(layouts=(0, 1, 2, 3, 4), styles=(0, 1)).task_for(per_scene=1)
    assert len(held.scenes) == len(varied.scenes) == 10
    assert len({(s.metadata["layout"], s.metadata["style"]) for s in held.scenes}) == 1
    assert len({(s.metadata["layout"], s.metadata["style"]) for s in varied.scenes}) == 10


def test_the_action_width_comes_from_the_environment():
    """A fixed Panda is seven numbers, a mobile one twelve, the humanoid
    different again. Assuming any of them turns a body swap into a run of
    truncated commands."""
    assert evaluator(width=12).action().shape == (12,)
    assert evaluator(width=7).action().shape == (7,)


def test_an_environment_that_reports_no_action_dimension_is_refused():
    class Mute(FakeKitchen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_dim = None
            self.action_spec = None

    made = RobocasaEvaluator(
        "PickPlaceCounterToCabinet",
        factory=lambda task, **kw: Mute(task, kw["robot"], kw["layout"], kw["style"]),
    )
    with pytest.raises(ConfigError, match="action dimension"):
        made.action()


def test_the_action_width_can_come_from_action_spec_instead():
    class SpecOnly(FakeKitchen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.action_dim = None
            self.action_spec = (np.zeros(7), np.ones(7))

    made = RobocasaEvaluator(
        "PickPlaceCounterToCabinet",
        factory=lambda task, **kw: SpecOnly(task, kw["robot"], kw["layout"], kw["style"]),
    )
    assert made.action().shape == (7,)


def test_success_comes_from_the_environments_own_predicate():
    made = evaluator(win_at=2)
    record = made.run(Chunker(), made.task_for(per_scene=3), Protocol())
    assert all(episode.labels.success for episode in record.episodes)
    assert record.episodes[0].array("action").shape[0] == 2


def test_only_the_cameras_and_proprioception_asked_for_are_recorded():
    made = evaluator(win_at=2)
    record = made.run(Chunker(), made.task_for(per_scene=1), Protocol())
    names = set(record.episodes[0].channel_names)
    assert "object" not in names
    assert set(CAMERAS) <= names


def test_the_camera_channels_are_declared_rather_than_inferred():
    """Inference gives shape and dtype; it cannot say 'this is rgb', and
    semantics are what the adapter plane matches on."""
    made = evaluator(win_at=2)
    record = made.run(Chunker(), made.task_for(per_scene=1), Protocol())
    camera = record.episodes[0].channel(CAMERAS[0])
    assert camera.semantics == "rgb"
    assert camera.kind == "image"
    assert camera.rate_hz == 20.0


def test_the_body_is_one_argument_and_reaches_the_environment():
    panda = evaluator(win_at=2)
    humanoid = panda.for_embodiment("GR1FixedLowerBody")
    assert humanoid.descriptor().provides["hosts_embodiment"] is True
    record = humanoid.run(Chunker(), humanoid.task_for(per_scene=1), Protocol())
    assert record.episodes[0].meta.embodiment == "GR1FixedLowerBody"
    # The factory is shared, so the kitchen it actually built is visible here —
    # the body has to reach the environment, not just the record.
    assert panda.built[0].robot == "GR1FixedLowerBody"


def test_a_robocasa_evaluator_conforms():
    made = evaluator(win_at=2)
    verdict = check_evaluator(made, Chunker(), made.task_for(per_scene=2))
    assert verdict.ok, verdict.explain()


def test_close_releases_every_kitchen():
    made = evaluator(layouts=(0, 1), styles=(0,), win_at=2)
    made.run(Chunker(), made.task_for(per_scene=1), Protocol())
    made.close()
    assert all(kitchen.closed for kitchen in made.built)


@pytest.mark.skip(reason="needs RoboCasa and its kitchen asset pack")
def test_against_the_real_simulator():  # pragma: no cover
    made = RobocasaEvaluator("PickPlaceCounterToCabinet", robot="PandaOmron")
    assert made.action().shape[0] >= 7


def test_task_ids_are_the_ones_robocasa_registers_not_the_ones_the_paper_uses():
    """The paper writes PnPCounterToCab; the registry says
    PickPlaceCounterToCabinet. The code wins, and an unresolvable id otherwise
    surfaces as a robosuite exception several layers down, after the scene has
    started building, with a message listing three hundred names."""
    from gantry_evaluator_robocasa import KITCHEN_CORE, TASKS

    assert "PickPlaceCounterToCabinet" in TASKS
    assert not any(t.startswith("PnP") for t in TASKS)
    assert set(KITCHEN_CORE) <= set(TASKS)

    with pytest.raises(ConfigError, match="does not register"):
        evaluator(task="PnPCounterToCab").task_for()


def test_a_near_miss_task_id_gets_a_suggestion():
    with pytest.raises(ConfigError, match="did you mean"):
        evaluator(task="PickPlaceCounterToFridge").task_for()
