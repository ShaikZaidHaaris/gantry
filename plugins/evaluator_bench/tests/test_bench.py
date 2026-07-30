"""A bench, checked without a bench.

The machine and the operator are both injected, so the whole hardware path runs
in a test with no robot and no person — which is the only way this plugin's logic
gets tested at all, since the alternative is a lab visit per assertion.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from gantry_evaluator_bench import (
    Attending,
    Bench,
    BenchEvaluator,
    Recording,
    scenes_from_file,
)

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol, Scene
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ConfigError
from gantry.resolve import requires_channels
from gantry.spine import ChannelSpec

ACTION = ChannelSpec("action", "vector", (7,), "float32", semantics="actuation")

SCENES = [
    Scene(id="near", instruction="pick up the cube", setup="cube at the near mark, gripper open"),
    Scene(id="far", instruction="pick up the cube", setup="cube at the far mark, gripper open"),
]


class FakeArm:
    def __init__(self):
        self.commands: list[np.ndarray] = []
        self.halts = 0

    def observe(self):
        return {
            "camera": np.zeros((4, 4, 3), dtype="uint8"),
            "joints": np.zeros(7, dtype="float32"),
        }

    def command(self, action):
        self.commands.append(np.asarray(action))

    def halt(self):
        self.halts += 1


class Scripted:
    """An operator whose answers are a list, so a run is deterministic."""

    def __init__(self, answers=(), interrupt_after=None):
        self.answers = list(answers)
        self.interrupt_after = interrupt_after
        self.prepared: list[str] = []
        self.steps = 0

    def prepare(self, scene):
        self.prepared.append(scene.setup or scene.id)

    def interrupt(self):
        self.steps += 1
        return self.interrupt_after is not None and self.steps >= self.interrupt_after

    def judge(self, scene, trial):
        return self.answers.pop(0) if self.answers else None


class Chunker(Policy):
    def descriptor(self):
        return policy_descriptor(
            name="chunker", version="0.1", horizon=4, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ACTION

    def observes(self):
        return requires_channels("chunker", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        return np.zeros((4, 7), dtype="float32")


def evaluator(operator=None, **kwargs):
    return BenchEvaluator(
        FakeArm(),
        action=ACTION,
        scenes=SCENES,
        operator=operator or Scripted(),
        rate_hz=10.0,
        embodiment="so100",
        observes=("camera", "joints"),
        horizon=8,
        **kwargs,
    )


def test_a_bench_run_will_not_start_without_its_scenes_written_down():
    """The refusal that keeps a hardware run from being 'we tried it a few times'."""
    with pytest.raises(ConfigError, match="scenes declared"):
        BenchEvaluator(FakeArm(), action=ACTION, scenes=())


def test_it_never_claims_to_be_seedable():
    """Every paired analysis here rests on same-seed-same-scene, and on a bench
    that is false. Claiming it would put a tight interval around a comparison
    nobody made."""
    provides = evaluator().descriptor().provides
    assert provides["seedable"] is False
    assert provides["closed_loop"] is True


def test_recording_reports_no_outcomes_at_all():
    """Not zero outcomes — no outcome capability. The labels do not exist yet,
    and anything needing them should be refused rather than read an empty column
    as a row of failures."""
    bench = evaluator(operator=Recording(prompt=lambda _: ""))
    assert bench.descriptor().provides["outcomes"] is False

    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert [e.labels.success for e in record.episodes] == [None, None]
    assert record.metrics == {}


def test_attending_reports_outcomes_and_records_abstentions_as_abstentions():
    operator = Scripted(answers=[True, None])
    bench = evaluator(operator=operator)
    assert bench.descriptor().provides["outcomes"] is True

    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert [e.labels.success for e in record.episodes] == [True, None]
    # One judged win out of one judged trial. The abstention is dropped, not
    # counted against the policy.
    assert record.metrics["success_rate"].value == 1.0
    assert record.metrics["success_rate"].n == 1


def test_the_operator_is_asked_to_set_up_every_scene_by_its_written_arrangement():
    operator = Scripted()
    bench = evaluator(operator=operator)
    bench.run(Chunker(), bench.task_for(), Protocol())
    assert operator.prepared == [
        "cube at the near mark, gripper open",
        "cube at the far mark, gripper open",
    ]


def test_an_interrupt_stops_the_attempt_and_halts_the_machine():
    arm = FakeArm()
    operator = Scripted(interrupt_after=3)
    bench = BenchEvaluator(arm, action=ACTION, scenes=SCENES[:1], operator=operator, horizon=20)
    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert len(arm.commands) == 3
    assert arm.halts >= 1
    assert record.episodes[0].labels.annotations["steps"] == 3


def test_the_arm_is_halted_before_the_operator_is_asked_anything():
    """Judging while the arm is still moving is how somebody gets hurt reaching
    for the cube to see where it ended up."""
    order: list[str] = []

    class Watching(FakeArm):
        def halt(self):
            order.append("halt")
            super().halt()

    class Asking(Scripted):
        def judge(self, scene, trial):
            order.append("judge")
            return True

    bench = BenchEvaluator(
        Watching(), action=ACTION, scenes=SCENES[:1], operator=Asking(), horizon=4
    )
    bench.run(Chunker(), bench.task_for(), Protocol())
    assert order.index("halt") < order.index("judge")


def test_the_recorded_episode_carries_the_arrangement_and_the_body():
    bench = evaluator(operator=Scripted(answers=[True, False]))
    record = bench.run(Chunker(), bench.task_for(), Protocol())
    episode = record.episodes[0]
    assert episode.meta.embodiment == "so100"
    assert episode.labels.annotations["instruction"] == "pick up the cube"
    assert episode.channel("joints").rate_hz == 10.0
    assert episode.array("action").shape == (8, 7)


def test_a_declared_channel_beats_inference():
    bench = evaluator(
        declares=[ChannelSpec("joints", "vector", (7,), "float32", semantics="joint_position")]
    )
    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert record.episodes[0].channel("joints").semantics == "joint_position"


def test_the_run_sheet_copies_the_rubric_rather_than_summarising_it():
    """A shortened rubric on paper is a different standard than the one the
    simulator checked, and the sim-to-real number that follows compares two
    different questions."""
    bench = evaluator()
    task = bench.task_for()
    rubric = [
        "the cube is off the table by at least 5 cm and still held at the end",
        "the gripper did not touch anything but the cube",
    ]
    path = bench.run_sheet(task, "sheet.txt", rubric=rubric)
    text = path.read_text()
    for line in rubric:
        assert line in text
    for scene in task.scenes:
        assert scene.id in text
        assert (scene.setup or "")[:30] in text
    assert "'?', not a failure" in text
    path.unlink()


def test_the_run_sheet_says_so_when_there_is_no_rubric():
    bench = evaluator()
    path = bench.run_sheet(bench.task_for(), "bare.txt")
    assert "(no rubric supplied)" in path.read_text()
    path.unlink()


def test_scene_lists_load_from_json_and_a_list_of_names_is_refused(tmp_path):
    good = tmp_path / "scenes.json"
    good.write_text(json.dumps([{"id": "a", "setup": "cube left", "instruction": "pick it up"}]))
    scenes = scenes_from_file(good)
    assert scenes[0].setup == "cube left"

    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ConfigError, match="record no arrangement"):
        scenes_from_file(bare)


def test_attending_shows_the_rubric_verbatim_at_the_moment_of_judging():
    shown: list[str] = []

    def prompt(text):
        shown.append(text)
        return "y" if "met the standard" in text else ""

    operator = Attending(prompt=prompt, rubric="lifted 5 cm and still held")
    bench = BenchEvaluator(
        FakeArm(), action=ACTION, scenes=SCENES[:1], operator=operator, horizon=4
    )
    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert record.episodes[0].labels.success is True
    assert any("lifted 5 cm and still held" in text for text in shown)


def test_attending_keeps_asking_until_the_answer_is_one_it_understands():
    answers = iter(["maybe", "dunno", "?"])

    def prompt(text):
        return next(answers) if "met the standard" in text else ""

    operator = Attending(prompt=prompt)
    bench = BenchEvaluator(
        FakeArm(), action=ACTION, scenes=SCENES[:1], operator=operator, horizon=4
    )
    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert record.episodes[0].labels.success is None


def test_skip_is_an_abstention_and_not_a_failure():
    def prompt(text):
        return "skip" if "met the standard" in text else ""

    bench = BenchEvaluator(
        FakeArm(),
        action=ACTION,
        scenes=SCENES[:1],
        operator=Attending(prompt=prompt),
        horizon=4,
    )
    record = bench.run(Chunker(), bench.task_for(), Protocol())
    assert record.episodes[0].labels.success is None


def test_a_bench_evaluator_conforms():
    bench = evaluator(operator=Scripted(answers=[True, False]))
    verdict = check_evaluator(bench, Chunker(), bench.task_for())
    assert verdict.ok, verdict.explain()


def test_the_world_halts_on_close():
    arm = FakeArm()
    world = Bench(arm, Scripted())
    world.close()
    assert arm.halts == 1
