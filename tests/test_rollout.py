"""The shared closed-loop trial, checked on a world made of dictionaries.

Nothing here imports a simulator, which is the point: every suite adapter in
every plugin inherits this loop, so a bug caught here is caught for all of them
and a bug missed here is missed by all of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from gantry.conformance import check_evaluator
from gantry.contracts.evaluator import Protocol, Scene, TaskSpec, evaluator_descriptor
from gantry.contracts.policy import Policy, policy_descriptor
from gantry.errors import ComponentError, ConfigError
from gantry.resolve import requires_channels
from gantry.rollout import ClosedLoop, Step, Trial, imported, infer_spec
from gantry.spine import ChannelSpec

ACTION = ChannelSpec("action", "vector", (2,), "float32", semantics="actuation")


class Counter:
    """A world whose observation says which step it is on.

    Chosen so that a one-step offset between observations and actions is visible
    in the recorded arrays rather than requiring a simulator to notice.
    """

    def __init__(self, *, win_at: int | None = None, milestone_at: int | None = None):
        self.win_at = win_at
        self.milestone_at = milestone_at
        self.step = 0
        self.seen: list[np.ndarray] = []
        self.closed = False

    def _observation(self) -> dict[str, np.ndarray]:
        return {"clock": np.asarray([float(self.step)], dtype="float32")}

    def begin(self, scene):
        self.step = 0
        return self._observation()

    def advance(self, action):
        self.seen.append(np.asarray(action))
        self.step += 1
        won = self.win_at is not None and self.step >= self.win_at
        return Step(
            observation=self._observation(),
            reward=1.0 if won else 0.0,
            done=won,
            success=True if won else None,
            reached=(
                ("touched",)
                if self.milestone_at is not None and self.step >= self.milestone_at
                else ()
            ),
        )

    def close(self):
        self.closed = True


class Suite(ClosedLoop):
    observes = ("clock",)
    rate_hz = 10.0
    embodiment = "twojoint"

    def __init__(self, world=None, **kwargs):
        self.world = world or Counter(**kwargs)

    def descriptor(self):
        return evaluator_descriptor(
            name="counter",
            version="0.1",
            stage_events=True,
            outcomes=True,
            seedable=True,
            closed_loop=True,
        )

    def action(self):
        return ACTION

    def task_for(self, name: str = "", scenes: int = 2, horizon: int = 6) -> TaskSpec:
        return TaskSpec(
            name=name or "counter",
            horizon=horizon,
            stages=("touched",),
            scenes=tuple(
                Scene(id=f"scene-{index}", instruction="count") for index in range(scenes)
            ),
        )

    def world_for(self, scene):
        return self.world

    def close(self):
        self.world.close()


class Steady(Policy):
    """Emits a chunk whose rows are distinguishable, so ordering is checkable."""

    def __init__(self, chunk: int = 3, raises: bool = False, shape=(3, 2)):
        self._chunk = chunk
        self._raises = raises
        self._shape = shape
        self.calls = 0

    def descriptor(self):
        return policy_descriptor(
            name="steady", version="0.1", horizon=self._chunk, chunk=True, deterministic=True
        )

    def action_spec(self):
        return ACTION

    def observes(self):
        return requires_channels("steady", "policy")

    def reset(self, context):
        self.context = context

    def act(self, observation):
        self.calls += 1
        if self._raises:
            raise RuntimeError("no")
        base = float(observation.channels["clock"][0])
        return np.asarray(
            [[base + offset, -1.0] for offset in range(self._shape[0])], dtype="float32"
        ).reshape(self._shape)


def test_the_observation_is_recorded_before_the_action_taken_from_it():
    """The invariant the module exists for.

    Offset these by one and every exported episode teaches a policy to predict
    the action it has already taken — a corruption with no error message.
    """
    suite = Suite()
    policy = Steady(chunk=1, shape=(1, 2))
    record = suite.run(policy, suite.task_for(scenes=1, horizon=4), Protocol())
    episode = record.episodes[0]

    clock = episode.array("clock")
    action = episode.array("action")
    assert [float(value[0]) for value in clock] == [0.0, 1.0, 2.0, 3.0]
    # The policy encodes the clock it saw into its action, so equality here is
    # the pairing being right.
    assert [float(row[0]) for row in action] == [0.0, 1.0, 2.0, 3.0]


def test_execute_plays_only_a_prefix_of_the_chunk():
    """Chunk size moved a measured rate by fourteen points in this project's own
    benchmark, so the prefix arithmetic is worth a test of its own."""
    suite = Suite()
    policy = Steady(chunk=3)
    record = suite.run(policy, suite.task_for(scenes=1, horizon=6), Protocol(execute=2))
    assert record.episodes[0].array("action").shape == (6, 2)
    # Six steps at two executed per query is three queries, not two and not six.
    assert policy.calls == 3


def test_a_policy_that_raises_loses_one_trial_and_not_the_run():
    suite = Suite()
    record = suite.run(Steady(raises=True), suite.task_for(scenes=3), Protocol())
    assert len(record.episodes) == 3
    for episode in record.episodes:
        assert episode.labels.success is None
        assert "RuntimeError" in episode.labels.annotations["error"]
    # No successes established, so no rate — rather than a rate of zero, which
    # would report a harness bug as a policy result.
    assert record.metrics == {}
    assert record.provenance.notes


def test_a_world_that_raises_halts_the_run():
    """The opposite call from a failing policy, and deliberately so: a broken
    simulator makes every later trial meaningless, so continuing would fill a
    record with plausible zeros."""

    class Broken(Counter):
        def advance(self, action):
            raise ComponentError("the physics went away")

    suite = Suite(world=Broken())
    with pytest.raises(ComponentError):
        suite.run(Steady(), suite.task_for(scenes=3), Protocol())


def test_a_wrong_chunk_shape_is_refused_by_name():
    """Not a failed trial — a refusal. A policy emitting the wrong width is
    misconfigured for this world, and every trial after it would be too."""

    class WrongWidth(Steady):
        def act(self, observation):
            return np.zeros((3, 5), dtype="float32")

    suite = Suite()
    with pytest.raises(ComponentError, match="chunk of shape"):
        suite.run(WrongWidth(), suite.task_for(scenes=1), Protocol())

    class NotAChunk(Steady):
        def act(self, observation):
            return np.zeros(2, dtype="float32")  # one action, unbatched

    with pytest.raises(ComponentError, match="chunk of shape"):
        Suite().run(NotAChunk(), suite.task_for(scenes=1), Protocol())


def test_running_out_of_horizon_is_not_a_failure():
    """A trial that simply ran long has established nothing. Recording it as a
    loss is the difference between "did not succeed within 6 steps" and "failed",
    and only one of those is what the record would then claim."""
    suite = Suite(win_at=99)
    record = suite.run(Steady(), suite.task_for(scenes=2, horizon=4), Protocol())
    for episode in record.episodes:
        assert episode.labels.success is None
        assert episode.labels.annotations["truncated"] is True
    assert record.metrics == {}


def test_success_ends_the_trial_and_is_counted():
    suite = Suite(win_at=2)
    record = suite.run(Steady(), suite.task_for(scenes=4, horizon=10), Protocol())
    assert all(episode.labels.success for episode in record.episodes)
    assert record.metrics["success_rate"].value == 1.0
    assert record.metrics["success_rate"].n == 4
    assert record.episodes[0].array("action").shape[0] == 2


def test_a_milestone_is_recorded_once_at_the_step_it_was_reached():
    suite = Suite(milestone_at=3)
    record = suite.run(Steady(), suite.task_for(scenes=1, horizon=6), Protocol())
    events = record.episodes[0].labels.stage_events
    assert [(event.name, event.step) for event in events] == [("touched", 3)]


def test_epochs_give_one_episode_per_attempt_with_distinct_ids():
    suite = Suite(win_at=2)
    record = suite.run(Steady(), suite.task_for(scenes=2, horizon=6), Protocol(epochs=3))
    assert len(record.episodes) == 6
    assert len({episode.meta.id for episode in record.episodes}) == 6


def test_max_trials_truncates_the_plan_not_the_scene_list():
    suite = Suite(win_at=2)
    record = suite.run(Steady(), suite.task_for(scenes=5), Protocol(max_trials=2))
    assert len(record.episodes) == 2


def test_the_schema_carries_the_suite_declaration_over_inference():
    """Inference yields shape and dtype and cannot yield semantics. A suite that
    knows a channel is a joint position has to be able to say so, because
    semantics are what the adapter plane matches on."""

    class Declaring(Suite):
        def declares(self):
            return {
                "clock": ChannelSpec("clock", "vector", (1,), "float32", semantics="joint_position")
            }

    suite = Declaring(win_at=2)
    record = suite.run(Steady(), suite.task_for(scenes=1), Protocol())
    clock = record.episodes[0].channel("clock")
    assert clock.semantics == "joint_position"

    plain = Suite(win_at=2)
    record = plain.run(Steady(), plain.task_for(scenes=1), Protocol())
    assert record.episodes[0].channel("clock").semantics is None
    assert record.episodes[0].channel("clock").rate_hz == 10.0


def test_provenance_names_both_components_and_the_protocol():
    suite = Suite(win_at=2)
    record = suite.run(Steady(), suite.task_for(scenes=1), Protocol(execute=2))
    assert len(record.provenance.components) == 2
    assert record.provenance.protocol["execute"] == 2
    assert record.provenance.protocol["horizon"] == 6


def test_a_closed_loop_evaluator_conforms():
    # milestone_at as well as win_at: the descriptor says stage_events=True, and
    # conformance checks declarations against the record rather than taking them
    # on trust. A suite that claims a funnel and emits none is caught here.
    suite = Suite(win_at=3, milestone_at=2)
    verdict = check_evaluator(suite, Steady(), suite.task_for(scenes=2))
    assert verdict.ok, verdict.explain()


def test_observes_narrows_what_is_recorded():
    class Noisy(Counter):
        def _observation(self):
            return {
                "clock": np.asarray([float(self.step)], dtype="float32"),
                "junk": np.zeros(64, dtype="float32"),
            }

    suite = Suite(world=Noisy(win_at=2))
    record = suite.run(Steady(), suite.task_for(scenes=1), Protocol())
    assert "junk" not in record.episodes[0].channel_names

    class Everything(Suite):
        observes = None

    suite = Everything(world=Noisy(win_at=2))
    record = suite.run(Steady(), suite.task_for(scenes=1), Protocol())
    assert "junk" in record.episodes[0].channel_names


def test_non_array_observations_are_skipped_rather_than_crashing():
    """Suites put language strings and nested dicts in observations. Those are
    not channels, and the alternative to skipping them is a numpy object array
    that fails much later, somewhere unrelated."""

    class Chatty(Counter):
        def _observation(self):
            return {
                "clock": np.asarray([float(self.step)], dtype="float32"),
                "language": "pick up the cube",
                "nested": {"a": 1},
            }

    class Everything(Suite):
        observes = None

    suite = Everything(world=Chatty(win_at=2))
    record = suite.run(Steady(), suite.task_for(scenes=1), Protocol())
    assert record.episodes[0].channel_names == ("clock", "action", "reward")


# -- the small helpers -------------------------------------------------------


def test_imported_refuses_clearly_rather_than_raising_from_inside_a_package():
    with pytest.raises(ConfigError, match="importable"):
        imported("numpy.zeros")
    with pytest.raises(ConfigError, match="cannot import"):
        imported("not_a_real_module_anywhere:thing")
    with pytest.raises(ConfigError, match="has no"):
        imported("numpy:definitely_not_here")
    assert imported("numpy:zeros") is np.zeros


@pytest.mark.parametrize(
    "sample,kind",
    [
        (np.zeros((16, 16, 3), dtype="uint8"), "image"),
        (np.zeros((16, 16, 1), dtype="uint8"), "image"),
        (np.zeros(7, dtype="float32"), "vector"),
        (np.asarray(1.0, dtype="float32"), "scalar"),
        # Rank three but not an image shape — a stack, not a picture.
        (np.zeros((4, 4, 7), dtype="float32"), "vector"),
    ],
)
def test_infer_spec_reads_rank_and_says_nothing_about_meaning(sample, kind):
    spec = infer_spec("thing", sample)
    assert spec.kind == kind
    assert spec.shape == tuple(sample.shape)
    assert spec.semantics is None


def test_a_trial_reports_milestones_in_the_order_reached():
    trial = Trial()
    trial.steps = 2
    trial.note(("a", "b"))
    trial.steps = 5
    trial.note(("b", "c"))
    assert trial.reached == ("a", "b", "c")
    assert [event.step for event in trial.events] == [2, 2, 5]
