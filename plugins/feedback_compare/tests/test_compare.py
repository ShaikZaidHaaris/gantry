"""Ranking policies, and refusing to rank anything else."""

from __future__ import annotations

import pytest
from gantry_feedback_compare import Floor, PolicyComparison, paired_counts, scene_of

from gantry.conformance import check_feedback
from gantry.contracts.feedback import Cohort
from gantry.errors import ConfigError
from gantry.spine import ComponentRef, EpisodeLabels, IncompatibleError, Provenance
from gantry.spine.episode import episode_from_labels

RIG = ComponentRef("evaluation", "robosuite", "1.0")


def arm(name, outcomes, *, policy="p", evaluation=RIG, offset=0, trained_on=None):
    episodes = tuple(
        episode_from_labels(
            id=f"scene-{i + offset:02d}",
            source=name,
            labels=EpisodeLabels(success=won),
        )
        for i, won in enumerate(outcomes)
    )
    protocol = {"execute": 8}
    if trained_on:
        protocol["trained_on"] = trained_on
    return Cohort(
        name,
        episodes,
        provenance=Provenance(
            components=(ComponentRef("policy", policy, "1.0"), evaluation),
            protocol=protocol,
        ),
    )


@pytest.fixture
def arms():
    """Ten shared scenes; ph wins six that mg loses and loses none.

    Six, not five, on purpose: five disagreements all in one direction gives
    p=0.0625 and does not clear the bar. The exact test is doing real work at
    this sample size, which is the reason for using it.
    """
    ph = arm("ph", [True] * 8 + [False] * 2, policy="ft-ph", trained_on="lift/ph")
    mg = arm("mg", [True, True] + [False] * 8, policy="ft-mg", trained_on="lift/mg")
    return [ph, mg]


def test_conforms(arms):
    verdict = check_feedback(PolicyComparison(), arms, strict=True)
    assert verdict.ok, verdict.explain()


def test_it_holds_the_world_and_not_the_policy():
    """Every other comparative module holds the policy. This is the inverse."""
    module = PolicyComparison()
    assert "policy" not in module.holds()
    assert "evaluation" in module.holds()


def test_it_names_the_winner_with_a_paired_test(arms):
    (finding,) = PolicyComparison().run(arms).findings
    assert finding.code == "compare.best_policy"
    assert finding.evidence["best"] == "ph"
    assert finding.evidence["won"] == 6 and finding.evidence["lost"] == 0
    assert finding.evidence["p"] < 0.05
    assert finding.severity == "strong"
    assert "Use ph" in finding.prescription


def test_every_arm_gets_a_rate_with_an_interval(arms):
    report = PolicyComparison().run(arms)
    rate = report.measurements["ph.success_rate"]
    assert rate.value == pytest.approx(0.8)
    assert rate.n == 10 and rate.ci[0] < 0.8 < rate.ci[1]


def test_the_training_set_is_context_not_a_conclusion(arms):
    report = PolicyComparison().run(arms)
    (finding,) = report.findings
    assert finding.evidence["best_trained_on"] == "lift/ph"
    assert "cannot separate the training data from the training seed" in " ".join(report.notes)
    assert "data is better" not in (finding.prescription or "")


def test_a_small_difference_is_reported_without_a_prescription():
    close = [
        arm("a", [True] * 5 + [False] * 5, policy="a"),
        arm("b", [True] * 4 + [False] * 6, policy="b"),
    ]
    (finding,) = PolicyComparison().run(close).findings
    assert finding.severity == "weak"
    assert finding.prescription is None


def test_arms_sharing_no_scene_are_ranked_unpaired():
    left = arm("a", [True, True, False], policy="a")
    right = arm("b", [False, False, False], policy="b", offset=50)
    (finding,) = PolicyComparison().run([left, right]).findings
    assert finding.evidence["paired_scenes"] == 0
    assert "share no scene" in finding.summary
    assert finding.prescription is None, "unpaired is not convincing on its own"


def test_all_zero_ranks_nothing_and_says_why():
    """The failure mode this project cares most about not faking."""
    flat = [arm("a", [False] * 8, policy="a"), arm("b", [False] * 8, policy="b")]
    report = PolicyComparison().run(flat)
    assert not report.findings
    assert any("no resolving power" in note for note in report.notes)


def test_a_different_world_is_refused(arms):
    other = arm(
        "mg", [True] * 10, policy="ft-mg", evaluation=ComponentRef("evaluation", "libero", "1.0")
    )
    verdict = PolicyComparison().check_inputs([arms[0], other])
    assert "feedback.incomparable" in verdict.codes()
    with pytest.raises(IncompatibleError, match="differ on"):
        PolicyComparison().run([arms[0], other])


def test_comparing_one_policy_against_itself_is_noted():
    same = [arm("a", [True, False], policy="same"), arm("b", [True, True], policy="same")]
    verdict = PolicyComparison().check_inputs(same)
    assert "compare.same_policy" in verdict.codes()
    assert verdict.ok, "a note, not a refusal, it is a legitimate variance check"


def test_cohorts_without_outcomes_are_refused():
    empty = [
        Cohort(
            n,
            (episode_from_labels(id="s", source=n, labels=EpisodeLabels()),),
            provenance=Provenance(components=(ComponentRef("policy", n, "1"), RIG)),
        )
        for n in ("a", "b")
    ]
    verdict = PolicyComparison().check_inputs(empty)
    assert "compare.no_outcomes" in verdict.codes()


def test_pairing_counts_only_shared_scenes():
    left = arm("a", [True, False, True], policy="a")
    right = arm("b", [True, True], policy="b")
    from gantry_feedback_compare import arm_of

    both, only_l, only_r, neither = paired_counts(arm_of(left), arm_of(right))
    assert both + only_l + only_r + neither == 2


def test_the_scene_is_read_from_an_annotation_when_there_is_one():
    episode = episode_from_labels(
        id="run-9",
        source="x",
        labels=EpisodeLabels(success=True, annotations={"initial_state": 4}),
    )
    assert scene_of(episode) == "4"


# -- the floor ---------------------------------------------------------------
#
# Added after a real failure. Two checkpoints were compared on held-out data,
# one won by 8% on both scenes, and it was reported. Then a model that had
# learned nothing turned out to beat both of them.


def floor_of(**scores):
    return Floor(
        scores=scores, higher_is_better=True, method="no-learning predictors on the same scenes"
    )


def test_a_ranking_below_the_floor_is_refused_rather_than_reported():
    """The failure this exists for: the winner lost to a one-line heuristic, and
    "8% better than the control" and "worse than doing nothing" were the same
    result told from opposite ends."""
    report = PolicyComparison(floor=floor_of(does_nothing=0.40)).analyse(
        [arm("a", [True] * 3 + [False] * 7), arm("b", [True] * 1 + [False] * 9)]
    )
    codes = [f.code for f in report.findings]
    assert codes == ["compare.below_floor"]

    finding = report.findings[0]
    assert finding.severity == "strong"
    assert "required no learning at all" in finding.summary
    assert "does_nothing" in finding.evidence["hardest_trivial_predictor"]
    assert finding.evidence["floor"] == 0.4


def test_the_ranking_is_withheld_not_printed_beside_the_warning():
    """A number printed next to a caveat is a number somebody quotes without the
    caveat."""
    report = PolicyComparison(floor=floor_of(does_nothing=0.40)).analyse(
        [arm("a", [True] * 3 + [False] * 7), arm("b", [False] * 10)]
    )
    assert not any(f.code == "compare.best_policy" for f in report.findings)
    assert any("somebody will quote" in note for note in report.notes)
    # the per-arm rates are still measured -- refusing to rank is not refusing to
    # measure, and a reader needs the numbers to see how far below the floor it is
    assert "a.success_rate" in report.measurements


def test_clearing_the_floor_lets_the_ranking_through():
    report = PolicyComparison(floor=floor_of(does_nothing=0.10)).analyse(
        [arm("a", [True] * 8 + [False] * 2), arm("b", [True] * 3 + [False] * 7)]
    )
    assert any(f.code == "compare.best_policy" for f in report.findings)
    assert any("beat doing nothing" in note for note in report.notes)


def test_the_hardest_trivial_predictor_is_the_one_that_has_to_be_beaten():
    """Several trivial predictors, and the bar is the best of them -- beating the
    weakest while losing to a stronger one is not clearing the floor."""
    floor = floor_of(zeros=0.10, dataset_mean=0.25, copy_state=0.45)
    assert floor.best == ("copy_state", 0.45)
    assert not floor.cleared_by(0.30)
    assert floor.cleared_by(0.50)


def test_a_floor_works_for_an_error_metric_where_lower_is_better():
    """Prediction error, which is how this was found. The direction has to be
    declared because 0.50 beats 0.45 for a success rate and loses to it for an
    error."""
    floor = Floor(
        scores={"copy_state": 0.4252, "dataset_mean": 0.4434},
        higher_is_better=False,
        method="held-out MAE",
    )
    assert floor.best == ("copy_state", 0.4252)
    assert not floor.cleared_by(0.5029)  # the ego checkpoint, on the day
    assert not floor.cleared_by(0.5450)  # the scrambled control
    assert floor.cleared_by(0.3800)


def test_an_empty_floor_is_refused():
    with pytest.raises(ConfigError, match="not a floor"):
        Floor(scores={})


def test_no_floor_is_reported_rather_than_assumed_away():
    """A comparison without one is still a comparison. It simply cannot say
    whether the thing it ranked was worth ranking, and that gets said."""
    report = PolicyComparison().analyse(
        [arm("a", [True] * 8 + [False] * 2), arm("b", [True] * 3 + [False] * 7)]
    )
    assert any(f.code == "compare.best_policy" for f in report.findings)
    assert any("cannot say whether winning was worth anything" in n for n in report.notes)


def test_the_floor_lands_in_the_descriptor_so_a_run_records_its_bar():
    made = PolicyComparison(floor=floor_of(does_nothing=0.4))
    assert made.descriptor().metadata["floor"]["floor"] == 0.4
    assert PolicyComparison().descriptor().metadata["floor"] is None


# -- what two arms are paired on ------------------------------------------------
#
# This was collapsing every scene in a run into one. scene_of fell back to the
# part of the episode id before a "#" -- which is the *task*, identical for every
# scene -- and arm_of keyed a dict on it, so ten scenes became one entry and the
# other nine were dropped without a word. McNemar then ran on a single pair,
# which is not a test.


def episode_at(scene, success, train_seed=None, annotations=None):
    extra = {"scene": scene, **(annotations or {})}
    if train_seed is not None:
        extra["train_seed"] = train_seed
    return episode_from_labels(
        id=f"task#{scene}",
        source="t",
        task="t",
        labels=EpisodeLabels(success=success, annotations=extra),
    )


def test_scenes_are_not_collapsed_into_the_task():
    from gantry_feedback_compare.compare import scene_of

    bare = [
        episode_from_labels(
            id=f"pick_dual_bottles#{i:03d}",
            source="t",
            task="t",
            labels=EpisodeLabels(success=True),
        )
        for i in range(3)
    ]
    assert len({scene_of(e) for e in bare}) == 3


def test_the_recorded_scene_wins_over_the_id():
    from gantry_feedback_compare.compare import scene_of

    assert scene_of(episode_at("s7", True)) == "s7"


def test_several_training_seeds_on_one_scene_are_distinct_units():
    """Three seeds over ten scenes is thirty paired comparisons rather than ten,
    which is the cheapest real power available -- but only if the pairing key
    tells them apart."""
    from gantry_feedback_compare.compare import unit_of

    units = {unit_of(episode_at("s0", True, train_seed=s)) for s in (0, 1, 2)}
    assert len(units) == 3


def test_two_outcomes_for_one_unit_are_refused_not_silently_resolved():
    """Keeping the last is how a three-seed run quietly becomes a one-seed run."""
    from gantry_feedback_compare.compare import arm_of

    from gantry.contracts.feedback import Cohort
    from gantry.errors import ConfigError

    cohort = Cohort(name="a", episodes=(episode_at("s0", True), episode_at("s0", False)))
    with pytest.raises(ConfigError, match="record `train_seed`"):
        arm_of(cohort)


def test_a_rollout_writes_down_the_scene_it_attempted():
    """Nothing else in the record identifies it: an episode id is a convention,
    and a module that parses one is a rename away from pairing everything with
    everything."""
    import numpy as np

    from gantry.contracts.evaluator import Protocol, Scene, TaskSpec
    from gantry.contracts.policy import Policy, policy_descriptor
    from gantry.resolve import requires_channels
    from gantry.rollout import ClosedLoop, Step
    from gantry.spine import ChannelSpec

    class World:
        def begin(self, scene):
            self.n = 0
            return {"c": np.zeros(1)}

        def advance(self, action):
            self.n += 1
            return Step(observation={"c": np.zeros(1)}, reward=0.0, done=True, success=True)

        def close(self):
            pass

    class Suite(ClosedLoop):
        embodiment = "e"

        def descriptor(self):
            from gantry.contracts.evaluator import evaluator_descriptor

            return evaluator_descriptor(
                name="s",
                version="0.1",
                stage_events=False,
                outcomes=True,
                seedable=True,
                closed_loop=True,
                hosts_embodiment=False,
            )

        def action(self):
            return ChannelSpec("action", "vector", (2,), "float32", semantics="actuation")

        def world_for(self, scene):
            return World()

        def task_for(self, name="t", scenes=2, horizon=None):
            return TaskSpec(name, scenes=tuple(Scene(id=f"s{i}", seed=i) for i in range(scenes)))

    class P(Policy):
        def descriptor(self):
            return policy_descriptor(
                name="p", version="0.1", horizon=1, chunk=1, deterministic=True
            )

        def action_spec(self):
            return ChannelSpec("action", "vector", (2,), "float32", semantics="actuation")

        def observes(self):
            return requires_channels("p", "policy")

        def reset(self, context):
            pass

        def act(self, observation):
            return np.zeros((1, 2), dtype="float32")

    task = TaskSpec("t", scenes=(Scene(id="alpha", seed=7), Scene(id="beta", seed=9)))
    record = Suite().run(P(), task, Protocol())
    scenes = [e.labels.annotations["scene"] for e in record.episodes]
    assert scenes == ["alpha", "beta"]
    assert record.episodes[0].labels.annotations["scene_seed"] == 7
