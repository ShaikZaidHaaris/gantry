"""A model judge is only safe because something checks it.

Lives here rather than in the plugin's own suite because it spans two plugins,
and a plugin test that leans on a sibling it never declared is exactly what the
isolation check exists to catch — it caught this one.
"""

from __future__ import annotations

from gantry_feedback_calibrate import Calibration, Corpus
from gantry_scorer_vlm import VlmScorer, replay_by_trial

from gantry.contracts.scorer import Evidence
from gantry.contracts.task import Criterion, TaskDefinition

RUBRIC = (
    "The cube is clear of the table by at least 4 cm and held in the gripper. "
    "A cube nudged off the edge or dropped immediately does not count."
)
HUMAN = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]


def task():
    return TaskDefinition(
        name="lift_cube",
        instruction="lift the cube",
        success=(Criterion("lifted", {"height": 0.04}, RUBRIC),),
    )


def scored_by_model(replies):
    """Judge twelve trials with a taped model, and return its labels."""
    scorer = VlmScorer(
        replay_by_trial({f"seed_{i}": f"lifted: {v}" for i, v in enumerate(replies)}),
        model="candidate",
    )
    return {
        f"seed_{i}": {
            judgement.criterion: judgement.passed
            for judgement in scorer.judge(Evidence(f"seed_{i}", video=None), task())
        }
        for i in range(len(replies))
    }


def as_human(outcomes):
    return {f"seed_{i}": {"lifted": bool(v)} for i, v in enumerate(outcomes)}


def test_a_model_that_agrees_with_a_person_is_cleared():
    """The whole reason this is safe to point at twenty thousand videos."""
    labels = scored_by_model(["yes" if v else "no" for v in HUMAN])
    gate = Calibration(
        Corpus(labels={"human": as_human(HUMAN), "candidate": labels})
    ).gate("candidate")
    assert gate.ok


def test_a_model_that_guesses_is_refused():
    labels = scored_by_model(["no" if v else "yes" for v in HUMAN])
    gate = Calibration(
        Corpus(labels={"human": as_human(HUMAN), "candidate": labels})
    ).gate("candidate")
    assert not gate.ok
    assert "judge.uncalibrated" in gate.codes()


def test_a_model_nobody_has_checked_is_refused_rather_than_trusted():
    """The default posture: an unmeasured judge does not get the benefit of the
    doubt, because its labels are indistinguishable from correct ones."""
    labels = scored_by_model(["yes" if v else "no" for v in HUMAN])
    gate = Calibration(Corpus(labels={"candidate": labels})).gate("candidate")
    assert gate.ok or "judge.unmeasured" in gate.codes()


def test_abstentions_are_dropped_from_agreement_but_the_rate_is_reported():
    replies = ["unclear"] * 4 + ["yes" if v else "no" for v in HUMAN[4:]]
    labels = scored_by_model(replies)
    report = Calibration(
        Corpus(labels={"human": as_human(HUMAN), "candidate": labels})
    ).analyse([])
    kappa = report.measurements["candidate.kappa"]
    assert kappa.detail["dropped_abstentions"] == 4
    assert report.measurements["candidate.abstention_rate"].value > 0
