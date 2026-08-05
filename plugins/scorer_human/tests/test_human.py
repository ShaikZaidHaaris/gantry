"""A person's judgements, recorded and read back.

The scorer itself is thin -- it looks up what somebody wrote down. What is worth
testing is the discipline around that: an unscored trial abstains rather than
failing, an anonymous session is refused because agreement needs to know who
judged, and the rubric reaches the page unabridged because the sentence is the
thing being measured.
"""

from __future__ import annotations

import json

import pytest
from gantry_scorer_human import HumanScorer, Session, read_session, write_page
from gantry_scorer_human.human import merge

from gantry.conformance import check_scorer
from gantry.contracts.scorer import VIDEO, Evidence
from gantry.contracts.task import Criterion, TaskDefinition

RUBRIC = (
    "The cube is clear of the table by at least 4 cm and held in the gripper. "
    "A cube nudged off the edge or dropped immediately does not count."
)


def task(*criteria):
    return TaskDefinition(
        name="lift_cube",
        instruction="lift the cube",
        success=criteria or (Criterion("lifted", {"height": 0.04}, RUBRIC),),
    )


def session(outcomes, rater="zaid"):
    return Session(
        rater=rater,
        judgements={f"seed_{i}": {"lifted": v} for i, v in enumerate(outcomes)},
    )


# -- reading a session back --------------------------------------------------


def test_a_session_round_trips_through_the_file_the_page_writes(tmp_path):
    path = tmp_path / "judgements.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "rater": "zaid",
                    "task": "lift_cube",
                    "trial": f"seed_{i}",
                    "criterion": "lifted",
                    "passed": v,
                    "note": "clean" if i == 0 else "",
                }
            )
            for i, v in enumerate([True, False, None])
        )
        + "\n"
    )
    back = read_session(path)
    assert back.rater == "zaid"
    assert back.judgements["seed_0"]["lifted"] is True
    assert back.judgements["seed_1"]["lifted"] is False
    assert back.judgements["seed_2"]["lifted"] is None
    assert back.notes["seed_0"] == "clean"


def test_an_empty_file_is_an_empty_session_not_a_crash(tmp_path):
    path = tmp_path / "nothing.jsonl"
    path.write_text("")
    assert read_session(path).judgements == {}


# -- judging -----------------------------------------------------------------


def test_it_returns_what_the_person_decided():
    scorer = HumanScorer(session([True, False]))
    judged = scorer.score(Evidence("seed_0", video="a.mp4"), task())
    assert judged[0].passed is True
    assert "zaid" in judged[0].rationale


def test_an_unscored_trial_abstains_rather_than_failing():
    """An incomplete session must not look like a bad policy."""
    scorer = HumanScorer(session([True]))
    judged = scorer.score(Evidence("seed_99", video="a.mp4"), task())
    assert judged[0].passed is None
    assert "not scored" in judged[0].rationale


def test_a_recorded_cannot_tell_stays_an_abstention():
    scorer = HumanScorer(session([None]))
    assert scorer.score(Evidence("seed_0", video="a.mp4"), task())[0].passed is None


def test_it_needs_video_and_says_so():
    scorer = HumanScorer(session([True]))
    assert scorer.needs == (VIDEO,)
    assert scorer.cost == "human"
    # A person is not deterministic, and this does not pretend otherwise.
    assert scorer.deterministic is False


def test_it_passes_the_conformance_kit():
    verdict = check_scorer(HumanScorer(session([True])), Evidence("seed_0", video="a.mp4"), task())
    assert verdict.ok, verdict.explain()


# -- what a session refuses --------------------------------------------------


def test_an_anonymous_session_is_refused():
    """Agreement needs the labels grouped by who produced them."""
    verdict = session([True], rater="").validate()
    assert not verdict.ok
    assert "session.anonymous" in verdict.codes()


def test_a_session_that_judged_nothing_is_refused():
    verdict = Session(rater="zaid").validate()
    assert not verdict.ok
    assert "session.empty" in verdict.codes()


def test_frequent_abstention_is_noted_as_a_rubric_problem():
    """Cheaper to learn from twenty videos than from a bench."""
    verdict = session([None, None, None, None, True]).validate()
    assert verdict.ok
    assert "session.abstains_often" in verdict.codes()
    assert session([None] * 4 + [True]).abstention_rate == pytest.approx(0.8)


# -- the page ----------------------------------------------------------------


def test_the_page_prints_the_rubric_verbatim(tmp_path):
    """Summarising it would measure agreement about a different sentence."""
    page = write_page([{"trial": "seed_0", "video": "seed_0.mp4"}], task(), tmp_path / "index.html")
    text = page.read_text()
    assert RUBRIC in text


def test_the_page_offers_an_abstention_button(tmp_path):
    page = write_page([{"trial": "s", "video": "s.mp4"}], task(), tmp_path / "i.html")
    assert "Can't tell" in page.read_text()


def test_the_page_carries_every_criterion(tmp_path):
    two = task(
        Criterion("lifted", {}, RUBRIC),
        Criterion("held", {}, "The cube stays in the gripper for a full second."),
    )
    page = write_page([{"trial": "s", "video": "s.mp4"}], two, tmp_path / "i.html")
    text = page.read_text()
    assert RUBRIC in text and "full second" in text


def test_the_page_needs_no_server_or_build_step(tmp_path):
    """A scoring tool with a build step is one nobody uses on a Friday."""
    page = write_page([{"trial": "s", "video": "s.mp4"}], task(), tmp_path / "i.html")
    text = page.read_text()
    assert "<script>" in text
    assert "http://" not in text and "https://" not in text


# -- merging -----------------------------------------------------------------


def test_two_sessions_from_one_rater_combine():
    first = Session(rater="zaid", judgements={"a": {"lifted": True}})
    second = Session(rater="zaid", judgements={"b": {"lifted": False}})
    merged = merge([first, second])
    assert set(merged["zaid"].judgements) == {"a", "b"}


def test_sessions_from_different_raters_stay_apart():
    """Which is what inter-rater agreement is computed from."""
    merged = merge(
        [
            Session(rater="a", judgements={"t": {"lifted": True}}),
            Session(rater="b", judgements={"t": {"lifted": False}}),
        ]
    )
    assert set(merged) == {"a", "b"}


def test_an_anonymous_session_is_dropped_rather_than_folded_in():
    """Folding it into somebody else's labels would corrupt the one number
    this is all for."""
    merged = merge(
        [
            Session(rater="zaid", judgements={"t": {"lifted": True}}),
            Session(rater="", judgements={"t": {"lifted": False}}),
        ]
    )
    assert set(merged) == {"zaid"}
