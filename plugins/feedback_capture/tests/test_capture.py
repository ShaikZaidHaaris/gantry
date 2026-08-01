"""Filming advice, and the promise it is not allowed to make."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_feedback_capture import (
    CAUSAL_WORDS,
    CHECKS,
    Capture,
    Check,
    measured,
    top_fixes,
    variety,
)

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeLabels, episode_from_arrays

SPEC = ChannelSpec("x", "vector", (1,), "float32")


def cohort(clips, name="theirs"):
    """`clips` is a list of annotation dicts, one per clip."""
    episodes = tuple(
        episode_from_arrays(
            {"x": np.zeros((2, 1), dtype="float32")},
            [SPEC],
            id=f"{name}-{index}",
            source=name,
            labels=EpisodeLabels(annotations=dict(annotations)),
        )
        for index, annotations in enumerate(clips)
    )
    return Cohort(name=name, episodes=episodes)


def varied(index, **over):
    """A clip from a well-made upload: its own location and its own description."""
    return clean(scene=f"room-{index}", instruction=f"do thing {index}", **over)


def clean(**over):
    base = {
        "hands_visible": 0.98,
        "in_reach": 0.95,
        "usable_length": 1.0,
        "motion_ok": 0.95,
        "stabilized": 0.0,
        "labelled": 1.0,
        "scene": "kitchen-1",
        "instruction": "pick up the mug",
    }
    base.update(over)
    return base


def codes(report):
    return [finding.code for finding in report.findings]


# -- the honesty guard -------------------------------------------------------


def test_no_prescription_promises_a_gain():
    """The association is across datasets and says nothing causal about this one.
    A prescription *wants* to promise something, which is why this is easier to
    break here than anywhere else in the feedback plane."""
    report = Capture(better={"hands_visible": 0.96, "in_reach": 0.93}).analyse(
        [cohort([clean(hands_visible=0.6, in_reach=0.4)] * 8)]
    )
    assert report.findings
    for finding in report.findings:
        text = (finding.prescription or "").lower()
        for phrase in CAUSAL_WORDS:
            assert phrase not in text, f"{finding.code} promised: {phrase}"


def test_the_comparison_is_phrased_as_an_observation_about_other_datasets():
    report = Capture(better={"hands_visible": 0.96}).analyse(
        [cohort([clean(hands_visible=0.6)] * 5)]
    )
    finding = next(f for f in report.findings if f.code == "capture.hands_offscreen")
    assert "uploads that scored above this one average 96%" in finding.prescription
    assert finding.evidence["datasets_above_this_one"] == 0.96


def test_it_is_useful_with_no_comparison_available():
    """The findings simply say what is wrong rather than how far from the field."""
    report = Capture().analyse([cohort([clean(hands_visible=0.6)] * 5)])
    finding = next(f for f in report.findings if f.code == "capture.hands_offscreen")
    assert "uploads that scored above" not in finding.prescription
    assert "datasets_above_this_one" not in finding.evidence
    assert finding.prescription


# -- ordering is the product -------------------------------------------------


def test_findings_come_out_ordered_by_measured_cost_not_by_how_bad_they_sound():
    """A partly-bad expensive problem outranks a completely-bad cheap one.

    Hands at 0.7 are 22% short of their bar; outcomes at 0.0 are 100% short of
    theirs. Dropped frames still win, because they weigh three times what missing
    metadata does — one costs training data, the other costs a column.
    """
    report = Capture().analyse(
        [cohort([varied(index, hands_visible=0.7, labelled=0.0) for index in range(6)])]
    )
    assert codes(report)[0] == "capture.hands_offscreen"
    assert "capture.unlabelled_outcomes" in codes(report)


def test_cost_is_a_fraction_of_the_bar_not_a_raw_distance():
    """The thresholds are not on one scale — 0.9 for hand visibility and 0.05 for
    instruction variety are both "the bar" — so a raw gap would make whichever
    check has the highest threshold always look worst."""
    strict = Check(key="a", code="a", threshold=0.9, weight=1.0)
    lenient = Check(key="b", code="b", threshold=0.1, weight=1.0)
    # Both a tenth of their own bar short: the same amount of wrong.
    assert strict.cost(0.81) == pytest.approx(lenient.cost(0.09), rel=1e-6)


def test_only_the_leading_few_are_strong():
    """A list of eleven equally urgent things is not a list."""
    report = Capture(lead=3).analyse(
        [
            cohort(
                [
                    varied(
                        index,
                        hands_visible=0.5,
                        in_reach=0.4,
                        usable_length=0.5,
                        motion_ok=0.5,
                        labelled=0.0,
                    )
                    for index in range(6)
                ]
            )
        ]
    )
    severities = [finding.severity for finding in report.findings]
    assert severities[:3] == ["strong"] * 3
    assert set(severities[3:]) == {"weak"}


def test_top_fixes_is_three_sentences_for_a_gui():
    report = Capture().analyse(
        [cohort([clean(hands_visible=0.5, in_reach=0.4, usable_length=0.5)] * 6)]
    )
    fixes = top_fixes(report)
    assert len(fixes) == 3
    # Each fix is the finding and its remedy in one line. It used to be checked
    # by looking for an em dash between them; the copy reads better without one,
    # so the check is that both halves are present rather than the punctuation
    # that happened to join them.
    assert all(". " in fix and len(fix) > 40 for fix in fixes)
    assert "out of frame" in fixes[0]


# -- not measured is not fine ------------------------------------------------


def test_a_signal_nobody_measured_is_absent_rather_than_zero():
    """'Your hands were never visible' and 'nobody looked' are different
    sentences, and it would be especially cruel to confuse them here."""
    report = Capture().analyse([cohort([{"scene": "k", "instruction": "x"}] * 4)])
    assert "capture.hands_offscreen" not in codes(report)
    assert any("ot measured is not the same as fine" in note for note in report.notes)


def test_a_cohort_with_nothing_wrong_says_so():
    report = Capture().analyse([cohort([varied(index) for index in range(6)])])
    assert codes(report) == ["capture.clean"]


# -- the individual checks ---------------------------------------------------


def test_hands_off_frame_is_the_heaviest_check():
    """A frame with no hand in it cannot produce a hand pose, so those steps are
    dropped before anything is trained."""
    report = Capture().analyse([cohort([clean(hands_visible=0.78)] * 5)])
    finding = next(f for f in report.findings if f.code == "capture.hands_offscreen")
    # 22%, not 12%. The number a person recognises is the actual shortfall
    # (1 - 0.78), not the distance below a 0.9 bar. The earlier version printed
    # the latter, so footage with no hands at all read as "90% out of frame".
    assert "out of frame for 22%" in finding.summary
    assert "cannot produce a hand pose" in finding.prescription


def test_out_of_workspace_reaching_is_reported_from_the_retargeters_number():
    """Read off the record rather than recomputed — this module never opens a
    video, which is what keeps it cheap enough to run on every upload."""
    report = Capture().analyse([cohort([clean(in_reach=0.55)] * 5)])
    finding = next(f for f in report.findings if f.code == "capture.out_of_workspace")
    assert "where the arm cannot go" in finding.summary
    assert "Stand closer" in finding.prescription


def test_one_location_fires_whether_the_upload_is_large_or_small():
    """Usually the largest single difference between the uploads that transfer
    and the ones that do not — and the advice is the same at six clips as at
    forty, which a per-clip ratio got wrong."""
    for size in (6, 40):
        report = Capture().analyse([cohort([clean()] * size)])
        finding = next(f for f in report.findings if f.code == "capture.single_scene")
        assert f"1 location(s) across {size} clips" in finding.summary


def test_many_locations_does_not_fire():
    report = Capture().analyse([cohort([clean(scene=f"room-{i}") for i in range(10)])])
    assert "capture.single_scene" not in codes(report)


def test_one_instruction_everywhere_means_the_language_carries_nothing():
    report = Capture().analyse([cohort([clean()] * 40)])
    finding = next(f for f in report.findings if f.code == "capture.one_instruction")
    assert "1 distinct instruction(s) across 40 clips" in finding.summary


def test_several_instructions_do_not_fire_however_few_clips_there_are():
    report = Capture().analyse(
        [cohort([clean(scene=f"r-{i}", instruction=f"do thing {i}") for i in range(4)])]
    )
    assert "capture.one_instruction" not in codes(report)


def test_stabilisation_fires_upward_rather_than_downward():
    """The only check where more is worse, so the direction is worth pinning."""
    report = Capture().analyse([cohort([clean(stabilized=1.0)] * 6)])
    finding = next(f for f in report.findings if f.code == "capture.stabilized")
    assert "6 clip(s)" in finding.summary
    assert "destroys the head trajectory" in finding.prescription

    assert "capture.stabilized" not in codes(
        Capture().analyse([cohort([clean(stabilized=0.0)] * 6)])
    )


def test_unlabelled_outcomes_fire_and_say_why_failures_are_welcome():
    report = Capture().analyse([cohort([clean(labelled=0.0)] * 6)])
    finding = next(f for f in report.findings if f.code == "capture.unlabelled_outcomes")
    assert "Failures are useful" in finding.prescription
    assert "quietly all failures" in finding.prescription


def test_booleans_are_read_as_well_as_fractions():
    """A per-clip flag and a per-cohort fraction are both natural things for an
    earlier stage to have written."""
    report = Capture().analyse(
        [cohort([clean(stabilized=True)] * 3 + [clean(stabilized=False)] * 1)]
    )
    assert report.measurements["theirs.stabilized"].value == pytest.approx(0.75)


# -- the checks are data -----------------------------------------------------


def test_a_new_check_is_an_entry_rather_than_a_code_change():
    """The set of things worth checking will grow, and every one is the same
    shape: read a number, compare it to a threshold, say what it costs."""
    extra = Check(
        key="lighting",
        code="capture.too_dark",
        threshold=0.6,
        weight=1.2,
        summary="{missing:.0%} of the footage is underexposed",
        fix="Film with the light in front of you.",
        why="A dark frame gives the estimator nothing to work with.",
    )
    report = Capture(checks=(*CHECKS, extra)).analyse(
        [cohort([varied(index, lighting=0.2) for index in range(5)])]
    )
    assert codes(report) == ["capture.too_dark"]
    assert "underexposed" in report.findings[0].summary


def test_every_shipped_check_has_a_fix_and_a_reason():
    for check in CHECKS:
        assert check.fix, check.code
        assert check.why, check.code
        assert check.summary, check.code


def test_the_measurements_carry_the_threshold_so_a_user_can_see_the_bar():
    report = Capture().analyse([cohort([clean(hands_visible=0.6)] * 5)])
    measurement = report.measurements["theirs.hands_visible"]
    assert measurement.value == 0.6
    assert measurement.detail["threshold"] == 0.9
    assert measurement.detail["fires"] is True
    assert measurement.n == 5


# -- the plumbing ------------------------------------------------------------


def test_the_helpers_read_off_annotations_and_metadata_alike():
    signals = measured(cohort([clean(hands_visible=0.5), clean(hands_visible=0.9)]))
    assert signals["hands_visible"] == pytest.approx(0.7)

    counts = variety(cohort([clean(scene="a"), clean(scene="b"), clean(scene="a")]))
    assert counts["scenes"] == 2
    assert counts["scene_variety"] == pytest.approx(2 / 3)


def test_it_analyses_one_cohort_and_prescribes():
    made = Capture()
    assert made.descriptor().provides["min_cohorts"] == 1
    assert made.descriptor().provides["prescribes"] is True
    assert made.descriptor().metadata["comparative"] is False
    assert Capture(better={"hands_visible": 0.9}).descriptor().metadata["comparative"] is True


def test_history_supplies_the_comparison_when_no_table_is_passed():
    report = Capture(history=lambda key: 0.97 if key == "hands_visible" else None).analyse(
        [cohort([clean(hands_visible=0.6)] * 5)]
    )
    finding = next(f for f in report.findings if f.code == "capture.hands_offscreen")
    assert finding.evidence["datasets_above_this_one"] == 0.97


def test_several_cohorts_are_each_assessed():
    report = Capture().analyse(
        [cohort([clean()] * 3, "a"), cohort([clean(hands_visible=0.4)] * 3, "b")]
    )
    assert any(finding.cohorts == ("b",) for finding in report.findings)
    assert "a.hands_visible" in report.measurements


# -- absent is not zero, and getting this wrong accuses the contributor ---------
#
# measured() says two functions above variety() that a signal nobody measured is
# absent rather than zero. variety() did the opposite: it counted len(set()) and
# reported it as a measurement. A corpus filmed in ten kitchens whose scene
# labels were lost on the way through LeRobot was told, at strong severity, to
# go and film somewhere else.


def episode_with(annotations=None, task=None, extra=None):
    from gantry.spine import EpisodeLabels
    from gantry.spine.episode import episode_from_labels

    return episode_from_labels(
        id="e",
        source="s",
        task=task,
        labels=EpisodeLabels(success=None, annotations=annotations or {}),
        **({"extra": extra} if extra else {}),
    )


def test_a_corpus_that_declares_no_scene_reports_nothing_rather_than_zero():
    from gantry_feedback_capture.capture import variety

    from gantry.contracts.feedback import Cohort

    quiet = Cohort(name="q", episodes=tuple(episode_with() for _ in range(6)))
    signals = variety(quiet)
    assert "scenes" not in signals
    assert "instructions" not in signals


def test_a_corpus_that_does_declare_them_is_counted():
    from gantry_feedback_capture.capture import variety

    from gantry.contracts.feedback import Cohort

    loud = Cohort(
        name="l",
        episodes=tuple(
            episode_with(annotations={"scene": f"kitchen{i}", "instruction": f"do thing {i}"})
            for i in range(4)
        ),
    )
    signals = variety(loud)
    assert signals["scenes"] == 4
    assert signals["instructions"] == 4


def test_the_sentence_is_found_where_lerobot_leaves_it():
    """A round trip stores it as meta.task and as a plural 'tasks' list. Reading
    only 'instruction' reported a corpus of ten kitchens as having none."""
    from gantry_feedback_capture.capture import variety

    from gantry.contracts.feedback import Cohort

    viaTask = Cohort(name="t", episodes=(episode_with(task="open the fridge"),))
    assert variety(viaTask)["instructions"] == 1

    viaList = Cohort(name="p", episodes=(episode_with(annotations={"tasks": ["chop vegetables"]}),))
    assert variety(viaList)["instructions"] == 1


def test_an_undeclared_scene_does_not_fire_the_filming_advice():
    """The whole point: no accusation without a measurement behind it."""
    from gantry_feedback_capture import Capture

    from gantry.contracts.feedback import Cohort

    quiet = Cohort(name="q", episodes=tuple(episode_with(task="do the thing") for _ in range(8)))
    report = Capture().analyse([quiet])
    codes = {f.code for f in report.findings}
    assert "capture.single_scene" not in codes
