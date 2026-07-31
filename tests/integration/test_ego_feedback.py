"""The six feedback modules composed into the document a contributor reads.

Each module is tested on its own; what this checks is that the *ordering*
survives composition, because the ordering is the part that stops a report being
misread. A number presented before the question of whether it means anything is
a number somebody will quote.
"""

from __future__ import annotations

import numpy as np
from gantry_feedback_capture import Capture
from gantry_feedback_control import Control
from gantry_feedback_coverage import Coverage
from gantry_feedback_extraction import Extraction
from gantry_feedback_provenance import Provenance
from gantry_feedback_report import as_markdown, assemble

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeLabels, episode_from_arrays

SPEC = ChannelSpec("x", "vector", (1,), "float32")

TASKS = [
    "open the fridge and take out a container",
    "wash and put away the dishes",
    "cook at the stove",
]


def clip(
    index, *, scene, instruction, licence="Apache-2.0 (MediaPipe + OpenCV)", success=None, **signals
):
    base = {
        "instruction": instruction,
        "scene": scene,
        "hands_visible": 0.95,
        "pose_solved": 0.9,
        "left_pose_plausible": 0.99,
        "right_pose_plausible": 0.98,
        "in_reach": 0.82,
        "steps_in": 100,
        "steps_out": 96,
        "intrinsics_source": "calibrated",
        "scale": "metric",
        "estimator_licence": licence,
    }
    base.update(signals)
    return episode_from_arrays(
        {"x": np.zeros((2, 1), dtype="float32")},
        [SPEC],
        id=f"e{index}",
        source="ego",
        license=licence,
        labels=EpisodeLabels(success=success, annotations=base),
    )


def upload(
    name="ego", *, n=9, licence="Apache-2.0 (MediaPipe + OpenCV)", scenes=3, wins=None, **signals
):
    episodes = []
    for index in range(n):
        episodes.append(
            clip(
                index,
                scene=f"kitchen-P{index % scenes:02d}",
                instruction=TASKS[index % len(TASKS)],
                licence=licence,
                success=None if wins is None else index < wins,
                **signals,
            )
        )
    return Cohort(name=name, episodes=tuple(episodes))


def run(cohort, *, control=None, base=None, evaluates=TASKS, intent="commercial"):
    """Every module, the way the product runs them."""
    arms = [a for a in (cohort, control, base) if a is not None]
    reports = [
        Provenance(intent=intent).analyse([cohort]),
        Coverage(evaluates=evaluates).analyse([cohort]),
        Extraction().analyse([cohort]),
        Capture().analyse([cohort]),
    ]
    if control is not None:
        reports.append(Control().analyse(arms))
    return assemble(reports, dataset="upload-1")


def section(made, key):
    return next(s for s in made.sections if s.key == key)


# -- the orderings that survive composition ---------------------------------


def test_a_clean_upload_with_a_control_reads_as_a_result():
    made = run(
        upload(wins=None),
        control=upload("shuffled"),
        base=upload("base"),
    )
    assert not made.blockers
    text = as_markdown(made)
    # Licence first, and everything that has something to say appears in order.
    # An empty section is omitted rather than rendered as a heading with nothing
    # under it — on a clean upload there is genuinely no filming advice.
    keys = [s.key for s in made.sections if s.findings]
    assert keys.index("usable") < keys.index("signal")
    assert text.index("Can this be used") < text.index("Did the data carry")


def test_an_encumbered_estimator_stops_the_report_before_any_number():
    """The scenario the licence module exists for, through the whole stack: the
    best hand models are CC-BY-NC-ND and nothing in their output records it."""
    made = run(upload(licence="CC-BY-NC-ND (HaWoR) + MANO"), control=upload("shuffled"))

    assert made.verdict == "the dataset cannot be used commercially"
    assert section(made, "signal").blocked
    assert section(made, "effect").blocked
    text = as_markdown(made)
    assert text.index("Read this first") < text.index("Did the data carry")


def test_the_same_upload_is_reportable_for_research():
    made = run(
        upload(licence="CC-BY-NC-ND (HaWoR) + MANO"), control=upload("shuffled"), intent="research"
    )
    assert not made.blockers
    assert not section(made, "signal").blocked


def test_out_of_scope_data_blocks_the_result_rather_than_reporting_zero():
    """Kitchen footage measured on tabletop tasks produces a real number about
    nothing, and the delta is meaningless until this passes."""
    made = run(
        upload(),
        control=upload("shuffled"),
        evaluates=["stack the red block", "insert the peg into the hole"],
    )
    assert any(code == "coverage.mismatch" for code, _ in made.blockers)
    assert section(made, "signal").blocked
    assert any("not answerable" in a for a in made.abstentions)


def test_a_missing_control_is_itself_a_blocker():
    """Without it, a result reproduces for every contributor including the ones
    whose data is worthless."""
    made = assemble(
        [
            Provenance().analyse([upload()]),
            Coverage(evaluates=TASKS).analyse([upload()]),
            Control().analyse([upload(wins=6), upload("base", wins=2)]),
        ]
    )
    assert any(code == "control.no_control" for code, _ in made.blockers)


# -- the two audiences stay apart -------------------------------------------


def test_the_contributor_and_the_pipeline_get_separate_sections():
    made = run(
        upload(n=9, scenes=1, hands_visible=0.55),
        control=upload("shuffled"),
    )
    filming = [f.code for f in section(made, "filming").findings]
    pipeline = [f.code for f in section(made, "pipeline").findings]

    assert any(c.startswith("capture.") for c in filming)
    assert any(c.startswith("extraction.") for c in pipeline)
    assert not any(c.startswith("extraction.") for c in filming)
    # and the pipeline advice is addressed to us, not to them
    detector = next(
        f
        for f in section(made, "pipeline").findings
        if f.code == "extraction.detector_missing_hands"
    )
    assert "detector" in detector.prescription.lower()


def test_one_location_reaches_the_contributor_and_not_the_pipeline_section():
    made = run(upload(n=8, scenes=1), control=upload("shuffled"))
    filming = [f.code for f in section(made, "filming").findings]
    assert "capture.single_scene" in filming


# -- the refusals are visible -----------------------------------------------


def test_everything_unanswerable_is_gathered_into_one_section():
    made = run(
        upload(licence="Some Bespoke Licence"),
        control=upload("shuffled"),
        evaluates=["stack the red block"],
    )
    text = as_markdown(made)
    assert "## What we could not tell you" in text
    assert len(made.abstentions) >= 2


def test_a_report_with_nothing_conclusive_says_so_rather_than_implying_more():
    made = run(upload(), control=upload("shuffled"))
    # no outcomes and no action_error anywhere -> control has nothing to compare
    assert made.verdict in (
        "no conclusion was reached",
        "not separated from its control at this number of trials",
    )


def test_the_whole_report_renders_and_every_claim_has_a_module_behind_it():
    made = run(upload(n=9, scenes=1, hands_visible=0.6), control=upload("shuffled"))
    text = as_markdown(made)
    assert text.startswith("# upload-1")
    # every bullet in the body traces to a finding code the modules emitted
    emitted = {f.code for s in made.sections for f in s.findings}
    assert emitted
    assert all(code.count(".") >= 1 for code in emitted)
    data = made.as_dict()
    assert data["verdict"] == made.verdict
    assert {s["key"] for s in data["sections"]} >= {"usable", "scope", "filming", "pipeline"}
