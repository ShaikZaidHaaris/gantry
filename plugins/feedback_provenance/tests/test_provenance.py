"""Licence tracing, and the two asymmetries it rests on."""

from __future__ import annotations

import numpy as np
import pytest
from gantry_feedback_provenance import (
    ATTRIBUTION,
    NON_COMMERCIAL,
    PERMISSIVE,
    UNKNOWN,
    Provenance,
    chain,
    classify,
    usable_for,
    worst,
)

from gantry.contracts.feedback import Cohort
from gantry.spine import ChannelSpec, EpisodeLabels, episode_from_arrays

SPEC = ChannelSpec("x", "vector", (1,), "float32")


def episode(index=0, license=None, **extra):
    return episode_from_arrays(
        {"x": np.zeros((2, 1), dtype="float32")},
        [SPEC],
        id=f"e{index}",
        source="ego",
        license=license,
        labels=EpisodeLabels(annotations=dict(extra)),
    )


def cohort(episodes, name="theirs"):
    return Cohort(name=name, episodes=tuple(episodes))


def codes(report):
    return [f.code for f in report.findings]


# -- the two asymmetries -----------------------------------------------------


def test_the_most_restrictive_licence_governs_rather_than_the_commonest():
    """A dataset derived through five components carries the worst licence in the
    chain. One CC-BY-NC step makes the whole thing non-commercial however
    permissive everything else was."""
    claims = [
        episode(0, license="Apache-2.0"),
        episode(1, license="MIT", estimator_licence="Apache-2.0"),
        episode(2, license="Apache-2.0", depth_licence="CC-BY-NC-4.0"),
    ]
    report = Provenance().analyse([cohort(claims)])
    assert "provenance.non_commercial" in codes(report)
    assert report.measurements["theirs.restriction"].detail["governing"] == "non-commercial"


def test_undeclared_outranks_non_commercial():
    """Surprising and the right way round. A stated restriction can be swapped or
    disclosed; an unstated one cannot be planned around at all."""
    assert UNKNOWN > NON_COMMERCIAL > ATTRIBUTION > PERMISSIVE
    assert worst([]) == UNKNOWN

    report = Provenance().analyse(
        [
            cohort(
                [
                    episode(0, license="CC-BY-NC-4.0"),
                    episode(1, license="Some Bespoke Licence v3"),
                ]
            )
        ]
    )
    undeclared = next(f for f in report.findings if f.code == "provenance.undeclared")
    assert undeclared.severity == "strong"
    assert "unbounded" in undeclared.prescription


def test_an_unrecognised_licence_is_unknown_rather_than_assumed_permissive():
    """Reading an unfamiliar licence as permissive produces a confident wrong
    answer; reading it as unknown produces a question."""
    assert classify("Weird Corp Community Licence") == UNKNOWN
    assert classify("") == UNKNOWN
    assert classify(None) == UNKNOWN
    assert classify("Apache-2.0") == PERMISSIVE


# -- the real cases this was written for ------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Apache-2.0 (MediaPipe hand landmarker + OpenCV solvePnP)", PERMISSIVE),
        ("Apache-2.0 (RTMPose / rtmlib)", PERMISSIVE),
        ("MIT", PERMISSIVE),
        ("BSD-3-Clause", PERMISSIVE),
        ("CC-BY-NC-4.0 (non-commercial)", NON_COMMERCIAL),
        ("CC-BY-NC-ND", NON_COMMERCIAL),
        ("research only", NON_COMMERCIAL),
        ("requires the MANO model licence", NON_COMMERCIAL),
        ("CC-BY-4.0", ATTRIBUTION),
        ("GPL-3.0", ATTRIBUTION),
    ],
)
def test_the_licences_this_pipeline_actually_encounters(text, expected):
    assert classify(text) == expected


def test_a_hawor_style_dataset_is_caught():
    """The scenario the module exists for: the best hand models are CC-BY-NC-ND
    on a registration-gated MANO, and nothing in the arrays they produce records
    it."""
    report = Provenance(intent="commercial").analyse(
        [cohort([episode(0, license="Apache-2.0", estimator_licence="CC-BY-NC-ND (HaWoR) + MANO")])]
    )
    finding = next(f for f in report.findings if f.code == "provenance.non_commercial")
    assert finding.severity == "strong"
    assert "cannot be used commercially" in finding.summary
    assert "rebuild" in finding.prescription


def test_the_same_dataset_is_fine_for_research():
    """A research benchmark is unbothered by CC-BY-NC; a product is stopped by
    it. Stating the intent is what lets one module serve both."""
    made = Provenance(intent="research")
    report = made.analyse(
        [cohort([episode(0, license="Apache-2.0", estimator_licence="CC-BY-NC-ND (HaWoR)")])]
    )
    finding = next(f for f in report.findings if f.code == "provenance.non_commercial")
    assert finding.severity == "info"
    assert "fine for research use" in finding.summary
    assert "later becomes a product" in finding.prescription


def test_this_pipelines_own_output_comes_out_clean():
    """The whole reason MediaPipe and OpenCV were chosen over better models."""
    report = Provenance().analyse(
        [
            cohort(
                [
                    episode(
                        0,
                        license="Apache-2.0 (MediaPipe hand landmarker + OpenCV solvePnP)",
                        estimator_licence="Apache-2.0 (RTMPose / rtmlib)",
                    )
                ]
            )
        ]
    )
    assert codes(report) == ["provenance.clean"]
    assert "may be used commercially" in report.findings[0].summary


# -- where it looks ----------------------------------------------------------


def test_licences_are_found_wherever_components_recorded_them():
    """Different components record theirs in different places and no single
    field is authoritative."""
    made = episode(
        0,
        license="MIT",
        estimator_licence="Apache-2.0",
        depth_license="CC-BY-NC",
        trajectory_licence="BSD",
    )
    rows = chain(made)
    assert len(rows) == 4
    assert {r["restriction"] for r in rows} == {"permissive", "non-commercial"}


def test_a_cohort_with_no_licence_anywhere_is_reported_as_unknown_not_free():
    report = Provenance().analyse([cohort([episode(0)])])
    assert codes(report) == ["provenance.nothing_declared"]
    assert report.findings[0].severity == "strong"
    assert "unknown rather than unrestricted" in report.findings[0].summary


def test_attribution_is_reported_when_it_governs():
    report = Provenance().analyse([cohort([episode(0, license="CC-BY-4.0")])])
    finding = next(f for f in report.findings if f.code == "provenance.attribution")
    assert "with attribution" in finding.summary
    assert "condition of use" in finding.prescription


def test_usable_for_is_one_word_for_a_badge():
    assert usable_for(cohort([episode(0, license="MIT")])) == "permissive"
    assert usable_for(cohort([episode(0, license="CC-BY-NC")])) == "non-commercial"
    assert usable_for(cohort([episode(0)])) == "undeclared"


# -- the plumbing ------------------------------------------------------------


def test_it_says_out_loud_that_it_is_not_legal_advice():
    """It will end up quoted in a report and somebody will otherwise take it for
    an opinion."""
    assert "not legal advice" in Provenance().descriptor().metadata["disclaimer"]


def test_an_unknown_intent_is_refused():
    with pytest.raises(ValueError, match="commercial.*research"):
        Provenance(intent="vibes")


def test_several_cohorts_are_each_traced():
    report = Provenance().analyse(
        [
            cohort([episode(0, license="MIT")], "clean"),
            cohort([episode(0, license="CC-BY-NC")], "encumbered"),
        ]
    )
    assert report.measurements["clean.restriction"].value == float(PERMISSIVE)
    assert report.measurements["encumbered.restriction"].value == float(NON_COMMERCIAL)
