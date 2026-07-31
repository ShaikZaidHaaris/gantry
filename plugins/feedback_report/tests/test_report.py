"""Assembly, and the orderings it refuses to let a reader get wrong."""

from __future__ import annotations

import json

from gantry_feedback_report import as_markdown, assemble, write

from gantry.contracts.feedback import Finding, Report


def report(module, *findings, notes=(), measurements=None):
    return Report(
        module=module,
        findings=tuple(findings),
        notes=tuple(notes),
        measurements=dict(measurements or {}),
    )


def finding(code, severity="info", summary="", prescription=None):
    return Finding(code=code, summary=summary or code, severity=severity, prescription=prescription)


def section(assembled, key):
    return next(s for s in assembled.sections if s.key == key)


# -- the ordering is the design ---------------------------------------------


def test_licence_comes_before_results_because_it_invalidates_them():
    """Putting it after means somebody reads the results first and remembers
    those."""
    made = assemble(
        [
            report("control", finding("control.data_carried_information", "strong")),
            report("provenance", finding("provenance.non_commercial", "strong")),
        ]
    )
    keys = [s.key for s in made.sections]
    assert keys.index("usable") < keys.index("signal")


def test_an_encumbered_dataset_leads_with_that_and_nothing_else():
    made = assemble(
        [
            report("provenance", finding("provenance.non_commercial", "strong")),
            report("control", finding("control.data_carried_information", "strong")),
        ]
    )
    assert made.verdict == "the dataset cannot be used commercially"
    assert made.blockers[0][0] == "provenance.non_commercial"


def test_a_blocker_suppresses_the_sections_it_invalidates_without_deleting_them():
    """The numbers are still computed and still in the record. They are just not
    presented as an answer, because they are not one."""
    made = assemble(
        [
            report("coverage", finding("coverage.mismatch", "strong")),
            report("control", finding("control.data_carried_information", "strong")),
        ]
    )
    signal = section(made, "signal")
    assert signal.blocked
    assert signal.suppressed_by == "coverage.mismatch"
    # still there, not deleted
    assert [f.code for f in signal.findings] == ["control.data_carried_information"]
    assert "still computed and are in the record" in as_markdown(made)


def test_a_missing_control_blocks_the_effect_section():
    made = assemble(
        [
            report("control", finding("control.no_control", "strong")),
            report("compare", finding("compare.delta", "strong", "up 9 points")),
        ]
    )
    assert section(made, "effect").blocked
    assert made.verdict == "there is no control, so nothing here is attributable to the data"


def test_a_control_that_wins_is_a_blocker_rather_than_a_result():
    made = assemble([report("control", finding("control.control_wins", "strong"))])
    assert made.blockers
    assert "something is wrong" in made.verdict


# -- what it will not do -----------------------------------------------------


def test_nothing_is_upgraded_in_transit():
    """The assembler can order, group and quote. It cannot turn a weak finding
    into a headline — every sentence is traceable to a module that signed it."""
    weak = finding("control.not_separated", "weak", "not separated at this n")
    made = assemble([report("control", weak)])
    out = section(made, "signal").findings[0]
    assert out.severity == "weak"
    assert out.summary == "not separated at this n"
    assert made.verdict == "not separated from its control at this number of trials"


def test_the_worst_finding_in_a_section_comes_first():
    made = assemble(
        [
            report(
                "capture",
                finding("capture.a", "info"),
                finding("capture.b", "strong"),
                finding("capture.c", "weak"),
            )
        ]
    )
    assert [f.severity for f in section(made, "filming").findings] == ["strong", "weak", "info"]


def test_a_code_from_a_module_nobody_has_written_yet_is_kept():
    made = assemble([report("future", finding("newthing.happened", "strong"))])
    other = section(made, "other")
    assert [f.code for f in other.findings] == ["newthing.happened"]


# -- the section that is not an appendix ------------------------------------


def test_refusals_are_a_section_rather_than_a_footnote():
    """A report whose abstentions are buried reads as more confident than it is,
    and the refusals are the reason to trust the rest."""
    made = assemble(
        [
            report("coverage", finding("coverage.mismatch", "strong")),
            report("control", finding("control.not_separated", "weak", "arms overlap")),
            report(
                "extraction",
                notes=("theirs: 2 stages wrote no signal — not measured is not the same as fine",),
            ),
        ]
    )
    text = as_markdown(made)
    assert "## What we could not tell you" in text
    assert any("not answerable" in a for a in made.abstentions)
    assert any("arms overlap" in a for a in made.abstentions)
    assert any("not measured is not the same as fine" in a for a in made.abstentions)


def test_filming_and_pipeline_advice_stay_apart():
    """Different owners. Telling a contributor to re-film because our detector
    was weak is the kind of wrong that loses a customer."""
    made = assemble(
        [
            report("capture", finding("capture.hands_offscreen", "strong", "hands out of frame")),
            report(
                "extraction",
                finding("extraction.detector_missing_hands", "strong", "the detector found 60%"),
            ),
        ]
    )
    assert [f.code for f in section(made, "filming").findings] == ["capture.hands_offscreen"]
    assert [f.code for f in section(made, "pipeline").findings] == [
        "extraction.detector_missing_hands"
    ]
    text = as_markdown(made)
    assert "What to change about the filming" in text
    assert "What we should change on our side" in text


# -- the whole thing ---------------------------------------------------------


def test_a_clean_run_reads_as_a_result():
    made = assemble(
        [
            report("provenance", finding("provenance.clean", "info", "permissive throughout")),
            report("coverage", finding("coverage.ample", "info", "covers 100% of tasks")),
            report(
                "control",
                finding(
                    "control.data_carried_information", "strong", "beat its control 41% to 12%"
                ),
            ),
            report("compare", finding("compare.delta", "strong", "up 9 points, p=0.003")),
            report("capture", finding("capture.single_scene", "strong", "one location")),
        ],
        dataset="upload-42",
    )
    assert not made.blockers
    assert made.verdict == ("the data carried information beyond what fine-tuning alone provides")
    text = as_markdown(made)
    assert text.startswith("# upload-42")
    assert "beat its control 41% to 12%" in text
    assert not any(s.blocked for s in made.sections)


def test_it_writes_both_forms_so_a_claim_can_be_traced(tmp_path):
    made = assemble(
        [report("control", finding("control.not_separated", "weak"))], dataset="upload-7"
    )
    path = write(made, tmp_path / "report.md")
    assert path.exists()
    data = json.loads(path.with_suffix(".json").read_text())
    assert data["dataset"] == "upload-7"
    assert data["verdict"] == made.verdict
    assert data["sections"]


def test_measurements_are_carried_through_for_the_system_that_consumes_them():
    from gantry.spine import Measurement

    made = assemble(
        [
            report(
                "control",
                finding("control.x"),
                measurements={"ego.success_rate": Measurement(value=0.4, n=60)},
            )
        ]
    )
    assert made.as_dict()["measurements"]["ego.success_rate"]["value"] == 0.4


def test_an_empty_run_says_no_conclusion_rather_than_implying_one():
    made = assemble([])
    assert made.verdict == "no conclusion was reached"


def test_a_blocking_code_marked_info_by_its_own_module_does_not_block():
    """Severity is the module's judgement of how much its finding matters, and
    the assembler is not entitled to overrule it. Provenance emits
    `non_commercial` at `info` for a research-intent run — blocking on the code
    alone made every research report lead with a stop its own module had called
    a note."""
    made = assemble(
        [
            report(
                "provenance",
                finding(
                    "provenance.non_commercial", "info", "non-commercial; fine for research use"
                ),
            ),
            report("control", finding("control.data_carried_information", "strong")),
        ]
    )
    assert not made.blockers
    assert not section(made, "signal").blocked
    assert made.verdict == ("the data carried information beyond what fine-tuning alone provides")


def test_the_same_code_marked_strong_does_block():
    made = assemble(
        [
            report("provenance", finding("provenance.non_commercial", "strong")),
            report("control", finding("control.data_carried_information", "strong")),
        ]
    )
    assert made.blockers
    assert section(made, "signal").blocked
