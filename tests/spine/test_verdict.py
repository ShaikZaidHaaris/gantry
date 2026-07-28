from __future__ import annotations

import pytest

from gantry.spine import IncompatibleError, Verdict


def test_yes_is_truthy_and_no_is_falsy():
    assert Verdict.yes()
    assert not Verdict.no("x.y", "nope")


def test_conjunction_keeps_every_reason():
    combined = Verdict.all(
        [Verdict.no("a.1", "first"), Verdict.yes(), Verdict.no("b.2", "second")]
    )
    assert not combined.ok
    assert combined.codes() == ("a.1", "b.2")


def test_a_note_passes_but_still_reports():
    verdict = Verdict.note("units.undeclared", "only one side declared units")
    assert verdict.ok
    assert verdict.codes() == ("units.undeclared",)


def test_detail_is_addressable_by_code():
    verdict = Verdict.no("rate.mismatch", "30 vs 20", provider=30.0, consumer=20.0)
    assert verdict.because("rate.mismatch")[0].detail["provider"] == 30.0
    assert verdict.because("nothing.here") == ()


def test_explain_is_multiline_and_names_the_codes():
    text = Verdict.all([Verdict.no("a.1", "first"), Verdict.no("b.2", "second")]).explain()
    assert text.splitlines()[0] == "refused"
    assert "[a.1] first" in text and "[b.2] second" in text


def test_hint_is_rendered_when_present():
    assert "(try this)" in str(Verdict.no("x", "broken", hint="try this").reasons[0])


def test_escalation_carries_the_verdict():
    verdict = Verdict.no("a.1", "first")
    with pytest.raises(IncompatibleError) as caught:
        verdict.raise_if_refused("while planning the run")
    assert caught.value.verdict is verdict
    assert "while planning the run" in str(caught.value)


def test_escalating_a_pass_does_nothing():
    Verdict.yes().raise_if_refused()
