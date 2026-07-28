"""Every shipped feedback module, held to the contract it claims."""

from __future__ import annotations

import pytest
from gantry.conformance import check_feedback, feedback_checks
from gantry.contracts.feedback import Cohort, Finding, Report, feedback_descriptor
from gantry.fixtures import make_clean, make_defective
from gantry.resolve import requires_channels
from gantry.spine import Measurement

from gantry_feedback_core import Attribution, Funnel, Harden, Screen

A = Cohort("a", make_defective("never_completes", n=30, fraction=0.5, seed=1).episodes)
B = Cohort("b", make_defective("never_completes", n=30, fraction=0.5, seed=2).episodes)
CLEAN = Cohort("clean", make_clean(n=30, seed=4).episodes)


@pytest.mark.parametrize(
    "module",
    [Screen("comparative"), Screen("reference", reference="a"), Screen("absolute"),
     Funnel(), Attribution(), Harden()],
    ids=["screen-comparative", "screen-reference", "screen-absolute",
         "funnel", "attribution", "harden"],
)
def test_conforms_strictly(module):
    verdict = check_feedback(module, [A, B], strict=True)
    assert verdict.ok, verdict.explain()


def test_the_kit_is_discoverable():
    codes = dict(feedback_checks())
    assert codes["pure"] == "the same cohorts give the same report"
    assert "min_cohorts" in codes


def test_purity_survives_the_bootstrapping_modules_do():
    """Screen resamples to build intervals; a seeded resampler stays pure."""
    assert check_feedback(Screen("comparative"), [A, CLEAN], only=["pure"]).ok


# -- the kit must catch a module that misbehaves ---------------------------


class _Base(Funnel):
    def descriptor(self):
        return feedback_descriptor("broken", "1.0", min_cohorts=1, prescribes=True)


class _Drifts(_Base):
    _count = 0

    def analyse(self, cohorts):
        _Drifts._count += 1
        return Report("broken", (), {"drift": Measurement(float(_Drifts._count), n=1)},
                      cohorts=tuple(c.name for c in cohorts))


class _BareNumber(_Base):
    def analyse(self, cohorts):
        return Report("broken", (), {"n": 0.5}, cohorts=tuple(c.name for c in cohorts))


class _UnevidencedAdvice(_Base):
    def analyse(self, cohorts):
        return Report(
            "broken",
            (Finding("x", "collect more of everything", prescription="collect more"),),
            cohorts=tuple(c.name for c in cohorts),
        )


class _ForgetsItsCohorts(_Base):
    def analyse(self, cohorts):
        return Report("broken", (), {})


class _AnswersAnyway(Funnel):
    def descriptor(self):
        return feedback_descriptor("greedy", "1.0", min_cohorts=3, prescribes=False)

    def check_inputs(self, cohorts):
        from gantry.spine import Verdict

        return Verdict.yes()


@pytest.mark.parametrize(
    "module,code",
    [
        (_Drifts(), "conformance.not_pure"),
        (_BareNumber(), "conformance.bare_number"),
        (_UnevidencedAdvice(), "conformance.unevidenced_prescription"),
        (_ForgetsItsCohorts(), "conformance.report_cohorts"),
    ],
)
def test_the_kit_catches_a_broken_module(module, code):
    verdict = check_feedback(module, [A, B])
    assert not verdict.ok
    assert code in verdict.codes(), verdict.explain()


def test_a_module_that_answers_below_its_minimum_is_caught():
    verdict = check_feedback(_AnswersAnyway(), [A, B, CLEAN])
    assert "conformance.answered_anyway" in verdict.codes()


def test_the_kit_refuses_to_run_with_too_few_cohorts():
    assert "conformance.too_few_for_the_kit" in check_feedback(Harden(), [A]).codes()
