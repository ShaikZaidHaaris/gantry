"""The counts and rates a plane declares, held against the value that is not one.

Every check in this file guards the same defect: a bound written as a plain
comparison is blind to NaN, because every comparison against NaN is False. So
``if horizon < 1`` reads as a bound and admits the one value that makes every
later bound admit it too.

This matters more here than anywhere else in the tree. ``contracts`` is what the
conformance kits validate a plugin against, so a blind check here does not fail
one module — it certifies every module that inherits it. That is not
hypothetical: an SO-101 embodiment declaring ``max_relative_target=nan``, a
per-tick motion clamp that bounds nothing, passed ``check_embodiment(strict=True)``
because nothing in this layer looked.

The failures are quiet by construction, which is the argument for pinning them:

* a NaN ``horizon`` makes ``step < horizon`` False on the first test, so the
  evaluator records a complete, well-formed episode in which the policy was
  never once asked for an action;
* a NaN ``execute`` falls through to the fully closed-loop path, which is a
  different experiment published under the same protocol;
* a NaN ``control_hz`` makes two byte-identical channels refuse to bind, because
  ``nan != nan`` reads to the spine as a rate mismatch;
* a NaN ``chunk`` or ``min_cohorts`` is published as a capability and matched by
  exact equality, so it cannot satisfy a requirement asking for that same NaN.

Infinity is refused alongside NaN at every site. Unlike the arm's
``SafetyLimits``, where ``inf`` is the declared encoding for "no limit", nothing
in this layer gives it a meaning: an infinite horizon never terminates and an
infinite rate names no period.
"""

from __future__ import annotations

import math

import pytest

from gantry.contracts.embodiment import EmbodimentDescriptor
from gantry.contracts.evaluator import Protocol, Scene, TaskSpec
from gantry.contracts.feedback import feedback_descriptor
from gantry.contracts.policy import policy_descriptor

NAN = float("nan")
INF = float("inf")

#: The two values a plain comparison cannot refuse, and the two it always could.
#: Parametrising over both is what keeps a fix from trading one hole for another.
NOT_A_COUNT = [NAN, INF]
ALREADY_REFUSED = [0, -1]


def _task(horizon):
    return TaskSpec("t", scenes=(Scene(id="s1"),), horizon=horizon)


def _machine(control_hz):
    return EmbodimentDescriptor(name="x", version="1", state=(), action=(), control_hz=control_hz)


# -- the evaluator's two declarations ---------------------------------------


@pytest.mark.parametrize("horizon", NOT_A_COUNT + ALREADY_REFUSED)
def test_a_horizon_that_is_not_a_number_of_steps_is_refused(horizon):
    """The quietest of the set: it does not crash, it produces a finished run.

    ``while step < horizon`` is False on the first test for NaN, so the loop
    body never executes. Nothing raises. The record has episodes, a denominator,
    and a provenance block, and the policy was never asked for an action.
    """
    verdict = _task(horizon).validate()
    assert not verdict.ok
    assert "task.horizon" in verdict.codes(), verdict.explain()


@pytest.mark.parametrize("epochs", NOT_A_COUNT + ALREADY_REFUSED)
def test_an_epoch_count_that_is_not_a_count_is_refused(epochs):
    verdict = Protocol(epochs=epochs).validate()
    assert not verdict.ok
    assert "protocol.epochs" in verdict.codes(), verdict.explain()


@pytest.mark.parametrize("execute", NOT_A_COUNT + ALREADY_REFUSED)
def test_an_execute_count_that_is_not_a_count_is_refused(execute):
    """``execute`` is the cheapest lever on a result, per its own docstring.

    It decides how many actions of each predicted chunk are carried out before
    the policy is asked again. A NaN does not disable it loudly — it falls
    through to fully closed-loop, so a run that was supposed to execute 8 of
    every 16 actions executes 1 of 16 and reports the protocol it was given.
    """
    verdict = Protocol(execute=execute).validate()
    assert not verdict.ok
    assert "protocol.execute" in verdict.codes(), verdict.explain()


def test_the_legal_evaluator_declarations_still_pass():
    """The positive control. A validator that refused everything would pass above."""
    assert _task(10).validate().ok
    assert Protocol(epochs=1).validate().ok
    assert Protocol(execute=4).validate().ok
    # None is the declared "not stated" value for execute and must stay legal:
    # it means "no chunk truncation", not "an unchecked number".
    assert Protocol(execute=None).validate().ok
    assert Protocol().validate().ok


# -- the machine's rate -----------------------------------------------------


@pytest.mark.parametrize("control_hz", NOT_A_COUNT + [0.0, -1.0])
def test_a_control_rate_that_is_not_a_rate_is_refused(control_hz):
    """A NaN rate is worse than an absent one, because ``None`` already means absent.

    Two channels carrying byte-identical NaN rates come back as a
    ``rate.mismatch``, so a spec stops matching itself and the resolver hunts
    for a resampler to close a gap that does not exist.
    """
    verdict = _machine(control_hz).validate()
    assert "embodiment.control_hz" in verdict.codes(), verdict.explain()


def test_an_undeclared_rate_stays_legal_and_a_real_one_passes():
    """``None`` is the documented "not stated" branch and is not a defect."""
    assert "embodiment.control_hz" not in _machine(None).validate().codes()
    assert "embodiment.control_hz" not in _machine(30.0).validate().codes()


# -- the two published capability counts ------------------------------------


@pytest.mark.parametrize("chunk", NOT_A_COUNT + ALREADY_REFUSED)
def test_a_chunk_that_is_not_a_count_is_refused(chunk):
    with pytest.raises(ValueError, match="chunk must be at least 1"):
        policy_descriptor("p", "1", chunk=chunk, deterministic=True)


@pytest.mark.parametrize("min_cohorts", NOT_A_COUNT + ALREADY_REFUSED)
def test_a_cohort_minimum_that_is_not_a_count_is_refused(min_cohorts):
    """The one published count that had no check at all.

    Its sibling factories validated theirs; this one went straight into
    ``provides``. The value is read back through ``int()`` in ``min_cohorts()``
    and in the conformance kit, so a non-finite one raised there instead —
    naming neither the module nor the parameter that was wrong.
    """
    with pytest.raises(ValueError, match="min_cohorts must be at least 1"):
        feedback_descriptor("f", "1", min_cohorts=min_cohorts, prescribes=False)


def test_the_legal_capability_counts_still_build_and_still_match():
    """And the reason a NaN capability is incoherent, not merely unvalidated.

    ``provides_all`` matches by exact equality, so a NaN chunk could not satisfy
    a requirement asking for that same NaN — a declared capability that nothing,
    including itself, can match.
    """
    policy = policy_descriptor("p", "1", chunk=4, deterministic=True)
    assert policy.provides["chunk"] == 4
    assert policy.provides_all({"chunk": 4}).ok
    assert not policy.provides_all({"chunk": 8}).ok

    module = feedback_descriptor("f", "1", min_cohorts=1, prescribes=False)
    assert module.provides["min_cohorts"] == 1

    assert NAN != NAN, "the equality that makes a NaN capability unmatchable"


# -- the rule, stated once --------------------------------------------------


def test_no_declared_count_in_contracts_accepts_a_non_finite_value():
    """One statement of what the checks above are instances of.

    A new declaration added in the ``if x < 1`` idiom passes its own test and
    fails this one, which is the only kind that catches the next occurrence
    rather than the last.
    """
    survived = []
    for value in (NAN, INF):
        if _task(value).validate().ok:
            survived.append(f"TaskSpec.horizon={value}")
        if Protocol(epochs=value).validate().ok:
            survived.append(f"Protocol.epochs={value}")
        if Protocol(execute=value).validate().ok:
            survived.append(f"Protocol.execute={value}")
        if "embodiment.control_hz" not in _machine(value).validate().codes():
            survived.append(f"EmbodimentDescriptor.control_hz={value}")
        for label, build in (
            (
                "policy_descriptor.chunk",
                lambda v=value: policy_descriptor("p", "1", chunk=v, deterministic=True),
            ),
            (
                "feedback_descriptor.min_cohorts",
                lambda v=value: feedback_descriptor("f", "1", min_cohorts=v, prescribes=False),
            ),
        ):
            try:
                build()
            except ValueError:
                continue
            survived.append(f"{label}={value}")
    assert not survived, f"these accepted a non-finite declaration: {survived}"


def test_the_guard_idiom_is_why_the_plain_comparison_could_not_work():
    """Proof the fixes are necessary rather than decorative."""
    assert not (NAN < 1), "the plain bound admits NaN"
    assert not (NAN <= 0), "and admits it on the other spelling too"
    assert not math.isfinite(NAN) and not math.isfinite(INF)
    assert math.isfinite(0) and math.isfinite(-1), "the old bound still does its half"
