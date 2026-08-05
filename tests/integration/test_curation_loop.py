"""The loop, end to end: propose, refuse to test it badly, judge, remember.

Core-only. No plugin is imported here -- the curator is defined in the test,
which is the point: the loop is a property of the contracts, not of any signal
that happens to ship with them.

What this demonstrates, in order, is the thing the design exists for. A signal
proposes a change and says what it expects. The change is refused if testing it
could not have meant anything -- too few trials, or scored on the scenes that
produced it. Otherwise it is retrained, paired, and judged by machinery that
never heard of the signal. And the verdict is kept whether or not it flattered
anybody.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from gantry.contracts.curation import (
    CurationAction,
    CurationOutcome,
    CurationPlan,
    Curator,
    Prediction,
    curator_descriptor,
)
from gantry.ledger import Ledger
from gantry.spine import Descriptor, Measurement, Verdict


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)
    stage_events: tuple = ()


@dataclass
class Meta:
    uid: str
    id: str = ""


@dataclass
class Ep:
    meta: Meta
    labels: Labels


class DropFailures(Curator):
    """A whole curation signal, in ten lines, defined outside core."""

    def descriptor(self) -> Descriptor:
        return curator_descriptor(
            name="drop_failures", version="1.0", rung="screening", per_episode=True
        )

    def propose(self, episodes: Sequence[Any], runs: Sequence[Any] = ()) -> CurationPlan:
        bad = tuple(e.meta.uid for e in episodes if e.labels.success is False)
        return CurationPlan(
            actions=(CurationAction("drop", episodes=bad),),
            signal="drop_failures",
            rung="screening",
            predicted=Prediction(magnitude=0.10, tasks=("lift_cube",)),
        )


def dataset(good: int, bad: int):
    return [Ep(Meta(f"d/{i}"), Labels(True)) for i in range(good)] + [
        Ep(Meta(f"d/{good + i}"), Labels(False)) for i in range(bad)
    ]


def outcomes(results):
    return tuple(Ep(Meta(f"r/{i}", f"seed_{i}"), Labels(ok)) for i, ok in enumerate(results))


def test_the_whole_loop(tmp_path):
    from gantry_feedback_verify import CurationVerifier, preflight

    from gantry.contracts.feedback import Cohort

    ledger = Ledger(tmp_path / "ledger")

    # 1. A signal reads the data and proposes a change, with a prediction.
    plan = DropFailures().plan(dataset(good=16, bad=84))
    assert len(plan.drops) == 84
    assert plan.proposes_a_change
    assert "success_rate +0.1 on lift_cube" in plan.summary()

    # 2. Testing it on twenty trials would not have meant anything, and that is
    #    said before the retrain rather than discovered after it.
    thin = preflight(plan, list(range(20)), baseline_rate=0.35)
    assert not thin.ok and "curation.underpowered" in thin.codes()

    # 3. With a real budget it is allowed to proceed.
    seeds = list(range(200))
    assert preflight(plan, seeds, baseline_rate=0.35).ok

    # 4. Two retrains later, the paired result is judged by a module that has
    #    never heard of the signal.
    baseline = outcomes([True] * 7 + [False] * 13)
    curated = outcomes([True] * 17 + [False] * 3)
    verifier = CurationVerifier(plan, plans_already_tested=ledger.tested("drop_failures"))
    report = verifier.analyse([Cohort("baseline", baseline), Cohort("curated", curated)])
    assert report.findings[0].code == "curation.verified"

    # 5. And the verdict is kept.
    outcome = verifier.outcome(
        [Cohort("baseline", baseline), Cohort("curated", curated)],
        baseline_run="runs/base",
        curated_run="runs/cur",
        cost={"gpu_minutes": 44},
    )
    ledger.record(outcome)
    assert len(ledger) == 1
    assert ledger.prior_for("drop_failures", "lift_cube") == 1.0


def test_a_refuted_plan_is_kept_too(tmp_path):
    """A ledger of successes is a brochure; the refutations are the information."""
    from gantry_feedback_verify import CurationVerifier

    from gantry.contracts.feedback import Cohort

    ledger = Ledger(tmp_path / "ledger")
    plan = DropFailures().plan(dataset(good=16, bad=84))
    verifier = CurationVerifier(plan)
    cohorts = [
        Cohort("baseline", outcomes([True] * 7 + [False] * 13)),
        Cohort("curated", outcomes([True] * 8 + [False] * 12)),
    ]
    ledger.record(verifier.outcome(cohorts, baseline_run="b", curated_run="c"))
    assert len(ledger) == 1
    assert ledger.prior_for("drop_failures") == 0.0
    assert "0/1 held" in ledger.report()


def test_the_ledger_tightens_the_threshold_as_a_signal_keeps_trying(tmp_path):
    """The tenth attempt is not making the same claim as the first."""
    from gantry_feedback_verify import CurationVerifier

    from gantry.contracts.feedback import Cohort

    ledger = Ledger(tmp_path / "ledger")
    plan = DropFailures().plan(dataset(good=16, bad=84))
    # Six wins, no losses: p = 0.031. Real once; not the best of forty tries.
    cohorts = [
        Cohort("baseline", outcomes([True] * 4 + [False] * 16)),
        Cohort("curated", outcomes([True] * 10 + [False] * 10)),
    ]
    first = CurationVerifier(plan, plans_already_tested=0).analyse(cohorts)
    assert first.findings[0].code == "curation.verified"

    for i in range(40):
        ledger.record(
            CurationOutcome(
                plan=plan,
                baseline_run=f"b{i}",
                curated_run=f"c{i}",
                delta=Measurement(value=0.0, n=20, method="paired"),
                p=0.9,
                verdict=Verdict.note("curation.refuted", "no"),
            )
        )
    later = CurationVerifier(plan, plans_already_tested=ledger.tested("drop_failures")).analyse(
        cohorts
    )
    assert ledger.tested("drop_failures") == 40
    assert later.findings[0].code == "curation.refuted"


def test_the_prior_stays_silent_until_there_is_evidence(tmp_path):
    """Inventing a default here would be the framework asserting what it does
    not know, which is the habit the ledger exists to break."""
    assert Ledger(tmp_path / "nothing").prior_for("anything") is None


def test_a_curator_needs_no_core_change_to_exist():
    """The plane is registered, so a signal defined in a test file is a citizen."""
    from gantry.conformance import check_curator
    from gantry.spine import known_planes

    assert "curation" in known_planes()
    verdict = check_curator(DropFailures(), dataset(good=5, bad=5))
    assert verdict.ok, verdict.explain()


def test_a_signal_that_reads_rollouts_and_hides_it_is_refused():
    """The guard no published method has: a plan tested where it was fitted."""

    class Sneaky(Curator):
        def descriptor(self) -> Descriptor:
            return curator_descriptor(
                name="sneaky",
                version="1.0",
                rung="influence",
                per_episode=True,
                needs_runs=True,
            )

        def propose(self, episodes, runs=()):
            return CurationPlan(
                actions=(CurationAction("drop", episodes=("d/0",)),),
                signal="sneaky",
                rung="influence",
                predicted=Prediction(magnitude=0.1),
                # names no evidence_seeds, despite having read rollouts
            )

    plan = Sneaky().propose(dataset(1, 1), runs=[object()])
    verdict = plan.validate()
    assert not verdict.ok
    assert "plan.unnamed_evidence" in verdict.codes()
