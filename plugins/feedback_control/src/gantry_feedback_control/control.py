"""Did the data carry information, or would any fine-tuning have done as well?

This is the module that decides whether the whole product means anything, and
the question it asks is not the obvious one.

The obvious experiment is baseline against treatment: a model without the
contributor's data, a model with it, and the difference between them. That
comparison is real and it is not enough, because fine-tuning a large pretrained
model on *anything* moves it. Show it five thousand frames of kitchen video with
the action labels attached to the wrong frames and the loss will still fall, the
behaviour will still change, and a baseline-versus-treatment table will still
show a difference. Reported as "their data helped", that is false in the most
expensive possible way: it is a result that reproduces for every contributor,
including the ones whose data is worthless.

So a third arm is needed
------------------------
``shuffled`` — the same frames, the same action distribution, the same number of
gradient steps, and no relationship between what the model saw and what it was
told to do. Whatever fine-tuning-in-general buys, this arm buys it too.

The comparison that matters is therefore **ego against shuffled**, not ego
against base. Ego beating base says the model changed. Ego beating *shuffled*
says the change came from the correspondence between the images and the actions —
which is the only thing a contributor's data can actually be selling.

What each outcome licenses you to say
-------------------------------------
Both arms beat base, ego beats shuffled     the data carried information
Both beat base, ego does not beat shuffled  fine-tuning helped; the data did not
Neither beats base                          nothing measurable happened
Shuffled beats ego                          something is wrong with the pipeline

That last row is worth keeping. A control that outperforms the real thing is not
a result about data quality — it is a signal that labels are misaligned, or that
the split leaked, and it should stop the report rather than appear in it.

On abstaining
-------------
Where the arms are not separated at the sample size available, this says so
rather than reporting the larger number. The point of a control is to be capable
of embarrassing you, and a control that can be rounded past is not one.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, proportion
from gantry.spine.inference import barnard, holm

VERSION = "0.1.0.dev0"

#: Cohort names this looks for. Any of them may be absent; what changes is which
#: questions can be answered, and the report says which.
TREATMENT, CONTROL, BASELINE = "ego", "shuffled", "base"

#: Below this, two arms are not separated and the module abstains rather than
#: reporting the larger number.
ALPHA = 0.05


def outcomes_of(cohort: Cohort) -> tuple[int, int]:
    """Successes and scored trials. Abstentions are dropped, never counted as losses."""
    scored = [o for o in cohort.outcomes if o is not None]
    return sum(1 for o in scored if o), len(scored)


def errors_of(cohort: Cohort, key: str = "action_error") -> np.ndarray:
    """Per-episode prediction error, where an offline evaluation recorded one.

    The fallback when there are no closed-loop outcomes. It measures fit rather
    than capability, and the report is required to say so — but a number that
    admits what it is beats no number.
    """
    out: list[float] = []
    for episode in cohort.episodes:
        labels = getattr(episode, "labels", None)
        value = (getattr(labels, "annotations", {}) or {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return np.asarray(out, dtype=float)


class Control(FeedbackModule):
    """Ego against its own control, which is the comparison that means something."""

    def __init__(self, *, alpha: float = ALPHA, error_key: str = "action_error"):
        self._alpha = float(alpha)
        self._key = error_key

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="control",
            version=VERSION,
            # Two arms minimum: the treatment and its control. A baseline is
            # welcome and is not what makes this answerable.
            min_cohorts=2,
            prescribes=True,
            # The arms differ only in their training data, so everything else
            # must be held. Declared so a run where the recipe also changed is
            # refused rather than reported as a data result.
            holds=("policy", "evaluation", "task", "embodiment"),
            arms=[TREATMENT, CONTROL, BASELINE],
            alpha=self._alpha,
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "control",
            "feedback",
            capabilities={"outcomes": True},
            description="a treatment arm and a shuffled-label control trained the same way",
        )

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        arms = {cohort.name: cohort for cohort in cohorts}
        treatment = _find(arms, TREATMENT)
        control = _find(arms, CONTROL)
        baseline = _find(arms, BASELINE)

        if treatment is None or control is None:
            missing = [
                name for name, arm in ((TREATMENT, treatment), (CONTROL, control)) if arm is None
            ]
            return Report(
                module="control",
                findings=(
                    Finding(
                        code="control.no_control",
                        summary=(
                            f"no {' and '.join(missing)} arm, so whether the data carried "
                            "information cannot be told apart from whether fine-tuning "
                            "helped"
                        ),
                        severity="strong",
                        prescription=(
                            "Train a control: the same frames, the same action "
                            "distribution, the same number of steps, and the actions "
                            "detached from the frames they belong to. Whatever "
                            "fine-tuning in general buys, that arm buys too — so the "
                            "difference between them is the only part attributable to "
                            "the data. Without it, a result reproduces for every "
                            "contributor including the ones whose data is worthless."
                        ),
                    ),
                ),
                cohorts=tuple(arms),
            )

        outcomes = {name: outcomes_of(arm) for name, arm in arms.items()}
        if all(total == 0 for _, total in outcomes.values()):
            return self._offline(arms, treatment, control, baseline)
        return self._closed_loop(arms, treatment, control, baseline, outcomes)

    # -- the real comparison ----------------------------------------------

    def _closed_loop(self, arms, treatment, control, baseline, outcomes) -> Report:
        measurements: dict[str, Measurement] = {}
        for name, (wins, total) in outcomes.items():
            if total:
                measurements[f"{name}.success_rate"] = proportion(wins, total)

        tw, tn = outcomes[treatment.name]
        cw, cn = outcomes[control.name]
        tests: dict[str, float] = {}
        if tn and cn:
            tests["ego_vs_shuffled"] = barnard(tw, tn - tw, cw, cn - cw)
        if baseline is not None:
            bw, bn = outcomes[baseline.name]
            if bn:
                if tn:
                    tests["ego_vs_base"] = barnard(tw, tn - tw, bw, bn - bw)
                if cn:
                    tests["shuffled_vs_base"] = barnard(cw, cn - cw, bw, bn - bw)

        # holm takes a sequence and returns a parallel tuple of booleans; keys
        # are re-attached here so the findings can name which comparison.
        names = list(tests)
        flags = holm([tests[n] for n in names], alpha=self._alpha) if names else ()
        corrected = dict(zip(names, flags))
        for name, p in tests.items():
            measurements[name] = Measurement(
                value=round(float(p), 6),
                n=tn + cn,
                method="Barnard exact test, Holm-corrected across the comparisons made",
                detail={"separated": bool(corrected.get(name, False))},
            )

        rate = lambda arm: (
            (outcomes[arm.name][0] / outcomes[arm.name][1]) if outcomes[arm.name][1] else 0.0
        )
        ego, shuffled = rate(treatment), rate(control)
        beats_control = bool(corrected.get("ego_vs_shuffled", False)) and ego > shuffled
        control_wins = bool(corrected.get("ego_vs_shuffled", False)) and shuffled > ego

        findings: list[Finding] = []
        if control_wins:
            # Not a data-quality result. Something is wrong.
            findings.append(
                Finding(
                    code="control.control_wins",
                    severity="strong",
                    summary=(
                        f"the shuffled control ({shuffled:.0%}) beat the real data "
                        f"({ego:.0%}), which is not a finding about the data"
                    ),
                    measurements={k: v for k, v in measurements.items() if k in tests},
                    prescription=(
                        "Stop and check the pipeline before reporting anything. A "
                        "control outperforming the real thing means labels are "
                        "misaligned with frames, or the split leaked, or the two arms "
                        "did not actually differ in the way intended. None of those "
                        "are results."
                    ),
                    cohorts=tuple(arms),
                )
            )
        elif beats_control:
            findings.append(
                Finding(
                    code="control.data_carried_information",
                    severity="strong",
                    summary=(
                        f"the data beat its own shuffled control ({ego:.0%} against "
                        f"{shuffled:.0%}), so the gain came from the correspondence "
                        "between the images and the actions rather than from "
                        "fine-tuning alone"
                    ),
                    measurements={k: v for k, v in measurements.items()},
                    prescription=(
                        "This is the claim worth making, and it is narrower than it "
                        "sounds: it says the data carried signal, on these tasks, at "
                        "this sample size. It does not say the data is better than an "
                        "equivalent amount of robot data, which is a different "
                        "experiment needing a different arm."
                    ),
                    cohorts=tuple(arms),
                )
            )
        else:
            findings.append(
                Finding(
                    code="control.not_separated",
                    severity="weak",
                    summary=(
                        f"the data ({ego:.0%}) and its shuffled control ({shuffled:.0%}) "
                        "are not separated at this number of trials, so the data has not "
                        "been shown to carry information"
                    ),
                    measurements={k: v for k, v in measurements.items()},
                    prescription=(
                        "Either the data carries little, or there are too few trials to "
                        "tell. The power module says which — and reporting the larger "
                        "number here would be reporting noise with a decimal point on it."
                    ),
                    cohorts=tuple(arms),
                )
            )

        if baseline is not None and outcomes[baseline.name][1]:
            findings.append(
                self._against_base(arms, outcomes, corrected, baseline, treatment, control)
            )

        return Report(
            module="control",
            findings=tuple(findings),
            measurements=measurements,
            notes=(f"arms: {', '.join(f'{n} n={outcomes[n][1]}' for n in arms)}",),
            cohorts=tuple(arms),
        )

    def _against_base(self, arms, outcomes, corrected, baseline, treatment, control) -> Finding:
        """What fine-tuning bought, separately from what the data bought."""
        bw, bn = outcomes[baseline.name]
        base = bw / bn if bn else 0.0
        ego = outcomes[treatment.name][0] / max(1, outcomes[treatment.name][1])
        shuffled = outcomes[control.name][0] / max(1, outcomes[control.name][1])
        both_moved = corrected.get("ego_vs_base", False) and corrected.get(
            "shuffled_vs_base", False
        )
        return Finding(
            code="control.fine_tuning_effect",
            severity="info",
            summary=(
                f"base {base:.0%}, shuffled {shuffled:.0%}, ego {ego:.0%}"
                + (
                    " — both arms moved off base, which is what fine-tuning does to a "
                    "pretrained model regardless of what it is shown"
                    if both_moved
                    else ""
                )
            ),
            evidence={
                "base": round(base, 4),
                "shuffled": round(shuffled, 4),
                "ego": round(ego, 4),
                "both_moved_off_base": bool(both_moved),
            },
            prescription=(
                "Report all three. A two-arm table showing base against ego is the "
                "table that would have been reported as a data result no matter what "
                "the data contained."
            ),
            cohorts=tuple(arms),
        )

    # -- the fallback, honest about being one -----------------------------

    def _offline(self, arms, treatment, control, baseline) -> Report:
        """Prediction error, when there are no closed-loop outcomes."""
        errors = {name: errors_of(arm, self._key) for name, arm in arms.items()}
        if not any(len(v) for v in errors.values()):
            return Report(
                module="control",
                findings=(
                    Finding(
                        code="control.nothing_measured",
                        summary=(
                            "no arm carries outcomes or prediction error, so there is "
                            "nothing to compare"
                        ),
                        severity="strong",
                        prescription=(
                            "Run the arms through an evaluator. Until then the training "
                            "loss is the only number available, and a falling loss says "
                            "the model fitted the data — which is true of the shuffled "
                            "control as well."
                        ),
                    ),
                ),
                cohorts=tuple(arms),
            )

        measurements = {
            f"{name}.action_error": Measurement(
                value=round(float(values.mean()), 6),
                n=len(values),
                ci=_interval(values),
                method="mean per-episode action prediction error, offline replay",
            )
            for name, values in errors.items()
            if len(values)
        }
        ego, shuffled = errors[treatment.name], errors[control.name]
        separated = _separated(ego, shuffled)
        better = len(ego) and len(shuffled) and ego.mean() < shuffled.mean()

        return Report(
            module="control",
            findings=(
                Finding(
                    code=(
                        "control.data_carried_information"
                        if (separated and better)
                        else "control.not_separated"
                    ),
                    severity="weak",
                    summary=(
                        f"offline: prediction error {ego.mean():.4f} against the shuffled "
                        f"control's {shuffled.mean():.4f}"
                        + ("; separated" if separated else "; not separated")
                    ),
                    measurements=measurements,
                    evidence={"offline": True},
                    prescription=(
                        "This measures fit, not capability. A model that predicts "
                        "held-out actions better has learned the distribution; it has "
                        "not been shown to do the task. Report it as a rung on the "
                        "ladder rather than as the answer, and run a closed-loop "
                        "evaluation before making any claim about behaviour."
                    ),
                    cohorts=tuple(arms),
                ),
            ),
            measurements=measurements,
            notes=("no closed-loop outcomes; fell back to offline prediction error",),
            cohorts=tuple(arms),
        )


def _find(arms: Mapping[str, Cohort], want: str) -> Cohort | None:
    """An arm by name, tolerating the prefixes a real run puts on cohort names."""
    for name, arm in arms.items():
        if name == want or name.lower().replace("-", "_").startswith(want):
            return arm
    return None


def _interval(values: np.ndarray) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    half = 1.96 * float(values.std(ddof=1)) / (len(values) ** 0.5)
    return (round(float(values.mean() - half), 6), round(float(values.mean() + half), 6))


def _separated(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two error samples' intervals fail to overlap."""
    if len(a) < 2 or len(b) < 2:
        return False
    lo_a, hi_a = _interval(a) or (0.0, 0.0)
    lo_b, hi_b = _interval(b) or (0.0, 0.0)
    return hi_a < lo_b or hi_b < lo_a
