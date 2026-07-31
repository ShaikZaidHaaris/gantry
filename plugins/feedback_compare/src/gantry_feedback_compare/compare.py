"""Which policy was better, and did the data make it so.

Every other comparative module in this project holds the policy fixed and asks
what the *data* was like. This one does the opposite: it holds the world fixed
and asks which *policy* won. That is the question a training run is for, and
until now nothing could read it — a run where three checkpoints faced the same
scenes was correctly refused by every module, because they all declare the
policy as something they hold constant.

Paired, when it can be
----------------------
Two policies run on the same scenes agree about most of them. The easy scenes
both solve and the hard ones both fail, and neither tells you anything about
which is better. Only the disagreements do. So where the arms share scene
identifiers this compares them pair by pair with McNemar's exact test, which
sees a real difference on far fewer trials than comparing two marginal rates
would. Where they share no scenes it says so and ranks them unpaired.

What it refuses
---------------
A comparison where the *world* also changed. Two policies measured in different
simulators, at different protocols, or against different scene sets are not
being compared — they are being described separately in one table. That reads
identically to a real result, which is why it is refused rather than noted.

What it will not conclude
-------------------------
That the training data caused the difference. This module sees policies and
outcomes; it does not see what anything was trained on. Where the run records
each policy's training set it is reported alongside the ranking as context, and
the prescription says "this checkpoint won", never "this data is better" — one
run of three checkpoints cannot separate the data from the seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gantry.contracts.feedback import Cohort, FeedbackModule, Finding, Report, feedback_descriptor
from gantry.errors import ConfigError
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Verdict, mcnemar, proportion

VERSION = "0.1.0.dev0"

#: The world must be the same, or the policies are not being compared.
#: The policy plane is deliberately absent: it is the thing that varies.
HELD = ("evaluation", "embodiment")

#: A difference smaller than this is reported without a prescription. Ten points
#: is not a law; it is the point below which one evaluation of this size should
#: not be sending anybody back to retrain.
MATERIAL = 0.10


@dataclass(frozen=True)
class Arm:
    """One policy's run, with what it scored and on which scenes."""

    name: str
    policy: str | None
    trained_on: str | None
    by_scene: Mapping[str, bool]

    @property
    def n(self) -> int:
        return len(self.by_scene)

    @property
    def wins(self) -> int:
        return sum(1 for won in self.by_scene.values() if won)

    @property
    def rate(self) -> float:
        return self.wins / self.n if self.n else 0.0


def scene_of(episode: Any) -> str:
    """The scene attempted, so two arms can be paired on it."""
    annotations = episode.labels.annotations
    for key in ("scene", "scene_id", "initial_state"):
        if key in annotations:
            return str(annotations[key])
    return episode.meta.id.split("#", 1)[0]


def arm_of(cohort: Cohort) -> Arm | None:
    scored = {
        scene_of(e): bool(e.labels.success) for e in cohort.episodes if e.labels.success is not None
    }
    if not scored:
        return None
    policy = cohort.provenance.component("policy") if cohort.provenance else None
    protocol = dict(cohort.provenance.protocol) if cohort.provenance else {}
    return Arm(
        cohort.name,
        policy.ref if policy else None,
        # Recorded where a run bothered to say. Context for the reader, never
        # an input to the ranking.
        str(protocol.get("trained_on")) if protocol.get("trained_on") else None,
        scored,
    )


def paired_counts(left: Arm, right: Arm) -> tuple[int, int, int, int] | None:
    shared = set(left.by_scene) & set(right.by_scene)
    if not shared:
        return None
    both = sum(left.by_scene[s] and right.by_scene[s] for s in shared)
    only_left = sum(left.by_scene[s] and not right.by_scene[s] for s in shared)
    only_right = sum(right.by_scene[s] and not left.by_scene[s] for s in shared)
    return both, only_left, only_right, len(shared) - both - only_left - only_right


@dataclass(frozen=True)
class Floor:
    """What a comparison has to clear before the comparison means anything.

    Added after a real failure in this project, which is the only reason it is
    worth the code. Two checkpoints were compared on held-out data: one trained
    on retargeted ego actions, one on the same frames with the actions
    deliberately scrambled. The ego one won by 8% on both held-out scenes, the
    difference was consistent, and it was reported as a finding.

    Then somebody asked what a model that had learned nothing would score.
    Copying the current state forward scored better than *both*. The winner had
    lost to a one-line heuristic, and "8% better than scrambled" and "worse than
    doing nothing" were the same result told from opposite ends.

    A control tells you the comparison is fair. A floor tells you it is worth
    having. They are different questions and the first does not imply the second:
    a scrambled-input control is a very low bar, and clearing it proves only that
    the winner is not actively harmful.

    ``scores`` maps a trivial predictor's name to what it achieved on the same
    data, in the same direction as the metric being compared.
    """

    scores: Mapping[str, float]
    #: True when a *larger* number is better, as for a success rate. False for an
    #: error, where the floor is beaten by scoring lower.
    higher_is_better: bool = True
    #: How the trivial predictors were computed, for the record. A floor whose
    #: provenance nobody can check is as bad as no floor.
    method: str = ""

    def __post_init__(self) -> None:
        if not self.scores:
            raise ConfigError(
                "a floor with no trivial predictors in it is not a floor. The point "
                "is to name something that required no learning and see whether the "
                "winner beat it"
            )

    @property
    def best(self) -> tuple[str, float]:
        """The hardest trivial predictor to beat."""
        pick = max if self.higher_is_better else min
        name = pick(self.scores, key=lambda k: self.scores[k])
        return name, float(self.scores[name])

    def cleared_by(self, score: float) -> bool:
        _, value = self.best
        return score > value if self.higher_is_better else score < value

    def as_dict(self) -> dict[str, Any]:
        name, value = self.best
        return {
            "floor_scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "hardest_trivial_predictor": name,
            "floor": round(value, 4),
            "higher_is_better": self.higher_is_better,
            "floor_method": self.method or "not stated",
        }


class PolicyComparison(FeedbackModule):
    """Rank policies measured in one world on one set of scenes."""

    def __init__(self, material: float = MATERIAL, floor: Floor | None = None):
        """``floor`` is optional and its absence is reported, not assumed away.

        A comparison run without one is still a comparison; it simply cannot say
        whether the thing it ranked was worth ranking. That gets said out loud in
        the notes rather than left for a reader to wonder about.
        """
        self.material = material
        self.floor = floor

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            "compare",
            VERSION,
            min_cohorts=2,
            prescribes=True,
            # The world, not the policy. Every other comparative module here
            # holds the policy; this one is the reason the capability is a list
            # rather than a flag.
            holds=HELD,
            material=self.material,
            floor=self.floor.as_dict() if self.floor else None,
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "compare",
            "feedback",
            capabilities={"outcomes": True},
            description="which policy solved more of the same scenes",
        )

    def check_inputs(self, cohorts: Sequence[Cohort]) -> Verdict:
        checks = [super().check_inputs(cohorts)]
        arms = [arm for arm in (arm_of(c) for c in cohorts) if arm is not None]
        if len(arms) < 2:
            checks.append(
                Verdict.no(
                    "compare.no_outcomes",
                    "at least two cohorts must carry outcomes; a run with no results "
                    "cannot be ranked",
                    hint="an open-loop evaluator reports error, not success — this needs a world",
                )
            )
            return Verdict.all(checks)
        policies = {arm.policy for arm in arms}
        if len(policies) == 1 and None not in policies:
            checks.append(
                Verdict.note(
                    "compare.same_policy",
                    f"every cohort names the same policy ({policies.pop()}), so this "
                    "ranks one policy against itself",
                    hint="differences here are run-to-run variation, not a comparison",
                )
            )
        return Verdict.all(checks)

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        arms = [arm for arm in (arm_of(c) for c in cohorts) if arm is not None]
        ranked = sorted(arms, key=lambda arm: arm.rate, reverse=True)
        measurements = {f"{arm.name}.success_rate": proportion(arm.wins, arm.n) for arm in arms}
        notes = ["ranked by success rate; every arm faced the same world"]

        if all(arm.wins == 0 for arm in arms):
            notes.append(
                "no arm solved a single scene, so this ranks nothing: the comparison "
                "has no resolving power at this level of performance, which is not the "
                "same as the arms being equal"
            )
            return Report("compare", (), measurements, tuple(notes), tuple(c.name for c in cohorts))

        best, worst = ranked[0], ranked[-1]

        # Before anything is said about who won, ask whether winning meant
        # anything. A ranking among things that all lost to a trivial predictor
        # is arithmetic, not a result.
        if self.floor is not None and not self.floor.cleared_by(best.rate):
            name, value = self.floor.best
            return Report(
                "compare",
                (
                    Finding(
                        code="compare.below_floor",
                        summary=(
                            f"the best arm ({best.name}, {best.rate:.4f}) does not beat "
                            f"{name} ({value:.4f}), which required no learning at all — "
                            "so the ranking between these arms is a comparison of things "
                            "that all lost to a one-line heuristic"
                        ),
                        severity="strong",
                        evidence={
                            "best": best.name,
                            "best_score": round(best.rate, 4),
                            "ranking_withheld": [arm.name for arm in ranked],
                            **self.floor.as_dict(),
                        },
                        prescription=(
                            "Do not report the difference between these arms as a "
                            "result. The honest statement is that none of them cleared "
                            f"{name}. A gap between two arms below the floor is still a "
                            "real gap and still means nothing about whether either is "
                            "useful — the two readings are the same number told from "
                            "opposite ends, and only one of them is the headline."
                        ),
                        cohorts=tuple(arm.name for arm in ranked),
                    ),
                ),
                measurements,
                tuple(notes)
                + (
                    "the ranking is withheld rather than reported alongside the "
                    "refusal, because a number printed next to a warning is a number "
                    "somebody will quote without the warning",
                ),
                tuple(c.name for c in cohorts),
            )

        findings = [self._verdict(best, worst)]
        if self.floor is None:
            notes.append(
                "no trivial-predictor floor was supplied, so this says which arm won "
                "and cannot say whether winning was worth anything. A control shows a "
                "comparison is fair; a floor shows it is worth having"
            )
        else:
            name, value = self.floor.best
            notes.append(
                f"the best arm clears the hardest trivial predictor ({name} at "
                f"{value:.4f}), so the ranking is between things that beat doing nothing"
            )
        if best.trained_on:
            notes.append(
                f"the winning checkpoint records trained_on={best.trained_on!r}; that is "
                "context, and one run of these checkpoints cannot separate the training "
                "data from the training seed"
            )
        return Report(
            "compare", tuple(findings), measurements, tuple(notes), tuple(c.name for c in cohorts)
        )

    def _verdict(self, best: Arm, worst: Arm) -> Finding:
        gain = best.rate - worst.rate
        counts = paired_counts(best, worst)
        evidence: dict[str, Any] = {
            "best": best.name,
            "worst": worst.name,
            "best_policy": best.policy,
            "worst_policy": worst.policy,
            "best_trained_on": best.trained_on,
            "worst_trained_on": worst.trained_on,
            "gain": gain,
        }
        if counts:
            both, only_best, only_worst, neither = counts
            p = mcnemar(only_best, only_worst)
            evidence.update(
                paired_scenes=both + only_best + only_worst + neither,
                won=only_best,
                lost=only_worst,
                agreed=both + neither,
                p=p,
            )
            how = (
                f"paired on {evidence['paired_scenes']} shared scenes: won {only_best}, "
                f"lost {only_worst}, agreed on {both + neither}, p={p:.3g}"
            )
            convincing = p < 0.05
        else:
            evidence["paired_scenes"] = 0
            how = "compared unpaired: the arms share no scene"
            convincing = False

        return Finding(
            code="compare.best_policy",
            summary=(
                f"{best.name} solved {best.rate:.1%} and {worst.name} {worst.rate:.1%}, "
                f"{gain:+.1%}; {how}"
            ),
            severity="strong" if convincing and gain >= self.material else "weak",
            measurements={
                best.name: proportion(best.wins, best.n),
                worst.name: proportion(worst.wins, worst.n),
            },
            evidence=evidence,
            prescription=(
                f"Use {best.name}. It solved {gain:+.1%} more of the same scenes in the same world."
                if convincing and gain >= self.material
                else None
            ),
            cohorts=(best.name, worst.name),
        )
