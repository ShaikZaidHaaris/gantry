"""The simulator's own answer, demoted to one opinion among several.

Nothing here is new behaviour — this is the predicate that has been deciding
every success rate in this project all along. What changes is its status. It
used to be *the* definition of success; it becomes a scorer with a name, a
declared evidence requirement, and a version, which makes three things possible
that were not:

Its agreement with a person can be measured, because agreement needs two named
judges and previously there was one unnamed one.

A number it produced can be told apart from a number a person produced, because
the run records which scorer decided.

It can be wrong. A predicate checks a pose; the rubric beside that predicate
says the cube must be "held for at least one second", and a policy that lifts
and immediately drops satisfies one and not the other. That gap was invisible
while the predicate was definitional. Now it is a measurable disagreement.

Why it declares ``final_state`` rather than video
------------------------------------------------
It cannot use a video and should not be handed one. A scorer that accepted
evidence it could not read would be able to fall back on something the bench
cannot supply, and the whole point of writing criteria twice is that the sim
lane and the hardware lane are held to the same sentence.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from gantry.contracts.scorer import (
    FINAL_STATE,
    Evidence,
    Judgement,
    Scorer,
    scorer_descriptor,
)
from gantry.spine import Descriptor

VERSION = "0.1.0.dev0"

#: Where a world puts its own verdict in the final state it hands over. Read
#: rather than recomputed: the environment already knows whether its task is
#: solved, and a second implementation of that check would be a second thing to
#: keep in step.
SUCCESS_KEY = "success"


class MachinePredicate(Scorer):
    """Reads the world's verdict out of the final state it reported.

    The default scorer for any closed-loop evaluation, and free: the answer was
    computed during the rollout and this just names who computed it.
    """

    def __init__(self, *, key: str = SUCCESS_KEY, name: str = "machine"):
        self._key = key
        self._name = name

    def descriptor(self) -> Descriptor:
        return scorer_descriptor(
            name=self._name,
            version=VERSION,
            evidence=(FINAL_STATE,),
            # A predicate over a fixed state is the same answer every time, and
            # unlike a person or a model it can honestly say so.
            deterministic=True,
            cost="free",
            # It reads a boolean or it does not find one. There is no evidence
            # from which it could be uncertain.
            abstains=False,
            reads=self._key,
        )

    def judge(self, evidence: Evidence, task: Any) -> tuple[Judgement, ...]:
        state = evidence.final_state or {}
        verdict = state.get(self._key)
        if isinstance(verdict, Mapping):
            # Some worlds report sub-goals alongside the overall answer. Only
            # the overall one is read; a sub-goal is a milestone and belongs to
            # the funnel, not to a success label.
            verdict = verdict.get("task")

        criteria = getattr(task, "success", ()) or ()
        if not criteria:
            return ()
        # One judgement per declared criterion. A predicate cannot tell the
        # criteria apart — the world reports one boolean for the whole task —
        # so it says the same thing about each and says that it did.
        return tuple(
            Judgement(
                criterion=criterion.check,
                passed=None if verdict is None else bool(verdict),
                rationale=(
                    f"{self._name}: the world reported {self._key}={verdict!r} for the "
                    "task as a whole; this predicate does not distinguish between "
                    "criteria"
                    if len(criteria) > 1
                    else f"{self._name}: the world reported {self._key}={verdict!r}"
                ),
                used=(FINAL_STATE,),
            )
            for criterion in criteria
        )


class ThresholdPredicate(Scorer):
    """Judges each criterion separately, from measured quantities.

    Where :class:`MachinePredicate` reads one boolean for the whole task, this
    evaluates a criterion at a time against numbers in the final state — so
    "lifted 4 cm" and "held for a second" become separable, which is what makes
    a disagreement with a rubric locatable rather than merely present.

    ``checks`` maps a criterion's ``check`` name onto a callable taking the
    criterion's arguments and the final state. Supplied rather than inferred:
    which measured quantity corresponds to which criterion is knowledge about a
    particular world, and a scorer that guessed would be inventing the thing it
    is supposed to be testing.
    """

    def __init__(
        self,
        checks: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool | None]],
        *,
        name: str = "threshold",
    ):
        self._checks = dict(checks)
        self._name = name

    def descriptor(self) -> Descriptor:
        return scorer_descriptor(
            name=self._name,
            version=VERSION,
            evidence=(FINAL_STATE,),
            deterministic=True,
            cost="free",
            # A criterion it has no check for is an abstention, not a failure.
            abstains=True,
            checks=sorted(self._checks),
        )

    def judge(self, evidence: Evidence, task: Any) -> tuple[Judgement, ...]:
        state = evidence.final_state or {}
        out = []
        for criterion in getattr(task, "success", ()) or ():
            check = self._checks.get(criterion.check)
            if check is None:
                out.append(
                    Judgement(
                        criterion=criterion.check,
                        passed=None,
                        rationale=(
                            f"{self._name}: no check is registered for "
                            f"{criterion.check!r}, so this scorer has no opinion "
                            "rather than a negative one"
                        ),
                        used=(FINAL_STATE,),
                    )
                )
                continue
            try:
                verdict = check(dict(criterion.args), dict(state))
            except Exception as error:  # noqa: BLE001 - an unreadable state abstains
                out.append(
                    Judgement(
                        criterion=criterion.check,
                        passed=None,
                        rationale=f"{self._name}: {type(error).__name__}: {error}",
                        used=(FINAL_STATE,),
                    )
                )
                continue
            out.append(
                Judgement(
                    criterion=criterion.check,
                    passed=None if verdict is None else bool(verdict),
                    rationale=f"{self._name}: {criterion.check}({dict(criterion.args)}) "
                    f"-> {verdict!r}",
                    used=(FINAL_STATE,),
                )
            )
        return tuple(out)


def lifted(args: Mapping[str, Any], state: Mapping[str, Any]) -> bool | None:
    """Whether the named object cleared the surface by the stated height.

    A worked example of a check, and of the gap a rubric can open: this tests
    the height at the end of the episode. The rubric beside it says the object
    must be *held*, which a policy that lifts and drops does not satisfy. The
    two disagree, deliberately, and that disagreement is what a calibration
    pass is for.
    """
    height = state.get("object_height_above_surface")
    if height is None:
        return None
    return float(height) >= float(args.get("height", 0.04))


def all_of(*names: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool | None]:
    """A check that every named flag in the final state is true."""

    def check(args: Mapping[str, Any], state: Mapping[str, Any]) -> bool | None:
        values = [state.get(name) for name in names]
        if any(value is None for value in values):
            return None
        return all(bool(value) for value in values)

    return check


def reached(stage: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], bool | None]:
    """A check that a named milestone appears in the reported stages."""

    def check(args: Mapping[str, Any], state: Mapping[str, Any]) -> bool | None:
        stages: Sequence[str] | None = state.get("stages")
        return None if stages is None else stage in stages

    return check
