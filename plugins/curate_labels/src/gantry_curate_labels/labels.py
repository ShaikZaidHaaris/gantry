"""The cheapest curation there is: stop training on the demonstrations that failed.

This is the bottom rung of the ladder and it is not a toy. Pointed at a public
multi-stage collection, the screening pass in this project found 244 of 1500
demonstrations succeed — sixteen percent. Everything else in that dataset is a
person trying and not managing it, and a policy fitted to it is being taught to
try and not manage it.

Nothing here is clever. That is the argument for it: it costs one pass over the
labels, needs no model, no rollouts and no GPU, and it is the first thing to
rule out before paying for a signal that needs any of those. A curation
pipeline that reaches for influence functions before checking whether the data
is labelled failed has skipped the free win.

What it will not do
-------------------
It refuses to guess. An episode with no success label is not evidence of
failure, and treating it as such would silently delete unlabelled data — so
unlabelled episodes are reported separately and dropped only when asked. That
distinction is the difference between "this dataset is 16% good" and "this
dataset is 16% *labelled* good", and they are not the same sentence.
"""

from __future__ import annotations

from typing import Any, Sequence

from gantry.contracts.curation import (
    CurationAction,
    CurationPlan,
    Curator,
    Prediction,
    curator_descriptor,
)
from gantry.spine import ComponentRef, Descriptor, Provenance

VERSION = "0.1.0.dev0"


def outcome_of(episode: Any) -> bool | None:
    """Whether this episode succeeded, or None if nobody said.

    Reads the labels a connector exposes rather than inferring from rewards or
    episode length: a short episode is not a failed one, and a shaped reward is
    positive for being close.
    """
    labels = getattr(episode, "labels", None)
    success = getattr(labels, "success", None)
    if success is None and isinstance(labels, dict):
        success = labels.get("success")
    return None if success is None else bool(success)


def uid_of(episode: Any) -> str:
    meta = getattr(episode, "meta", None)
    return str(getattr(meta, "uid", None) or getattr(episode, "uid", ""))


class FailedDemonstrations(Curator):
    """Drop what is labelled failed. Optionally drop what nobody labelled."""

    def __init__(self, *, drop_unlabelled: bool = False, expect: float = 0.05):
        #: Unlabelled is not failed. Off by default, because a dataset whose
        #: labels were never written would otherwise vanish entirely.
        self._drop_unlabelled = drop_unlabelled
        #: What removing failures is predicted to be worth, in success rate.
        #: An argument rather than a constant because it depends on how much
        #: failure is in there, and the caller has usually just been told.
        self._expect = expect

    def descriptor(self) -> Descriptor:
        return curator_descriptor(
            name="labels",
            version=VERSION,
            rung="screening",
            per_episode=True,
            needs_runs=False,
            drop_unlabelled=self._drop_unlabelled,
        )

    def propose(
        self, episodes: Sequence[Any], runs: Sequence[Any] = ()
    ) -> CurationPlan:
        failed, unlabelled, passed = [], [], []
        for episode in episodes:
            uid = uid_of(episode)
            if not uid:
                continue
            outcome = outcome_of(episode)
            if outcome is None:
                unlabelled.append(uid)
            elif outcome:
                passed.append(uid)
            else:
                failed.append(uid)

        drop = list(failed) + (list(unlabelled) if self._drop_unlabelled else [])
        scored = len(failed) + len(passed)
        rate = len(passed) / scored if scored else 0.0
        actions = []
        if drop:
            actions.append(
                CurationAction(
                    kind="drop",
                    episodes=tuple(drop),
                    detail={
                        "failed": len(failed),
                        "unlabelled_dropped": len(unlabelled) if self._drop_unlabelled else 0,
                        "success_rate_before": round(rate, 4),
                    },
                )
            )
        else:
            # Nothing to do is a legitimate answer and has to be sayable. An
            # empty plan is refused by the contract, so say "keep everything"
            # explicitly — it applies cleanly and records that this signal was
            # asked and had no objection.
            actions.append(
                CurationAction(
                    kind="keep",
                    episodes=tuple(passed + unlabelled),
                    detail={"reason": "no episode is labelled failed"},
                )
            )

        return CurationPlan(
            actions=tuple(actions),
            signal="labels",
            rung="screening",
            predicted=Prediction(
                metric="success_rate",
                direction="+",
                # Scaled by how much failure is actually present: removing 2%
                # failures and removing 84% of them are not the same claim.
                magnitude=round(self._expect * (1 - rate), 4) if drop else 0.0,
            ),
            evidence=Provenance(
                components=(ComponentRef("curation", "labels", VERSION),),
                protocol={
                    "episodes_read": len(episodes),
                    "labelled": scored,
                    "unlabelled": len(unlabelled),
                    "success_rate": round(rate, 4),
                },
            ),
            metadata={
                "unlabelled": len(unlabelled),
                "note": (
                    f"{len(passed)}/{scored} labelled demonstrations succeed"
                    if scored
                    else "no episode carries a success label"
                ),
            },
        )
