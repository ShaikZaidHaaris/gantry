"""Gate 2 — Signal check. Is there anything learnable here?

The money gate, and the only one whose job is to say no.

What it does
------------
Hold some episodes back. Fit a policy to the rest. Fit the *same* policy to the
same episodes with their actions detached and donated to other episodes. Score
both fits on the held-out episodes, which neither ever saw. Compare them, on
the same episodes, one episode at a time.

Why the comparison is against a shuffle and not against nothing
---------------------------------------------------------------
Fine-tuning a large model on anything moves it. Show it five thousand frames
with the labels attached to the wrong pictures and the loss still falls and the
behaviour still changes. So "it got better after training on your data" is a
result that reproduces for every contributor, including the ones whose footage
is worthless. Beating the shuffle is the narrower claim, and the only one worth
selling: the *correspondence* between what the camera saw and what the hands did
carried information.

Why this gate exists rather than going straight to the robot
------------------------------------------------------------
Because the robot test costs a day and this costs a minute. In this project a
full fine-tune and a closed-loop evaluation were run on data whose relationship
nobody had checked, and eight GPU-hours produced two zeros. Neither number said
anything about the data, because nothing had established there was anything
there to find. This gate is that check, run first, for the price of nothing.

What a pass does and does not mean
----------------------------------
A pass says a cheap linear probe found coarse structure. That is a low bar
deliberately: this gate should be wrong in the direction of letting something
through to a real test, never in the direction of refusing footage that was
fine. A fail is the confident direction -- if the cheapest possible model cannot
tell the data from its own shuffle, a bigger one is unlikely to be the answer,
and it certainly is not worth a day of GPU to find out.

Three outcomes, and the third is not a fail
-------------------------------------------
passed      the data beat its own shuffle
refused     the shuffle was as good or better, so there is nothing to buy
abstained   the two could not be separated on this many held-out episodes

The third is the one that gets misreported everywhere else. "Not separated" is
about the size of the upload, not about the footage, and telling a contributor
their data failed when the honest answer is that there was not enough of it to
tell is the kind of wrong that loses a customer who was right.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

Report = Callable[..., None]


def _quiet(*_args, **_kwargs) -> None:
    """Default report: say nothing. A gate must run without a worker."""


#: Fraction of episodes kept back to score on, when that is more than the floor.
HELD_OUT = 0.25

#: Fewest episodes to fit on. Below this the fit is about the handful of clips
#: it saw rather than about the upload.
FIT_FLOOR = 4

ALPHA = 0.05

#: Seed for the split and the derangement. Fixed, so a resubmission of the same
#: data gets the same held-out episodes and the two runs are comparable. A
#: random seed would make every rerun a slightly different experiment.
SEED = 20250801


def smallest_conclusive(alpha: float = ALPHA, cap: int = 64) -> int:
    """Held-out clips below which this gate cannot pass, however good the data.

    Derived from the test rather than chosen. With five paired clips a *clean
    sweep* -- the data winning every single one -- is p=0.0625, so five can
    never reach 0.05 and a gate configured to hold out five would refuse
    everything while looking like it was measuring something. Six is the first
    number where a sweep clears the bar.

    Computed here so the floor and the test can never drift apart. A constant
    would be right today and silently wrong the first time anybody changed
    ``alpha``.
    """
    for n in range(1, cap):
        if _sign_test(n, 0) <= alpha:
            return n
    return cap


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test. Ties carry no information and are excluded.

    The same test the robot gate uses on scenes, applied here to episodes, so a
    contributor is not asked to trust two different notions of significance in
    one report.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(extreme, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _error(fitted: Any, episode: Any, names: Sequence[str], action: str = "action") -> float:
    """Mean squared error of one fit on one held-out episode."""
    from gantry_policy_probe import features

    x = features(episode, names)
    predicted = x @ fitted._weights + fitted._bias
    actual = np.asarray(episode.array(action), np.float64).reshape(len(x), -1)
    return float(((predicted - actual) ** 2).mean())


def run(
    archive: Path,
    workdir: Path,
    report: Report = _quiet,
    params: Mapping[str, Any] | None = None,
) -> dict:
    """The gate contract. Intake already unpacked; read what it left."""
    from gantry_connector_lerobot import LeRobotConnector
    from gantry_curate_control import shuffled
    from gantry_curate_split import hold_out
    from gantry_policy_probe import ProbeLearner

    report("finding the dataset")
    root = next((workdir / "unpacked").rglob("meta/info.json"), None)
    if root is None:
        return {
            "status": "failed",
            "summary": "the unpacked dataset is missing. Intake did not leave it where this gate looks for it",
            "findings": [],
        }

    report("opening the clips")
    connector = LeRobotConnector(str(root.parents[1]))
    episodes = tuple(connector.open(i) for i in connector.episode_ids())

    # Enough held-out clips for a clean sweep to be significant, plus enough
    # left over to fit on. Refusing to *run* below this is the honest move: the
    # alternative is a check that always abstains while appearing to measure
    # something, which is worse than one that says it needs more footage.
    need_held = smallest_conclusive()
    minimum = need_held + FIT_FLOOR
    if len(episodes) < minimum:
        return {
            "status": "abstained",
            "summary": (
                f"{len(episodes)} clip(s) is too few for this check to conclude anything. "
                "That is about the size of the upload, not the footage"
            ),
            "findings": [
                {
                    "code": "signal.too_few_clips",
                    "severity": "info",
                    "summary": (
                        f"{len(episodes)} clip(s), and at least {minimum} are needed: "
                        f"{need_held} to score on and {FIT_FLOOR} to fit on"
                    ),
                    "prescription": (
                        f"Upload at least {minimum} clips. The comparison is one clip against "
                        f"itself, so with fewer than {need_held} held back, your data winning "
                        "every single clip would still not be past chance. The check would "
                        "abstain no matter how good the footage was."
                    ),
                    "module": "signal",
                }
            ],
        }

    report("splitting", note=f"holding back {int(HELD_OUT * 100)}% to score on")
    try:
        fit_part, held_part = hold_out(
            episodes,
            fraction=max(HELD_OUT, need_held / len(episodes)),
            seed=SEED,
            minimum=FIT_FLOOR,
        )
    except Exception as error:  # noqa: BLE001 - a refusal from the splitter is a refusal
        return {
            "status": "abstained",
            "summary": f"the clips could not be split into a fit and a held-out set: {error}",
            "findings": [
                {
                    "code": "signal.cannot_split",
                    "severity": "info",
                    "summary": str(error),
                    "prescription": None,
                }
            ],
        }

    # Images only. Proprioceptive state predicts the next action in *every*
    # dataset -- the next action is mostly where the arm already is -- so a
    # probe allowed to read it reports signal in footage whose pixels say
    # nothing, and reports it against the shuffled control too, because the
    # control's state comes from the donor episode while the real arm keeps its
    # own. That is a large measured effect entirely about proprioception, and
    # the contribution being screened here is the footage.
    learner = ProbeLearner(kinds=("image",))
    names = learner.observation_names(fit_part.episodes)
    if not names:
        return {
            "status": "abstained",
            "summary": (
                "these clips share no camera between them, so there is no imagery to check. "
                "This gate asks whether the footage predicts the actions"
            ),
            "findings": [
                {
                    "code": "signal.no_shared_camera",
                    "severity": "info",
                    "summary": "no image channel is present in every clip",
                    "prescription": (
                        "Upload clips that all carry the same camera. A channel only some "
                        "clips have cannot be fitted across the set. Proprioception on its "
                        "own predicts the next action in any dataset, so it would pass this "
                        "check without saying anything about your footage."
                    ),
                    "module": "signal",
                }
            ],
        }

    report("fitting your data", current=1, total=2, note=f"{len(fit_part)} clips")
    real = learner.fit(fit_part.episodes, seed=SEED)

    report("fitting the shuffled control", current=2, total=2, note="same clips, actions detached")
    control = learner.fit(shuffled(fit_part.episodes, seed=SEED), seed=SEED)

    report("scoring", current=0, total=len(held_part), note="on clips neither fit has seen")
    pairs: list[dict[str, Any]] = []
    for index, episode in enumerate(held_part.episodes):
        report("scoring", current=index + 1, total=len(held_part), note=str(episode.meta.uid))
        yours = _error(real, episode, names)
        theirs = _error(control, episode, names)
        pairs.append(
            {
                "episode": str(episode.meta.uid),
                "yours": round(yours, 6),
                "shuffled": round(theirs, 6),
                "better": bool(yours < theirs),
            }
        )

    wins = sum(1 for p in pairs if p["yours"] < p["shuffled"])
    losses = sum(1 for p in pairs if p["yours"] > p["shuffled"])
    p_value = _sign_test(wins, losses)

    yours_median = float(np.median([p["yours"] for p in pairs]))
    shuffled_median = float(np.median([p["shuffled"] for p in pairs]))
    # Guarded: a control that predicts perfectly would divide by zero, and a
    # reduction of "inf" is not a number anybody should be shown.
    reduction = (
        (shuffled_median - yours_median) / shuffled_median if shuffled_median > 0 else 0.0
    )

    from gantry.spine import proportion

    measures = {
        "held_out.clips_your_data_predicted_better": {
            **proportion(wins, len(pairs)).as_dict(),
            "module": "signal",
        },
        "held_out.error_your_data": {
            "value": round(yours_median, 6),
            "n": len(pairs),
            "ci": None,
            "units": "mean squared error",
            "method": "median over held-out clips",
            "module": "signal",
        },
        "held_out.error_shuffled_control": {
            "value": round(shuffled_median, 6),
            "n": len(pairs),
            "ci": None,
            "units": "mean squared error",
            "method": "median over held-out clips",
            "module": "signal",
        },
    }

    common = {
        "measures": measures,
        "detail": {
            "pairs": pairs,
            "p_value": round(p_value, 5),
            "fit_clips": len(fit_part),
            "held_out_clips": len(held_part),
            "read": list(names),
            # Said out loud: with nothing recording who filmed what, every clip
            # is its own group and the leakage check passed without looking at
            # anything. That is not the same as passing.
            "leakage_checkable": held_part.grouped_by is not None,
            "grouped_by": held_part.grouped_by,
        },
    }

    if p_value > ALPHA:
        return {
            **common,
            "status": "abstained",
            "summary": (
                f"your data predicted better on {wins} of {len(pairs)} held-out clips, which "
                f"is not enough to separate it from its own shuffle (p={p_value:.2f})"
            ),
            "findings": [
                {
                    "code": "signal.not_separated",
                    "severity": "info",
                    "summary": (
                        f"on {len(pairs)} held-out clip(s), your data and its shuffled control "
                        "could not be told apart"
                    ),
                    "prescription": (
                        "This is about how much footage there is, not how it was filmed. "
                        f"Your data won {wins} of {len(pairs)} held-out clips, and at this "
                        "many clips that is still within what chance produces. More clips of "
                        "the same quality would settle it, since the held-out set grows with "
                        "the upload."
                    ),
                    "module": "signal",
                }
            ],
        }

    if wins < losses:
        # The row worth keeping from the control's own truth table: a control
        # that outperforms the real thing is not a result about data quality.
        # It is a signal that labels are misaligned or the split leaked, and it
        # should stop the report rather than appear in it.
        return {
            **common,
            "status": "refused",
            "summary": (
                f"the shuffled control predicted better than your data on {losses} of "
                f"{len(pairs)} held-out clips (p={p_value:.3f})"
            ),
            "findings": [
                {
                    "code": "signal.control_wins",
                    "severity": "strong",
                    "summary": (
                        "scrambling which actions belong to which frames made prediction "
                        "*better*, which should be impossible if the pairing is right"
                    ),
                    "prescription": (
                        "This usually means the actions are offset from the frames they "
                        "belong to. An export that wrote actions one frame late, say, or a "
                        "timestamp column that was never used to line them up. Check the "
                        "alignment before running a robot test. Nothing downstream can "
                        "recover from it."
                    ),
                    "module": "signal",
                }
            ],
        }

    return {
        **common,
        "status": "passed",
        "summary": (
            f"your data beat its own shuffled control on {wins} of {len(pairs)} held-out "
            f"clips (p={p_value:.3f}), cutting prediction error by {reduction:.0%}"
        ),
        "findings": [
            {
                "code": "signal.present",
                "severity": "info",
                "summary": (
                    f"a fit on your clips predicts held-out actions {reduction:.0%} better "
                    "than the same fit with the actions detached"
                ),
                "prescription": (
                    "There is a relationship between what the camera saw and what the hands "
                    "did, and it holds up on clips the fit never saw. The robot test is what "
                    "measures how much it is worth. This check used a deliberately cheap model, "
                    "so it tells you the signal is there, not how strong it is."
                ),
                "module": "signal",
            }
        ],
    }
