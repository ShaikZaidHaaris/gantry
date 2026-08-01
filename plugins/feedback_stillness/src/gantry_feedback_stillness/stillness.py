"""Parts of the action that never moved for a whole episode.

The defect this exists for
--------------------------
When an estimator loses track of something -- a hand leaves the shot, a marker
is occluded -- the honest thing to record is nothing. What most ingests actually
record is the last value it had, repeated to the end of the clip. The result is
a block of numbers that is the right width, the right dtype, inside every
plausible range, and perfectly motionless. Every schema check passes. Every
statistic over the whole vector looks unremarkable. And for a task that needs
that part of the body, the clip teaches the opposite of what it appears to.

Why it is written in terms of blocks and not limbs
--------------------------------------------------
It would be shorter to split the action in half and call the halves left and
right. That would be a module about two-armed robots, and it would be wrong for
every embodiment that is not one -- a mobile base, a single arm with a wrist, a
five-fingered hand. What is actually true regardless of embodiment is narrower
and checkable: *these adjacent dimensions did not change*. Naming them is a
separate step, and it is only done when the channel declares ``dim_labels``, in
which case the label is read rather than assumed. Given labels, the finding says
"the seven `left_*` dimensions"; without them it says "dimensions 0-6", which is
less useful and still true.

A frozen block is not a verdict
-------------------------------
Plenty of dimensions are legitimately constant. A gripper that stays closed
through a clip that never grasps is constant and correct; so is a padding
dimension. Two things keep the noise down, and both are thresholds this module
owns rather than hides: a block must be at least :data:`MIN_BLOCK` wide before
it looks like a limb rather than a coincidence, and the finding is raised on the
*fraction of clips* affected rather than on any single clip. One still clip in
sixty is a clip. A third of them is a filming problem.

Absent is not zero, here too
----------------------------
An episode whose action channel cannot be read at all is counted as unread and
named in the notes, never as an episode with no frozen block. Silence from a
failed read looks exactly like a clean bill of health, and this module would be
a bad place to introduce that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from gantry.contracts.feedback import (
    Cohort,
    FeedbackModule,
    Finding,
    Report,
    feedback_descriptor,
)
from gantry.resolve import Requirement, requires_channels
from gantry.spine import Descriptor, Measurement, Verdict, proportion

VERSION = "0.1.0.dev0"

#: Total spread below which a dimension counts as unmoving. Not exactly zero:
#: a value held through a float32 round-trip, or resampled to a different rate,
#: can wobble in the last bits. Well below anything a real joint does and well
#: above representation noise.
STILL = 1e-9

#: Adjacent still dimensions before the block is worth mentioning. A single
#: constant number is ordinary -- an unused gripper, a padding slot. Three in a
#: row is the smallest thing that can be a position, and the smallest thing that
#: reads as a part of the body rather than a coincidence.
MIN_BLOCK = 3

#: Fraction of a cohort's clips that must be affected before this is a finding
#: about the dataset rather than a note about one clip.
AFFECTED = 0.1

#: Channels to look at, most-preferred first. The action is what a policy is
#: trained to produce, so a frozen block there is the expensive one; state is
#: checked when there is no action channel to read.
PREFERRED = ("action", "observation.state", "state")


@dataclass(frozen=True)
class Block:
    """A run of adjacent dimensions that did not change, and what they are called."""

    start: int
    stop: int  # exclusive
    label: str | None = None

    @property
    def width(self) -> int:
        return self.stop - self.start

    def describe(self) -> str:
        """What to call this block in a sentence a contributor reads."""
        if self.label:
            return f"the {self.width} {self.label} dimension(s)"
        return f"dimensions {self.start}-{self.stop - 1}"

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "stop": self.stop, "width": self.width, "label": self.label}


def still_dims(values: np.ndarray, *, still: float = STILL) -> np.ndarray:
    """Boolean mask over dimensions: did this one change at all?

    Spread over the whole episode rather than a sum of step sizes. A dimension
    that oscillates between two values has a large total step and no spread,
    but the failure being caught is a held value, and spread is the direct
    reading of it -- fewer assumptions, and it cannot be fooled by rounding
    that happens to alternate.
    """
    if values.ndim != 2 or len(values) < 2:
        return np.zeros(values.shape[-1] if values.ndim == 2 else 0, dtype=bool)
    finite = np.isfinite(values)
    spread = np.where(
        finite.all(axis=0),
        values.max(axis=0) - values.min(axis=0),
        np.inf,  # a dimension with a NaN in it is not something this module judges
    )
    return spread <= still


def blocks_in(
    mask: np.ndarray,
    labels: Sequence[str] | None = None,
    *,
    min_block: int = MIN_BLOCK,
) -> list[Block]:
    """Maximal runs of ``True`` in ``mask``, at least ``min_block`` wide.

    ``labels``, when the channel declares them, names the block by the longest
    prefix its members share up to a separator -- ``left_x, left_y, left_z``
    becomes ``left``. A block whose labels share no prefix keeps its indices,
    because inventing a name for it would be the module guessing at an
    embodiment it was careful not to assume.
    """
    out: list[Block] = []
    start: int | None = None
    for index, still in enumerate([*mask, False]):
        if still and start is None:
            start = index
        elif not still and start is not None:
            if index - start >= min_block:
                out.append(Block(start, index, _shared_prefix(labels, start, index)))
            start = None
    return out


def _shared_prefix(labels: Sequence[str] | None, start: int, stop: int) -> str | None:
    if not labels or stop > len(labels):
        return None
    parts = [str(labels[i]) for i in range(start, stop)]
    heads = {p.replace(".", "_").split("_")[0] for p in parts}
    return heads.pop() if len(heads) == 1 else None


def _channel(episode: Any, preferred: Sequence[str] = PREFERRED) -> str | None:
    names = set(getattr(episode, "channel_names", ()) or ())
    return next((name for name in preferred if name in names), None)


def _labels(episode: Any, name: str) -> tuple[str, ...] | None:
    try:
        return getattr(episode.channel(name), "dim_labels", None)
    except Exception:  # noqa: BLE001 - a channel that will not describe itself is unnamed
        return None


class Stillness(FeedbackModule):
    """Which parts of the action never moved, in how much of the footage."""

    def __init__(
        self,
        *,
        still: float = STILL,
        min_block: int = MIN_BLOCK,
        affected: float = AFFECTED,
        preferred: Sequence[str] = PREFERRED,
    ):
        self._still = float(still)
        self._min_block = int(min_block)
        self._affected = float(affected)
        self._preferred = tuple(preferred)

    def descriptor(self) -> Descriptor:
        return feedback_descriptor(
            name="stillness",
            version=VERSION,
            min_cohorts=1,
            prescribes=True,
            holds=(),
            still=self._still,
            min_block=self._min_block,
            channels=list(self._preferred),
        )

    def requirement(self) -> Requirement:
        return requires_channels(
            "stillness",
            "feedback",
            description="a control channel to read, one of " + ", ".join(self._preferred),
        )

    def check_inputs(self, cohorts: Sequence[Cohort]) -> Verdict:
        """Refuse when no cohort has a channel to read.

        Refusing beats reporting nothing found: "no frozen blocks" and "nothing
        was looked at" are the same empty list and opposite claims.
        """
        for cohort in cohorts:
            if any(_channel(episode, self._preferred) for episode in cohort.episodes):
                return Verdict.yes()
        return Verdict.no(
            "stillness.no_control_channel",
            "no cohort carries any of " + ", ".join(self._preferred) + " to read",
        )

    # -- the measurement ---------------------------------------------------

    def scan(self, cohort: Cohort) -> dict[str, Any]:
        """Per-episode blocks, plus how often each named block was frozen."""
        per_episode: list[dict[str, Any]] = []
        unread: list[str] = []
        for episode in cohort.episodes:
            name = _channel(episode, self._preferred)
            uid = str(getattr(getattr(episode, "meta", None), "uid", "?"))
            if name is None:
                unread.append(uid)
                continue
            try:
                values = np.asarray(episode.array(name), dtype=np.float64)
            except Exception:  # noqa: BLE001 - unread, which is not "nothing frozen"
                unread.append(uid)
                continue
            values = values.reshape(len(values), -1) if values.ndim > 1 else values[:, None]
            mask = still_dims(values, still=self._still)
            found = blocks_in(mask, _labels(episode, name), min_block=self._min_block)
            per_episode.append(
                {
                    "episode": uid,
                    "channel": name,
                    "frames": int(len(values)),
                    "width": int(values.shape[1]),
                    "still_dims": int(mask.sum()),
                    "whole_channel": bool(mask.all()),
                    "blocks": [b.as_dict() for b in found],
                }
            )
        return {"per_episode": per_episode, "unread": unread}

    def analyse(self, cohorts: Sequence[Cohort]) -> Report:
        findings: list[Finding] = []
        measurements: dict[str, Measurement] = {}
        notes: list[str] = []

        for cohort in cohorts:
            scan = self.scan(cohort)
            read = scan["per_episode"]
            if scan["unread"]:
                notes.append(
                    f"{cohort.name}: {len(scan['unread'])} clip(s) had no readable control "
                    "channel and were not judged either way"
                )
            if not read:
                continue

            dead = [e for e in read if e["whole_channel"]]
            # A whole channel that never changes is a different and worse
            # defect than a limb that did, and reporting it as "one frozen
            # block" would understate it.
            if dead:
                measurements[f"{cohort.name}.motionless"] = proportion(len(dead), len(read))
                findings.append(
                    Finding(
                        code="stillness.nothing_moved",
                        summary=(
                            f"{cohort.name}: {len(dead)} of {len(read)} clip(s) have an action "
                            "channel that never changes from its first value"
                        ),
                        severity="strong",
                        measurements={"clips": proportion(len(dead), len(read))},
                        evidence={"episodes": [e["episode"] for e in dead[:10]]},
                        prescription=(
                            "These clips contain no motion to learn from. Check that the export "
                            "wrote per-frame actions rather than repeating the first frame, and "
                            "drop them if it did not. A clip with no motion is not a hard "
                            "example; it is a constant the policy will fit and then obey."
                        ),
                        cohorts=(cohort.name,),
                    )
                )

            moving = [e for e in read if not e["whole_channel"]]
            counts: dict[str, list[str]] = {}
            for episode in moving:
                for block in episode["blocks"]:
                    key = Block(block["start"], block["stop"], block["label"]).describe()
                    counts.setdefault(key, []).append(episode["episode"])

            for key, episodes in sorted(counts.items(), key=lambda kv: -len(kv[1])):
                rate = proportion(len(episodes), len(moving))
                measurements[f"{cohort.name}.frozen[{key}]"] = rate
                if rate.value < self._affected:
                    continue
                findings.append(
                    Finding(
                        code="stillness.frozen_block",
                        summary=(
                            f"{cohort.name}: {key} never change in {len(episodes)} of "
                            f"{len(moving)} clip(s)"
                        ),
                        severity="strong" if rate.value >= 0.25 else "weak",
                        measurements={"clips": rate},
                        # Named, because "re-film some of your clips" is not
                        # actionable and "re-film these two" is.
                        evidence={"episodes": episodes[:10], "of": len(episodes)},
                        prescription=(
                            f"In {', '.join(episodes[:4])}"
                            + (f" and {len(episodes) - 4} more" if len(episodes) > 4 else "")
                            + f", {key} hold one value for the entire clip while the rest of the "
                            "action moves. That is what a lost track looks like after the ingest "
                            "fills it in. Re-film those clips with the whole body in shot for the "
                            "whole take. Until then the policy is being taught to hold that part "
                            "still while doing the task."
                        ),
                        cohorts=(cohort.name,),
                    )
                )

            if not findings or not any(f.cohorts == (cohort.name,) for f in findings):
                findings.append(
                    Finding(
                        code="stillness.none",
                        summary=(
                            f"{cohort.name}: every part of the action moves in all "
                            f"{len(read)} clip(s)"
                        ),
                        severity="info",
                        cohorts=(cohort.name,),
                    )
                )

        return Report(
            module="stillness",
            findings=tuple(findings),
            measurements=measurements,
            notes=tuple(notes),
            cohorts=tuple(cohort.name for cohort in cohorts),
        )
