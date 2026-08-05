"""The shuffled control: same frames, same actions, no correspondence.

The comparison this exists for
------------------------------
The obvious experiment is a model without the contributor's data against a model
with it. That comparison is real and it is not enough, because fine-tuning a
large pretrained model on *anything* moves it. Show it five thousand frames with
the action labels attached to the wrong frames and the loss still falls, the
behaviour still changes, and a before-and-after table still shows a difference.
Reported as "their data helped", that is false in the most expensive possible
way: it is a result that reproduces for every contributor, including the ones
whose footage is worthless.

So the control is built from the contributor's own data. Identical frames,
identical actions, identical count of gradient steps -- and the actions belong
to a different episode than the frames do. Whatever fine-tuning-in-general buys,
this arm buys too. Beating it is the only thing that says the *correspondence*
between what was seen and what was done carried information.

Why whole episodes rather than frames
-------------------------------------
Shuffling frame by frame within an episode leaves a control that is nearly as
learnable as the original: neighbouring frames look alike and their actions are
alike, so a model predicting the mean of a small neighbourhood does well without
any correspondence at all. The control has to be *harder* than that or it is not
a control. Donating a whole episode's action sequence to a different episode's
frames keeps every action trajectory smooth and physically plausible -- the
action distribution is untouched, which is the point -- while destroying the
relationship completely.

A derangement, not a shuffle
----------------------------
The permutation is required to move every episode. An ordinary shuffle leaves
some episodes matched with themselves, and those are not controls at all: they
are training data sitting inside the control arm, quietly making it look better
and the contributor's data look worse by comparison.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

VERSION = "0.1.0.dev0"

#: Channels whose contents are the *decision* rather than the observation.
#: These are what a donor episode supplies; everything else stays with the
#: frames it was recorded with.
DECISIONS = ("action",)


class DetachedSource:
    """One episode's observations with another's decisions.

    Lengths rarely match, so both are read to the shorter of the two. Padding
    or repeating to the longer length would invent steps that were never
    recorded, and a control built out of invented data cannot be compared with
    anything.
    """

    def __init__(self, keeps: Any, donor: Any, decisions: Sequence[str] = DECISIONS):
        self._keeps = keeps
        self._donor = donor
        self._decisions = tuple(d for d in decisions if d in donor.channels)
        self._num_steps = min(keeps.num_steps, donor.num_steps)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(self._keeps.channels)

    def read(
        self,
        channels: Sequence[str] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> dict[str, np.ndarray]:
        wanted = tuple(channels) if channels is not None else self.channels
        stop = self._num_steps if stop is None else min(stop, self._num_steps)
        start = min(start, stop)

        from_donor = [name for name in wanted if name in self._decisions]
        from_keeps = [name for name in wanted if name not in self._decisions]
        out: dict[str, np.ndarray] = {}
        if from_keeps:
            out.update(self._keeps.read(from_keeps, start, stop))
        if from_donor:
            out.update(self._donor.read(from_donor, start, stop))
        return out


def _derangement(count: int, seed: int) -> list[int]:
    """A permutation with no fixed point, by rejection.

    Rejection rather than a constructive shift, because a shift is the same
    permutation every time and a control that is always "the next episode
    along" can be gamed by an upload ordered to exploit it -- neighbouring
    clips of the same activity would make the control nearly as good as the
    real thing.
    """
    if count < 2:
        raise ValueError("a shuffled control needs at least two episodes to swap between")
    rng = np.random.default_rng(seed)
    order = list(range(count))
    for _ in range(1000):
        rng.shuffle(order)
        if all(index != position for position, index in enumerate(order)):
            return list(order)
    # Falls back to a rotation, which is a valid derangement, rather than
    # returning a permutation that leaves episodes matched with themselves.
    return [(i + 1) % count for i in range(count)]


def shuffled(
    episodes: Sequence[Any],
    *,
    seed: int = 0,
    decisions: Sequence[str] = DECISIONS,
    tag: str = "shuffled",
) -> tuple[Any, ...]:
    """The same episodes with their decisions donated by other episodes.

    Every returned episode keeps its own frames, its own state, its own
    schema and its own instruction, and takes its actions from a different
    episode. The set of actions in the cohort is unchanged -- only which frames
    they sit beside.

    The uid records where the actions came from, so a control episode can be
    traced back to both of its parents. A control that cannot be traced is
    indistinguishable from a bug.
    """
    if len(episodes) < 2:
        raise ValueError("a shuffled control needs at least two episodes to swap between")
    order = _derangement(len(episodes), seed)

    out = []
    for position, episode in enumerate(episodes):
        donor = episodes[order[position]]
        source = DetachedSource(episode.source, donor.source, decisions)
        meta = episode.meta
        extra: Mapping[str, Any] = {
            **dict(getattr(meta, "extra", {}) or {}),
            "control": tag,
            "actions_from": str(getattr(donor.meta, "uid", "?")),
        }
        out.append(
            replace(
                episode,
                meta=replace(meta, id=f"{meta.id}+{tag}", extra=extra),
                source=source,
            )
        )
    return tuple(out)
