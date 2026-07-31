"""Cut a corpus into cohorts that can honestly be compared with each other.

Ranking two contributors' data means training on each and comparing. Almost
every way of building those two halves quietly answers a different question than
the one asked, and none of the failures show up as an error:

* **Split on frames and the same clip lands in both halves.** The two cohorts
  then share the thing being measured, and any difference between them is
  attenuated toward zero by however much they overlap.
* **Split without matching sizes and the comparison is about quantity.** More
  data usually wins. A ranking that reproduces the size ordering has told you
  the sizes.
* **Match sizes by truncating episodes and the trajectories end mid-reach.** A
  policy trained on those learns that tasks stop halfway.
* **Split on a property measured after the fact and never write down which
  one.** Six weeks later the two cohorts are "A" and "B" and nobody can say what
  distinguished them.

So this splits on whole episodes, on a declared and measured property, matches
by dropping whole episodes rather than trimming any, and records the criterion,
the measurement per episode, and what the matching cost.

Why this is not a curator
-------------------------
The curation plane prescribes changes — drop these, collect more of that, and
here is what I predict will happen. This prescribes nothing. It describes a
corpus and cuts it where it was told to, and the cutting is not a claim that
either half is better. What each half is worth is decided downstream by a
retrain and a paired evaluation, by machinery that has never heard of the
property split on. Keeping those apart is the same discipline the curation plane
enforces on itself: a signal that scores its own effect is the Goodhart machine
in miniature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from gantry.errors import ConfigError

VERSION = "0.1.0.dev0"

#: A per-episode number the split is made on.
Measure = Callable[[Any], float]


@dataclass(frozen=True)
class Part:
    """One side of a split, and what it is."""

    name: str
    episodes: tuple[Any, ...]
    #: The measured value for each episode kept, in the same order.
    measured: tuple[float, ...] = ()
    #: Episodes dropped from this part to match another's size, and their frames.
    dropped: tuple[str, ...] = ()

    @property
    def frames(self) -> int:
        return sum(len(episode) for episode in self.episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def summary(self) -> dict[str, Any]:
        values = np.asarray(self.measured, dtype=float) if self.measured else np.zeros(0)
        return {
            "name": self.name,
            "episodes": len(self.episodes),
            "frames": self.frames,
            "measured_median": round(float(np.median(values)), 4) if values.size else None,
            "measured_range": (
                [round(float(values.min()), 4), round(float(values.max()), 4)]
                if values.size
                else None
            ),
            "dropped_to_match": list(self.dropped),
        }


def group_of(episode: Any) -> str:
    """What must not straddle a split.

    Two halves that share a participant, a kitchen or a recording session are
    not two contributors — they are one, cut in half. The grouping key is taken
    from the episode's own metadata where it says, and falls back to the episode
    id, which makes each episode its own group and is the safe reading rather
    than the useful one.
    """
    extra = dict(getattr(episode.meta, "extra", {}) or {})
    for key in ("participant", "subject", "scene", "session"):
        if key in extra:
            return str(extra[key])
    return str(episode.meta.id)


@dataclass
class Split:
    """Cut a corpus in two on a measured property.

    ``measure`` returns one number per episode. ``threshold`` decides the side.
    Both are recorded, because "we split on hand visibility above 0.5" is a
    sentence somebody can disagree with and "cohort A" is not.
    """

    measure: Measure
    threshold: float
    #: (at or above, below) — named so the ordering is legible in the report.
    names: tuple[str, str] = ("high", "low")
    #: What the measure is, in words. Required: a number with no name is not a
    #: criterion, it is a coincidence.
    criterion: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.criterion.strip():
            raise ConfigError(
                "a split has to say what it split on. 'cohort A' and 'cohort B' are "
                "not a criterion anybody can disagree with six weeks later"
            )
        if len(set(self.names)) != 2:
            raise ConfigError(f"the two parts need different names, got {self.names}")

    def cut(self, episodes: Sequence[Any]) -> dict[str, Part]:
        """Both halves, with each episode's measured value kept."""
        above, below = [], []
        for episode in episodes:
            value = float(self.measure(episode))
            (above if value >= self.threshold else below).append((episode, value))

        parts = {}
        for name, chosen in zip(self.names, (above, below)):
            parts[name] = Part(
                name=name,
                episodes=tuple(e for e, _ in chosen),
                measured=tuple(v for _, v in chosen),
            )
        self._check_leakage(parts)
        return parts

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _check_leakage(parts: Mapping[str, Part]) -> None:
        seen: dict[str, str] = {}
        for name, part in parts.items():
            for episode in part.episodes:
                group = group_of(episode)
                if seen.setdefault(group, name) != name:
                    raise ConfigError(
                        f"{group!r} appears in both {seen[group]!r} and {name!r}. Two "
                        "halves that share a participant or a session are one "
                        "contributor cut in half, and any difference between them is "
                        "pulled toward zero by however much they overlap"
                    )


def match_frames(
    parts: Mapping[str, Part], *, tolerance: float = 0.05, seed: int = 0
) -> dict[str, Part]:
    """Trim the larger parts to the smallest, by dropping whole episodes.

    Whole episodes, never a trim: truncating a trajectory to hit a frame count
    teaches a policy that reaching stops halfway. So the count lands near the
    target rather than on it, and how near is reported.

    Refuses if it cannot get within ``tolerance``, because a comparison between
    halves of visibly different size is a comparison of sizes.
    """
    target = min(part.frames for part in parts.values())
    rng = np.random.default_rng(seed)
    out: dict[str, Part] = {}

    for name, part in parts.items():
        if part.frames <= target:
            out[name] = part
            continue
        # Drop at random rather than shortest-first: dropping the shortest
        # episodes first would leave the part systematically longer-clipped than
        # the one it is being compared with.
        order = list(rng.permutation(len(part.episodes)))
        keep, dropped, total = [], [], 0
        for index in order:
            episode = part.episodes[index]
            if total + len(episode) <= target:
                keep.append(index)
                total += len(episode)
            else:
                dropped.append(str(episode.meta.id))
        out[name] = Part(
            name=name,
            episodes=tuple(part.episodes[i] for i in sorted(keep)),
            measured=tuple(part.measured[i] for i in sorted(keep)) if part.measured else (),
            dropped=tuple(dropped),
        )

    sizes = [part.frames for part in out.values()]
    spread = (max(sizes) - min(sizes)) / max(1, max(sizes))
    if spread > tolerance:
        raise ConfigError(
            f"could not match these parts within {tolerance:.0%}: got {dict((n, p.frames) for n, p in out.items())}, "
            f"{spread:.1%} apart. Whole episodes are the unit, so a part made of a few "
            "long clips cannot always be trimmed finely. Compare unmatched and say so, "
            "or split on something that divides more evenly"
        )
    return out


# --------------------------------------------------------------------------
# measures
# --------------------------------------------------------------------------


def moving_fraction(channel: str = "action", arms: int = 2, width: int = 7) -> Measure:
    """How much of an episode both arms were actually moving.

    An arm the tracker never found is held at its last value by the connector,
    so it reads as a frozen block of identical numbers rather than as missing.
    That is invisible to any shape or dtype check and it is the difference
    between two-handed footage and one-handed footage — which, for a two-armed
    task, is the difference between data that could teach it and data that
    cannot.

    Returns the *smaller* of the two arms' moving fractions, because a clip is
    only two-handed if both hands are.
    """

    def measure(episode: Any) -> float:
        values = np.asarray(episode.array(channel), dtype=float)
        live = []
        for index in range(arms):
            block = values[:, index * width : index * width + 3]
            if len(block) < 2:
                return 0.0
            step = np.linalg.norm(np.diff(block, axis=0), axis=1)
            live.append(float((step > 1e-9).mean()))
        return min(live)

    return measure
