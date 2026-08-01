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
    #: What identified the groups, or None if nothing did — in which case the
    #: two halves may share a participant and there is no way to tell.
    grouped_by: str | None = None

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
            "grouped_by": self.grouped_by,
            # Said out loud, because "no leakage found" and "leakage not
            # checkable" look the same in a table.
            "leakage_checkable": self.grouped_by is not None,
        }


#: Metadata keys that identify who or what a clip belongs to, in the order they
#: are trusted.
GROUPS = ("participant", "subject", "scene", "session")


def grouped_by(episodes: Sequence[Any]) -> str | None:
    """Which key identifies the groups, or ``None`` if nothing does.

    ``None`` is the case that matters. When no episode says who filmed it, every
    episode becomes its own group and the leakage check passes trivially -- it
    has nothing to compare. That is not "no leakage found", it is "leakage not
    checkable", and the two read identically in a report unless one of them says
    so.
    """
    for key in GROUPS:
        if any(key in dict(getattr(e.meta, "extra", {}) or {}) for e in episodes):
            return key
    return None


def group_of(episode: Any) -> str:
    """What must not straddle a split.

    Two halves that share a participant, a kitchen or a recording session are
    not two contributors — they are one, cut in half. The grouping key is taken
    from the episode's own metadata where it says, and falls back to the episode
    id, which makes each episode its own group and is the safe reading rather
    than the useful one.
    """
    extra = dict(getattr(episode.meta, "extra", {}) or {})
    for key in GROUPS:
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

        key = grouped_by(episodes)
        parts = {}
        for name, chosen in zip(self.names, (above, below)):
            parts[name] = Part(
                name=name,
                episodes=tuple(e for e, _ in chosen),
                measured=tuple(v for _, v in chosen),
                grouped_by=key,
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


def hold_out(
    episodes: Sequence[Any], *, fraction: float = 0.2, seed: int = 0, minimum: int = 2
) -> tuple[Part, Part]:
    """Keep some episodes back to score on. Returns ``(train, held_out)``.

    Different from :class:`Split`, which cuts a corpus on a measured property to
    make two cohorts to compare. This cuts it into something to fit and
    something to score on, and the only thing it cares about is that the two
    sides share nothing.

    Whole episodes, and grouped
    ---------------------------
    Splitting by frame would put step 400 of a clip in the training set and step
    401 in the held-out set. They are nearly the same picture, so the score
    would measure interpolation between neighbouring frames rather than anything
    that generalises -- and it would look excellent. Any two episodes that share
    a participant, a session or a scene go to the same side for the same reason
    one step further out.

    ``grouped_by`` is carried through, so a report can say whether that check
    had anything to check. When no episode records who filmed it, every episode
    is its own group and the check passes without looking at anything, which is
    not the same as passing.
    """
    if len(episodes) < minimum * 2:
        raise ConfigError(
            f"{len(episodes)} episode(s) cannot be split into something to fit and "
            f"something to score on; at least {minimum * 2} are needed"
        )
    key = grouped_by(episodes)

    # Whole groups move together, and which group goes where is decided by a
    # seeded shuffle of the *groups* -- not of the episodes, which would split
    # a group across both sides whenever it had more than one clip.
    groups: dict[str, list[Any]] = {}
    for episode in episodes:
        groups.setdefault(group_of(episode), []).append(episode)
    names = sorted(groups)
    np.random.default_rng(seed).shuffle(names)

    wanted = max(minimum, int(round(len(episodes) * fraction)))
    held: list[Any] = []
    for name in names:
        if len(held) >= wanted:
            break
        held.extend(groups[name])
    kept = [e for e in episodes if e not in held]

    if len(kept) < minimum:
        raise ConfigError(
            f"holding out {len(held)} of {len(episodes)} episode(s) leaves too few to "
            f"fit on; at least {minimum} are needed"
        )
    parts = (
        Part(name="fit", episodes=tuple(kept), grouped_by=key),
        Part(name="held_out", episodes=tuple(held), grouped_by=key),
    )
    Split._check_leakage({p.name: p for p in parts})
    return parts


def match_frames(
    parts: Mapping[str, Part], *, tolerance: float = 0.05, seed: int = 0
) -> dict[str, Part]:
    """Trim the larger parts to the smallest, by dropping whole episodes.

    Whole episodes, never a trim: truncating a trajectory to hit a frame count
    teaches a policy that reaching stops halfway. So the count lands near the
    target rather than on it, and how near is reported.

    Overshooting is allowed when it lands closer. Episodes in a real corpus are
    all about the same length, so a target rarely falls on an episode boundary:
    with ~300-frame clips and a target of 2294, seven clips is 2087 and eight is
    2376, and the second is the better match. Taking only the largest subset
    that fits under the target would have refused this split for being 9% apart
    when a 4% match was available.

    Refuses if it still cannot get within ``tolerance``, because a comparison
    between halves of visibly different size is a comparison of sizes.
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
            # Keep it if doing so lands nearer the target than stopping here.
            if abs(total + len(episode) - target) < abs(total - target):
                keep.append(index)
                total += len(episode)
            else:
                dropped.append(str(episode.meta.id))
        out[name] = Part(
            name=name,
            episodes=tuple(part.episodes[i] for i in sorted(keep)),
            measured=tuple(part.measured[i] for i in sorted(keep)) if part.measured else (),
            dropped=tuple(dropped),
            grouped_by=part.grouped_by,
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
