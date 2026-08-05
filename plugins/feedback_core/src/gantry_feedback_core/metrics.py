"""Per-episode statistics, over roles the module asked for by name.

Three roles, and a statistic reads whichever channel got bound to one:

``position``
    A point per step. Path efficiency and direction changes are defined over
    any number of dimensions, so this is deliberately width-agnostic.

``actuation``
    One scalar per step for whatever the machine grips, holds, or switches.

``command``
    The vector actually sent, whatever its width.

Why roles rather than a search
------------------------------
These statistics used to find their channels themselves, by looking up a
hardcoded semantics tag. That quietly broke the framework's central promise: a
dataset described with a *richer* vocabulary than the one hardcoded here got
*less* analysis than one described with the generic tags -- six statistics went
unavailable on real data described exactly correctly. Describing your data
better made the answer worse.

So the module declares a requirement, the resolver binds it -- by name, by alias,
by meaning, through a retargeter where the shapes need reconciling -- and the
episode arrives with its channels already renamed to these roles. The module
never looks anything up, which means it cannot fail to find something it should
have.

The quarantine
--------------
Every statistic still declares two things, and both exist because of a specific
way this analysis has been wrong before.

``duration_free``
    False for anything that grows with episode length on its own. A raw count
    rises with duration whether or not anything is wrong, so an analysis run on
    counts calls long episodes defective when they are merely long.

``outcome_derived``
    True for anything computed from the outcome rather than from behaviour. A
    truncated failed episode is short *because* it failed; "short episodes fail"
    is not a lesson about data collection.

Only statistics that are duration-free and not outcome-derived may become advice
someone spends a collection budget on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from gantry.resolve import Requirement, requires_channels
from gantry.spine import ChannelSpec, EpisodeRecord

#: The roles a statistic may read. These are the names channels arrive under
#: once the resolver has bound them, not names anything has to be called on disk.
POSITION, ACTUATION, COMMAND = "position", "actuation", "command"

#: What the module asks for. Widths are wildcards because these statistics are
#: defined in any dimension, and every role is optional: a dataset without a
#: gripper signal should lose the gripper statistics, not the whole analysis.
WANTS: tuple[ChannelSpec, ...] = (
    ChannelSpec(POSITION, "vector", (None,), "float32", optional=True),
    ChannelSpec(ACTUATION, "scalar", (), "float32", optional=True),
    ChannelSpec(COMMAND, "vector", (None,), "float32", optional=True),
)

#: Channel names commonly meaning each role, used when the caller says nothing.
#: A convenience and not an inference: whatever binds is reported in the run's
#: wiring, so a wrong default is visible rather than silent. Override it when
#: your channels are called something else.
DEFAULT_ALIASES: Mapping[str, tuple[str, ...]] = {
    POSITION: ("position", "eef_pos", "observation.position"),
    ACTUATION: ("actuation", "engagement", "gripper"),
    COMMAND: ("command", "action", "actions"),
}


def requirement(
    name: str, aliases: Mapping[str, Sequence[str]] | None = None, **kwargs
) -> Requirement:
    """The channels every statistic here reads, as a bindable requirement."""
    merged = {role: tuple(names) for role, names in DEFAULT_ALIASES.items()}
    for role, names in (aliases or {}).items():
        merged[role] = tuple(names) + merged.get(role, ())
    return requires_channels(name, "feedback", *WANTS, aliases=merged, **kwargs)


HIGHER, LOWER = "higher_is_better", "lower_is_better"


@dataclass(frozen=True)
class Statistic:
    name: str
    description: str
    compute: Callable[[EpisodeRecord], float | None]
    duration_free: bool
    outcome_derived: bool = False
    direction: str | None = None

    @property
    def prescribable(self) -> bool:
        """May a finding on this statistic become advice?"""
        return self.duration_free and not self.outcome_derived

    def __str__(self) -> str:
        return self.name


_STATISTICS: dict[str, Statistic] = {}


def register_statistic(statistic: Statistic) -> Statistic:
    existing = _STATISTICS.get(statistic.name)
    if existing is not None and existing != statistic:
        raise ValueError(f"statistic {statistic.name!r} already registered differently")
    _STATISTICS[statistic.name] = statistic
    return statistic


def known_statistics() -> tuple[Statistic, ...]:
    return tuple(_STATISTICS[name] for name in sorted(_STATISTICS))


def prescribable_statistics() -> tuple[Statistic, ...]:
    return tuple(s for s in known_statistics() if s.prescribable)


def get_statistic(name: str) -> Statistic:
    try:
        return _STATISTICS[name]
    except KeyError:
        raise KeyError(
            f"unknown statistic {name!r}; known: {', '.join(sorted(_STATISTICS))}"
        ) from None


# --------------------------------------------------------------------------
# reading a role
# --------------------------------------------------------------------------


def role(episode: EpisodeRecord, name: str) -> np.ndarray | None:
    """The channel bound to a role, or ``None`` if nothing was."""
    return episode.array(name) if episode.has(name) else None


def channel_named(episode: EpisodeRecord, semantics: str, kind: str | None = None) -> str | None:
    """Find a channel by declared meaning.

    Kept for callers analysing episodes that never went through the resolver.
    Nothing in this module uses it any more, and nothing new should: searching
    for a tag is how a correctly described dataset ends up unreadable.
    """
    for spec in episode.schema:
        if spec.semantics == semantics and (kind is None or spec.kind == kind):
            return spec.name
    return None


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------


def _path_efficiency(episode: EpisodeRecord) -> float | None:
    position = role(episode, POSITION)
    if position is None or len(position) < 2:
        return None
    travelled = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    if travelled < 1e-12:
        return None
    return float(np.linalg.norm(position[-1] - position[0]) / travelled)


def _actuation_transition_rate(episode: EpisodeRecord) -> float | None:
    signal = role(episode, ACTUATION)
    if signal is None or len(signal) < 2:
        return None
    engaged = signal > (float(np.min(signal)) + float(np.max(signal))) / 2.0
    return float(np.count_nonzero(np.diff(engaged)) / (len(signal) - 1))


def _actuation_jerk(episode: EpisodeRecord) -> float | None:
    command = role(episode, COMMAND)
    if command is None or len(command) < 2:
        return None
    return float(np.abs(np.diff(command, axis=0)).mean())


def _actuation_onset(episode: EpisodeRecord) -> float | None:
    signal = role(episode, ACTUATION)
    if signal is None or len(signal) == 0:
        return None
    engaged = signal > (float(np.min(signal)) + float(np.max(signal))) / 2.0
    if not engaged.any():
        return 1.0
    return float(int(np.argmax(engaged)) / len(engaged))


def _direction_change_rate(episode: EpisodeRecord) -> float | None:
    position = role(episode, POSITION)
    if position is None or len(position) < 3:
        return None
    velocity = np.diff(position, axis=0)
    return float(np.count_nonzero(np.diff(np.sign(velocity), axis=0)) / (len(position) - 2))


def _direction_change_count(episode: EpisodeRecord) -> float | None:
    position = role(episode, POSITION)
    if position is None or len(position) < 3:
        return None
    velocity = np.diff(position, axis=0)
    return float(np.count_nonzero(np.diff(np.sign(velocity), axis=0)))


def _command_magnitude(episode: EpisodeRecord) -> float | None:
    command = role(episode, COMMAND)
    if command is None or len(command) < 1:
        return None
    return float(np.linalg.norm(command, axis=1).mean())


def _episode_length(episode: EpisodeRecord) -> float:
    return float(len(episode))


def _stages_reached(episode: EpisodeRecord) -> float:
    return float(len(episode.labels.stage_events))


for _statistic in [
    Statistic(
        "path_efficiency",
        "net displacement over distance travelled",
        _path_efficiency,
        duration_free=True,
        direction=HIGHER,
    ),
    Statistic(
        "actuation_transition_rate",
        "actuator state changes per step",
        _actuation_transition_rate,
        duration_free=True,
        direction=LOWER,
    ),
    Statistic(
        "actuation_jerk",
        "mean absolute step-to-step change in command",
        _actuation_jerk,
        duration_free=True,
        direction=LOWER,
    ),
    Statistic(
        "actuation_onset",
        "when the actuator first engaged, as a fraction of the episode",
        _actuation_onset,
        duration_free=True,
        direction=None,
    ),
    Statistic(
        "direction_change_rate",
        "velocity sign changes per step",
        _direction_change_rate,
        duration_free=True,
        direction=LOWER,
    ),
    Statistic(
        "command_magnitude",
        "mean size of the commanded step",
        _command_magnitude,
        duration_free=True,
        direction=None,
    ),
    # Deliberate canaries. Both are legitimate context and neither may ever
    # become advice: the first grows with duration by construction, the second
    # is the duration.
    Statistic(
        "direction_change_count",
        "total velocity sign changes (grows with duration)",
        _direction_change_count,
        duration_free=False,
        direction=None,
    ),
    Statistic(
        "episode_length",
        "number of steps",
        _episode_length,
        duration_free=False,
        direction=None,
    ),
    Statistic(
        "stages_reached",
        "how many milestones the episode reached",
        _stages_reached,
        duration_free=True,
        outcome_derived=True,
        direction=HIGHER,
    ),
]:
    register_statistic(_statistic)


# --------------------------------------------------------------------------
# tabulation
# --------------------------------------------------------------------------


def tabulate(
    episodes: Sequence[EpisodeRecord],
    statistics: Sequence[Statistic] | None = None,
) -> tuple[dict[str, list[float]], tuple[str, ...]]:
    """Compute every statistic over every episode.

    Returns the columns that could be computed, plus the names of those that
    could not, so a caller reports "unavailable" rather than silently analysing
    a smaller set of statistics than it thinks it has.
    """
    chosen = tuple(statistics or known_statistics())
    columns: dict[str, list[float]] = {}
    unavailable: list[str] = []
    for statistic in chosen:
        values = [statistic.compute(episode) for episode in episodes]
        usable = [value for value in values if value is not None]
        if len(usable) != len(values) or not usable:
            unavailable.append(statistic.name)
            continue
        columns[statistic.name] = usable
    return columns, tuple(unavailable)


def outcomes_of(episodes: Sequence[EpisodeRecord]) -> tuple[list[int], int]:
    """Successes as 0/1, and how many episodes carried no label at all."""
    labelled = [e.labels.success for e in episodes]
    unlabelled = sum(1 for value in labelled if value is None)
    return [int(bool(value)) for value in labelled if value is not None], unlabelled
