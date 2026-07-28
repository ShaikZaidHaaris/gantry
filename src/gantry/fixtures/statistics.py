"""Minimal statistics, existing only so a fixture can prove its own claims.

These are **not** the feedback plane. They are a deliberately independent,
deliberately naive reimplementation, kept here so that
:meth:`~gantry.fixtures.truth.FixtureSuite.verify` can check a generated suite
really contains the defect it advertises.

Independence is the point. If the fixture were verified by the same code under
test, a bug shared by both would make a broken detector look correct on a
broken fixture, and nothing would ever fail. Feedback modules are plugins and
share not one line with this file.

The pair ``engagement_transitions`` / ``engagement_transition_rate`` is here for
a specific reason: a raw count grows with episode length whether or not
anything is wrong, so an analysis run on counts calls long episodes defective
when they are merely long. The rate does not. Both are provided so a fixture
can demonstrate the trap and the fix.
"""

from __future__ import annotations

import numpy as np

from ..spine import EpisodeRecord

POSITION = "position"
ENGAGEMENT = "engagement"
ACTION = "action"


def path_efficiency(episode: EpisodeRecord) -> float:
    """Net displacement over distance travelled. 1.0 is a straight line."""
    position = episode.array(POSITION)
    if len(position) < 2:
        return 0.0
    travelled = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
    if travelled < 1e-12:
        return 0.0
    net = float(np.linalg.norm(position[-1] - position[0]))
    return net / travelled


def engagement_transitions(episode: EpisodeRecord) -> float:
    """How many times the engagement signal flipped. Scales with duration."""
    engaged = episode.array(ENGAGEMENT) > 0.5
    return float(np.count_nonzero(np.diff(engaged)))


def engagement_transition_rate(episode: EpisodeRecord) -> float:
    """Flips per step. The duration-free version of the above."""
    return engagement_transitions(episode) / max(len(episode) - 1, 1)


def direction_changes(episode: EpisodeRecord) -> float:
    """How often velocity changed sign on any axis.

    Healthy jitter, not a flaw: any real trajectory wobbles, and a longer one
    wobbles more times for exactly the same quality of motion. That is what
    makes this the honest bait for the duration trap.
    """
    position = episode.array(POSITION)
    if len(position) < 3:
        return 0.0
    velocity = np.diff(position, axis=0)
    return float(np.count_nonzero(np.diff(np.sign(velocity), axis=0)))


def direction_change_rate(episode: EpisodeRecord) -> float:
    """Direction changes per step. The duration-free version of the above."""
    return direction_changes(episode) / max(len(episode) - 2, 1)


def action_jerk(episode: EpisodeRecord) -> float:
    """Mean absolute step-to-step change in the action."""
    action = episode.array(ACTION)
    if len(action) < 2:
        return 0.0
    return float(np.abs(np.diff(action, axis=0)).mean())


def engagement_onset(episode: EpisodeRecord) -> float:
    """When engagement first happened, as a fraction of the episode."""
    engaged = episode.array(ENGAGEMENT) > 0.5
    if not engaged.any():
        return 1.0
    return float(int(np.argmax(engaged)) / max(len(engaged), 1))


def completion(episode: EpisodeRecord) -> float:
    return 1.0 if episode.labels.success else 0.0


def duration(episode: EpisodeRecord) -> float:
    return float(len(episode))


def correlation(left: list[float], right: list[float]) -> float:
    """Pearson r, or 0.0 when either side is constant."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
