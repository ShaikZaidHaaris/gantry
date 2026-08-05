"""Failure taxonomy, split by who has to respond.

The usual axis for exceptions is severity, which is the wrong one here. What a
runner needs to know is *whether it may carry on*, and that is a different
question from how bad the failure was. A misconfigured manifest and a stuck
actuator are both fatal to the current run, but only one of them means a person
has to walk over to the rig before anything else happens.

So the classes below carry two class-level facts:

``halts``
    The run stops, and no policy setting overrides it. Reserved for states
    where continuing means acting in a world that is no longer understood: a
    faulted machine, a vetoed motion. A framework that lets these be swallowed
    by ``continue_on_error`` will, at three in the morning, advance to the next
    scene with a jammed arm and record the result as data.

``recoverable``
    The failure belongs to one unit of work. The runner records it, marks that
    unit failed, and may proceed depending on its own error policy.

This split is taken from the Inspect Robots error taxonomy, which arrived at it
from running real hardware unattended. It is one of those distinctions that
only shows up after something expensive happens, so it is worth inheriting
rather than rediscovering.
"""

from __future__ import annotations

from typing import Any, ClassVar


class GantryError(Exception):
    """Base for every Gantry failure.

    ``partial`` carries whatever the runner had assembled when the failure
    happened -- the steps taken, the episodes finished. A run that dies halfway
    is still evidence, and discarding it because an exception was raised loses
    the most informative part of the failure.
    """

    #: Always stops the run, whatever the caller's error policy says.
    halts: ClassVar[bool] = True
    #: May be recorded as one failed unit while the run continues.
    recoverable: ClassVar[bool] = False

    def __init__(self, message: str, partial: Any = None):
        super().__init__(message)
        self.partial = partial


class ConfigError(GantryError):
    """A manifest, config, or argument is wrong. Fails before anything runs."""


class ComponentError(GantryError):
    """A plugin raised while doing its job.

    Recoverable: one episode, one rollout, one analysis failed. The unit is
    marked failed and the runner decides whether to go on. This is the ordinary
    case of a policy that threw or a connector that hit a corrupt record.
    """

    halts = False
    recoverable = True


class FaultError(GantryError):
    """The machine or world is in a state nobody has confirmed is safe.

    Always halts. A faulted rig that keeps being handed scenes produces
    plausible-looking numbers describing nothing.
    """


class SafetyAbort(GantryError):
    """A guard refused an action, or a human stopped the run. Always halts."""


def disposition(error: BaseException) -> str:
    """``"halt"``, ``"record"``, or ``"unknown"`` for any exception.

    Anything that is not a :class:`GantryError` is ``"unknown"`` and must be
    treated as halting. A failure the framework cannot classify is not a
    failure it may decide to continue through.
    """
    if not isinstance(error, GantryError):
        return "unknown"
    if error.halts:
        return "halt"
    return "record" if error.recoverable else "halt"


def must_halt(error: BaseException) -> bool:
    """Whether this failure ends the run no matter what the caller prefers."""
    return disposition(error) != "record"
