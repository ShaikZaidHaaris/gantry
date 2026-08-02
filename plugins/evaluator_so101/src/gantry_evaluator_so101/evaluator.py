"""Plane 4, backend family 2: a physical leader-follower SO-ARM101 pair.

One rig, two jobs, one loop
---------------------------
This evaluator does the two things a hardware station is for:

**Evaluation.** A policy drives the follower at the pair's control rate, a person
resets the scene between trials and says what happened, and the result is a
:class:`~gantry.spine.run.RunRecord` shaped exactly like a simulator's.

**Recording.** The leader drives the follower and the demonstrations become
EpisodeRecords for the corpus factory.

These are not two code paths. ARCHITECTURE.md §2 says a rollout is just an
EpisodeRecord that a policy generated instead of a human, so the human is
modelled as a policy: :class:`LeaderTeleop` implements the plane-3 contract and
returns the leader's current pose as a one-action chunk. ``record()`` is then
literally ``run(LeaderTeleop(pair), task, protocol)``. Everything the evaluation
path gets — the safety clamp, the tracking-error measurement, the calibration
stamp, the rate audit — the corpus gets too, because there is nothing else to
get it from.

The three declarations that matter
----------------------------------
``closed_loop = True`` and ``outcomes = True`` are easy: the world responds, and
a person at the rig can always say what happened.

``seedable = False``, permanently. The contract's wording is that a seedable
evaluator "returns identical records for identical seeds, which is what a paired
comparison rests on". A physical rig cannot do that. The object is placed by a
hand, and that placement carries its own error; nothing downstream can subtract
it. Declaring ``True`` here would not produce a wrong number, it would let the
resolver *plan a paired analysis that is not valid* — and the paired-seed power
calculation in CORPUS-PROTOCOL.md (n=127 unpaired collapsing to n=76 paired) is
computed on exactly that assumption. This single boolean is what stops a
sim-tier statistical design from being applied to hardware without anyone
noticing. Scenes may still carry a seed: it names which pre-registered start
pose the operator should *draw*, which is an instruction to a human, not a
guarantee about a machine.

``stage_events = False`` unless a pose source is explicitly configured. A funnel
needs to know where the object is at every step. On this rig nothing does: six
STS3215 servos report position and nothing else, there is no force sensing
anywhere, and the perception rig that would supply object pose does not exist
yet. The gripper-stall proxy can suggest contact, but contact is one rung;
lifted and placed need the object. An evaluator that claimed stage events it
cannot emit would hand the funnel a table with one rung at a hundred percent,
which reads as a finished analysis. So the default is a flat no, and
``pose_source=`` is the only thing that changes it.

What this refuses to do
-----------------------
* Run without a per-tick motion limit. LeRobot's ``max_relative_target``
  defaults to ``None``, which is no clamp at all; an unclamped bad action drives
  the follower into the table at full servo speed. Passing ``None`` here is
  refused by name.
* Run without both arms' calibration hashes. LeRobot issue #3758 is a ~17°
  calibration offset that produced ~6 cm of gripper error and was invisible
  during training — the policy learns the biased mapping and the loss looks
  healthy. The hashes go into the embodiment's ``artifact_digest``, so a run
  recorded before a recalibration is not comparable to one recorded after it,
  structurally, without anyone having to remember.
* Guess which end of the gripper's range is closed. That depends on the
  calibration, and a guess produces a contact label that is exactly backwards.
* Pad, truncate, or reinterpret an action vector. Six values in the LeRobot
  order or nothing.

The siblings this is written against
------------------------------------
``SO101Pair`` (``.arm``) is the machine, and this module calls exactly:

===============================  ==========================================
``pair.joints``                  ``tuple[str, ...]``, the six LeRobot names
``pair.control_hz``              ``float``, the rate the pair is driven at
``pair.calibration``             ``{"leader": sha256, "follower": sha256}``
``pair.read_follower()``         ``float32[6]``, measured, RANGE_0_100
``pair.read_leader()``           ``float32[6]``, measured, RANGE_0_100
``pair.write_follower(target)``  commands ``float32[6]``, returns what it
                                 actually sent — the arm's own guards may
                                 tighten it, and the record carries that value
``pair.connect()``               idempotent
``pair.disconnect()``            idempotent
===============================  ==========================================

``MockTransport`` (``.transport``) is a serial-bus stand-in with no hardware
behind it, so every line below runs in CI. :func:`mock_pair` wires two of them
into a pair; that is the constructor used by this plugin's own tests.

One description of the arm
--------------------------
``.embodiment`` is plane 2 and it is the only authority on what this machine
is: the joint order, the normalization, the calibration variant, the frame tag,
the meaning tags, which metadata keys are load-bearing, and the per-tick clamp.
Nothing below restates any of them. This module used to, and the two halves of
one package ended up describing two different machines — an
``embodiment:so101-follower@1.0`` that no installed plugin provided, joints in
frame ``base`` instead of ``so101_follower_joints``, and a clamp applied at 5
while the descriptor advertised 8.

Two things went wrong and only one of them is obvious. Every RunRecord carried
an embodiment ref the resolver cannot bind, so a corpus recorded here would not
bind to this package's own descriptor. The subtler one: the two halves spelled
the discriminator ``normalisation`` and ``normalization``, so each key appeared
on one side only, and :func:`~gantry.spine.compatible` can only report a
one-sided key as a ``metadata.undeclared`` **note**. The single mechanism built
to make a normalization mismatch a hard refusal was inert, and it was inert
because of a spelling. :func:`_agree` runs at import and refuses the package if
the two descriptions ever drift apart again — including on that note, which is
why it is not enough for the verdict to be ``ok``.
"""

from __future__ import annotations

import math
import socket
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from gantry.contracts.embodiment import EmbodimentDescriptor
from gantry.contracts.evaluator import (
    Evaluator,
    Scene,
    TaskSpec,
    evaluator_descriptor,
)
from gantry.contracts.evaluator import Protocol as RunProtocol
from gantry.contracts.policy import EpisodeContext, Observation, Policy, policy_descriptor
from gantry.errors import ComponentError, ConfigError, FaultError
from gantry.resolve import Requirement, requires_channels
from gantry.spine import (
    AdapterStep,
    ChannelSpec,
    Descriptor,
    EpisodeLabels,
    EpisodeRecord,
    Measurement,
    Provenance,
    RunRecord,
    StageEvent,
    compatible,
    digest_of,
    episode_from_arrays,
    episode_from_labels,
)

from .arm import SO101Pair, mock_arm
from .embodiment import ACTION as _EMBODIMENT_ACTION
from .embodiment import (
    CALIBRATION,
    CONTROL_HZ,
    FOLLOWER_JOINT_FRAME,
    GRIPPER_INDEX,
    JOINT_NAMES,
    NORMALIZATION,
    NORMALIZATION_RANGES,
    SEMANTICS_ACTION,
    joint_channel,
    so101_embodiment,
)
from .embodiment import DEFAULT_MAX_RELATIVE_TARGET as _EMBODIMENT_CLAMP
from .embodiment import STATE as _EMBODIMENT_STATE

VERSION = "0.1.0.dev0"

#: The six degrees of freedom, in the order LeRobot writes ``action`` and
#: ``observation.state`` for an SO-101. The order is the whole content of the
#: vector: two arms with these names permuted are compatible on every other axis
#: and produce garbage, which is why it is carried as ``dim_labels`` rather than
#: as a comment. Re-exported from ``.embodiment`` rather than spelled again —
#: a second copy of a joint order is a permutation waiting to be typed.
JOINTS: tuple[str, ...] = JOINT_NAMES

#: The gripper is the sixth DoF, not a separate binary channel.
GRIPPER = GRIPPER_INDEX

#: LeRobot's ``RANGE_0_100`` normalization: the ends of the calibrated travel,
#: read off ``.embodiment``'s table rather than written out here. Not radians —
#: these servos are commanded as a percentage of their calibrated travel, and
#: the mapping from percent to angle lives in the calibration file whose hash
#: this evaluator refuses to run without. Derived rather than pinned because
#: :meth:`SO101Evaluator._limit` bounds every command to these two numbers: a
#: rig recorded in ``RANGE_M100_100`` against a literal ``(0.0, 100.0)`` here
#: would clip away the whole negative half of every joint's travel and record
#: it as a policy that never went there.
RANGE: tuple[float, float] = NORMALIZATION_RANGES[NORMALIZATION]

#: The calibration variant these numbers are only meaningful under.
CALIBRATION_VARIANT = CALIBRATION

#: Per-tick motion limit, in units of the 0..100 range. The number, its units
#: and the argument for it belong to ``.embodiment``, which also publishes it on
#: the action channel and in the descriptor's ``safety`` metadata — so this is
#: the same object, not a second opinion. It matters that there is only one:
#: this module previously clamped at 5.0 while the descriptor beside it
#: advertised 8.0, which is a record describing a rig nobody ran. ``None`` is
#: still not one of the options; see :class:`SO101Evaluator`.
DEFAULT_MAX_RELATIVE_TARGET = _EMBODIMENT_CLAMP

#: Twenty seconds at 30 Hz. CORPUS-PROTOCOL.md's equal-horizon rule: giving a
#: sloppy corpus a longer cap is not a controlled experiment.
DEFAULT_HORIZON = 600

#: How far a commanded value may come back changed before the change counts as a
#: limit rather than as the encoder.
#:
#: A command in the 0..100 convention is written to an integer register, so what
#: the follower reports commanding is the request rounded to the nearest tick.
#: An SO-101 joint's calibrated travel is a few hundred to a few thousand of the
#: STS3215's 4096 counts; the coarsest joint on this rig, the gripper, spans
#: about 800, which is 0.125 of the 0..100 range per tick. A quarter of a unit
#: therefore absorbs the rounding on any plausible joint and is still well below
#: any clamp anyone would set — the default per-tick limit is 8. Without a
#: threshold the comparison is a value against its own round trip through an
#: integer, and every step in every episode is reported as clamped.
QUANTISATION = 0.25

#: Channel names. LeRobot's, because the corpus this produces is trained from.
#: The two the descriptor also names come from it: the arm, the evaluator and
#: the descriptor key their dictionaries by these strings, and a second spelling
#: in one of them is how an observation quietly stops being recorded.
STATE = _EMBODIMENT_STATE
ACTION = _EMBODIMENT_ACTION
REQUESTED = "action.requested"
ELAPSED = "elapsed"

#: The embodiment ref every record from this rig is stamped with, pinned here
#: and checked against :func:`~.embodiment.so101_embodiment` at import.
#:
#: This string is the only thing that binds a RunRecord back to the machine that
#: produced it: the resolver looks up the installed plugin that *provides* the
#: ref, so a corpus recorded under a ref no plugin publishes cannot be bound to
#: any description of any arm — which is the state this module shipped in, with
#: ``so101-follower`` here and ``so101_follower`` on the entry point. Renaming
#: or version-bumping the machine in plane 2 is therefore a decision about every
#: corpus ever recorded against it, and pinning the string makes that decision
#: fail loudly at import instead of quietly at the next resolve.
EMBODIMENT_REF = "embodiment:so101_follower@1.0"


def _machine(
    rate_hz: float = CONTROL_HZ,
    max_relative_target: float = DEFAULT_MAX_RELATIVE_TARGET,
) -> EmbodimentDescriptor:
    """The plane-2 descriptor, at the rate this pair is actually driven at.

    ``.embodiment`` is the authority on what this arm is. The one thing the
    evaluator knows that it does not is the pair's measured control rate, which
    is read off the hardware, so the descriptor is built there and only the rate
    is replaced here. Writing a second descriptor instead is precisely how this
    module came to publish an ``embodiment:so101-follower@1.0`` whose channels
    sat in frame ``base`` — same arm, same numbers, two descriptions, and every
    structural check passed on each of them separately.
    """
    machine = so101_embodiment(max_relative_target=max_relative_target)
    if rate_hz == machine.control_hz:
        return machine
    return replace(
        machine,
        control_hz=rate_hz,
        state=tuple(replace(spec, rate_hz=rate_hz) for spec in machine.state),
        action=tuple(replace(spec, rate_hz=rate_hz) for spec in machine.action),
    )


def action_spec(
    rate_hz: float = CONTROL_HZ,
    max_relative_target: float = DEFAULT_MAX_RELATIVE_TARGET,
) -> ChannelSpec:
    """What a policy must emit to drive this follower.

    Not a description of the action channel — it *is* the action channel the
    descriptor publishes, at this rig's rate. A policy needs something to
    declare against, and handing it anything other than the machine's own spec
    means the pair that binds in the resolver is not the pair that runs.
    """
    return _machine(rate_hz, max_relative_target).action[0]


def state_spec(rate_hz: float = CONTROL_HZ) -> ChannelSpec:
    """What the follower reports back: the descriptor's own state channel.

    ``state[0]`` is the joint vector; the camera streams follow it and are not
    what an observation of the arm's own position is.
    """
    return _machine(rate_hz).state[0]


def _requested_spec(rate_hz: float) -> ChannelSpec:
    """What the policy asked for, kept beside what the arm was told.

    Built through ``.embodiment``'s own channel builder rather than by hand, so
    its frame, its meaning tag and its discriminator *keys* are the descriptor's
    and cannot drift from them. Hand-building a companion channel is exactly how
    ``normalisation`` came to sit next to ``normalization``: each key then
    appeared on one side only, and :func:`~gantry.spine.compatible` can only
    report a one-sided key as a ``metadata.undeclared`` note. A mismatch that
    was supposed to be a hard refusal became a remark nobody reads.
    """
    return joint_channel(
        REQUESTED,
        frame=FOLLOWER_JOINT_FRAME,
        semantics=SEMANTICS_ACTION,
        rate_hz=rate_hz,
        # Pre-clamp, said in metadata rather than left to the channel name: this
        # column and ``action`` differ only in whether the safety limit has been
        # applied, and a reader holding one of them without knowing which cannot
        # tell a docile policy from a clamped one.
        extra={"clamp_applied": False},
    )


def _elapsed_spec(rate_hz: float) -> ChannelSpec:
    """Seconds since the trial started.

    The one channel ``.embodiment`` has no opinion on and cannot be asked for:
    its builders describe six-wide joint vectors and camera streams, and a
    duration is neither. So it is built here, with no frame — a duration is not
    expressed in one — and no discriminators, because there is nothing about a
    second that two rigs could disagree on and still both be right.
    """
    return ChannelSpec(
        ELAPSED,
        "scalar",
        (),
        "float32",
        units="s",
        semantics="time",
        rate_hz=rate_hz,
    )


def _agree(
    *,
    machine: EmbodimentDescriptor | None = None,
    action: ChannelSpec | None = None,
    state: ChannelSpec | None = None,
    clamp: float = DEFAULT_MAX_RELATIVE_TARGET,
    ref: str = EMBODIMENT_REF,
) -> None:
    """Refuse the package if its two descriptions of the follower differ.

    The package-level ``_agree`` in ``__init__`` compares three scalar constants
    across three modules — joint order, normalization, calibration variant — and
    caught none of what was actually wrong here, because the disagreement was in
    the parts nobody compared: the ref, the frame, the meaning tags, the
    discriminator key tuple and the clamp. This one compares the two things this
    package publishes about one arm: the channels the evaluator hands a policy
    and stamps into every episode, and the descriptor the entry point provides.

    ``ok`` is not the bar. :func:`~gantry.spine.compatible` downgrades a
    discriminator key it can see on one side only to a ``metadata.undeclared``
    *note*, so a verdict can be ``ok`` while the one check that would have
    caught a normalization mismatch is silently not running. A note on a
    load-bearing key is treated here as the refusal it could not be there.

    Every input is an argument so the guard can be perturbed and watched to
    fail; a guard nobody has seen refuse is not known to refuse.
    """
    # The descriptor is built at *its own* default clamp, not at this module's.
    # Passing the evaluator's number in would make the clamp comparison below
    # the evaluator's number against itself, which is the shape of check that
    # let a 5.0 here sit beside an 8.0 there for as long as it did.
    machine = so101_embodiment() if machine is None else machine
    action = action_spec() if action is None else action
    state = state_spec() if state is None else state
    if not machine.action or not machine.state:
        raise ConfigError(
            f"the SO-101 descriptor {machine.ref} declares no action or no state "
            "channel, so there is nothing for this evaluator to agree with; the "
            "follower is the arm that is commanded and scored"
        )
    published, observed = machine.action[0], machine.state[0]
    declared = published.metadata.get("max_relative_target")

    complaints = [
        f"{what}: evaluator says {mine!r}, descriptor says {theirs!r}"
        for what, mine, theirs in (
            ("the embodiment ref stamped on every record", ref, machine.ref),
            ("the action channel's frame", action.frame, published.frame),
            ("the state channel's frame", state.frame, observed.frame),
            ("the action channel's semantics", action.semantics, published.semantics),
            ("the state channel's semantics", state.semantics, observed.semantics),
            (
                "the action channel's discriminator keys",
                action.discriminators,
                published.discriminators,
            ),
            (
                "the state channel's discriminator keys",
                state.discriminators,
                observed.discriminators,
            ),
            ("the joint order", JOINTS, published.dim_labels),
            ("the per-tick clamp", float(clamp), declared if declared is None else float(declared)),
        )
        if mine != theirs
    ]

    # The end-to-end question, asked the way the resolver asks it. Both legs run
    # even when the itemised comparisons above already found something, because
    # each answers a different question: those say which field differs, this
    # says whether a run recorded here binds to this package's own descriptor.
    for side, mine, theirs in (("action", action, published), ("state", state, observed)):
        verdict = compatible(mine, theirs)
        undeclared = verdict.because("metadata.undeclared")
        if verdict.ok and not undeclared:
            continue
        complaints.append(
            f"the evaluator's {side} channel does not bind to the descriptor's: "
            + "; ".join(str(reason) for reason in verdict.reasons)
        )

    if complaints:
        raise ConfigError(
            "gantry-evaluator-so101 describes two different machines. The evaluator "
            "and the embodiment module are one physical arm and every record this "
            "package emits is stamped with the descriptor's ref, so a disagreement "
            "here is a corpus that cannot be bound to the machine that produced it:\n"
            + "\n".join(f"  {complaint}" for complaint in complaints)
        )


_agree()


# --------------------------------------------------------------------------
# pacing
# --------------------------------------------------------------------------


class Clock(Protocol):
    """Wall time, or something that pretends to be it.

    Injected rather than called directly so the whole loop runs at full speed
    with no hardware and no waiting. A test that had to spend twenty real
    seconds per trial is a test nobody runs.
    """

    def now(self) -> float: ...

    def sleep_until(self, deadline: float) -> None: ...


class WallClock:
    """Real time. What the rig runs on."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - self.now()
        if remaining > 0:
            time.sleep(remaining)


class VirtualClock:
    """Time that jumps to every deadline. What CI runs on.

    Reports the *nominal* rate, so a record produced under it says 30 Hz because
    that is what was asked for, not because it was achieved. Nothing in a
    virtual-clock run is evidence about timing, and the record says so through
    ``paced=False``.
    """

    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def sleep_until(self, deadline: float) -> None:
        self._t = max(self._t, deadline)


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------


class Console(Protocol):
    """The person at the rig, as a callable surface.

    Three things only: tell them what to set up, let them say something went
    wrong, and ask them what happened. Everything expensive about real
    evaluation is on the far side of this interface.
    """

    def guide(self, scene: Scene, task: TaskSpec, trial: int, of: int) -> None: ...

    def outcome(self, scene: Scene, summary: Mapping[str, Any]) -> bool | None: ...

    def note(self, text: str) -> None: ...


class TerminalConsole:
    """A prompt at the rig. Blocks until the operator answers.

    ``y``/``n`` are the outcome; anything else voids the trial, which is a real
    answer — an operator who bumped the object mid-rollout should be able to say
    "this one does not count" rather than being forced into a label.
    """

    def __init__(self, prompt: Callable[[str], str] = input, show: Callable[[str], None] = print):
        self._prompt, self._show = prompt, show

    def guide(self, scene: Scene, task: TaskSpec, trial: int, of: int) -> None:
        # ASCII only in anything the operator sees: the rig's console is a
        # Windows terminal in cp1252 and a stray em-dash prints as a question
        # mark in the middle of a safety instruction.
        self._show(f"\n[{trial}/{of}] {task.name} - scene {scene.id}")
        if scene.setup:
            self._show(f"  set up: {scene.setup}")
        if scene.instruction:
            self._show(f"  instruction: {scene.instruction}")
        if scene.seed is not None:
            # The seed names a drawn start pose in the pre-registered list. It
            # does not make the trial reproducible; a hand placed the object.
            self._show(f"  draw start pose #{scene.seed} from the pre-registered list")
        self._prompt("  press enter when the scene is set ")

    def outcome(self, scene: Scene, summary: Mapping[str, Any]) -> bool | None:
        self._show(f"  {summary}")
        answer = self._prompt(f"  did {scene.id} succeed? [y/n/other=void] ").strip().lower()
        return True if answer == "y" else False if answer == "n" else None

    def note(self, text: str) -> None:
        self._show(f"  {text}")


class ScriptedConsole:
    """A rehearsed operator, for tests and dry runs.

    Outcomes are consumed in order and the script running out is a refusal, not
    a shrug: a run that silently voided its last trials would report a success
    rate over a denominator nobody chose.
    """

    def __init__(self, outcomes: Sequence[bool | None] = (), repeat: bool = False):
        self._outcomes = tuple(outcomes)
        self._repeat = repeat
        self.guided: list[str] = []
        self.notes: list[str] = []
        self._index = 0

    def guide(self, scene: Scene, task: TaskSpec, trial: int, of: int) -> None:
        self.guided.append(scene.id)

    def outcome(self, scene: Scene, summary: Mapping[str, Any]) -> bool | None:
        if not self._outcomes:
            raise ConfigError("ScriptedConsole was given no outcomes to report")
        if self._index >= len(self._outcomes):
            if not self._repeat:
                raise ConfigError(
                    f"ScriptedConsole ran out of outcomes at trial {self._index + 1}; "
                    f"it was given {len(self._outcomes)}"
                )
            self._index = 0
        answer = self._outcomes[self._index]
        self._index += 1
        return answer

    def note(self, text: str) -> None:
        self.notes.append(text)


# --------------------------------------------------------------------------
# what a pose source would be, if one existed
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One tick as a detector sees it."""

    step: int
    elapsed: float
    state: np.ndarray
    sent: np.ndarray
    requested: np.ndarray


@dataclass(frozen=True)
class PoseSource:
    """A perception rig that can say which milestones a trial reached.

    The only thing that turns ``stage_events`` on. ``names`` is the vocabulary
    the run was *looking for* and is recorded whether or not anything reaches
    them, because a funnel that inferred its rungs from what happened would read
    a policy failing at rung two as a one-rung funnel at a hundred percent.

    Nothing implements this yet, and that is the honest state of the hardware:
    object pose needs a camera rig with a stated error budget, and until one
    exists this evaluator reports outcomes and no funnel.
    """

    milestones: Callable[[Sample], Sequence[str]]
    names: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.names:
            raise ConfigError(
                "a pose source must name the milestones it looks for; without them a "
                "funnel infers its own rungs from whatever happened to be reached"
            )


def gripper_stall(
    closes_toward: float,
    *,
    margin: float = 3.0,
    motion: float = 0.5,
    ticks: int = 5,
) -> Callable[[Sequence[Sample]], int | None]:
    """Contact, inferred from a gripper that stopped following its command.

    These servos report position and nothing else. When the jaws meet an object
    the commanded position keeps pushing toward closed and the measured position
    stops changing, so contact is the conjunction of two things: a command
    beyond where the jaws are, *and* jaws that are no longer moving.

    Both halves are load-bearing. Tracking error alone is not a stall — every
    fast close produces one, because a position-controlled servo always trails
    its target — and a proxy built on error alone reports contact on the way
    down, before the jaws have touched anything. Requiring the measured position
    to be stuck as well is what separates "pushing against an object" from
    "moving quickly".

    ``closes_toward`` has no default. Which end of the 0..100 range is closed is
    decided by the calibration, and a wrong guess labels every *release* as a
    grasp. If you do not know, do not configure this.

    Returns the first step of a sustained stall, or ``None``. It is deliberately
    not a stage event: one rung is not a funnel, and reporting it as one would
    let the resolver plan a diagnosis this rig cannot support. It is also not a
    force reading, and nothing about it says how hard the jaws are pushing.
    """
    if closes_toward not in RANGE:
        raise ConfigError(
            f"closes_toward must be {RANGE[0]} or {RANGE[1]} (the ends of the "
            f"calibrated range), got {closes_toward}"
        )
    if ticks < 1:
        raise ConfigError(f"ticks must be at least 1, got {ticks}")

    def detect(samples: Sequence[Sample]) -> int | None:
        run = 0
        previous: float | None = None
        for sample in samples:
            command, measured = float(sample.sent[GRIPPER]), float(sample.state[GRIPPER])
            pushing = (
                command < measured - margin
                if closes_toward == RANGE[0]
                else command > measured + margin
            )
            stuck = previous is not None and abs(measured - previous) < motion
            if pushing and stuck:
                run += 1
                if run >= ticks:
                    return sample.step - ticks + 1
            else:
                run = 0
            previous = measured
        return None

    return detect


# --------------------------------------------------------------------------
# the human, wearing the policy contract
# --------------------------------------------------------------------------


class LeaderTeleop(Policy):
    """The operator at the leader arm, as a plane-3 policy.

    This is the whole trick that keeps recording and evaluation on one path. The
    leader's measured pose *is* the action the follower should take, a chunk of
    one, produced at the control rate — which is precisely what a policy does.
    So the corpus factory is not a second loop with its own clamp and its own
    bugs; it is this evaluator's ordinary run loop with a different action
    source.

    ``deterministic=False`` and ``stateful=True`` are not hedges. A person is
    both.
    """

    def __init__(self, pair: SO101Pair, operator: str, *, rate_hz: float = 30.0):
        if not operator:
            raise ConfigError(
                "a teleoperated corpus must name its operator; the operator is the "
                "independent variable in half of CORPUS-PROTOCOL.md's contrasts"
            )
        self._pair = pair
        self._operator = operator
        self._rate_hz = rate_hz

    def descriptor(self) -> Descriptor:
        return policy_descriptor(
            "human-teleop",
            VERSION,
            chunk=1,
            deterministic=False,
            stateful=True,
            operator=self._operator,
            via="so101-leader",
        )

    def action_spec(self) -> ChannelSpec:
        return action_spec(self._rate_hz)

    def observes(self) -> Requirement:
        # A person looks at the workspace, not at the record. Nothing from the
        # observation dict is read, and saying so keeps the resolver from
        # planning a camera it does not need.
        return requires_channels(
            "human-teleop",
            "policy",
            description="nothing: the operator observes the workspace directly",
        )

    def act(self, observation: Observation) -> np.ndarray:
        return np.asarray(self._pair.read_leader(), dtype="float32").reshape(1, len(JOINTS))


# --------------------------------------------------------------------------
# the run loop
# --------------------------------------------------------------------------


@dataclass
class _Trial:
    states: list[np.ndarray] = field(default_factory=list)
    sent: list[np.ndarray] = field(default_factory=list)
    requested: list[np.ndarray] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    events: list[StageEvent] = field(default_factory=list)
    reached: set[str] = field(default_factory=set)
    limited: int = 0
    out_of_range: int = 0
    overruns: int = 0
    calls: int = 0

    def record(self, sample: Sample) -> None:
        self.states.append(sample.state)
        self.sent.append(sample.sent)
        self.requested.append(sample.requested)
        self.elapsed.append(sample.elapsed)
        self.samples.append(sample)

    @property
    def tracking_error(self) -> np.ndarray:
        """Per-joint |commanded - achieved|, one row per step.

        Cheap, always available, and the one number that tells an operator the
        follower is not keeping up with the leader — which CORPUS-PROTOCOL.md
        requires per episode for exactly that reason.
        """
        if not self.sent:
            return np.zeros((0, len(JOINTS)), dtype="float32")
        # Step k's command is answered by step k+1's measurement; comparing a
        # command against the state that preceded it would measure the command.
        sent = np.asarray(self.sent[:-1], dtype="float32")
        reached = np.asarray(self.states[1:], dtype="float32")
        return np.abs(sent - reached)


class SO101Evaluator(Evaluator):
    """Runs a policy, or a person, on a physical SO-ARM101 follower."""

    def __init__(
        self,
        pair: SO101Pair,
        *,
        console: Console | None = None,
        name: str = "so101",
        task: str = "so101",
        max_relative_target: float | None = DEFAULT_MAX_RELATIVE_TARGET,
        pose_source: PoseSource | None = None,
        contact: Callable[[Sequence[Sample]], int | None] | None = None,
        clock: Clock | None = None,
        scenes: int = 10,
        horizon: int = DEFAULT_HORIZON,
        setup: str | None = None,
        instruction: str | None = None,
        session: Mapping[str, Any] | None = None,
    ):
        """``pose_source`` is the only argument that changes a capability.

        ``max_relative_target`` is in units of the 0..100 range, per tick.
        ``None`` is LeRobot's own default and is refused here: it means no
        limit, and no limit means one bad action reaches the table.

        ``contact`` is the gripper-stall proxy, off unless configured, because
        it needs to know which end of the range is closed and this code does
        not. What it finds is an annotation, never a stage event.

        ``session`` is campaign metadata — corpus id, condition label, session
        and block ids, the RNG seed that produced the block order — carried
        verbatim onto every episode. It is not parsed, because the schema
        belongs to the campaign, not to the rig.

        There is no ``kinematics`` argument. The URDF reference and the list of
        its known defects are declared together in ``.embodiment``, and letting
        the evaluator swap in a different model would put a reference in the
        record whose defects are not the ones recorded beside it — a reader
        would check the gripper against a model that has no gripper joint and
        be told, correctly for the wrong file, that nothing is missing.
        """
        if max_relative_target is None:
            raise ConfigError(
                "max_relative_target=None is LeRobot's own default and means no clamp "
                "at all: a single bad action drives the follower into the table at full "
                "servo speed. Pass a per-tick limit in units of the 0..100 range "
                f"(the default is {DEFAULT_MAX_RELATIVE_TARGET})."
            )
        if max_relative_target <= 0:
            raise ConfigError(
                f"max_relative_target={max_relative_target} permits no motion at all"
            )
        # Infinity and NaN pass every `<= 0` test and are not clamps. Infinity in
        # particular is the dangerous one: it produces a run in which the limit
        # never binds, so `clamped_steps` is zero and the provenance declares no
        # loss — a record that reads exactly like a well-behaved policy on a
        # properly guarded rig. A limit at or above the full 0..100 travel is the
        # same thing more slowly: it permits the full-range slam in one tick that
        # this argument exists to prevent. ``transport.SafetyLimits`` already
        # refuses all of these, and the two layers have to agree about what
        # counts as a clamp.
        span = RANGE[1] - RANGE[0]
        if not math.isfinite(max_relative_target) or max_relative_target >= span:
            raise ConfigError(
                f"max_relative_target={max_relative_target} is not a clamp: a per-tick "
                f"limit that is infinite, NaN, or as large as the whole {span:g}-unit "
                "travel never binds, and a run under it reports zero clamped steps and "
                "no declared loss while the follower is free to cross its entire range "
                f"in one tick. Pass a real limit (the default is "
                f"{DEFAULT_MAX_RELATIVE_TARGET})."
            )
        if scenes < 1:
            raise ValueError(f"a task needs at least one scene, got {scenes}")
        if horizon < 1:
            raise ValueError(f"a horizon of {horizon} runs nothing")

        self._pair = pair
        self._rate_hz = _checked_pair(pair, name)
        self._console: Console = console or TerminalConsole()
        self._name = name
        self._task = task
        self._pose = pose_source
        self._contact = contact
        self._clock: Clock = clock or WallClock()
        self._scenes = scenes
        self._horizon = horizon
        self._setup = setup
        self._instruction = instruction
        self._session = dict(session or {})
        # One clamp, and the number applied is the number stamped. The
        # descriptor is built with the limit this rig was configured with, and
        # the limit every tick is bounded by is then read back out of the
        # channel that publishes it — so the two cannot be different values.
        # They were: 5.0 applied here, 8.0 advertised on the descriptor, which
        # is a record describing motion limits the arm never ran under.
        self._embodiment = _machine(self._rate_hz, float(max_relative_target))
        self._action = self._embodiment.action[0]
        self._state = self._embodiment.state[0]
        self._max_relative = float(self._action.metadata["max_relative_target"])
        self._requested = _requested_spec(self._rate_hz)
        self._elapsed = _elapsed_spec(self._rate_hz)
        self._connected = False

    # -- contract ----------------------------------------------------------

    def descriptor(self) -> Descriptor:
        """Identity and the four capability claims.

        Three of them are constants of the hardware and one is a function of the
        configuration. See the module docstring for why each is what it is; the
        short version is that a physical rig cannot be seeded, and a rig with no
        perception cannot emit a funnel.
        """
        return evaluator_descriptor(
            name=self._name,
            version=VERSION,
            closed_loop=True,
            outcomes=True,
            seedable=False,
            stage_events=self._pose is not None,
            embodiment=self._embodiment.ref,
            control_hz=self._rate_hz,
            max_relative_target=self._max_relative,
            # Read off the channel rather than off a module constant, so what
            # the descriptor advertises is what the recorded numbers are
            # actually relative to. LeRobot #3758 is invisible in every other
            # field of this record.
            calibration_variant=self._action.metadata["calibration_variant"],
            stages=list(self._pose.names) if self._pose else [],
            paced=isinstance(self._clock, WallClock),
        )

    def requires(self) -> Requirement:
        return requires_channels(
            self._name,
            "evaluation",
            self._action,
            description=(
                f"absolute joint targets for {len(JOINTS)} DoF in LeRobot order, "
                f"{self._action.metadata['normalization']}, at {self._rate_hz:g} Hz"
            ),
        )

    def task_for(self, name: str = "", **options: Any) -> TaskSpec:
        """Scenes a person can actually set up.

        A real rig's scenes are instructions to an operator, so each carries the
        setup text and the index of the start pose to draw from the
        pre-registered list. The seed is that index and nothing more: it does not
        make the trial reproducible, and this evaluator does not claim it does.
        """
        unknown = set(options) - {"scenes", "horizon", "setup", "instruction", "seed_base"}
        if unknown:
            raise ConfigError(
                f"{self._name}.task_for got unknown option(s) {sorted(unknown)}; "
                "known: scenes, horizon, setup, instruction, seed_base"
            )
        count = int(options.get("scenes") or self._scenes)
        seed_base = int(options.get("seed_base") or 0)
        setup = options.get("setup") or self._setup
        instruction = options.get("instruction") or self._instruction
        return TaskSpec(
            name=name or self._task,
            scenes=tuple(
                Scene(
                    id=f"scene-{index:03d}",
                    instruction=instruction,
                    seed=seed_base + index,
                    setup=setup,
                    metadata={"placement": "drawn from the pre-registered list, set by hand"},
                )
                for index in range(count)
            ),
            horizon=int(options.get("horizon") or self._horizon),
            stages=self._pose.names if self._pose else (),
            metadata={
                "embodiment": self._embodiment.ref,
                "control_hz": self._rate_hz,
                "seconds": round(int(options.get("horizon") or self._horizon) / self._rate_hz, 3),
            },
        )

    # -- the two entry points, one loop ------------------------------------

    def run(self, policy: Any, task: TaskSpec, protocol: RunProtocol) -> RunRecord:
        """Evaluate ``policy`` on the follower and return one record.

        ``policy`` being :class:`LeaderTeleop` is what makes this the recording
        path as well; nothing below branches on it.
        """
        self.connect()
        trials = [
            (index, scene, epoch)
            for index, scene in enumerate(task.scenes)
            for epoch in range(protocol.epochs)
        ]
        if protocol.max_trials is not None:
            trials = trials[: protocol.max_trials]

        episodes: list[EpisodeRecord] = []
        failures = 0
        limited = 0
        out_of_range = 0
        for position, (index, scene, epoch) in enumerate(trials, start=1):
            self._console.guide(scene, task, position, len(trials))
            trial, error = self._one(policy, scene, task, protocol)
            failures += error is not None
            limited += trial.limited
            out_of_range += trial.out_of_range
            episodes.append(
                self._episode(trial, scene, epoch, position - 1, task, protocol, error)
            )

        return RunRecord(
            provenance=self._provenance(
                policy, task, protocol, failures, limited, out_of_range, len(trials)
            ),
            episodes=tuple(episodes),
            metrics=self._metrics(episodes),
        )

    def record(
        self,
        task: TaskSpec,
        protocol: RunProtocol | None = None,
        *,
        operator: str,
    ) -> RunRecord:
        """Drive the follower from the leader and keep what the human did.

        The corpus factory's entry point. It returns a RunRecord because a
        demonstration corpus is a run whose policy happened to be a person: the
        episodes inside it are EpisodeRecords, identical in type to a rollout's,
        and they arrive carrying the same calibration stamp, the same
        tracking-error measurement and the same protocol echo. Reach for
        ``.episodes`` when you want the corpus.
        """
        return self.evaluate(
            LeaderTeleop(self._pair, operator, rate_hz=self._rate_hz), task, protocol
        )

    # -- one trial ---------------------------------------------------------

    def _one(
        self, policy: Any, scene: Scene, task: TaskSpec, protocol: RunProtocol
    ) -> tuple[_Trial, str | None]:
        trial = _Trial()
        policy.reset(EpisodeContext(scene.id, scene.instruction, scene.seed))

        started = self._clock.now()
        period = 1.0 / self._rate_hz
        chunk: np.ndarray = np.zeros((0, len(JOINTS)), dtype="float32")
        cursor = 0
        error: str | None = None

        for step in range(task.horizon):
            state = self._read()
            if cursor >= len(chunk):
                try:
                    chunk = self._act(policy, step, state)
                except Exception as failure:  # noqa: BLE001 - one trial, not the run
                    error = f"{type(failure).__name__}: {failure}"
                    break
                trial.calls += 1
                cursor = 0
            requested = chunk[cursor]
            cursor += 1
            if protocol.execute is not None and cursor >= protocol.execute:
                # Re-plan after ``execute`` actions rather than exhausting the
                # chunk. The single cheapest lever on a result, so it is honoured
                # here and echoed into the record.
                chunk = chunk[:0]

            sent, limited, clipped = self._limit(requested, state)
            # What the arm reports it actually commanded, which is not always
            # what it was handed: the follower carries its own envelope check and
            # its own relative guard, and either may bind after this one has. The
            # record has to carry the value that reached the servos — a corpus
            # whose ``action`` column is a number the arm never executed teaches a
            # policy a mapping that was never demonstrated.
            executed = self._write(sent)
            limited = limited or bool(np.any(np.abs(executed - sent) > QUANTISATION))
            trial.limited += limited
            trial.out_of_range += clipped
            trial.record(
                Sample(
                    step,
                    self._clock.now() - started,
                    state,
                    executed,
                    requested.astype("float32"),
                )
            )

            deadline = started + (step + 1) * period
            if self._clock.now() > deadline:
                trial.overruns += 1
            self._clock.sleep_until(deadline)

        self._detect(trial)
        return trial, error

    def _act(self, policy: Any, step: int, state: np.ndarray) -> np.ndarray:
        """Ask the policy, and check what came back before anything is sent.

        Nothing here is padded, truncated or reshaped into fitting. A policy
        that emits five values for a six-jointed arm has a wiring bug, and
        inventing the sixth is how a gripper command becomes a wrist command.
        """
        raw = np.asarray(policy.act(Observation(step, {STATE: state})))
        if raw.ndim != 2 or raw.shape[1] != len(JOINTS) or len(raw) < 1:
            raise ComponentError(
                f"{getattr(policy, 'name', 'policy')} returned a chunk of shape "
                f"{raw.shape}; expected (horizon, {len(JOINTS)}) for {list(JOINTS)}. "
                "Nothing is padded to fit."
            )
        if not np.can_cast(raw.dtype, np.dtype("float32"), casting="same_kind"):
            raise ComponentError(
                f"{getattr(policy, 'name', 'policy')} returned {raw.dtype}, which is not "
                "same-kind castable to float32"
            )
        if not np.all(np.isfinite(raw)):
            raise ComponentError(
                f"{getattr(policy, 'name', 'policy')} returned a non-finite action; "
                "the follower is not commanded and the trial is recorded as failed"
            )
        return raw.astype("float32")

    def _limit(self, requested: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """The safety clamp, and it is always engaged.

        Two limits, and they are different things. The *relative* one bounds how
        far the follower may move in one tick, which is what stops a wild command
        becoming a collision. The *range* one is a hard property of the servos:
        a target outside 0..100 is not a position on this arm. Both are counted,
        and the counts travel into provenance as a declared loss, because a run
        where the clamp bound on most steps measured the clamp, not the policy.
        """
        floor = state - self._max_relative
        ceiling = state + self._max_relative
        stepped = np.clip(requested, floor, ceiling)
        limited = bool(np.any(stepped != requested))
        bounded = np.clip(stepped, RANGE[0], RANGE[1])
        # Asked against the *request*, not against what the relative clamp left
        # of it. A policy emitting 1e9 with a 5-per-tick clamp has its command
        # pulled inside the range by the first limit, and asking the second one
        # whether anything was out of range then answers no — which reports zero
        # off-scale commands for a policy that emitted nothing else. The two
        # limits mean different things and each is asked about the thing it means.
        clipped = bool(np.any((requested < RANGE[0]) | (requested > RANGE[1])))
        return bounded.astype("float32"), limited, clipped

    def _detect(self, trial: _Trial) -> None:
        """Milestones, if and only if something can supply them."""
        if self._pose is not None:
            for sample in trial.samples:
                for reached in self._pose.milestones(sample):
                    if reached in trial.reached:
                        continue
                    trial.reached.add(reached)
                    trial.events.append(StageEvent(reached, sample.step))

    # -- the machine -------------------------------------------------------

    def connect(self) -> None:
        if not self._connected:
            self._pair.connect()
            self._connected = True

    def close(self) -> None:
        if self._connected:
            self._pair.disconnect()
            self._connected = False

    def _read(self) -> np.ndarray:
        """Measured follower position. A transport failure halts the run.

        Not recoverable, deliberately. Continuing to hand scenes to an arm whose
        bus stopped answering produces plausible-looking numbers describing
        nothing.
        """
        try:
            state = np.asarray(self._pair.read_follower(), dtype="float32")
        except Exception as failure:  # noqa: BLE001
            raise FaultError(
                f"{self._name}: reading the follower raised "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        if state.shape != (len(JOINTS),):
            raise FaultError(
                f"{self._name}: the follower reported {state.shape}, expected "
                f"({len(JOINTS)},) for {list(JOINTS)}"
            )
        return state

    def _write(self, target: np.ndarray) -> np.ndarray:
        """Command the follower and return what it says it actually sent.

        The return value is the record's ``action`` column, so a pair that will
        not say what it commanded is refused rather than recorded from the
        request. The difference is small and always in the same direction — a
        clamp only ever tightens — which is exactly the shape of bias that
        survives training and surfaces as a policy confidently reaching a little
        further than the arm ever did.
        """
        try:
            executed = self._pair.write_follower(target)
        except Exception as failure:  # noqa: BLE001
            raise FaultError(
                f"{self._name}: commanding the follower raised "
                f"{type(failure).__name__}: {failure}"
            ) from failure
        executed = np.asarray(executed, dtype="float32")
        if executed.shape != (len(JOINTS),):
            raise FaultError(
                f"{self._name}: the follower reported commanding {executed.shape}, "
                f"expected ({len(JOINTS)},). Without it the record cannot say what the "
                "arm was told, and the difference from what it was asked is exactly "
                "what a clamp removes."
            )
        return executed

    # -- records -----------------------------------------------------------

    def _episode(
        self,
        trial: _Trial,
        scene: Scene,
        epoch: int,
        index: int,
        task: TaskSpec,
        protocol: RunProtocol,
        error: str | None,
    ) -> EpisodeRecord:
        summary = self._summary(trial, error)
        # A trial the policy broke out of is not asked about. The arm stopped
        # mid-motion, so neither answer is true: labelling it a failure would
        # count a crashed backend as a task the policy cannot do, and labelling
        # it a success is absurd. It stays unknown, with the error beside it,
        # and drops out of the success-rate denominator rather than out of the
        # record.
        success = None if error else self._console.outcome(scene, summary)
        annotations = {
            "epoch": epoch,
            "trial_index": index,
            "steps": len(trial.sent),
            "policy_calls": trial.calls,
            "scene_seed_drawn": scene.seed,
            "placement": "set by hand; not verified, and not reproducible",
            "clamped_steps": trial.limited,
            "out_of_range_steps": trial.out_of_range,
            "late_ticks": trial.overruns,
            "calibration": dict(self._pair.calibration),
            "started_at": summary["started_at"],
            "ended_at": summary["ended_at"],
            **summary["tracking"],
            **self._session,
            **({"error": error} if error else {}),
            **({"contact_step": summary["contact_step"]} if self._contact else {}),
        }
        if not trial.sent:
            # A trial that failed before its first command is a record of that,
            # not an episode to be dropped from the denominator.
            return episode_from_labels(
                id=self._uid(scene, epoch, protocol),
                source=task.name,
                labels=EpisodeLabels(success=success, annotations=annotations),
                task=task.name,
                embodiment=self._embodiment.ref,
                collected_by=self._name,
            )
        return episode_from_arrays(
            {
                STATE: np.asarray(trial.states, dtype="float32"),
                ACTION: np.asarray(trial.sent, dtype="float32"),
                REQUESTED: np.asarray(trial.requested, dtype="float32"),
                ELAPSED: np.asarray(trial.elapsed, dtype="float32"),
            },
            (
                self._state,
                self._action,
                # What the policy asked for, kept beside what the arm was told.
                # Without both, a clamped run is indistinguishable from a docile
                # one and the clamp's effect on the result is unrecoverable.
                self._requested,
                self._elapsed,
            ),
            id=self._uid(scene, epoch, protocol),
            source=task.name,
            labels=EpisodeLabels(
                success=success,
                stage_events=tuple(trial.events),
                annotations=annotations,
            ),
            task=task.name,
            embodiment=self._embodiment.ref,
            collected_by=self._name,
        )

    def _uid(self, scene: Scene, epoch: int, protocol: RunProtocol) -> str:
        return f"{scene.id}#{epoch}" if protocol.epochs > 1 else scene.id

    def _summary(self, trial: _Trial, error: str | None) -> dict[str, Any]:
        """What the operator is shown before being asked for a label.

        Facts the rig knows and the person cannot see: how often the clamp bound,
        whether the loop kept up, how far the follower lagged its command. An
        operator labelling a trial where the clamp bound on eighty percent of
        steps should know that before answering.
        """
        error_rows = trial.tracking_error
        tracking: dict[str, Any] = {}
        if len(error_rows):
            tracking = {
                "tracking_error_mean": [round(v, 4) for v in error_rows.mean(axis=0).tolist()],
                "tracking_error_max": [round(v, 4) for v in error_rows.max(axis=0).tolist()],
            }
        now = datetime.now(timezone.utc).isoformat()
        return {
            "steps": len(trial.sent),
            "seconds": round(trial.elapsed[-1], 3) if trial.elapsed else 0.0,
            "clamped_steps": trial.limited,
            "late_ticks": trial.overruns,
            "contact_step": self._contact(trial.samples) if self._contact else None,
            "tracking": tracking,
            "started_at": now,
            "ended_at": now,
            **({"error": error} if error else {}),
        }

    def _provenance(
        self,
        policy: Any,
        task: TaskSpec,
        protocol: RunProtocol,
        failures: int,
        limited: int,
        out_of_range: int,
        trials: int,
    ) -> Provenance:
        """Who ran what, on which arms, under which calibration.

        The calibration digest rides on the embodiment's ``artifact_digest``,
        which puts it in the component's ``ref``. Two runs across a
        recalibration are therefore not comparable while holding the embodiment
        fixed, and nobody has to remember that — ``RunSet.comparable`` refuses
        on its own. That is LeRobot #3758 turned into a structural property
        rather than a thing to be careful about.
        """
        calibration = dict(self._pair.calibration)
        adapters: list[AdapterStep] = []
        if limited or out_of_range:
            losses = []
            if limited:
                losses.append(
                    f"{limited} commanded target(s) were limited before reaching the "
                    f"servos: this evaluator caps motion at {self._max_relative:g} of "
                    f"the {RANGE[0]:g}..{RANGE[1]:g} range per tick, and the follower "
                    "applies its own "
                    "envelope and relative guard after that. The policy asked for "
                    "motion the follower was not permitted to make"
                )
            if out_of_range:
                losses.append(
                    f"{out_of_range} commanded target(s) fell outside the servos' "
                    f"{RANGE[0]:g}..{RANGE[1]:g} range and were bounded to it"
                )
            adapters.append(AdapterStep("so101-safety-clamp", VERSION, tuple(losses)))

        notes: list[str] = []
        if failures:
            notes.append(f"{failures} trial(s) failed to complete")
        if not isinstance(self._clock, WallClock):
            notes.append(
                "run on a virtual clock: timing in this record is nominal, not measured"
            )
        if self._pose is None:
            notes.append(
                "no pose source configured, so no stage events; a funnel cannot be run "
                "on this record and the resolver will say so"
            )

        return Provenance(
            components=(
                policy.descriptor().component_ref(),
                self.descriptor().component_ref(),
                self._embodiment.descriptor().component_ref(
                    artifact_digest=digest_of(calibration)
                ),
            ),
            protocol={
                "task": task.name,
                "horizon": task.horizon,
                # What the run was *looking for*, which is not the same as what
                # it found. A funnel that inferred its rungs from the milestones
                # that happened to fire would read a policy failing at rung two
                # as a one-rung task at a hundred percent. The task declares the
                # vocabulary; a hand-built task that forgot to falls back to the
                # pose source's, because the run still looked for those.
                "stages": list(task.stages or (self._pose.names if self._pose else ())),
                "control_hz": self._rate_hz,
                "max_relative_target": self._max_relative,
                # Spelled the way every channel, every discriminator and the
                # descriptor spell it. The British spelling sat here and on the
                # evaluator's channels while the descriptor used the American
                # one, which meant the key existed on one side only and the
                # spine had nothing to compare.
                "normalization": self._action.metadata["normalization"],
                "calibration_variant": self._action.metadata["calibration_variant"],
                "calibration": calibration,
                "trials": trials,
                "reset": "human, from the task spec's scenes; placement not verified",
                "outcomes": "human label at the rig",
                **self._session,
                **protocol.as_dict(),
            },
            adapters=tuple(adapters),
            created_at=datetime.now(timezone.utc).isoformat(),
            host=socket.gethostname(),
            notes=tuple(notes),
        )

    def _metrics(self, episodes: Sequence[EpisodeRecord]) -> dict[str, Measurement]:
        """Aggregates, each carrying its n. No bare floats leave here.

        The success rate gets a Wilson interval rather than a bare proportion
        because hardware n is small by construction — twenty trials is an
        expensive afternoon — and at n=20 the normal approximation is not
        defensible while Wilson still is.
        """
        metrics: dict[str, Measurement] = {}
        scored = [e for e in episodes if e.labels.success is not None]
        if scored:
            successes = sum(1 for e in scored if e.labels.success)
            metrics["success_rate"] = Measurement(
                successes / len(scored),
                n=len(scored),
                ci=_wilson(successes, len(scored)),
                units="fraction",
                method="operator label at the rig; Wilson 95%",
                detail={"voided": len(episodes) - len(scored)},
            )

        maxima = [
            max(e.labels.annotations["tracking_error_max"])
            for e in episodes
            if "tracking_error_max" in e.labels.annotations
        ]
        if maxima:
            metrics["tracking_error_max"] = Measurement(
                float(np.max(maxima)),
                n=len(maxima),
                units="percent",
                method="worst per-joint |commanded - achieved| over every episode",
            )

        steps = [int(e.labels.annotations.get("steps", 0)) for e in episodes]
        clamped = [int(e.labels.annotations.get("clamped_steps", 0)) for e in episodes]
        if sum(steps):
            metrics["clamped_fraction"] = Measurement(
                sum(clamped) / sum(steps),
                n=sum(steps),
                units="fraction",
                method=(
                    f"steps whose command was limited to {self._max_relative:g} per tick; "
                    "a high value means the clamp was measured, not the policy"
                ),
            )

        # Achieved rate, read from the timestamps the loop actually recorded
        # rather than from the rate it was asked for. A rig that quietly ran at
        # 22 Hz produced a corpus whose action spacing does not match its
        # declared 30 Hz, and nothing downstream can tell unless this is
        # measured here.
        # Over the *span between samples*, not from zero. The first sample is
        # taken after that step's bus traffic, so its timestamp is already a few
        # milliseconds past the start; dividing by it charges one step's serial
        # latency to the whole episode and reports a rate that drifts with
        # episode length. n-1 intervals between n stamps is the whole of it.
        spans = [
            (len(e) - 1, float(e.array(ELAPSED)[-1]) - float(e.array(ELAPSED)[0]))
            for e in episodes
            if e.has(ELAPSED) and len(e) > 1
        ]
        rates = [steps / span for steps, span in spans if span > 0]
        if rates:
            metrics["control_rate_hz"] = Measurement(
                float(np.median(rates)),
                n=len(rates),
                units="Hz",
                method=(
                    f"achieved, median over episodes; nominal {self._rate_hz:g} Hz"
                    + ("" if isinstance(self._clock, WallClock) else " (virtual clock)")
                ),
            )
        return metrics


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _checked_pair(pair: Any, name: str) -> float:
    """Refuse a pair that cannot be driven, before anything is commanded.

    Every one of these is a wiring mistake that would otherwise surface as
    motion. The joint check in particular: all STS3215 servos ship with id=1, so
    a daisy chain is assigned one servo at a time on an otherwise empty bus, and
    getting that order wrong produces a pair whose fourth value drives the wrong
    joint. Comparing the declared names against the LeRobot order catches it
    here rather than at the elbow.
    """
    missing = [
        attribute
        for attribute in (
            "joints",
            "control_hz",
            "calibration",
            "read_follower",
            "read_leader",
            "write_follower",
            "connect",
            "disconnect",
        )
        if not hasattr(pair, attribute)
    ]
    if missing:
        raise ConfigError(f"{name}: the pair has no {missing}; see SO101Pair")

    joints = tuple(pair.joints)
    if joints != JOINTS:
        raise ConfigError(
            f"{name}: the pair declares joints {list(joints)}, but this evaluator "
            f"writes {list(JOINTS)}. Order is the content of the vector, so a "
            "permutation is not a rename and nothing here will reorder it for you."
        )

    rate = float(pair.control_hz)
    if rate <= 0:
        raise ConfigError(f"{name}: the pair declares control_hz={rate}, which is not a rate")

    calibration = dict(pair.calibration or {})
    absent = [arm for arm in ("leader", "follower") if not calibration.get(arm)]
    if absent:
        raise ConfigError(
            f"{name}: no calibration hash for {absent}. Both are required: a joint "
            "offset is invisible in a training loss (LeRobot #3758: ~17 degrees, "
            "~6 cm of gripper error, a healthy-looking loss curve), so the only way "
            "to find one retrospectively is to know which calibration each episode "
            "was recorded under."
        )
    return rate


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval. Behaves at n=10 and at 0/n, unlike the normal one."""
    if n <= 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z / denominator * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mock_pair(*, bus_clock: Clock | None = None, **kwargs: Any) -> SO101Pair:
    """A pair with two :class:`MockTransport` buses and no hardware anywhere.

    Every capability of this evaluator is exercisable through this, which is not
    a convenience: a hardware plugin that can only be tested on the hardware is a
    plugin that is tested the week the hardware is free.

    The two arms get *different* calibration files, because the pair refuses two
    arms sharing one — that would be one arm running the other's homing offsets,
    which is #3758 installed deliberately, and it would also collapse the two
    hashes the episode metadata carries into one indistinguishable value.

    Neither arm can be built without a clamp: :func:`~.arm.mock_arm` passes one
    to the bus, and the bus constructor has no default for it.

    ``bus_clock`` is the evaluator's own clock, handed down to the simulated
    servos so they integrate their motion against the same time the loop is
    pacing to. Without it the two layers keep separate time — the loop jumping
    from deadline to deadline while the servos move by however many microseconds
    the interpreter spent — and two identical runs come back with different
    numbers. Pass the clock you passed the evaluator.
    """
    timing: dict[str, Any] = {}
    if bus_clock is not None:
        timing = {
            "clock": bus_clock.now,
            # The bus's per-transaction latency, spent on the shared clock. A
            # virtual clock then charges serial time to the control period the
            # way a real bus does, instead of it being free.
            "sleep": lambda seconds: bus_clock.sleep_until(bus_clock.now() + seconds),
        }
    return SO101Pair(
        leader=mock_arm("leader", widen=0, **timing),
        # Two SO-101s off the same print do not share travel limits. A few counts
        # of difference is what a second measured calibration actually looks like.
        follower=mock_arm("follower", widen=12, **timing),
        **kwargs,
    )


__all__ = [
    "ACTION",
    "CALIBRATION_VARIANT",
    "DEFAULT_HORIZON",
    "DEFAULT_MAX_RELATIVE_TARGET",
    "ELAPSED",
    "EMBODIMENT_REF",
    "GRIPPER",
    "JOINTS",
    "RANGE",
    "REQUESTED",
    "STATE",
    "VERSION",
    "Clock",
    "Console",
    "LeaderTeleop",
    "PoseSource",
    "SO101Evaluator",
    "Sample",
    "ScriptedConsole",
    "TerminalConsole",
    "VirtualClock",
    "WallClock",
    "action_spec",
    "gripper_stall",
    "mock_pair",
    "state_spec",
]
