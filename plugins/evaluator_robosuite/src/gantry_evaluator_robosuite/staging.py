"""Staging a declared task in robosuite: the task file's regions become placement.

A task file says where an object may start as a rectangle on a surface. This
world says the same thing as a ``UniformRandomSampler``. Translating between
them is the whole of this module, and it matters more than its size suggests:
without it a task file's regions are decoration, the real placement lives in
robosuite's source, and "the same task on a real bench" has no numbers to
reproduce. With it, the rectangle in the file is the rectangle in the simulator
and the rectangle taped to the table.

Two things are deliberately kept apart.

**Translation is pure.** Turning a region into sampler arguments imports no
simulator, so a task can be checked for stageability on a laptop with nothing
installed. Only :func:`materialise` -- called from inside the env builder, where
robosuite is already imported -- turns those arguments into objects.

**Whether it worked is measured, not assumed.** Some environments here build
their own placement unconditionally and discard an injected sampler. Nothing in
the description says which, and a run that silently used the simulator's layout
while reporting the task file's would be exactly the failure this design exists
to prevent. So the sampler is checked for identity after the world is built, and
an environment that threw it away is refused by name rather than guessed at from
a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gantry.contracts.task import TaskDefinition
from gantry.errors import ConfigError
from gantry.spine import Verdict

#: This world's name, as a task file's ``staging`` block keys it.
REALISATION = "robosuite"

#: How far above the surface an object is dropped, when the task does not say.
#: robosuite's own environments use a small positive offset so an object does
#: not spawn intersecting the table; zero is not a safe default.
Z_OFFSET = 0.01


@dataclass(frozen=True)
class Placement:
    """One object's start region, in the terms this world uses.

    Plain data. Holding it as a value rather than as a live sampler is what
    lets a task be checked, printed and diffed without a simulator present.
    """

    #: What robosuite calls this object. Environments here bind objects to
    #: sub-samplers by name, so this has to be their name, not the task's.
    object_name: str
    #: The task's own id for the same object, kept so a refusal can name the
    #: thing the author wrote rather than the thing the simulator calls it.
    task_id: str
    x: tuple[float, float]
    y: tuple[float, float]
    yaw: tuple[float, float] | None = None
    z_offset: float = Z_OFFSET

    @property
    def sampler_name(self) -> str:
        """The sub-sampler name this world looks for when binding objects.

        Environments here call ``add_objects_to_sampler(f"{name}Sampler", ...)``
        with their own object name. A composite sampler missing that exact key
        raises inside robosuite, which is the correct outcome -- loud, and at
        build time.
        """
        return f"{self.object_name}Sampler"

    def as_kwargs(self, reference: Any) -> dict[str, Any]:
        """Sampler arguments, given where this world puts the surface.

        ``reference`` is measured from the built world rather than written into
        a task file: it is a property of the environment, and a constant copied
        by hand drifts the moment the environment changes its table.
        """
        return {
            "name": self.sampler_name,
            "x_range": list(self.x),
            "y_range": list(self.y),
            # None is robosuite's "uniform over a full turn". Passed through
            # rather than substituted, so a task that says nothing about
            # orientation gets this world's own behaviour and a note saying so.
            "rotation": list(self.yaw) if self.yaw else None,
            "rotation_axis": "z",
            "ensure_object_boundary_in_range": False,
            "ensure_valid_placement": True,
            "reference_pos": reference,
            "z_offset": self.z_offset,
        }


def block_of(task: TaskDefinition) -> Mapping[str, Any] | None:
    """This world's staging block, or None if it does not know this task."""
    return task.staged_by(REALISATION)


def placements_for(task: TaskDefinition) -> tuple[Placement, ...]:
    """Every start region in the task, as this world's placements. Pure."""
    block = block_of(task) or {}
    places = dict(block.get("places") or {})
    offsets = dict(block.get("z_offsets") or {})
    found = []
    for thing in task.things:
        if thing.start is None or thing.id not in places:
            continue
        found.append(
            Placement(
                object_name=str(places[thing.id]),
                task_id=thing.id,
                x=tuple(thing.start.x),
                y=tuple(thing.start.y),
                yaw=tuple(thing.start.yaw) if thing.start.yaw else None,
                z_offset=float(offsets.get(thing.id, Z_OFFSET)),
            )
        )
    return tuple(found)


def check_staging(task: TaskDefinition) -> Verdict:
    """Whether this world can stage this task, as far as reading can tell.

    Everything here is decidable from the task file alone. Whether the
    environment actually *honours* the placement is a different question with a
    different answer, and it is measured in :func:`verify_honoured` once a world
    exists.
    """
    block = block_of(task)
    if block is None:
        return Verdict.no(
            f"{REALISATION}.unstaged",
            f"{task.name!r} has no {REALISATION!r} staging block, so this world does "
            "not know how to set it up",
            hint="it can still be scored from video by its rubric",
        )
    checks = []
    if not block.get("env_name"):
        checks.append(
            Verdict.no(
                f"{REALISATION}.no_env_name",
                f"{task.name!r} is staged by {REALISATION} but names no environment",
            )
        )
    places = dict(block.get("places") or {})
    unknown = sorted(set(places) - {thing.id for thing in task.things})
    if unknown:
        checks.append(
            Verdict.no(
                f"{REALISATION}.unknown_object",
                f"{task.name!r} maps {unknown} onto this world, but the task places no such object",
            )
        )
    for thing in task.things:
        if thing.start is None:
            continue
        if thing.id not in places:
            checks.append(
                Verdict.note(
                    f"{REALISATION}.unmapped_region",
                    f"{task.name!r}: {thing.id!r} declares a start region that this "
                    "world will ignore, because the staging block does not say which "
                    "of its objects that is",
                    hint=f"add it to staging.{REALISATION}.places",
                )
            )
        elif thing.start.yaw is None:
            checks.append(
                Verdict.note(
                    f"{REALISATION}.yaw_unspecified",
                    f"{task.name!r}: {thing.id!r} declares no starting orientation, so "
                    "this world picks one uniformly at random",
                    hint="declare yaw if a real bench is meant to reproduce this",
                )
            )
    return Verdict.all(checks)


def env_meta_for(task: TaskDefinition, **env_kwargs: Any) -> dict[str, Any]:
    """The recipe for this task's world, in the shape the evaluator already takes.

    Same structure a dataset's ``env_args`` has, so a task-driven world and a
    recorded one are built by the same code and neither is the special case.
    """
    check_staging(task).raise_if_refused(f"cannot stage {task.name!r} in {REALISATION}")
    block = block_of(task) or {}
    settings = {k: v for k, v in block.items() if k not in ("env_name", "places", "z_offsets")}
    settings.update(env_kwargs)
    return {
        "env_name": str(block["env_name"]),
        "env_kwargs": settings,
        "placement": tuple(placements_for(task)),
        "type": 1,
    }


# -- the simulator-facing half -------------------------------------------------
# Called from inside the env builder, where robosuite is already imported.


def shared_region(placements: tuple[Placement, ...]) -> bool:
    """Whether every object here starts in the same rectangle.

    This decides whether one sampler can faithfully stand for all of them, and
    the answer is not cosmetic: a single sampler holds one rectangle, so using
    one where the task declared two would quietly place both objects in the
    first object's region and report the file's numbers regardless.
    """
    return len({(p.x, p.y, p.yaw, p.z_offset) for p in placements}) <= 1


def materialise(placements: tuple[Placement, ...], reference: Any, *, composite: bool) -> Any:
    """Turn placements into a sampler, in one of the two shapes this world takes.

    Environments here bind their objects to an injected sampler in two
    incompatible ways, and neither is discoverable from a description: some call
    ``add_objects`` -- which a composite sampler refuses outright -- and some call
    ``add_objects_to_sampler`` with a name, which only a composite sampler has.
    Which one an environment uses is a fact about its source, so it is found by
    offering a shape and seeing whether the world accepts it, rather than by a
    list of environment names that is wrong as soon as anyone adds one.
    """
    from robosuite.utils.placement_samplers import (
        SequentialCompositeSampler,
        UniformRandomSampler,
    )

    if composite:
        sampler = SequentialCompositeSampler(name="ObjectSampler")
        for placement in placements:
            sampler.append_sampler(sampler=UniformRandomSampler(**placement.as_kwargs(reference)))
        return sampler
    # One rectangle for everything the environment adds. Only reachable when
    # every placement agrees on it -- see :func:`shared_region`.
    kwargs = placements[0].as_kwargs(reference)
    kwargs["name"] = "ObjectSampler"
    return UniformRandomSampler(**kwargs)


def sampler_shapes(placements: tuple[Placement, ...]) -> tuple[bool, ...]:
    """Which sampler shapes could faithfully carry these placements, best first.

    Per-object regions can only be expressed by the composite shape, so a task
    declaring different regions for different objects gets one candidate rather
    than two -- an environment that will not take it is refused, which is the
    correct outcome. Coercing it into a single region would be the failure this
    module exists to prevent.
    """
    return (False, True) if shared_region(placements) else (True,)


def accepts_placement(env_class: Any) -> bool:
    """Whether this environment can be handed a placement at all.

    Not every one can: some take no such argument, so passing one is a
    ``TypeError`` from deep inside a constructor rather than an answer. Asked of
    the signature instead, which is both a clearer refusal and a general one --     an environment written next week is judged the same way.
    """
    import inspect

    try:
        parameters = inspect.signature(env_class.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level or unusual init
        return True
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return True
    return "placement_initializer" in parameters


def surface_origin(env: Any) -> Any:
    """Where this world puts the surface objects rest on.

    Measured from the built environment. Regions in a task file are relative to
    the surface -- that is what makes them printable onto a real table -- and this
    is the offset that turns them into this world's coordinates.
    """
    origin = getattr(env, "table_offset", None)
    if origin is None:
        raise ConfigError(
            f"{type(env).__name__} exposes no surface origin, so a task file's regions "
            "cannot be placed in it; this world stages its objects some other way"
        )
    return origin


def verify_honoured(env: Any, sampler: Any, env_name: str) -> None:
    """Check the world kept the placement it was given, rather than assuming it.

    Some environments build their own placement unconditionally and overwrite
    an injected sampler. That is silent: the world runs, the run succeeds, and
    every number it reports is against a layout the task file did not specify.
    Detected by identity rather than by a list of environment names, so an
    environment added later -- by robosuite or by you -- is judged on what it
    does rather than on whether anyone remembered to list it.
    """
    kept = getattr(env, "placement_initializer", None)
    if kept is None:
        raise ConfigError(
            f"{env_name} has no placement to give, so a task file's start regions "
            "cannot drive it; this environment lays its objects out its own way"
        )
    if kept is not sampler:
        raise ConfigError(
            f"{env_name} discarded the placement it was given and built its own, so "
            "the task file's start regions would not be what ran; this environment "
            "cannot be driven from a task file"
        )
