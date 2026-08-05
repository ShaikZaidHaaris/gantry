"""Tasks read from files, and refused when they cannot be judged.

A directory of JSON files is the whole source. That is deliberate: a task you
can diff, review and commit is a task two people can disagree about *before* a
robot spends a day on it, which is the only cheap moment to disagree.

What the validator is actually for
----------------------------------
Not schema checking -- that is the easy half. The expensive failures in real
evaluation are tasks that are *scorable in principle and ambiguous in practice*:
a rubric two people read differently, a start region so tight the task measures
memorisation, a criterion no world has been taught to check. Each of those
produces data that looks fine and means nothing, and each is caught here for
free.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from gantry.contracts.task import (
    Criterion,
    Region,
    TaskDefinition,
    TaskSource,
    Thing,
    task_descriptor,
)
from gantry.errors import ConfigError
from gantry.spine import Descriptor, Verdict

VERSION = "0.1.0.dev0"

#: A start region narrower than this in both axes is a fixed placement wearing a
#: rectangle's clothes. Not wrong -- sometimes you mean it -- but worth saying.
TIGHT = 0.01


def _region(payload: Mapping[str, Any] | None, where: str) -> Region | None:
    if payload is None:
        return None
    try:
        return Region(
            surface=str(payload.get("surface", "table")),
            x=(float(payload["x"][0]), float(payload["x"][1])),
            y=(float(payload["y"][0]), float(payload["y"][1])),
            yaw=(tuple(float(v) for v in payload["yaw"]) if payload.get("yaw") else None),
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ConfigError(f"{where}: a start region needs x and y ranges ({error})") from error


def definition_from(payload: Mapping[str, Any], where: str = "<task>") -> TaskDefinition:
    """One task, parsed. Structure only -- judgement is :meth:`validate`."""
    for required in ("name", "instruction"):
        if not payload.get(required):
            raise ConfigError(f"{where}: a task needs a {required!r}")
    things = tuple(
        Thing(
            id=str(t["id"]),
            kind=str(t.get("kind", t["id"])),
            start=_region(t.get("start"), f"{where}.{t.get('id')}"),
            metadata=t.get("metadata") or {},
        )
        for t in payload.get("objects") or ()
    )
    success = tuple(
        Criterion(
            check=str(c.get("check", "")),
            args=c.get("args") or {},
            rubric=str(c.get("rubric", "")),
        )
        for c in payload.get("success") or ()
    )
    return TaskDefinition(
        name=str(payload["name"]),
        instruction=str(payload["instruction"]),
        things=things,
        success=success,
        horizon=int(payload.get("horizon", 300)),
        trials=int(payload.get("trials", 20)),
        surfaces=tuple(payload.get("surfaces") or ("table",)),
        staging=payload.get("staging") or {},
        metadata=payload.get("metadata") or {},
    )


class DeclaredTasks(TaskSource):
    """Every ``*.json`` in a directory, read as a task."""

    def __init__(self, path: str | os.PathLike[str], *, name: str = "declared"):
        self._root = Path(path)
        if not self._root.exists():
            raise ConfigError(f"no task directory at {self._root}")
        self._name = name
        self._files = {}
        for file in sorted(self._root.glob("*.json")):
            try:
                payload = json.loads(file.read_text())
            except json.JSONDecodeError as error:
                raise ConfigError(f"{file}: not valid JSON ({error})") from error
            task = definition_from(payload, str(file))
            if task.name in self._files:
                raise ConfigError(
                    f"{file}: task {task.name!r} is already defined in "
                    f"{self._files[task.name][0]}; a name is how a result is addressed"
                )
            self._files[task.name] = (file, task)
        if not self._files:
            raise ConfigError(f"{self._root}: no task files")

    def descriptor(self) -> Descriptor:
        return task_descriptor(
            self._name,
            VERSION,
            tasks=len(self._files),
            rubrics=all(task.rubrics for _, task in self._files.values()),
            path=str(self._root),
            worlds=sorted({w for _, t in self._files.values() for w in t.staging}),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._files)

    def task(self, name: str) -> TaskDefinition:
        try:
            return self._files[name][1]
        except KeyError:
            raise KeyError(f"{self._name}: no task {name!r}; has {list(self._files)}") from None

    def path_of(self, name: str) -> Path:
        return self._files[name][0]

    # -- the checks worth having -------------------------------------------

    def audit(self) -> Verdict:
        """Everything :meth:`validate` checks, plus what makes a task *usable*.

        Separate from ``validate`` because these are judgements rather than
        errors: a fixed start position is a legitimate choice and usually a
        mistake, and the difference is something only the author knows.
        """
        checks = [self.validate()]
        for name, (file, task) in self._files.items():
            for thing in task.things:
                if thing.start and thing.start.fixed:
                    checks.append(
                        Verdict.note(
                            "task.fixed_start",
                            f"{name}: {thing.id!r} starts at exactly one point",
                            hint="a task with no variation measures memorisation; give "
                            "it a region unless you mean this",
                        )
                    )
                elif thing.start and (
                    thing.start.x[1] - thing.start.x[0] < TIGHT
                    and thing.start.y[1] - thing.start.y[0] < TIGHT
                ):
                    checks.append(
                        Verdict.note(
                            "task.tight_start",
                            f"{name}: {thing.id!r} varies by under {TIGHT} m in both axes",
                        )
                    )
            named = {thing.id for thing in task.things}
            for criterion in task.success:
                missing = sorted(set(criterion.objects()) - named)
                if missing:
                    checks.append(
                        Verdict.no(
                            "task.criterion_unknown_object",
                            f"{name}: success refers to {missing}, which the task does "
                            f"not place; it places {sorted(named)}",
                        )
                    )
            if not task.staging:
                checks.append(
                    Verdict.note(
                        "task.unstaged",
                        f"{name}: no world knows how to set this up yet",
                        hint="it can still be scored from video by its rubric",
                    )
                )
        return Verdict.all(checks)

    def scorable_by_hand(self) -> tuple[str, ...]:
        """Tasks a person could judge from video today, with no simulator."""
        return tuple(name for name, (_, t) in self._files.items() if t.rubrics)
