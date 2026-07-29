"""A robot described by a file, or by the data it produced.

The embodiment plane had a contract and no implementation, which meant the one
check that asks *can this policy actually drive this machine* could never fire.
This closes that, and it does it without privileging any robot: there is no
Panda here, no humanoid, no arm at all. There is a format for saying what a
machine is, and two ways to fill it in.

**From a file.** You write down what your robot's actuators accept and what its
sensors report. That is knowledge nobody can derive — joint limits, gripper
polarity, control rate — and writing it once is what lets every later run be
checked against it rather than against somebody's memory.

**From a dataset's schema.** A connector already describes its channels, with
one label per dimension and units where the format declares them. That is most
of an embodiment description, obtained from data that already exists. It is
weaker than a written spec — a recording shows what a robot *did*, not what it
*can* do, so no limits come out of it — and it is enormously cheaper, so it is
the honest starting point rather than the finished article.

What this refuses to invent
---------------------------
Limits. A file that omits them produces an embodiment with none, and a policy
checked against it is checked on shape, units and meaning but not on range.
Deriving a limit from observed data would mean claiming a robot cannot exceed
what a demonstrator happened to do, which is how a benchmark quietly becomes a
description of its own dataset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from gantry.contracts.embodiment import EmbodimentDescriptor
from gantry.errors import ConfigError
from gantry.spine import ChannelSpec

VERSION = "0.1.0.dev0"

#: Capability names the contract already knows. Anything else in a file's
#: ``capabilities`` list is carried through untouched — this plugin does not
#: adjudicate what a machine can do.
KNOWN_CAPABILITIES = ("resettable", "seedable", "simulated")


def _spec(payload: Mapping[str, Any], where: str) -> ChannelSpec:
    if "name" not in payload:
        raise ConfigError(f"{where}: a channel needs a 'name'")
    shape = tuple(int(value) for value in payload.get("shape") or ())
    labels = payload.get("dim_labels")
    return ChannelSpec(
        name=str(payload["name"]),
        kind=str(payload.get("kind", "vector" if shape else "scalar")),
        shape=shape,
        dtype=str(payload.get("dtype", "float32")),
        units=payload.get("units"),
        frame=payload.get("frame"),
        rate_hz=payload.get("rate_hz"),
        semantics=payload.get("semantics"),
        dim_labels=tuple(str(label) for label in labels) if labels else None,
        discriminators=tuple(payload.get("discriminators") or ()),
        metadata=payload.get("metadata") or {},
    )


def from_file(path: str | os.PathLike[str]) -> EmbodimentDescriptor:
    """Read a machine description.

    The file is plain JSON and its shape is the contract's own fields, so
    nothing here is a private format::

        {"name": "panda", "version": "1.0", "control_hz": 20,
         "state":  [{"name": "...", "kind": "vector", "shape": [7], "units": "rad"}],
         "action": [{"name": "...", "kind": "vector", "shape": [7]}],
         "kinematics": "path/to/panda.urdf",
         "notes": "gripper: 1 open, -1 closed"}
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no embodiment description at {path}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path}: not valid JSON ({error})") from error
    return from_mapping(payload, source=str(path))


def from_mapping(payload: Mapping[str, Any], *, source: str = "<mapping>") -> EmbodimentDescriptor:
    """The same description, already parsed. What a manifest passes inline."""
    for required in ("name", "version"):
        if not payload.get(required):
            raise ConfigError(f"{source}: an embodiment needs a {required!r}")
    described = EmbodimentDescriptor(
        name=str(payload["name"]),
        version=str(payload["version"]),
        state=tuple(_spec(s, f"{source}:state") for s in payload.get("state") or ()),
        action=tuple(_spec(a, f"{source}:action") for a in payload.get("action") or ()),
        control_hz=payload.get("control_hz"),
        kinematics=payload.get("kinematics"),
        notes=payload.get("notes"),
        capabilities=frozenset(payload.get("capabilities") or ()),
        metadata={**(payload.get("metadata") or {}), "described_by": source},
    )
    described.validate().raise_if_refused(f"{source} does not describe a usable machine")
    return described


def from_schema(
    schema: Sequence[ChannelSpec],
    *,
    name: str,
    version: str = "derived",
    action: Sequence[str] = ("action",),
    state: Sequence[str] | None = None,
    **kwargs: Any,
) -> EmbodimentDescriptor:
    """Describe a machine from channels a connector already reported.

    Cheap and weaker, and the docstring above says how. Everything the schema
    declares is carried — shape, dtype, units, meaning, dimension labels — and
    nothing it does not is invented. In particular there are no limits, because
    a recording cannot establish one.
    """
    by_name = {spec.name: spec for spec in schema}
    missing = [key for key in action if key not in by_name]
    if missing:
        raise ConfigError(
            f"cannot derive an embodiment: the schema has no action channel(s) {missing}; "
            f"it has {sorted(by_name)}"
        )
    action_specs = tuple(by_name[key] for key in action)
    state_names = list(state) if state is not None else [
        spec.name for spec in schema if spec.name not in set(action)
    ]
    rates = {spec.rate_hz for spec in schema if spec.rate_hz}
    return EmbodimentDescriptor(
        name=name,
        version=version,
        state=tuple(by_name[key] for key in state_names if key in by_name),
        action=action_specs,
        control_hz=kwargs.pop("control_hz", next(iter(rates)) if len(rates) == 1 else None),
        metadata={
            **kwargs.pop("metadata", {}),
            "described_by": "schema",
            # Said in the record, not only in the docs: anything reading this
            # embodiment can tell it was inferred from a recording and carries
            # no operating limits.
            "derived_from_recording": True,
            "limits": "not established; a recording shows what was done, not what is allowed",
        },
        **kwargs,
    )


def to_file(embodiment: EmbodimentDescriptor, path: str | os.PathLike[str]) -> Path:
    """Write a description back out, so a derived one can be edited by hand.

    The intended path from cheap to real: derive from a dataset, write it down,
    add the limits only you know, and from then on every run is checked against
    the written article.
    """
    from gantry.serial import spec_to_dict

    payload = {
        "name": embodiment.name,
        "version": embodiment.version,
        "control_hz": embodiment.control_hz,
        "state": [spec_to_dict(spec) for spec in embodiment.state],
        "action": [spec_to_dict(spec) for spec in embodiment.action],
        "kinematics": embodiment.kinematics,
        "notes": embodiment.notes,
        "capabilities": sorted(embodiment.capabilities),
        "metadata": dict(embodiment.metadata),
    }
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
