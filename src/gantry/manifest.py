"""The unit of reproducibility: one file describing a whole run.

A command line is not a record. Flags get retyped from memory, a default shifts
between releases, and six weeks later nobody can say what produced the number in
the slide. A manifest is a file, so it can be committed, diffed, reviewed, and
re-run.

JSON, because core depends on nothing but numpy and a YAML parser is a
dependency. That rule is absolute rather than convenient: an optional import is
still an import, and "core installs anywhere" stops being true the first time it
is bent. Anyone who prefers YAML parses it themselves and calls
:meth:`Manifest.from_dict`, which is public for exactly that reason.

Every field that changes an answer belongs here, execution protocol included.
The alternative is a protocol that lives in someone's shell history.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .spine import Verdict

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ComponentSpec:
    """One named component and the configuration it is built with."""

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: Any, where: str) -> "ComponentSpec":
        if isinstance(payload, str):
            return cls(payload, {})
        if isinstance(payload, Mapping):
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise ConfigError(f"{where}: needs a non-empty 'name'")
            config = payload.get("config", {})
            if not isinstance(config, Mapping):
                raise ConfigError(f"{where}: 'config' must be an object")
            return cls(name, dict(config))
        raise ConfigError(f"{where}: expected a name or an object, got {type(payload).__name__}")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "config": dict(self.config)}


@dataclass(frozen=True)
class Manifest:
    """A complete, re-runnable description of one piece of work."""

    name: str
    cohorts: Mapping[str, ComponentSpec] = field(default_factory=dict)
    #: Every single-valued plane, keyed by plane name. A plane registered by a
    #: plugin lands here with no edit to this class — which is the difference
    #: between a manifest that supports six planes and one that supports planes.
    components: Mapping[str, ComponentSpec] = field(default_factory=dict)
    feedback: tuple[ComponentSpec, ...] = ()
    protocol: Mapping[str, Any] = field(default_factory=dict)
    version: int = MANIFEST_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # The three core single-valued planes read through, so existing manifests
    # and callers are unaffected by the generalisation.

    @property
    def policy(self) -> ComponentSpec | None:
        return self.components.get("policy")

    @property
    def evaluation(self) -> ComponentSpec | None:
        return self.components.get("evaluation")

    @property
    def embodiment(self) -> ComponentSpec | None:
        return self.components.get("embodiment")

    # -- reading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str = "manifest") -> "Manifest":
        version = payload.get("version", MANIFEST_VERSION)
        if version != MANIFEST_VERSION:
            raise ConfigError(
                f"{where}: manifest version {version!r} is not supported; "
                f"this Gantry reads version {MANIFEST_VERSION}"
            )
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{where}: needs a non-empty 'name'")

        raw_cohorts = payload.get("cohorts", {})
        if not isinstance(raw_cohorts, Mapping):
            raise ConfigError(f"{where}: 'cohorts' must be an object of name -> component")
        cohorts = {
            key: ComponentSpec.parse(value, f"{where}.cohorts.{key}")
            for key, value in raw_cohorts.items()
        }

        raw_feedback = payload.get("feedback", ())
        if isinstance(raw_feedback, (str, Mapping)):
            raw_feedback = [raw_feedback]
        feedback = tuple(
            ComponentSpec.parse(value, f"{where}.feedback[{index}]")
            for index, value in enumerate(raw_feedback)
        )

        # Any single-valued plane may appear as a top-level key. Asking the
        # plane registry rather than listing names here is what lets a plugin
        # add a plane and have manifests understand it immediately.
        from .spine import get_plane, known_planes

        components: dict[str, ComponentSpec] = {}
        for plane in known_planes():
            described = get_plane(plane)
            if described is None or described.many:
                continue
            if payload.get(plane) is not None:
                components[plane] = ComponentSpec.parse(payload[plane], f"{where}.{plane}")

        reserved = {"version", "name", "cohorts", "feedback", "protocol", "metadata"}
        unknown = [
            key
            for key in payload
            if key not in reserved and key not in known_planes()
        ]
        if unknown:
            raise ConfigError(
                f"{where}: unknown key(s) {sorted(unknown)}; "
                f"expected a plane or one of {sorted(reserved)}"
            )

        protocol = payload.get("protocol", {})
        if not isinstance(protocol, Mapping):
            raise ConfigError(f"{where}: 'protocol' must be an object")

        return cls(
            name=name,
            cohorts=cohorts,
            components=components,
            feedback=feedback,
            protocol=dict(protocol),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Manifest":
        target = Path(path)
        if not target.exists():
            raise ConfigError(f"no manifest at {target}")
        if target.suffix in {".yaml", ".yml"}:
            raise ConfigError(
                f"{target}: manifests are JSON. Core takes no parser dependency, so "
                "parse the YAML yourself and pass the result to Manifest.from_dict."
            )
        try:
            payload = json.loads(target.read_text())
        except json.JSONDecodeError as error:
            raise ConfigError(f"{target}: not valid JSON ({error})") from error
        if not isinstance(payload, Mapping):
            raise ConfigError(f"{target}: expected an object at the top level")
        return cls.from_dict(payload, where=str(target))

    # -- writing -----------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "cohorts": {key: spec.as_dict() for key, spec in self.cohorts.items()},
            "feedback": [spec.as_dict() for spec in self.feedback],
            "protocol": dict(self.protocol),
        }
        for plane, spec in self.components.items():
            payload[plane] = spec.as_dict()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        target.write_text(self.to_json() + "\n")
        return target

    # -- checking ----------------------------------------------------------

    @property
    def evaluates(self) -> bool:
        """Whether this manifest runs a policy, rather than only reading data."""
        return self.policy is not None and self.evaluation is not None

    def validate(self) -> Verdict:
        checks = []
        if not self.cohorts:
            checks.append(
                Verdict.no("manifest.no_cohorts", f"{self.name}: no cohorts to work with")
            )
        if not self.feedback:
            checks.append(
                Verdict.note(
                    "manifest.no_feedback",
                    f"{self.name}: no feedback modules, so this will read data and "
                    "report nothing about it",
                )
            )
        if (self.policy is None) != (self.evaluation is None):
            missing = "evaluation" if self.policy is not None else "policy"
            checks.append(
                Verdict.no(
                    "manifest.half_an_evaluation",
                    f"{self.name}: a run needs both a policy and an evaluation; "
                    f"{missing!r} is missing",
                    hint="drop both to analyse the data as recorded",
                )
            )
        return Verdict.all(checks)


def manifest_from(path: str | os.PathLike[str]) -> Manifest:
    return Manifest.load(path)
