"""What a plugin says about itself, before it is asked to do anything.

Every implementation on every plane ships a Descriptor: identity, the contract
version it implements, what it needs, what it provides, and how it wants to be
isolated. The resolver reads descriptors and nothing else, which is why it can
plan a run without importing a single plugin -- and why an unresolvable
combination fails in milliseconds instead of after a model finishes loading.

``requires`` and ``provides`` are free-form maps. Core defines no well-known
keys; the contracts do. That keeps this module ignorant of what a policy or an
evaluator actually is, which is the point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .plane import known_planes
from .provenance import ComponentRef, digest_of
from .verdict import Verdict

#: How a plugin wants to be run. Dependency stacks in this domain conflict
#: routinely, so isolation is declared by the plugin, not guessed by the host.
ISOLATION = ("in-process", "venv", "container")


@dataclass(frozen=True)
class ContractVersion:
    """A semver pin for one of the plane interfaces."""

    name: str
    major: int
    minor: int

    @classmethod
    def parse(cls, text: str) -> "ContractVersion":
        name, _, version = text.partition("@")
        major, _, minor = version.partition(".")
        try:
            return cls(name, int(major), int(minor or 0))
        except ValueError as error:
            raise ValueError(
                f"malformed contract version {text!r}; want 'name@major.minor'"
            ) from error

    def __str__(self) -> str:
        return f"{self.name}@{self.major}.{self.minor}"

    def satisfies(self, required: "ContractVersion") -> Verdict:
        """Can an implementation of ``self`` be used where ``required`` is wanted?

        Semver: same major, and at least the required minor.
        """
        if self.name != required.name:
            return Verdict.no(
                "contract.name",
                f"implements {self.name!r}, host wants {required.name!r}",
            )
        if self.major != required.major:
            return Verdict.no(
                "contract.major",
                f"implements {self} but host speaks {required}",
                hint="a major bump is a breaking change; upgrade the plugin or pin an older host",
            )
        if self.minor < required.minor:
            return Verdict.no(
                "contract.minor",
                f"implements {self}, host needs at least {required}",
                hint="the plugin predates a feature the host relies on",
            )
        return Verdict.yes()


@dataclass(frozen=True)
class Descriptor:
    """A plugin's self-description."""

    plane: str
    name: str
    version: str
    contract: str
    requires: Mapping[str, Any] = field(default_factory=dict)
    provides: Mapping[str, Any] = field(default_factory=dict)
    isolation: str = "in-process"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.plane not in known_planes():
            raise ValueError(f"unknown plane {self.plane!r}; expected one of {known_planes()}")
        if self.isolation not in ISOLATION:
            raise ValueError(f"unknown isolation {self.isolation!r}; expected one of {ISOLATION}")
        ContractVersion.parse(self.contract)

    @property
    def contract_version(self) -> ContractVersion:
        return ContractVersion.parse(self.contract)

    @property
    def ref(self) -> str:
        return f"{self.plane}:{self.name}@{self.version}"

    def capability(self, key: str, default: Any = None) -> Any:
        return self.provides.get(key, default)

    def needs(self, key: str, default: Any = None) -> Any:
        return self.requires.get(key, default)

    def provides_all(self, keys: Mapping[str, Any]) -> Verdict:
        """Does this descriptor provide every listed capability, as required?

        A required value of ``None`` means "must be present, value unchecked".
        Anything else must match exactly. Richer matching (ranges, subsets)
        belongs to the contract that defines the key, not here.
        """
        checks = []
        for key, wanted in keys.items():
            if key not in self.provides:
                checks.append(
                    Verdict.no(
                        "capability.missing",
                        f"{self.ref} does not provide {key!r}",
                        hint=f"provides: {sorted(self.provides)}",
                        capability=key,
                    )
                )
            elif wanted is not None and self.provides[key] != wanted:
                checks.append(
                    Verdict.no(
                        "capability.value",
                        f"{self.ref} provides {key}={self.provides[key]!r}, wanted {wanted!r}",
                        capability=key,
                    )
                )
        return Verdict.all(checks)

    def component_ref(
        self, config: Mapping[str, Any] | None = None, artifact_digest: str | None = None
    ) -> ComponentRef:
        """The provenance stamp for a run that used this plugin."""
        return ComponentRef(
            plane=self.plane,
            name=self.name,
            version=self.version,
            config_digest=digest_of(config) if config else None,
            artifact_digest=artifact_digest,
        )

    def validate(self) -> Verdict:
        checks = []
        if not self.name:
            checks.append(Verdict.no("descriptor.name", "descriptor has an empty name"))
        if self.isolation == "in-process" and self.metadata.get("conflicts"):
            checks.append(
                Verdict.note(
                    "isolation.risky",
                    f"{self.ref} declares dependency conflicts but asks to run in-process",
                    hint="declare isolation='venv' or 'container' instead",
                )
            )
        return Verdict.all(checks)

    # -- serialisation -----------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane,
            "name": self.name,
            "version": self.version,
            "contract": self.contract,
            "requires": dict(self.requires),
            "provides": dict(self.provides),
            "isolation": self.isolation,
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Descriptor":
        missing = [key for key in ("plane", "name", "version", "contract") if key not in payload]
        if missing:
            raise ValueError(f"descriptor is missing required field(s): {missing}")
        return cls(
            plane=payload["plane"],
            name=payload["name"],
            version=payload["version"],
            contract=payload["contract"],
            requires=payload.get("requires", {}),
            provides=payload.get("provides", {}),
            isolation=payload.get("isolation", "in-process"),
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "Descriptor":
        return cls.from_dict(json.loads(text))
