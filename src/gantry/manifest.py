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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .spine import Verdict

MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ComponentSpec:
    """One named component and the configuration it is built with.

    ``over`` makes this a link in a chain: the component this one reads from,
    built first and handed in. A chain is therefore a linked list of specs
    rather than a separate type, which keeps one thing to parse, one thing to
    serialise, and one thing for the runner to walk.

    Chains exist because the interesting pipelines are compositions. Reading raw
    ego video, estimating hands onto it, and turning those into robot actions is
    three connectors stacked, and until a manifest could say so the whole
    pipeline lived in hand-written scripts -- declarative in principle and
    bespoke in practice. A manifest that cannot express the thing it is for is
    not a manifest.
    """

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    #: The stage this one is built on top of, if any.
    over: "ComponentSpec | None" = None
    #: Whether the run continues without this link when it cannot be built.
    #:
    #: For stages that genuinely may not apply -- lifting to a world frame needs
    #: a camera trajectory, and half of real uploads do not have one. An
    #: optional link that fails is a note in the record, not a failed run; a
    #: required one that fails is a refusal.
    optional: bool = False

    @classmethod
    def parse(cls, payload: Any, where: str) -> "ComponentSpec":
        if isinstance(payload, str):
            return cls(payload, {})
        if isinstance(payload, Mapping):
            if "chain" in payload:
                return cls._chain(payload["chain"], where)
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise ConfigError(f"{where}: needs a non-empty 'name'")
            config = payload.get("config", {})
            if not isinstance(config, Mapping):
                raise ConfigError(f"{where}: 'config' must be an object")
            return cls(name, dict(config), optional=bool(payload.get("optional", False)))
        raise ConfigError(f"{where}: expected a name or an object, got {type(payload).__name__}")

    @classmethod
    def _chain(cls, payload: Any, where: str) -> "ComponentSpec":
        """A list of stages, innermost first, folded into a linked spec."""
        if not isinstance(payload, list) or not payload:
            raise ConfigError(
                f"{where}: 'chain' must be a non-empty list of components, "
                "innermost first, the one that reads from disk comes first and "
                "each later one reads from the one before it"
            )
        built: "ComponentSpec | None" = None
        for index, item in enumerate(payload):
            link = cls.parse(item, f"{where}.chain[{index}]")
            if index == 0 and link.optional:
                raise ConfigError(
                    f"{where}.chain[0]: the first stage cannot be optional; it is "
                    "what everything after it reads from"
                )
            built = replace(link, over=built)
        assert built is not None
        return built

    @property
    def chain(self) -> tuple["ComponentSpec", ...]:
        """This spec and everything under it, innermost first."""
        out: list[ComponentSpec] = []
        node: ComponentSpec | None = self
        while node is not None:
            out.append(node)
            node = node.over
        return tuple(reversed(out))

    @property
    def chained(self) -> bool:
        return self.over is not None

    @property
    def ref(self) -> str:
        """How this reads in a plan: ``egovideo -> handpose -> egoactions``."""
        return " -> ".join(link.name for link in self.chain)

    def as_dict(self) -> dict[str, Any]:
        if self.chained:
            return {
                "chain": [
                    {
                        "name": link.name,
                        "config": dict(link.config),
                        **({"optional": True} if link.optional else {}),
                    }
                    for link in self.chain
                ]
            }
        return {"name": self.name, "config": dict(self.config)}


@dataclass(frozen=True)
class Manifest:
    """A complete, re-runnable description of one piece of work."""

    name: str
    cohorts: Mapping[str, ComponentSpec] = field(default_factory=dict)
    #: Which plane the cohorts are components of -- the axis under comparison.
    #:
    #: Defaulted to the dataset plane because that is the common case, not
    #: because it is privileged. Comparing three checkpoints in one world is the
    #: same shape of question as comparing three datasets under one policy, and
    #: a manifest that could only express the second was quietly saying the
    #: dataset plane matters more. It does not.
    #:
    #: Whatever this names is the one thing that varies; every other plane is
    #: single-valued and therefore held. That is the same distinction a feedback
    #: module makes with ``holds``, seen from the other side, and the two are
    #: checked against each other from provenance at analysis time.
    varies: str = "dataset"
    #: Every single-valued plane, keyed by plane name. A plane registered by a
    #: plugin lands here with no edit to this class -- which is the difference
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
        varies = payload.get("varies", "dataset")
        if isinstance(raw_cohorts, Mapping) and "plane" in raw_cohorts and "of" in raw_cohorts:
            # {"cohorts": {"plane": "policy", "of": {...}}} -- the explicit form.
            varies = str(raw_cohorts["plane"])
            raw_cohorts = raw_cohorts["of"]
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

        # Keys with a dedicated shape of their own. ``feedback`` is a plane and
        # a list, so it must not also be read as a single component here.
        reserved = {"version", "name", "cohorts", "varies", "feedback", "protocol", "metadata"}

        components: dict[str, ComponentSpec] = {}
        for plane in known_planes():
            if plane in reserved:
                continue
            described = get_plane(plane)
            if described is None:
                continue
            # A many-valued plane used to be skipped here, because the only
            # many-valued plane was the one cohorts were always built on. Now
            # that any plane can be the axis, naming one as a single component
            # is meaningful -- "the dataset every policy is measured on". The
            # conflict check below catches naming it both ways.
            if payload.get(plane) is not None:
                components[plane] = ComponentSpec.parse(payload[plane], f"{where}.{plane}")

        unknown = [key for key in payload if key not in reserved and key not in known_planes()]
        if unknown:
            raise ConfigError(
                f"{where}: unknown key(s) {sorted(unknown)}; "
                f"expected a plane or one of {sorted(reserved)}"
            )

        protocol = payload.get("protocol", {})
        if not isinstance(protocol, Mapping):
            raise ConfigError(f"{where}: 'protocol' must be an object")

        from .spine import get_plane as _plane_named

        if _plane_named(str(varies)) is None:
            raise ConfigError(
                f"{where}: cohorts vary on {varies!r}, which is not a plane; "
                f"known planes are {sorted(known_planes())}"
            )
        if str(varies) in components:
            raise ConfigError(
                f"{where}: {varies!r} is named both as the varying axis and as a "
                f"single component. It is one or the other, if it varies, every value "
                f"belongs in 'cohorts'."
            )
        return cls(
            name=name,
            cohorts=cohorts,
            varies=str(varies),
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
        """Whether this manifest runs a policy, rather than only reading data.

        A plane that varies is supplied by the cohorts rather than by a single
        component, so it counts as present either way.
        """
        return self.provides("policy") and self.provides("evaluation")

    def provides(self, plane: str) -> bool:
        """Is this plane supplied at all -- as one component or as the axis?"""
        return self.varies == plane or self.components.get(plane) is not None

    def spec_for(self, plane: str, cohort: str) -> ComponentSpec | None:
        """The component for one plane in one cohort.

        The varying plane changes per cohort; every other plane is the same one
        throughout. Callers ask this rather than reaching for ``.policy``, so
        adding a new varying axis costs nothing at the call site.
        """
        if self.varies == plane:
            return self.cohorts.get(cohort)
        return self.components.get(plane)

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
        # Asked through provides(), because a plane may be supplied by the
        # cohorts rather than by a single component. Reading .policy directly
        # would call a policy-varying run half an evaluation.
        if self.provides("policy") != self.provides("evaluation"):
            missing = "evaluation" if self.provides("policy") else "policy"
            checks.append(
                Verdict.no(
                    "manifest.half_an_evaluation",
                    f"{self.name}: a run needs both a policy and an evaluation; "
                    f"{missing!r} is missing",
                    hint="drop both to analyse the data as recorded",
                )
            )
        if self.varies != "dataset" and not self.provides("dataset"):
            checks.append(
                Verdict.no(
                    "manifest.no_dataset",
                    f"{self.name}: cohorts vary on {self.varies!r}, so the dataset plane "
                    "has to be named, otherwise there is nothing for them to be "
                    "measured on",
                )
            )
        return Verdict.all(checks)


def manifest_from(path: str | os.PathLike[str]) -> Manifest:
    return Manifest.load(path)
