"""Where a number came from.

No naked floats. A success rate that does not know which policy, checkpoint,
embodiment, task, evaluator, protocol and adapter chain produced it is not a
result, it is a rumour. :class:`Measurement` makes that structural, and
:class:`Provenance` makes it comparable: two runs are comparable exactly when
their provenance matches on the axes being compared, which is the mechanism
that stops cross-embodiment numbers being averaged into one meaningless table.

Execution protocol lives in here rather than in a config file, because it
changes results. Action-chunk size alone moved a measured policy by fourteen
points in this project's own benchmark; a record that omits it is
irreproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .plane import CORE_PLANES, known_planes

#: Kept as the historical name for the planes core itself declares. Anything
#: validating a plane should ask :func:`~gantry.spine.plane.known_planes`
#: instead, so a plane a plugin registered is accepted like any other.
PLANES = CORE_PLANES


def digest_of(payload: Any) -> str:
    """A stable short digest of any JSON-able payload."""
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def seed_from(*parts: Any) -> int:
    """A reproducible RNG seed derived from arbitrary values.

    Here because the obvious way to do this is wrong in a way that hides. The
    builtin ``hash()`` is salted per process for strings, so
    ``default_rng(abs(hash((seed, episode_id, step))))`` gives a different
    stream in every interpreter -- while the object still reports itself as
    deterministic, and still *is* deterministic within one process, which is
    the only place a conformance check can look.

    Both reference policies shipped with that bug. Nothing failed loudly: they
    passed their determinism checks, and two runs of the same seeded experiment
    on different days simply disagreed. Paired comparison rests on this, so it
    belongs in core where it can be got right once.
    """
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return int.from_bytes(hashlib.blake2b(encoded.encode(), digest_size=4).digest(), "big")


#: 95% two-sided normal quantile.
Z95 = 1.959963984540054


def proportion(successes: int, n: int, *, z: float = Z95) -> "Measurement":
    """A success rate that carries its own interval.

    Here rather than in a feedback module because *every* evaluator reports one
    and the rule is that no number travels naked. A rate of 1.0 from four
    trials and a rate of 1.0 from four hundred are the same float and not the
    same claim, and only the interval says so.

    Wilson rather than the textbook normal interval: at the small n an
    evaluation actually runs, and at rates near zero or one where policies
    usually sit, the normal interval runs past the ends of the scale and gives
    negative lower bounds. Wilson does not.
    """
    if n <= 0:
        return Measurement(0.0, n=0, ci=(0.0, 1.0), units="fraction", method="wilson")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return Measurement(
        p,
        n=n,
        ci=(max(0.0, centre - half), min(1.0, centre + half)),
        units="fraction",
        method="wilson",
    )


def mcnemar(only_left: int, only_right: int) -> float:
    """Exact two-sided p-value over the trials where two arms disagreed.

    Paired, and that is the whole point. Two policies run on the same scenes
    agree on most of them -- the easy ones both solve, the hard ones both fail --     and those trials carry no information about which is better. Comparing
    marginal rates throws that pairing away and needs far more trials to see the
    same difference. Only the disagreements count here.

    Exact rather than the chi-squared approximation, because an evaluation that
    disagrees on nine scenes is normal and the approximation is not trustworthy
    there.
    """
    from math import comb

    discordant = only_left + only_right
    if discordant == 0:
        return 1.0
    smaller = min(only_left, only_right)
    tail = sum(comb(discordant, k) for k in range(smaller + 1)) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True)
class ComponentRef:
    """A pinned reference to one plugin as it was actually used."""

    plane: str
    name: str
    version: str
    config_digest: str | None = None
    artifact_digest: str | None = None  # e.g. a checkpoint hash
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.plane not in known_planes():
            raise ValueError(f"unknown plane {self.plane!r}; expected one of {known_planes()}")

    @property
    def ref(self) -> str:
        base = f"{self.plane}:{self.name}@{self.version}"
        return f"{base}+{self.artifact_digest}" if self.artifact_digest else base

    def as_dict(self) -> dict[str, Any]:
        return {
            "plane": self.plane,
            "name": self.name,
            "version": self.version,
            "config_digest": self.config_digest,
            "artifact_digest": self.artifact_digest,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class AdapterStep:
    """One transform applied between planes, with what it cost.

    ``losses`` is free text on purpose, but it is *mandatory* to state
    something when a transform is lossy. A resampler that drops from 30 Hz to
    20 Hz should say so, and that sentence travels with every number the run
    produces.
    """

    name: str
    version: str
    losses: tuple[str, ...] = ()

    @property
    def lossy(self) -> bool:
        return bool(self.losses)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "losses": list(self.losses)}


@dataclass(frozen=True)
class Provenance:
    """The full origin of a run."""

    components: tuple[ComponentRef, ...] = ()
    protocol: Mapping[str, Any] = field(default_factory=dict)
    adapters: tuple[AdapterStep, ...] = ()
    created_at: str | None = None
    host: str | None = None
    notes: tuple[str, ...] = ()

    def component(self, plane: str) -> ComponentRef | None:
        for component in self.components:
            if component.plane == plane:
                return component
        return None

    @property
    def losses(self) -> tuple[str, ...]:
        return tuple(loss for adapter in self.adapters for loss in adapter.losses)

    @property
    def lossy(self) -> bool:
        return any(adapter.lossy for adapter in self.adapters)

    def as_dict(self) -> dict[str, Any]:
        return {
            "components": [component.as_dict() for component in self.components],
            "protocol": dict(self.protocol),
            "adapters": [adapter.as_dict() for adapter in self.adapters],
            "created_at": self.created_at,
            "host": self.host,
            "notes": list(self.notes),
        }

    @property
    def digest(self) -> str:
        """Identity of the *experiment*, ignoring when and where it ran."""
        payload = self.as_dict()
        payload.pop("created_at")
        payload.pop("host")
        payload.pop("notes")
        return digest_of(payload)

    def agreement(self, other: "Provenance", plane: str) -> str:
        """How two runs stand on one plane: ``same``, ``differs``, ``one_sided``
        or ``absent``.

        Four answers rather than two, because :meth:`comparable_to` collapses
        three of them into a bare ``False`` and they call for different
        responses. Two runs that both name the same policy agree. Two that name
        different ones are a confound and the comparison is void. One that names
        a policy against one that does not is *unverifiable*, which is a
        different problem -- somebody may have changed it and not written it
        down. And two that both name nothing had no policy at all, which is the
        ordinary state of a screen over raw recordings and not a defect.

        Treating that last case as a failure is what made a raw-dataset
        comparison look suspicious when it was merely early.
        """
        mine, theirs = self.component(plane), other.component(plane)
        if mine is None and theirs is None:
            return "absent"
        if mine is None or theirs is None:
            return "one_sided"
        return "same" if mine.ref == theirs.ref else "differs"

    def comparable_to(self, other: "Provenance", holding: Iterable[str]) -> bool:
        """Do two runs agree on every plane in ``holding``?

        The guard against nonsense aggregation: comparing two policies is only
        meaningful when embodiment, task and evaluator are held fixed, and this
        is how a caller states which axes it is holding.
        """
        for plane in holding:
            mine, theirs = self.component(plane), other.component(plane)
            if mine is None or theirs is None or mine.ref != theirs.ref:
                return False
        return True


@dataclass(frozen=True)
class Measurement:
    """A number that knows what it is."""

    value: float
    n: int | None = None
    ci: tuple[float, float] | None = None
    units: str | None = None
    method: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "n": self.n,
            "ci": list(self.ci) if self.ci else None,
            "units": self.units,
            "method": self.method,
            "detail": dict(self.detail),
        }

    def __str__(self) -> str:
        text = f"{self.value:g}"
        if self.units:
            text += f" {self.units}"
        if self.ci:
            text += f" [{self.ci[0]:g}, {self.ci[1]:g}]"
        if self.n is not None:
            text += f" (n={self.n})"
        return text
