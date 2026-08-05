"""What one stream of data is, and whether two of them fit together.

The neutrality rule lives here. A channel is described on two axes:

``kind``
    The *modality* -- what shape of thing this is. Core owns this vocabulary
    because it is genuinely universal: an image is an image on a humanoid, a
    drone, or a microscope. Small, extensible, and nothing robot-specific.

``semantics``
    What the numbers *mean* -- ``joint_position``, ``eef_pose``, ``wrench``,
    anything. This registry is open and core ships only a starter vocabulary.
    **No logic in core ever branches on a semantics value.** Retargeters do,
    and retargeters are plugins.

That split is what lets an embodiment nobody has imagined register its own
semantics without patching the framework, while still getting shape, dtype
and dimensional checking for free.

:func:`compatible` answers only "do these fit with no adapter?". Mismatches
come back as codes rather than a flat no, so the resolver can go looking for
an adapter that closes each specific gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from . import units
from .verdict import Verdict

# --------------------------------------------------------------------------
# modality vocabulary (core-owned, extensible)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    """A modality: the shape family of a single timestep's value."""

    name: str
    ranks: frozenset[int] | None
    description: str = ""

    def rank_ok(self, shape: tuple[int | None, ...]) -> bool:
        return self.ranks is None or len(shape) in self.ranks


_KINDS: dict[str, Kind] = {}


def register_kind(
    name: str, ranks: frozenset[int] | set[int] | None, description: str = ""
) -> Kind:
    """Register a modality. Idempotent for identical definitions."""
    kind = Kind(name, None if ranks is None else frozenset(ranks), description)
    existing = _KINDS.get(name)
    if existing is not None and existing != kind:
        raise ValueError(f"kind {name!r} already registered with a different definition")
    _KINDS[name] = kind
    return kind


def known_kinds() -> tuple[str, ...]:
    return tuple(sorted(_KINDS))


def get_kind(name: str) -> Kind | None:
    return _KINDS.get(name)


for _name, _ranks, _description in [
    ("scalar", {0}, "a single number per step"),
    ("vector", {1}, "a 1-D array per step"),
    ("matrix", {2}, "a 2-D array per step"),
    ("tensor", None, "an array of any rank per step"),
    ("image", {2, 3}, "H×W or H×W×C per step"),
    ("depth_image", {2, 3}, "per-pixel range"),
    ("point_cloud", {2}, "N×D points per step"),
    ("audio", {1}, "a sample window per step"),
    ("text", {0}, "a string per step"),
    ("boolean", {0}, "a flag per step"),
    ("categorical", {0}, "a discrete label per step"),
    ("timestamp", {0}, "a clock reading per step"),
    ("blob", None, "opaque bytes; carried, never interpreted"),
]:
    register_kind(_name, _ranks, _description)


# --------------------------------------------------------------------------
# semantics vocabulary (open; core never branches on it)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Semantics:
    """A meaning tag, optionally pinned to a physical dimension."""

    name: str
    description: str = ""
    dimension: units.Dimension | None = None


_SEMANTICS: dict[str, Semantics] = {}


def register_semantics(
    name: str, description: str = "", dimension: units.Dimension | None = None
) -> Semantics:
    """Register a meaning tag. Plugins call this for their own vocabulary."""
    semantics = Semantics(name, description, dimension)
    existing = _SEMANTICS.get(name)
    if existing is not None and existing != semantics:
        raise ValueError(f"semantics {name!r} already registered with a different definition")
    _SEMANTICS[name] = semantics
    return semantics


def known_semantics() -> tuple[str, ...]:
    return tuple(sorted(_SEMANTICS))


def get_semantics(name: str) -> Semantics | None:
    return _SEMANTICS.get(name)


# A starter vocabulary, deliberately generic. Nothing here is privileged: an
# embodiment plugin may ignore all of it and register its own.
for _name, _description, _dimension in [
    ("position", "a position in some frame", units.LENGTH),
    ("orientation", "an orientation in some frame", None),
    ("pose", "position and orientation together", None),
    ("linear_velocity", "rate of change of position", None),
    ("angular_velocity", "rate of change of orientation", None),
    ("force", "a linear force", units.FORCE),
    ("torque", "a moment", units.TORQUE),
    ("joint_position", "per-joint configuration", None),
    ("joint_velocity", "per-joint rate", None),
    ("effort", "commanded or measured actuation effort", None),
    ("actuation", "a generic actuator command", None),
    ("observation", "an uninterpreted sensor reading", None),
    ("instruction", "a task instruction", None),
    ("reward", "a scalar objective signal", None),
    ("time", "elapsed or absolute time", units.TIME),
]:
    register_semantics(_name, _description, _dimension)


# --------------------------------------------------------------------------
# the channel
# --------------------------------------------------------------------------

Shape = tuple[int | None, ...]

#: Refusal codes emitted when a declared tag differs between two channels.
#: Spelled out rather than composed from an f-string: plugins branch on these,
#: and a code that only exists at runtime cannot be found by anyone reading the
#: source to learn what they may rely on.
TAG_CODES: dict[str, tuple[str, str]] = {
    "frame": ("frame.mismatch", "frame.undeclared"),
    "semantics": ("semantics.mismatch", "semantics.undeclared"),
}


@dataclass(frozen=True)
class ChannelSpec:
    """One named stream within an episode.

    ``shape`` describes a *single timestep*, never the time axis. ``None`` in
    a shape is a wildcard, which is how a consumer says "any width".
    """

    name: str
    kind: str
    shape: Shape = ()
    dtype: str = "float32"
    units: str | None = None
    frame: str | None = None
    rate_hz: float | None = None
    semantics: str | None = None
    #: A name per element of a single timestep, in order. Lets tooling address
    #: a dimension by what it is rather than by where it happens to sit, which
    #: is the difference between "element 9 is out of range" and "the right
    #: gripper is out of range".
    dim_labels: tuple[str, ...] | None = None
    optional: bool = False
    #: Metadata keys that are load-bearing: two channels are only compatible if
    #: they agree on every one of them.
    #:
    #: Core has no idea what any of them mean, and does not need to. It exists
    #: because some distinctions live in a vocabulary core must not own -- a
    #: quaternion stored scalar-first versus scalar-last is four floats either
    #: way, the same width, the same units, the same meaning tag, and executing
    #: one as the other is silently wrong. Listing the key here is how a plugin
    #: says "this must match" without core learning what a quaternion is.
    discriminators: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # -- validation --------------------------------------------------------

    def validate(self) -> Verdict:
        """Is this spec internally coherent? Says nothing about any data."""
        checks = [
            self._check_kind(),
            self._check_dtype(),
            self._check_units(),
            self._check_semantics_dimension(),
            self._check_dim_labels(),
        ]
        return Verdict.all(checks)

    @property
    def width(self) -> int:
        """Elements in one timestep. 1 for a scalar."""
        total = 1
        for dimension in self.shape:
            total *= dimension or 1
        return total

    def _check_dim_labels(self) -> Verdict:
        if self.dim_labels is None:
            return Verdict.yes()
        if any(dimension is None for dimension in self.shape):
            return Verdict.note(
                "dim_labels.wildcard",
                f"channel {self.name!r} labels its dimensions but has a wildcard shape",
                hint="the label count cannot be checked against an unknown width",
            )
        if len(self.dim_labels) != self.width:
            return Verdict.no(
                "dim_labels.count",
                f"channel {self.name!r} has {len(self.dim_labels)} labels "
                f"for {self.width} element(s)",
                channel=self.name,
            )
        if len(set(self.dim_labels)) != len(self.dim_labels):
            duplicates = sorted({n for n in self.dim_labels if self.dim_labels.count(n) > 1})
            return Verdict.no(
                "dim_labels.duplicate",
                f"channel {self.name!r} repeats dimension label(s) {duplicates}",
                hint="labels address dimensions, so duplicates make one unaddressable",
                channel=self.name,
            )
        return Verdict.yes()

    def _check_kind(self) -> Verdict:
        kind = get_kind(self.kind)
        if kind is None:
            return Verdict.no(
                "kind.unknown",
                f"channel {self.name!r} has unknown kind {self.kind!r}",
                hint=f"known kinds: {', '.join(known_kinds())}",
                channel=self.name,
            )
        if not kind.rank_ok(self.shape):
            allowed = sorted(kind.ranks or ())
            return Verdict.no(
                "kind.rank",
                f"channel {self.name!r} is kind {self.kind!r} but has rank {len(self.shape)}",
                hint=f"{self.kind} allows rank {allowed}",
                channel=self.name,
            )
        return Verdict.yes()

    def _check_dtype(self) -> Verdict:
        try:
            np.dtype(self.dtype)
        except TypeError:
            return Verdict.no(
                "dtype.unknown",
                f"channel {self.name!r} has unusable dtype {self.dtype!r}",
                channel=self.name,
            )
        return Verdict.yes()

    def _check_units(self) -> Verdict:
        try:
            units.parse(self.units)
        except units.UnknownUnitError as error:
            return Verdict.no(
                "units.unknown",
                f"channel {self.name!r}: {error}",
                channel=self.name,
            )
        return Verdict.yes()

    def _check_semantics_dimension(self) -> Verdict:
        if self.semantics is None:
            return Verdict.yes()
        semantics = get_semantics(self.semantics)
        if semantics is None:
            # Unregistered semantics is a note, not a failure: an embodiment
            # may legitimately be first to use a tag. The resolver decides
            # whether it cares.
            return Verdict.note(
                "semantics.unregistered",
                f"channel {self.name!r} uses unregistered semantics {self.semantics!r}",
                hint="register it with gantry.spine.channel.register_semantics",
                channel=self.name,
            )
        if semantics.dimension is None or self.units is None:
            return Verdict.yes()
        declared = units.parse(self.units).dimension
        if declared != semantics.dimension:
            return Verdict.no(
                "units.dimension",
                f"channel {self.name!r} is {self.semantics!r} "
                f"(expects {semantics.dimension}) but declares units {self.units!r} "
                f"({declared})",
                channel=self.name,
            )
        return Verdict.yes()

    # -- data agreement ----------------------------------------------------

    def accepts(self, array: np.ndarray) -> Verdict:
        """Does an array of shape ``(steps, *self.shape)`` match this spec?"""
        if array.ndim != len(self.shape) + 1:
            return Verdict.no(
                "data.rank",
                f"channel {self.name!r} expects rank {len(self.shape) + 1} "
                f"(steps + {len(self.shape)}), got {array.ndim}",
                channel=self.name,
            )
        for axis, (expected, actual) in enumerate(zip(self.shape, array.shape[1:]), start=1):
            if expected is not None and expected != actual:
                return Verdict.no(
                    "data.shape",
                    f"channel {self.name!r} axis {axis}: expected {expected}, got {actual}",
                    channel=self.name,
                )
        if not np.can_cast(array.dtype, np.dtype(self.dtype), casting="same_kind"):
            return Verdict.no(
                "data.dtype",
                f"channel {self.name!r} holds {array.dtype}, "
                f"which is not same-kind castable to {self.dtype}",
                channel=self.name,
            )
        return Verdict.yes()


def _shapes_fit(provider: Shape, consumer: Shape) -> bool:
    if len(provider) != len(consumer):
        return False
    return all(want is None or want == have for have, want in zip(provider, consumer))


def compatible(provider: ChannelSpec, consumer: ChannelSpec) -> Verdict:
    """Can ``provider`` feed ``consumer`` directly, with no adapter?

    A refusal itemises *what* does not line up, using stable codes. The
    resolver reads those codes to look for adapters (``rate.mismatch`` wants a
    resampler, ``frame.mismatch`` wants a transform, and so on), so a "no"
    here is the start of a search, not the end of one.
    """
    checks: list[Verdict] = []

    if provider.kind != consumer.kind:
        checks.append(
            Verdict.no(
                "kind.mismatch",
                f"{provider.name!r} is {provider.kind!r}, {consumer.name!r} wants {consumer.kind!r}",
                provider=provider.kind,
                consumer=consumer.kind,
            )
        )

    if not _shapes_fit(provider.shape, consumer.shape):
        checks.append(
            Verdict.no(
                "shape.mismatch",
                f"{provider.name!r} has shape {provider.shape}, "
                f"{consumer.name!r} wants {consumer.shape}",
                provider=provider.shape,
                consumer=consumer.shape,
            )
        )

    if not np.can_cast(np.dtype(provider.dtype), np.dtype(consumer.dtype), casting="same_kind"):
        checks.append(
            Verdict.no(
                "dtype.mismatch",
                f"{provider.dtype} is not same-kind castable to {consumer.dtype}",
                provider=provider.dtype,
                consumer=consumer.dtype,
            )
        )

    checks.append(_units_fit(provider, consumer))
    checks.append(_tag_fit("frame", provider.frame, consumer.frame, provider, consumer))
    checks.append(_tag_fit("semantics", provider.semantics, consumer.semantics, provider, consumer))
    checks.append(_rate_fit(provider, consumer))
    checks.append(_labels_fit(provider, consumer))
    checks.append(_discriminators_fit(provider, consumer))

    return Verdict.all(checks)


def _discriminators_fit(provider: ChannelSpec, consumer: ChannelSpec) -> Verdict:
    """Every load-bearing metadata key either side declared must agree.

    Declared by either side, not both: a consumer that says an encoding matters
    is entitled to an answer from a provider that never thought about it, and
    the answer is "undeclared", not "fine".
    """
    keys = dict.fromkeys((*provider.discriminators, *consumer.discriminators))
    checks = []
    for key in keys:
        mine = provider.metadata.get(key)
        theirs = consumer.metadata.get(key)
        if mine is None or theirs is None:
            if mine != theirs:
                checks.append(
                    Verdict.note(
                        "metadata.undeclared",
                        f"{key!r} is load-bearing but declared on only one side "
                        f"({provider.name!r}={mine!r}, {consumer.name!r}={theirs!r})",
                        hint="declare it on both sides to get a real answer",
                        key=key,
                    )
                )
            continue
        if mine != theirs:
            checks.append(
                Verdict.no(
                    "metadata.mismatch",
                    f"{key} {mine!r} vs {theirs!r}",
                    hint="the two channels differ in something declared to matter",
                    key=key,
                    provider=mine,
                    consumer=theirs,
                )
            )
    return Verdict.all(checks)


def _labels_fit(provider: ChannelSpec, consumer: ChannelSpec) -> Verdict:
    """Same names in the same order, when both sides bother to say.

    Two arms with identical widths and identical joint names in opposite order
    are compatible on every other axis and produce garbage. Order is the whole
    content of a dimension label.
    """
    if provider.dim_labels is None or consumer.dim_labels is None:
        if provider.dim_labels != consumer.dim_labels:
            return Verdict.note(
                "dim_labels.undeclared",
                "dimension labels declared on only one side",
                hint="cannot be checked; declare them on both sides to get a real answer",
            )
        return Verdict.yes()
    if provider.dim_labels != consumer.dim_labels:
        if set(provider.dim_labels) == set(consumer.dim_labels):
            return Verdict.no(
                "dim_labels.order",
                f"{provider.name!r} and {consumer.name!r} label the same dimensions "
                "in a different order",
                hint="needs a permutation, not a rename",
                provider=list(provider.dim_labels),
                consumer=list(consumer.dim_labels),
            )
        return Verdict.no(
            "dim_labels.mismatch",
            f"{provider.name!r} and {consumer.name!r} label different dimensions",
            provider=list(provider.dim_labels),
            consumer=list(consumer.dim_labels),
        )
    return Verdict.yes()


def _units_fit(provider: ChannelSpec, consumer: ChannelSpec) -> Verdict:
    if provider.units is None or consumer.units is None:
        if provider.units != consumer.units:
            return Verdict.note(
                "units.undeclared",
                f"units declared on only one side "
                f"({provider.name!r}={provider.units!r}, {consumer.name!r}={consumer.units!r})",
                hint="cannot be checked; declare them on both sides to get a real answer",
            )
        return Verdict.yes()
    try:
        provider_unit = units.parse(provider.units)
        consumer_unit = units.parse(consumer.units)
    except units.UnknownUnitError as error:
        return Verdict.no("units.unknown", str(error))
    if provider_unit.dimension != consumer_unit.dimension:
        return Verdict.no(
            "units.dimension",
            f"{provider.units!r} is {provider_unit.dimension}, "
            f"{consumer.units!r} is {consumer_unit.dimension}",
            hint="these are different physical quantities; no scaling fixes this",
        )
    if provider_unit.factor != consumer_unit.factor:
        return Verdict.no(
            "units.scale",
            f"{provider.units!r} and {consumer.units!r} differ by a constant factor",
            hint="needs a unit conversion",
            factor=provider_unit.factor / consumer_unit.factor,
        )
    return Verdict.yes()


def _tag_fit(
    label: str,
    provider_value: str | None,
    consumer_value: str | None,
    provider: ChannelSpec,
    consumer: ChannelSpec,
) -> Verdict:
    mismatch_code, undeclared_code = TAG_CODES[label]
    if provider_value is None or consumer_value is None:
        if provider_value != consumer_value:
            return Verdict.note(
                undeclared_code,
                f"{label} declared on only one side "
                f"({provider.name!r}={provider_value!r}, {consumer.name!r}={consumer_value!r})",
                hint="cannot be checked; declare it on both sides to get a real answer",
            )
        return Verdict.yes()
    if provider_value != consumer_value:
        return Verdict.no(
            mismatch_code,
            f"{label} {provider_value!r} vs {consumer_value!r}",
            provider=provider_value,
            consumer=consumer_value,
        )
    return Verdict.yes()


def _rate_fit(provider: ChannelSpec, consumer: ChannelSpec) -> Verdict:
    if provider.rate_hz is None or consumer.rate_hz is None:
        return Verdict.yes()
    if provider.rate_hz != consumer.rate_hz:
        return Verdict.no(
            "rate.mismatch",
            f"{provider.rate_hz} Hz vs {consumer.rate_hz} Hz",
            hint="needs a resampler",
            provider=provider.rate_hz,
            consumer=consumer.rate_hz,
        )
    return Verdict.yes()
