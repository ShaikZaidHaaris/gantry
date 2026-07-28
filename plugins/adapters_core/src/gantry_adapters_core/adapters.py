"""Unit conversion, dimension permutation, and rate resampling."""

from __future__ import annotations

import numpy as np
from gantry.resolve import Adapter, AdapterRegistry
from gantry.spine import ChannelSpec, Verdict, units

VERSION = "0.1.0.dev0"


# --------------------------------------------------------------------------
# units: exact, reversible, and refuses anything that is not
# --------------------------------------------------------------------------


def _units_guard(source: ChannelSpec, target: ChannelSpec) -> Verdict:
    if source.units is None or target.units is None:
        return Verdict.no(
            "adapter.units_undeclared",
            "cannot convert units that are not declared on both sides",
        )
    try:
        factor = units.conversion_factor(source.units, target.units)
    except units.UnknownUnitError as error:
        return Verdict.no("adapter.units_unconvertible", str(error))
    return Verdict.note(
        "adapter.units_factor", f"{source.units} -> {target.units} is x{factor:g}"
    )


def convert_units(values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
    factor = units.conversion_factor(source.units, target.units)
    return np.asarray(values, dtype=float) * factor


UNIT_CONVERT = Adapter(
    name="unit-convert",
    version=VERSION,
    closes=("units.scale",),
    guard=_units_guard,
    # Exact rescaling within one dimension. Nothing is discarded, so nothing is
    # claimed: an empty loss list here is a statement, not an omission.
    cost=lambda source, target: (),
    transform=convert_units,
    metadata={"reversible": True},
)


# --------------------------------------------------------------------------
# permutation: reordering, never renaming
# --------------------------------------------------------------------------


def permutation_between(source: ChannelSpec, target: ChannelSpec) -> tuple[int, ...] | None:
    """Index map taking source order to target order, or ``None`` if impossible."""
    if source.dim_labels is None or target.dim_labels is None:
        return None
    if sorted(source.dim_labels) != sorted(target.dim_labels):
        return None
    position = {label: index for index, label in enumerate(source.dim_labels)}
    return tuple(position[label] for label in target.dim_labels)


def _permute_guard(source: ChannelSpec, target: ChannelSpec) -> Verdict:
    if permutation_between(source, target) is None:
        return Verdict.no(
            "adapter.not_a_permutation",
            "the two channels do not label the same set of dimensions",
            hint="a differing set needs a retargeter that decides what maps to what",
        )
    return Verdict.yes()


def permute(values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
    order = permutation_between(source, target)
    if order is None:
        raise ValueError(f"{source.name!r} and {target.name!r} are not a permutation")
    return np.asarray(values)[:, list(order)]


PERMUTE = Adapter(
    name="permute",
    version=VERSION,
    closes=("dim_labels.order",),
    guard=_permute_guard,
    cost=lambda source, target: (),
    transform=permute,
    metadata={"reversible": True},
)


# --------------------------------------------------------------------------
# resampling: the one that costs something
# --------------------------------------------------------------------------


def _resample_guard(source: ChannelSpec, target: ChannelSpec) -> Verdict:
    if source.rate_hz is None or target.rate_hz is None:
        return Verdict.no(
            "adapter.rate_undeclared",
            "cannot resample between rates that are not declared on both sides",
        )
    if source.rate_hz <= 0 or target.rate_hz <= 0:
        return Verdict.no("adapter.rate_invalid", "rates must be positive")
    if source.kind not in {"scalar", "vector"}:
        return Verdict.no(
            "adapter.rate_kind",
            f"linear resampling is only defined for scalars and vectors, not {source.kind!r}",
            hint="an image or a discrete label needs a purpose-built resampler",
        )
    return Verdict.yes()


def _resample_cost(source: ChannelSpec, target: ChannelSpec) -> tuple[str, ...]:
    """State the loss in words a reader months later can act on."""
    if source.rate_hz is None or target.rate_hz is None:
        return ("resampled between undeclared rates",)
    if target.rate_hz < source.rate_hz:
        kept = target.rate_hz / source.rate_hz
        return (
            f"resampled {source.rate_hz:g} Hz to {target.rate_hz:g} Hz, keeping "
            f"{kept:.0%} of the timeline; detail faster than {target.rate_hz:g} Hz is gone",
        )
    if target.rate_hz > source.rate_hz:
        return (
            f"resampled {source.rate_hz:g} Hz up to {target.rate_hz:g} Hz by linear "
            "interpolation; the added samples are invented, not measured",
        )
    return ()


def resample_to(values: np.ndarray, source: ChannelSpec, target: ChannelSpec) -> np.ndarray:
    """Linear resampling along time. Endpoints are preserved exactly."""
    data = np.asarray(values, dtype=float)
    if source.rate_hz is None or target.rate_hz is None:
        raise ValueError("both channels must declare a rate")
    if len(data) < 2 or source.rate_hz == target.rate_hz:
        return data

    duration = (len(data) - 1) / source.rate_hz
    steps = max(2, int(round(duration * target.rate_hz)) + 1)
    source_time = np.linspace(0.0, duration, len(data))
    target_time = np.linspace(0.0, duration, steps)

    if data.ndim == 1:
        return np.interp(target_time, source_time, data)
    return np.stack(
        [np.interp(target_time, source_time, data[:, column]) for column in range(data.shape[1])],
        axis=1,
    )


RESAMPLE = Adapter(
    name="resample",
    version=VERSION,
    closes=("rate.mismatch",),
    guard=_resample_guard,
    cost=_resample_cost,
    transform=resample_to,
    preserves_length=False,
    metadata={"reversible": False},
)


def default_registry() -> AdapterRegistry:
    """Every adapter here, ready for the resolver."""
    return AdapterRegistry([UNIT_CONVERT, PERMUTE, RESAMPLE])
