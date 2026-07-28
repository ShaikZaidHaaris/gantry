"""Dimensional analysis, small enough to keep core numpy-only.

Core knows about *physical dimensions*, never about robots. That is what makes
unit checking neutral: a length is a length whether it came from a gripper, a
drone, or a soft actuator.

Existing to prevent one specific class of bug: silently comparing quantities
that are not the same quantity. A spatial metric that reads metres from one
source and millimetres from another is wrong in a way no test catches unless
units travel with the data.

New units register at runtime, so a plugin working in some exotic quantity is
never blocked by this table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

#: Order of exponents in a :class:`Dimension` vector.
BASE_DIMENSIONS = ("length", "mass", "time", "angle", "current", "temperature")


@dataclass(frozen=True)
class Dimension:
    """Exponents over the base dimensions. ``Dimension()`` is dimensionless."""

    exponents: tuple[int, ...] = (0,) * len(BASE_DIMENSIONS)

    def __post_init__(self) -> None:
        if len(self.exponents) != len(BASE_DIMENSIONS):
            raise ValueError(f"expected {len(BASE_DIMENSIONS)} exponents")

    def __mul__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: int) -> "Dimension":
        return Dimension(tuple(a * power for a in self.exponents))

    @property
    def dimensionless(self) -> bool:
        return all(exponent == 0 for exponent in self.exponents)

    def __str__(self) -> str:
        parts = [
            f"{name}^{exponent}" if exponent != 1 else name
            for name, exponent in zip(BASE_DIMENSIONS, self.exponents)
            if exponent
        ]
        return "·".join(parts) if parts else "dimensionless"


def _dim(**kwargs: int) -> Dimension:
    return Dimension(tuple(kwargs.get(name, 0) for name in BASE_DIMENSIONS))


DIMENSIONLESS = Dimension()
LENGTH = _dim(length=1)
MASS = _dim(mass=1)
TIME = _dim(time=1)
ANGLE = _dim(angle=1)
FORCE = _dim(mass=1, length=1, time=-2)
TORQUE = _dim(mass=1, length=2, time=-2)


@dataclass(frozen=True)
class Unit:
    """A parsed unit: how to reach SI, and what dimension it carries."""

    symbol: str
    factor: float
    dimension: Dimension

    def __str__(self) -> str:
        return self.symbol


# Atomic symbols. Compound expressions ("m/s^2", "N*m") are parsed from these.
_ATOMS: dict[str, tuple[float, Dimension]] = {
    "m": (1.0, LENGTH),
    "cm": (1e-2, LENGTH),
    "mm": (1e-3, LENGTH),
    "km": (1e3, LENGTH),
    "in": (0.0254, LENGTH),
    "ft": (0.3048, LENGTH),
    "kg": (1.0, MASS),
    "g": (1e-3, MASS),
    "s": (1.0, TIME),
    "ms": (1e-3, TIME),
    "us": (1e-6, TIME),
    "min": (60.0, TIME),
    "rad": (1.0, ANGLE),
    "deg": (0.017453292519943295, ANGLE),
    "rev": (6.283185307179586, ANGLE),
    "N": (1.0, FORCE),
    "A": (1.0, _dim(current=1)),
    "K": (1.0, _dim(temperature=1)),
    "Hz": (1.0, TIME**-1),
    "px": (1.0, DIMENSIONLESS),
    "count": (1.0, DIMENSIONLESS),
    "step": (1.0, DIMENSIONLESS),
    "fraction": (1.0, DIMENSIONLESS),
    "percent": (0.01, DIMENSIONLESS),
}

_DIMENSIONLESS_SPELLINGS = {"", "1", "unitless", "dimensionless", "none", "norm"}

_TOKEN = re.compile(r"([A-Za-z]+)(?:\^(-?\d+))?")


class UnknownUnitError(ValueError):
    """Raised for a symbol that is not registered."""


def register_unit(symbol: str, factor: float, dimension: Dimension) -> None:
    """Teach the parser a new atomic unit.

    Plugins call this at import time for quantities core has never heard of.
    Re-registering the same symbol with the same meaning is a no-op; changing
    a symbol's meaning is refused, because two plugins disagreeing about what
    "N" means is exactly the silent corruption this module exists to stop.
    """
    existing = _ATOMS.get(symbol)
    if existing is not None and existing != (factor, dimension):
        raise ValueError(
            f"unit {symbol!r} is already registered as {existing[0]}·{existing[1]}; "
            "pick a different symbol rather than redefining an existing one"
        )
    _ATOMS[symbol] = (factor, dimension)


def known_units() -> tuple[str, ...]:
    return tuple(sorted(_ATOMS))


def _terms(expression: str) -> Iterator[tuple[str, int]]:
    """Yield (symbol, signed exponent) for a compound expression."""
    sign = 1
    position = 0
    while position < len(expression):
        character = expression[position]
        if character in " ·":
            position += 1
            continue
        if character == "*":
            sign = 1
            position += 1
            continue
        if character == "/":
            sign = -1
            position += 1
            continue
        match = _TOKEN.match(expression, position)
        if not match:
            raise UnknownUnitError(f"cannot parse unit expression {expression!r}")
        symbol, exponent = match.group(1), int(match.group(2) or 1)
        yield symbol, sign * exponent
        position = match.end()


def parse(expression: str | None) -> Unit:
    """Parse a unit expression such as ``"m"``, ``"m/s^2"`` or ``"N*m"``.

    ``None`` and the dimensionless spellings all yield the dimensionless unit,
    so "no units declared" and "explicitly a ratio" are the same thing here.
    Whether *declaring* units is required is a policy question, decided by
    :mod:`gantry.spine.channel`, not by the parser.
    """
    text = (expression or "").strip()
    if text.lower() in _DIMENSIONLESS_SPELLINGS:
        return Unit(text or "1", 1.0, DIMENSIONLESS)

    factor = 1.0
    dimension = DIMENSIONLESS
    for symbol, exponent in _terms(text):
        atom = _ATOMS.get(symbol)
        if atom is None:
            raise UnknownUnitError(
                f"unknown unit {symbol!r} in {text!r}; "
                f"register it with gantry.spine.units.register_unit"
            )
        atom_factor, atom_dimension = atom
        factor *= atom_factor**exponent
        dimension = dimension * (atom_dimension**exponent)
    return Unit(text, factor, dimension)


def same_dimension(left: str | None, right: str | None) -> bool:
    """Do two unit expressions describe the same physical quantity?"""
    return parse(left).dimension == parse(right).dimension


def conversion_factor(source: str | None, target: str | None) -> float:
    """Multiplier taking a value in ``source`` units to ``target`` units."""
    source_unit, target_unit = parse(source), parse(target)
    if source_unit.dimension != target_unit.dimension:
        raise UnknownUnitError(
            f"cannot convert {source_unit} ({source_unit.dimension}) to "
            f"{target_unit} ({target_unit.dimension})"
        )
    return source_unit.factor / target_unit.factor
