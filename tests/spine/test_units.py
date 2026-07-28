from __future__ import annotations

import pytest

from gantry.spine import units


def test_dimensionless_spellings_agree():
    for text in ["", "1", "unitless", "none", None]:
        assert units.parse(text).dimension.dimensionless


def test_compound_expressions():
    assert units.parse("m/s").dimension == units.LENGTH / units.TIME
    assert units.parse("m/s^2").dimension == units.LENGTH / (units.TIME**2)
    assert units.parse("N*m").dimension == units.TORQUE
    assert units.parse("Hz").dimension == units.TIME**-1


def test_conversion_factor_round_trip():
    assert units.conversion_factor("mm", "m") == pytest.approx(1e-3)
    assert units.conversion_factor("m", "mm") == pytest.approx(1e3)
    assert units.conversion_factor("deg", "rad") == pytest.approx(0.0174532925, rel=1e-6)


def test_cannot_convert_across_dimensions():
    with pytest.raises(units.UnknownUnitError):
        units.conversion_factor("m", "s")


def test_same_dimension_ignores_scale():
    assert units.same_dimension("mm", "km")
    assert not units.same_dimension("m", "kg")


def test_unknown_symbol_names_the_fix():
    with pytest.raises(units.UnknownUnitError, match="register_unit"):
        units.parse("smoots")


def test_register_unit_is_idempotent_but_refuses_redefinition():
    units.register_unit("widget", 2.0, units.LENGTH)
    units.register_unit("widget", 2.0, units.LENGTH)  # same meaning, fine
    assert units.parse("widget").factor == 2.0
    with pytest.raises(ValueError, match="already registered"):
        units.register_unit("widget", 3.0, units.LENGTH)
