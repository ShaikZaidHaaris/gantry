"""The taxonomy is about who must respond, not how bad it was."""

from __future__ import annotations

import pytest

from gantry.errors import (
    ComponentError,
    ConfigError,
    FaultError,
    GantryError,
    SafetyAbort,
    disposition,
    must_halt,
)


def test_a_component_failure_is_one_failed_unit():
    assert disposition(ComponentError("policy threw")) == "record"
    assert not must_halt(ComponentError("policy threw"))


@pytest.mark.parametrize("error", [FaultError("stuck"), SafetyAbort("vetoed")])
def test_a_faulted_world_always_halts(error):
    """No caller preference overrides these.

    A rig that keeps being handed scenes after a fault produces
    plausible-looking numbers describing nothing, and it does so overnight
    while nobody is watching.
    """
    assert disposition(error) == "halt"
    assert must_halt(error)
    assert error.halts and not error.recoverable


def test_configuration_fails_before_anything_runs():
    assert must_halt(ConfigError("no such dataset"))


def test_an_unclassified_exception_is_treated_as_halting():
    """A failure the framework cannot classify is not one it may continue through."""
    assert disposition(RuntimeError("who knows")) == "unknown"
    assert must_halt(RuntimeError("who knows"))


def test_a_partial_result_survives_the_exception():
    """A run that dies halfway is still evidence."""
    error = ComponentError("died at scene 12", partial={"episodes": 11})
    assert error.partial == {"episodes": 11}


def test_partial_defaults_to_absent():
    assert GantryError("plain").partial is None
