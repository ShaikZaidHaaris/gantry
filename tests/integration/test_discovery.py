"""Discovery, which needs something to discover.

These live here rather than beside the rest of the resolver tests because they
assert that an *installed plugin* is found. With core alone they do not fail
meaningfully -- they fail because the premise is absent -- and the core CI job
installs core alone on purpose.
"""

from __future__ import annotations

from gantry.resolve import Registry


def test_discovery_finds_an_installed_plugin():
    """The CSV plugin declares an entry point, so it appears without editing core."""
    reg = Registry()
    found = reg.discover()
    assert "dataset:csv" in found
    assert reg.get("dataset", "csv").origin == "entry-point:gantry.connectors"


def test_discovery_does_not_import_the_plugin_until_it_is_built():
    reg = Registry()
    reg.discover()
    registration = reg.get("dataset", "csv")
    assert "lazy" in repr(registration.factory)
