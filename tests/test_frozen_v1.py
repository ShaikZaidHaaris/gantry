"""The v1 surface, pinned so a breaking change has to be deliberate.

The rule of three has been met: every plane has at least three real
implementations or a published contract exercised by more than one, so the
interfaces have been shaped by use rather than by imagination. That is the point
at which freezing means something.

What this file does is make a break *loud*. Renaming a method, dropping a
capability key, or changing a refusal code is not forbidden — it is a major
version bump, and this test is what forces someone to say so out loud instead of
discovering it in a plugin six weeks later.

Refusal codes are pinned too, and deliberately. Plugins branch on them: the
resolver maps them to adapters, and a third party may map them to remediation of
their own. A code is as much part of the contract as a method name.
"""

from __future__ import annotations

import inspect

import pytest

from gantry import contracts
from gantry.contracts.connector import CONNECTOR_CONTRACT, Connector
from gantry.contracts.embodiment import EMBODIMENT_CONTRACT, Retargeter
from gantry.contracts.evaluator import EVALUATOR_CONTRACT, Evaluator
from gantry.contracts.feedback import FEEDBACK_CONTRACT, FeedbackModule
from gantry.contracts.policy import POLICY_CONTRACT, Policy
from gantry.spine import PLANES, SPINE_CONTRACT

# --------------------------------------------------------------------------
# contract versions
# --------------------------------------------------------------------------

#: Every contract, pinned. Moving one is allowed and must be recorded here in
#: the same commit, which is what makes it a decision rather than a drift.
#:
#: evaluator@1.1 made ``task_for`` required. It had been a hook the runner
#: probed for, so an evaluator could pass every check and still be undriveable.
FROZEN_CONTRACTS = {
    "spine": (SPINE_CONTRACT, "1.1"),
    "connector": (CONNECTOR_CONTRACT, "1.0"),
    "embodiment": (EMBODIMENT_CONTRACT, "1.0"),
    "policy": (POLICY_CONTRACT, "1.0"),
    "evaluator": (EVALUATOR_CONTRACT, "1.1"),
    "feedback": (FEEDBACK_CONTRACT, "1.0"),
}


@pytest.mark.parametrize("name,pinned", sorted(FROZEN_CONTRACTS.items()))
def test_contract_versions_are_pinned(name, pinned):
    actual, expected = pinned
    assert actual == f"{name}@{expected}", (
        f"{name} moved to {actual}. That is allowed, but it is a contract change: "
        "update this test in the same commit so the break is on the record."
    )


def test_the_core_planes_are_frozen():
    """The planes core itself declares.

    Plugins may register more — that is the point of the plane registry — so
    this pins what core ships, not what a running system contains.
    """
    from gantry.spine import CORE_PLANES

    assert CORE_PLANES == (
        "dataset",
        "embodiment",
        "policy",
        "evaluation",
        "feedback",
        "adapter",
        "retargeter",
    )
    assert PLANES == CORE_PLANES


def test_every_core_plane_is_fully_described():
    """A plane the framework cannot reason about generically is not a plane."""
    from gantry.spine import CORE_PLANES, get_plane

    for name in CORE_PLANES:
        plane = get_plane(name)
        assert plane is not None and plane.description
        assert plane.entry_point_group, f"{name} has no entry-point group, so nothing installs into it"


# --------------------------------------------------------------------------
# required methods
# --------------------------------------------------------------------------

REQUIRED = {
    Connector: {"descriptor", "episode_ids", "schema", "open"},
    Policy: {"descriptor", "action_spec", "observes", "act", "reset"},
    Evaluator: {"descriptor", "requires", "run", "evaluate"},
    FeedbackModule: {"descriptor", "requirement", "analyse", "run", "check_inputs"},
    Retargeter: {"name", "version", "accepts", "losses", "apply", "check"},
}


@pytest.mark.parametrize("interface,methods", [(k, v) for k, v in REQUIRED.items()])
def test_required_methods_are_present(interface, methods):
    missing = {name for name in methods if not hasattr(interface, name)}
    assert not missing, f"{interface.__name__} lost {sorted(missing)}"


@pytest.mark.parametrize("interface,methods", [(k, v) for k, v in REQUIRED.items()])
def test_every_public_method_is_documented(interface, methods):
    """A contract whose methods have no docstring is not a contract."""
    undocumented = [
        name
        for name in methods
        if not (getattr(inspect.getattr_static(interface, name, None), "__doc__", None)
                or getattr(getattr(interface, name, None), "__doc__", None))
    ]
    assert not undocumented, f"{interface.__name__}: undocumented {undocumented}"


# --------------------------------------------------------------------------
# capability vocabulary
# --------------------------------------------------------------------------

FROZEN_CAPABILITIES = {
    "connector": {"lazy", "stage_events", "outcomes", "media"},
    "policy": {"chunk", "deterministic", "stateful"},
    "evaluator": {"stage_events", "outcomes", "seedable", "closed_loop"},
    "feedback": {"min_cohorts", "prescribes"},
}


def test_capability_keys_are_frozen():
    from gantry.contracts import connector, evaluator, feedback, policy

    actual = {
        "connector": set(connector.CAPABILITIES),
        "policy": {policy.CAP_CHUNK, policy.CAP_DETERMINISTIC, policy.CAP_STATEFUL},
        "evaluator": {
            evaluator.CAP_STAGE_EVENTS,
            evaluator.CAP_OUTCOMES,
            evaluator.CAP_SEEDABLE,
            evaluator.CAP_CLOSED_LOOP,
        },
        "feedback": {feedback.CAP_MIN_COHORTS, feedback.CAP_PRESCRIBES},
    }
    assert actual == FROZEN_CAPABILITIES


# --------------------------------------------------------------------------
# refusal codes
# --------------------------------------------------------------------------

#: Codes plugins are entitled to branch on. Adding one is a minor bump;
#: renaming or removing one breaks anything that mapped it to a remedy.
FROZEN_CODES = {
    # channel compatibility -> what the resolver looks for an adapter to close
    "kind.mismatch",
    "shape.mismatch",
    "dtype.mismatch",
    "units.dimension",
    "units.scale",
    "units.undeclared",
    "frame.mismatch",
    "frame.undeclared",
    "semantics.mismatch",
    "semantics.undeclared",
    "rate.mismatch",
    "dim_labels.order",
    "dim_labels.mismatch",
    "dim_labels.undeclared",
    "metadata.mismatch",
    "metadata.undeclared",
    # resolution
    "resolve.no_candidate",
    "resolve.no_adapter",
    "resolve.channel_incompatible",
    "resolve.capability",
    "resolve.not_installed",
    # persistence and manifests
    "manifest.not_installed",
    "plan.module_refused",
    # transforms
    "retarget.none_installed",
    "retarget.no_match",
    "isolate.unsupported_plane",
    # contract versioning
    "contract.name",
    "contract.major",
    "contract.minor",
}


def test_every_frozen_code_is_still_emitted_somewhere():
    """Each pinned code must still exist in the source that raises it.

    A code nobody emits is a promise to plugins that has quietly lapsed.
    """
    import pathlib

    root = pathlib.Path(contracts.__file__).parent.parent
    source = "\n".join(
        path.read_text() for path in root.rglob("*.py") if "tests" not in str(path)
    )
    missing = sorted(code for code in FROZEN_CODES if f'"{code}"' not in source)
    assert not missing, f"these pinned refusal codes are no longer emitted: {missing}"


def test_codes_are_namespaced():
    """Every code is dotted, so its origin is readable at a glance."""
    assert all("." in code for code in FROZEN_CODES)


# --------------------------------------------------------------------------
# the neutrality promise
# --------------------------------------------------------------------------


def test_core_is_usable_with_no_plugins_installed():
    """The whole public surface imports in a bare interpreter."""
    import subprocess
    import sys

    probe = (
        "import gantry, gantry.spine, gantry.contracts, gantry.conformance, "
        "gantry.resolve, gantry.fixtures, gantry.manifest, gantry.runner, gantry.cli; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
