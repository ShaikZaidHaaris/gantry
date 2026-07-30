"""The regression gate, against this project's own measurements.

What matters here is not that it computes a p-value — it is which way it fails.
A gate that blocks on noise gets switched off within a week and then protects
nothing, and a gate that blocks the very first run on a task blocks the commit
that would establish its baseline. Both are tested.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

import pytest

from gantry.history import History
from gantry.spine import ComponentRef, Provenance, proportion


@dataclass
class Labels:
    success: bool | None = None
    annotations: dict = field(default_factory=dict)


@dataclass
class Meta:
    id: str
    task: str | None = None


@dataclass
class Ep:
    meta: Meta
    labels: Labels


@dataclass
class Run:
    provenance: Provenance
    episodes: tuple
    metrics: dict


def run(task, policy, outcomes):
    components = (
        ComponentRef("policy", policy, "1.0"),
        ComponentRef("embodiment", "panda", "1.0"),
        ComponentRef("scorer", "machine", "1.0"),
    )
    episodes = tuple(Ep(Meta(f"seed_{i}", task), Labels(o)) for i, o in enumerate(outcomes))
    return Run(
        Provenance(components=components),
        episodes,
        {"success_rate": proportion(sum(1 for o in outcomes if o), len(outcomes))},
    )


def gantry(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gantry.cli", *args], capture_output=True, text=True
    )


@pytest.fixture
def measured(tmp_path):
    """ph as baseline, mh as an improvement, mg as a real regression."""
    root = str(tmp_path / "history")
    h = History(root)
    keys = {
        "ph": h.put(run("lift_cube", "ph_official", [True] * 28 + [False] * 22), keep_record=False),
        "mh": h.put(run("lift_cube", "mh_official", [True] * 33 + [False] * 17), keep_record=False),
        "mg": h.put(run("lift_cube", "mg_official", [False] * 50), keep_record=False),
    }
    return root, h, keys


def test_the_first_run_on_a_task_does_not_block(measured):
    """Otherwise the commit that establishes a baseline can never land."""
    root, _, keys = measured
    result = gantry("ci", keys["mh"], "--history", root)
    assert result.returncode == 0
    assert "no baseline pinned" in result.stdout


def test_pinning_makes_a_run_the_reference(measured):
    root, _, keys = measured
    result = gantry("ci-pin", keys["ph"], "--history", root)
    assert result.returncode == 0
    assert "baseline" in result.stdout


def test_a_significant_drop_fails_the_build(measured):
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    result = gantry("ci", keys["mg"], "--history", root)
    assert result.returncode == 1
    assert "REGRESSION" in result.stdout
    assert "-56.0%" in result.stdout


def test_an_improvement_passes(measured):
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    result = gantry("ci", keys["mh"], "--history", root)
    assert result.returncode == 0
    assert "Up +10.0%" in result.stdout


def test_a_drop_that_is_only_noise_does_not_block(measured):
    """A gate that blocks on noise is switched off within a week."""
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    # One scene worse out of fifty: real direction, no evidence.
    barely = h.put(
        run("lift_cube", "barely_worse", [True] * 27 + [False] * 23), keep_record=False
    )
    result = gantry("ci", barely, "--history", root)
    assert result.returncode == 0
    assert "not separable" in result.stdout


def test_the_threshold_tightens_as_a_task_accumulates_runs(measured):
    """The tenth checkpoint on a task is not making the first one's claim."""
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    result = gantry("ci", keys["mg"], "--history", root)
    assert "3 run(s) recorded on this task" in result.stdout
    assert "threshold p<0.0167" in result.stdout


def test_it_reports_the_pairing_rather_than_the_marginals_alone(measured):
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    result = gantry("ci", keys["mh"], "--history", root)
    assert "paired on 50 shared scene(s)" in result.stdout
    assert "won 5, lost 0" in result.stdout


def test_runs_sharing_no_scene_are_refused_rather_than_compared(measured):
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    other = History(root).put(
        Run(
            Provenance(components=(ComponentRef("policy", "elsewhere", "1.0"),)),
            tuple(Ep(Meta(f"different_{i}", "lift_cube"), Labels(True)) for i in range(10)),
            {"success_rate": proportion(10, 10)},
        ),
        keep_record=False,
    )
    result = gantry("ci", other, "--history", root)
    assert result.returncode == 2
    assert "share no scene" in result.stdout


def test_an_unknown_run_is_an_error_not_a_pass(measured):
    root, _, _ = measured
    assert gantry("ci", "nope", "--history", root).returncode == 2


def test_the_markdown_summary_can_be_written_for_a_pull_request(measured, tmp_path):
    root, h, keys = measured
    h.pin(keys["ph"], task="lift_cube")
    out = tmp_path / "summary.md"
    gantry("ci", keys["mg"], "--history", root, "--summary", str(out))
    assert out.exists()
    assert "REGRESSION" in out.read_text()


def test_history_is_inspectable_from_the_command_line(measured):
    root, _, _ = measured
    result = gantry("history", root)
    assert "3 run(s) recorded" in result.stdout
    assert "lift_cube" in result.stdout
