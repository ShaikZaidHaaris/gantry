"""Reading tasks, and refusing the ones nobody could score."""

from __future__ import annotations

import json

import pytest
from gantry_tasks_declared import DeclaredTasks, definition_from

from gantry.contracts.task import TaskDefinition
from gantry.errors import ConfigError

LIFT = {
    "name": "lift_cube",
    "instruction": "lift the cube off the table",
    "surfaces": ["table"],
    "objects": [
        {"id": "cube", "kind": "box_20mm",
         "start": {"surface": "table", "x": [-0.03, 0.03], "y": [-0.03, 0.03]}}
    ],
    "success": [
        {"check": "lifted", "args": {"object": "cube", "height": 0.04},
         "rubric": "The cube is fully clear of the table surface, held in the "
                   "gripper, and remains held for at least one second. A cube "
                   "that is dropped immediately does not count."}
    ],
    "horizon": 300,
    "trials": 20,
    "staging": {"robosuite": {"env_name": "Lift"}},
}


def write(tmp_path, *payloads):
    for payload in payloads:
        (tmp_path / f"{payload['name']}.json").write_text(json.dumps(payload))
    return tmp_path


@pytest.fixture
def tasks(tmp_path):
    return DeclaredTasks(write(tmp_path, LIFT))


# -- reading ---------------------------------------------------------------


def test_a_directory_of_files_is_a_task_source(tasks):
    assert tasks.names() == ("lift_cube",)
    assert len(tasks) == 1
    task = tasks.task("lift_cube")
    assert isinstance(task, TaskDefinition)
    assert task.instruction.startswith("lift the cube")
    assert task.things[0].kind == "box_20mm"


def test_it_names_no_particular_machine(tasks):
    """The property that lets the same task reach hardware later.

    Generic words are fine and often necessary — a rubric saying "held in the
    gripper" describes what a person watching would see, and is true of a
    parallel jaw, a three-finger hand or a suction cup alike. What must never
    appear is a *particular* machine or simulator, because that is what would
    stop this file being staged somewhere else.
    """
    blob = json.dumps(LIFT).lower()
    for named in ("panda", "sawyer", "ur5e", "iiwa", "jaco", "kinova",
                  "mujoco", "isaac", "franka"):
        assert named not in blob, f"a task should not name {named!r}"
    # And the staging block is where a world-specific detail is allowed to
    # live, quarantined under that world's name.
    assert set(LIFT["staging"]) == {"robosuite"}


def test_the_descriptor_says_what_it_offers(tasks):
    d = tasks.descriptor()
    assert d.plane == "task"
    assert d.provides["tasks"] == 1
    assert d.provides["rubrics"] is True
    assert d.metadata["worlds"] == ["robosuite"]


def test_staging_is_per_world_and_optional(tasks):
    assert tasks.task("lift_cube").staged_by("robosuite") == {"env_name": "Lift"}
    assert tasks.task("lift_cube").staged_by("libero") is None
    assert tasks.staged_by("robosuite") == ("lift_cube",)


def test_an_unknown_task_lists_what_there_is(tasks):
    with pytest.raises(KeyError, match="lift_cube"):
        tasks.task("nope")


def test_duplicate_names_are_refused(tmp_path):
    other = {**LIFT}
    (tmp_path / "a.json").write_text(json.dumps(LIFT))
    (tmp_path / "b.json").write_text(json.dumps(other))
    with pytest.raises(ConfigError, match="already defined"):
        DeclaredTasks(tmp_path)


def test_an_empty_directory_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="no task files"):
        DeclaredTasks(tmp_path)


# -- the refusals that matter ----------------------------------------------


def test_a_criterion_without_a_rubric_is_refused():
    """The one that decides whether this survives contact with hardware."""
    task = definition_from({**LIFT, "success": [{"check": "lifted", "args": {}}]})
    verdict = task.validate()
    assert "criterion.no_rubric" in verdict.codes()
    assert not verdict.ok


def test_a_task_nobody_can_score_is_refused():
    verdict = definition_from({**LIFT, "success": []}).validate()
    assert "task.no_success" in verdict.codes()


def test_a_terse_rubric_is_flagged_without_blocking():
    task = definition_from({**LIFT, "success": [
        {"check": "lifted", "rubric": "cube up"}]})
    verdict = task.validate()
    assert "criterion.terse_rubric" in verdict.codes()
    assert verdict.ok, "a judgement, not an error"


def test_success_referring_to_an_absent_object_is_refused(tmp_path):
    bad = {**LIFT, "name": "bad", "success": [
        {"check": "on", "args": {"object": "bowl", "target": "plate"},
         "rubric": "The bowl is resting stably on the plate and not held."}]}
    verdict = DeclaredTasks(write(tmp_path, bad)).audit()
    assert "task.criterion_unknown_object" in verdict.codes()


def test_a_fixed_start_is_flagged_as_measuring_memorisation(tmp_path):
    fixed = {**LIFT, "name": "fixed", "objects": [
        {"id": "cube", "kind": "box_20mm",
         "start": {"surface": "table", "x": [0.0, 0.0], "y": [0.0, 0.0]}}]}
    verdict = DeclaredTasks(write(tmp_path, fixed)).audit()
    assert "task.fixed_start" in verdict.codes()
    assert verdict.ok


def test_an_object_on_an_undeclared_surface_is_refused():
    verdict = definition_from({**LIFT, "objects": [
        {"id": "cube", "kind": "box",
         "start": {"surface": "shelf", "x": [0, 1], "y": [0, 1]}}]}).validate()
    assert "task.unknown_surface" in verdict.codes()


def test_a_clean_task_audits_clean(tasks):
    verdict = tasks.audit()
    assert verdict.ok, verdict.explain()
    assert not [c for c in verdict.codes() if c.startswith("task.criterion")]


# -- what carries to hardware ----------------------------------------------


def test_the_rubrics_are_the_artifact_that_moves(tasks):
    """Scorable by a person, from video, with no simulator anywhere."""
    task = tasks.task("lift_cube")
    assert len(task.rubrics) == 1
    assert "does not count" in task.rubrics[0]
    assert tasks.scorable_by_hand() == ("lift_cube",)


def test_a_task_with_no_world_is_still_scorable_by_hand(tmp_path):
    unstaged = {**LIFT, "name": "unstaged", "staging": {}}
    source = DeclaredTasks(write(tmp_path, unstaged))
    verdict = source.audit()
    assert "task.unstaged" in verdict.codes()
    assert verdict.ok
    assert source.scorable_by_hand() == ("unstaged",)
