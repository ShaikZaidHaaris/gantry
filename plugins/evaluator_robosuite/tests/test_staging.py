"""Translating a declared task into this world's placement.

Every test here runs with no simulator installed, which is the point: whether a
task file can be staged is a question about the file, and answering it should
not need MuJoCo, a GL context, or a GPU.
"""

from __future__ import annotations

import pytest
from gantry_evaluator_robosuite.staging import (
    Placement,
    check_staging,
    env_meta_for,
    placements_for,
    verify_honoured,
)

from gantry.contracts.task import Criterion, Region, TaskDefinition, Thing
from gantry.errors import ConfigError
from gantry.spine.verdict import IncompatibleError

RUBRIC = "The cube is clear of the table and held in the gripper for a second."


def task(**over):
    base = dict(
        name="lift_cube",
        instruction="lift the cube",
        things=(
            Thing("cube", "cube_20mm", Region("table", (-0.03, 0.03), (-0.03, 0.03))),
        ),
        success=(Criterion("lifted", {"object": "cube", "height": 0.04}, RUBRIC),),
        trials=5,
        horizon=250,
        staging={"robosuite": {"env_name": "Lift", "places": {"cube": "cube"}}},
    )
    base.update(over)
    return TaskDefinition(**base)


# -- translation ---------------------------------------------------------------


def test_a_region_becomes_a_sampler_over_the_same_numbers():
    (placement,) = placements_for(task())
    kwargs = placement.as_kwargs(reference=[0.0, 0.0, 0.8])
    assert kwargs["x_range"] == [-0.03, 0.03]
    assert kwargs["y_range"] == [-0.03, 0.03]
    assert kwargs["reference_pos"] == [0.0, 0.0, 0.8]


def test_the_sampler_is_named_the_way_this_world_binds_objects():
    # robosuite environments call add_objects_to_sampler(f"{name}Sampler", ...)
    # with their own object name, so the name has to be theirs, not the task's.
    (placement,) = placements_for(task())
    assert placement.task_id == "cube"
    assert placement.object_name == "cube"
    assert placement.sampler_name == "cubeSampler"


def test_an_objects_own_name_in_this_world_may_differ_from_the_tasks():
    t = task(
        things=(Thing("nut", "square_nut", Region("table", (0.0, 0.1), (0.0, 0.1))),),
        success=(Criterion("on_peg", {"object": "nut"}, RUBRIC),),
        staging={"robosuite": {"env_name": "NutAssembly", "places": {"nut": "SquareNut"}}},
    )
    (placement,) = placements_for(t)
    assert placement.sampler_name == "SquareNutSampler"


def test_an_object_with_no_start_region_is_not_placed():
    t = task(
        things=(Thing("door", "hinged_door", None),),
        staging={"robosuite": {"env_name": "Door", "places": {}}},
        success=(Criterion("opened", {"object": "door"}, RUBRIC),),
    )
    assert placements_for(t) == ()


def test_translation_needs_no_simulator():
    # Nothing above imported robosuite; assert it, so a future import that
    # sneaks into the pure half is caught rather than merely slowing things.
    import sys

    placements_for(task())
    check_staging(task())
    env_meta_for(task())
    assert "robosuite" not in sys.modules


# -- what the file alone can be checked for ------------------------------------


def test_a_task_this_world_does_not_know_is_refused_not_guessed():
    v = check_staging(task(staging={}))
    assert not v.ok
    assert v.reasons[0].code == "robosuite.unstaged"


def test_staging_without_an_environment_is_refused():
    v = check_staging(task(staging={"robosuite": {"places": {}}}))
    assert not v.ok
    assert "robosuite.no_env_name" in [r.code for r in v.reasons]


def test_mapping_an_object_the_task_never_places_is_refused():
    v = check_staging(
        task(staging={"robosuite": {"env_name": "Lift", "places": {"sphere": "cube"}}})
    )
    assert not v.ok
    assert "robosuite.unknown_object" in [r.code for r in v.reasons]


def test_a_region_this_world_would_silently_ignore_is_noted():
    # The task says where the cube starts; the staging block does not say which
    # of this world's objects that is, so the region would do nothing.
    v = check_staging(task(staging={"robosuite": {"env_name": "Lift", "places": {}}}))
    assert v.ok  # a note, not a refusal — it still runs, just not as written
    assert [r.code for r in v.reasons] == ["robosuite.unmapped_region"]


def test_an_undeclared_orientation_is_recorded_rather_than_invented():
    v = check_staging(task())
    assert v.ok
    assert [r.code for r in v.reasons] == ["robosuite.yaw_unspecified"]


def test_a_declared_orientation_produces_no_note_and_reaches_the_sampler():
    t = task(
        things=(
            Thing("cube", "cube_20mm", Region("table", (-0.03, 0.03), (-0.03, 0.03), (0.0, 1.57))),
        )
    )
    assert check_staging(t).reasons == ()
    (placement,) = placements_for(t)
    assert placement.as_kwargs([0, 0, 0.8])["rotation"] == [0.0, 1.57]


# -- the recipe ----------------------------------------------------------------


def test_env_meta_carries_the_environment_and_the_placement():
    meta = env_meta_for(task())
    assert meta["env_name"] == "Lift"
    assert len(meta["placement"]) == 1
    # places and z_offsets are instructions for building placement, not
    # arguments to the environment; passing them through would break robosuite.
    assert "places" not in meta["env_kwargs"]


def test_env_meta_passes_extra_settings_to_the_environment():
    meta = env_meta_for(task(), control_freq=20)
    assert meta["env_kwargs"]["control_freq"] == 20


def test_a_staging_block_may_carry_this_worlds_own_settings():
    t = task(
        staging={
            "robosuite": {
                "env_name": "PickPlaceCan",
                "places": {"cube": "Can"},
                "table_friction": [1.0, 0.005, 0.0001],
            }
        }
    )
    assert env_meta_for(t)["env_kwargs"]["table_friction"] == [1.0, 0.005, 0.0001]


def test_building_a_world_for_an_unstageable_task_refuses_up_front():
    with pytest.raises(IncompatibleError, match="robosuite.unstaged"):
        env_meta_for(task(staging={}))


# -- what only a built world can answer ----------------------------------------


class _Env:
    def __init__(self, kept):
        self.placement_initializer = kept


def test_a_world_that_kept_the_placement_passes():
    sampler = object()
    verify_honoured(_Env(sampler), sampler, "Lift")


def test_a_world_that_threw_the_placement_away_is_refused_by_name():
    # PickPlace and ToolHang build their own placement unconditionally. Caught
    # by identity rather than by a list of names, so an environment added later
    # is judged on what it does.
    with pytest.raises(ConfigError, match="discarded the placement"):
        verify_honoured(_Env(object()), object(), "PickPlace")


def test_a_world_with_no_placement_at_all_is_refused():
    # Wipe lays its markers out itself and has no placement_initializer.
    with pytest.raises(ConfigError, match="no placement to give"):
        verify_honoured(_Env(None), object(), "Wipe")


def test_a_placement_carries_the_task_id_so_a_refusal_can_name_it():
    p = Placement(object_name="SquareNut", task_id="nut", x=(0, 0.1), y=(0, 0.1))
    assert p.task_id == "nut"
    assert p.z_offset > 0  # zero spawns an object intersecting the table


# -- the two sampler shapes ----------------------------------------------------


def test_one_region_for_everything_may_be_carried_by_either_shape():
    from gantry_evaluator_robosuite.staging import sampler_shapes, shared_region

    both = (
        Placement("cubeA", "cube_a", (-0.03, 0.03), (-0.03, 0.03)),
        Placement("cubeB", "cube_b", (-0.03, 0.03), (-0.03, 0.03)),
    )
    assert shared_region(both)
    # Single first: environments that call add_objects() reject the composite
    # form outright, and the ones that need composite say so by rejecting this.
    assert sampler_shapes(both) == (False, True)


def test_per_object_regions_can_only_be_carried_by_the_composite_shape():
    from gantry_evaluator_robosuite.staging import sampler_shapes, shared_region

    apart = (
        Placement("SquareNut", "square_nut", (0.0, 0.1), (0.11, 0.22)),
        Placement("RoundNut", "round_nut", (0.0, 0.1), (-0.22, -0.11)),
    )
    assert not shared_region(apart)
    # One candidate, not two. A single sampler holds one rectangle, so offering
    # it here would place both nuts in the first nut's region while reporting
    # the file's numbers — a world that refuses composite is refused instead.
    assert sampler_shapes(apart) == (True,)


def test_an_environment_that_takes_no_placement_is_recognised_from_its_signature():
    from gantry_evaluator_robosuite.staging import accepts_placement

    class Takes:
        def __init__(self, robots, placement_initializer=None): ...

    class TakesNone:
        def __init__(self, robots): ...

    class TakesAnything:
        def __init__(self, robots, **kwargs): ...

    assert accepts_placement(Takes)
    assert accepts_placement(TakesAnything)
    # ToolHang is this shape: passing a placement is a TypeError from inside a
    # constructor rather than an answer, so it is asked before it is offered.
    assert not accepts_placement(TakesNone)
