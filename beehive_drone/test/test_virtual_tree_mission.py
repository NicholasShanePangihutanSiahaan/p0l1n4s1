"""Unit tests for the optional virtual-tree mission strategy."""

from types import SimpleNamespace

from beehive_drone.missions.virtual_tree_test import VirtualTreeTestMission
from beehive_drone.mission_state_machine import MissionStateMachine


def tree(tree_id=7):
    return SimpleNamespace(
        id=tree_id, x=8.0, y=0.0, z=3.0, confidence=1.0,
        inspected=False, validated=True, orbit_count=0)


def fsm(mode='home_relative'):
    return SimpleNamespace(
        home_pose=(1.0, -3.0, 0.0), virtual_tree_id=9001,
        virtual_tree_position_mode=mode,
        virtual_tree_position_x=6.0, virtual_tree_position_y=3.0,
        virtual_tree_offset_toward_home=6.0)


def test_home_relative_coordinates_are_based_on_recorded_home():
    mission = VirtualTreeTestMission()
    target = mission.tree_completed(tree(), fsm())
    assert target.id == 9001
    assert target.x == 7.0
    assert target.y == 0.0


def test_map_coordinates_are_absolute():
    mission = VirtualTreeTestMission()
    target = mission.tree_completed(tree(), fsm('map'))
    assert target.x == 6.0
    assert target.y == 3.0


def test_virtual_target_is_created_only_once():
    mission = VirtualTreeTestMission()
    assert mission.tree_completed(tree(), fsm()) is not None
    assert mission.tree_completed(tree(), fsm()) is None


def test_decorate_map_does_not_duplicate_virtual_tree():
    mission = VirtualTreeTestMission()
    mission.tree_completed(tree(), fsm())
    once = mission.decorate_tree_map([tree()])
    twice = mission.decorate_tree_map(once)
    assert [item.id for item in twice].count(9001) == 1


def test_virtual_target_bypasses_real_tree_direction_and_range_filters():
    mission = VirtualTreeTestMission()
    virtual = mission.tree_completed(tree(), fsm('map'))
    virtual.x = -20.0
    virtual.y = 0.0
    state = SimpleNamespace(
        current_pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=0.0, y=0.0))),
        trees=[virtual], completed_tree_ids=set(), explore_dir_x=1.0,
        require_tree_ahead=True, mission_strategy=mission)
    state.distance = MissionStateMachine.distance.__get__(state)

    selected = MissionStateMachine.find_uninspected_tree(state)

    assert selected.id == 9001
