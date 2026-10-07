"""Tests for the mission state shared by all multi-tree strategies."""

from types import SimpleNamespace

from beehive_drone.mission_state_machine import MissionStateMachine


class LoggerStub:
    def info(self, _message):
        pass


def make_fsm(max_trees=0, mission_mode='multi_tree'):
    fsm = SimpleNamespace(
        mission_mode=mission_mode,
        max_trees=max_trees,
        completed_tree_ids=set(),
        target_tree=object(),
        frozen_target_tree=object(),
        verification_retries=2,
        receiving_flower_pose=True,
        done_receiving_flower_pose=True,
        flower_pose=object(),
        state='POST_ORBIT_HOVER',
        last_tree_x=0.0,
        last_tree_y=0.0,
    )
    fsm.transition = lambda state: setattr(fsm, 'state', state)
    fsm.get_logger = lambda: LoggerStub()
    return fsm


def test_completed_tree_is_recorded_once_and_updates_row_reference():
    fsm = make_fsm()
    tree = SimpleNamespace(id=7, x=12.5, y=-3.0)

    MissionStateMachine.record_completed_tree(fsm, tree)
    MissionStateMachine.record_completed_tree(fsm, tree)

    assert fsm.completed_tree_ids == {7}
    assert fsm.last_tree_x == 12.5
    assert fsm.last_tree_y == -3.0


def test_unlimited_mission_continues_to_explore_and_resets_tree_state():
    fsm = make_fsm(max_trees=0)
    fsm.completed_tree_ids = {1}

    MissionStateMachine.advance_after_tree(fsm)

    assert fsm.state == 'EXPLORE_ROW'
    assert fsm.target_tree is None
    assert fsm.frozen_target_tree is None
    assert fsm.flower_pose is None
    assert not fsm.receiving_flower_pose
    assert not fsm.done_receiving_flower_pose


def test_tree_limit_returns_home():
    fsm = make_fsm(max_trees=2)
    fsm.completed_tree_ids = {1, 2}

    MissionStateMachine.advance_after_tree(fsm)

    assert fsm.state == 'ALIGN_HOME'


def test_single_tree_returns_home_after_first_tree_even_without_limit():
    fsm = make_fsm(max_trees=0, mission_mode='single_tree')
    fsm.completed_tree_ids = {1}

    MissionStateMachine.advance_after_tree(fsm)

    assert fsm.state == 'ALIGN_HOME'


def test_completed_tree_cannot_be_selected_again():
    fsm = SimpleNamespace(
        current_pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0))),
        trees=[
            SimpleNamespace(id=1, x=2.0, y=0.0, inspected=False),
            SimpleNamespace(id=2, x=4.0, y=0.0, inspected=False),
        ],
        completed_tree_ids={1},
        explore_dir_x=1.0,
    )
    fsm.distance = MissionStateMachine.distance.__get__(fsm)

    selected = MissionStateMachine.find_uninspected_tree(fsm)

    assert selected.id == 2
