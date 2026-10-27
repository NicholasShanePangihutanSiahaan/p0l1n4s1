"""Altitude gates use home-relative local Z and do not gate the orbit itself."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from beehive_drone.missions.basic_orbit import BasicOrbitMission


class Stamp:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)

    def __sub__(self, other):
        return Stamp((self.nanoseconds - other.nanoseconds) / 1e9)


def altitude_fsm(state):
    clock = SimpleNamespace(now=Mock(return_value=Stamp(0)))
    result = SimpleNamespace(
        state=state, altitude_mode='home_relative', hold_x=3.0, hold_y=4.0, hold_yaw=0.5,
        navigation_altitude=12.0, altitude_transition_tolerance=0.15,
        altitude_transition_hold_time=1.5, altitude_reached_since=None,
        current_pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(z=12.0))),
        publish_goal=Mock(), transition=Mock(), advance_after_tree=Mock(),
        get_clock=lambda: clock, clock=clock)
    result.navigation_altitude_error = lambda: abs(result.current_pose.pose.position.z - result.navigation_altitude)
    return result


@pytest.mark.parametrize('state', ['PRE_ORBIT_ALTITUDE', 'RESTORE_FLIGHT_ALTITUDE'])
def test_altitude_must_remain_within_tolerance_before_continuing(state):
    mission = BasicOrbitMission()
    fsm = altitude_fsm(state)
    mission.execute(fsm, True, 0, 3, 4)
    fsm.transition.assert_not_called()
    fsm.advance_after_tree.assert_not_called()
    fsm.clock.now.return_value = Stamp(1)
    fsm.current_pose.pose.position.z = 11.5
    mission.execute(fsm, True, 1, 3, 4)
    assert fsm.altitude_reached_since is None
    fsm.current_pose.pose.position.z = 12.0
    fsm.clock.now.return_value = Stamp(2)
    mission.execute(fsm, True, 2, 3, 4)
    fsm.clock.now.return_value = Stamp(3.6)
    mission.execute(fsm, True, 3.6, 3, 4)
    fsm.publish_goal.assert_called_with(3.0, 4.0, 0.5)
    if state == 'PRE_ORBIT_ALTITUDE':
        fsm.transition.assert_called_once_with('START_ORBIT')
    else:
        fsm.advance_after_tree.assert_called_once()


def test_orbit_wait_does_not_check_altitude():
    fsm = SimpleNamespace(state='WAIT_ORBIT', orbit_status='ORBIT_RUNNING')
    BasicOrbitMission().execute(fsm, True, 0, 0, 0)
