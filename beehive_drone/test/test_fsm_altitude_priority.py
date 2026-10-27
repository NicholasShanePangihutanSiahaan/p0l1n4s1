"""Mission altitude must remain authoritative regardless of range availability."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from geometry_msgs.msg import PoseStamped
from beehive_drone.position_setpoint_controller import PositionSetpointController


class Stamp:
    nanoseconds = 0

    def __sub__(self, other):
        return self

    def to_msg(self):
        return PoseStamped().header.stamp


@pytest.mark.parametrize('target_z', [2.0, 1.5])
def test_fsm_climb_and_return_follow_mission_target(target_z):
    now = Stamp()
    pose = PoseStamped()
    pose.pose.position.z = 1.5 if target_z == 2.0 else 2.0
    target = PoseStamped()
    target.pose.position.z = target_z
    fsm = SimpleNamespace(
        pose=pose, enabled=True, altitude_mode="home_relative", commanded_pose=None, target=target,
        target_time=now, timeout=0.75, 
        max_xy=0.1, max_z=0.05, output_frame='map',
        get_clock=lambda: SimpleNamespace(now=lambda: now),
        pub=Mock(),
        copy_pose=PositionSetpointController.copy_pose)
    PositionSetpointController.loop(fsm)
    actual = fsm.pub.publish.call_args.args[0].pose.position.z
    assert actual == pytest.approx(1.55 if target_z == 2.0 else 1.95)


def test_terrain_mode_holds_position_without_fresh_range():
    now = Stamp()
    pose = PoseStamped()
    pose.pose.position.z = 1.5
    target = PoseStamped()
    target.pose.position.x = 5.0
    target.pose.position.z = 99.0
    controller = SimpleNamespace(
        pose=pose, enabled=True, altitude_mode='terrain_following',
        commanded_pose=None, target=target, target_time=now, timeout=.75,
        agl_target=2.0, agl_target_time=0., output_frame="map",
        get_clock=lambda: SimpleNamespace(now=lambda: now),
        pub=Mock(), copy_pose=PositionSetpointController.copy_pose)
    PositionSetpointController.loop(controller)
    output = controller.pub.publish.call_args.args[0]
    assert output.pose.position.x == 0.
    assert output.pose.position.z == 1.5
