import math

from beehive_drone.frame_alignment import (
    circular_mean,
    multiply_quaternion,
    transform_xy,
    wrap_angle,
    yaw_quaternion,
)


def test_circular_mean_handles_wraparound():
    mean = circular_mean([math.radians(179), math.radians(-179)])
    assert abs(abs(math.degrees(mean)) - 180.0) < 1.0e-6


def test_transform_xy_rotates_then_translates():
    x, y = transform_xy(1.0, 0.0, math.pi / 2.0, 2.0, 3.0)
    assert abs(x - 2.0) < 1.0e-9
    assert abs(y - 4.0) < 1.0e-9


def test_yaw_quaternion_composition_adds_yaw():
    result = multiply_quaternion(
        yaw_quaternion(math.radians(30)),
        yaw_quaternion(math.radians(15)))
    yaw = math.atan2(
        2.0 * (result[3] * result[2] + result[0] * result[1]),
        1.0 - 2.0 * (result[1] ** 2 + result[2] ** 2))
    assert abs(wrap_angle(yaw - math.radians(45))) < 1.0e-9
