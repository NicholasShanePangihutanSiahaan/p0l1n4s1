import math

from beehive_drone.mission_state_machine import landing_command_due, yaw_aligned


def test_first_land_request_is_sent():
    assert landing_command_due('GUIDED', float('inf'), 1.0)


def test_land_request_is_throttled_while_mode_change_is_pending():
    assert not landing_command_due('GUIDED', 0.2, 1.0)


def test_land_request_stops_after_flight_controller_reports_land():
    assert not landing_command_due('LAND', float('inf'), 1.0)


def test_yaw_alignment_uses_shortest_path_across_wrap():
    assert yaw_aligned(math.radians(179), math.radians(-179), math.radians(3))


def test_yaw_alignment_rejects_large_error():
    assert not yaw_aligned(0.0, math.radians(12), math.radians(7.5))
