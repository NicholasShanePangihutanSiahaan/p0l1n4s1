"""FSM AGL changes and terrain changes must both affect the vertical target."""
from types import SimpleNamespace
import pytest
from beehive_drone.terrain_following import TerrainFollower
from beehive_drone.mission_state_machine import MissionStateMachine


def follower():
    return TerrainFollower(timeout=.5, filter_alpha=1., max_target_rate=.5,
                           max_correction=1., min_tilt_cosine=.7, median_window=1)


def test_fsm_agl_change_overrides_previous_phase():
    f = follower()
    q = SimpleNamespace(x=0., y=0., z=0., w=1.)
    f.ingest(1.5, q, 0.)
    assert f.compute_target(11.5, 11.5, 0., 1.5)[0] == 11.5
    f.ingest(1.5, q, .2)
    # Y rises to 2 m AGL; target climbs with the configured rate limit.
    assert f.compute_target(11.5, 11.5, .2, 2.)[0] == pytest.approx(11.6)
    f.ingest(2., q, .4)
    # Returning to X causes descent, despite the old orbit target.
    assert f.compute_target(12., 12., .4, 1.5)[0] == pytest.approx(11.5)


def test_hill_changes_local_z_target_at_constant_fsm_agl():
    f = follower()
    q = SimpleNamespace(x=0., y=0., z=0., w=1.)
    f.ingest(1.1, q, 0.)
    assert f.compute_target(11.5, 11.5, 0., 1.5)[0] == pytest.approx(11.9)
    held, status = f.compute_target(11.6, 11.5, 1., 1.5)
    assert held == pytest.approx(11.9)
    assert status == f.HOLD_RANGE_STALE


def test_terrain_altitude_gate_uses_agl_and_rejects_stale_measurement():
    stamp = SimpleNamespace(__unused=0)
    class Time:
        nanoseconds = 0
        def __sub__(self, other):
            return self
    now = Time()
    fsm = SimpleNamespace(altitude_mode='terrain_following', navigation_altitude=2.,
                          measured_agl=2., measured_agl_time=now,
                          get_clock=lambda: SimpleNamespace(now=lambda: now))
    assert MissionStateMachine.navigation_altitude_error(fsm) == 0.
    now.nanoseconds = 600000000
    assert MissionStateMachine.navigation_altitude_error(fsm) == float('inf')
