"""Deterministic tests for terrain-following altitude math."""

import math
from types import SimpleNamespace

import pytest

from beehive_drone.terrain_following import (
    TerrainFollower,
    body_vertical_cosine,
)


def quaternion_from_roll(roll):
    """Return a minimal quaternion-shaped object for a roll rotation."""
    return SimpleNamespace(
        x=math.sin(roll * 0.5), y=0.0, z=0.0,
        w=math.cos(roll * 0.5))


def upright_quaternion():
    """Return an identity quaternion-shaped object."""
    return quaternion_from_roll(0.0)


def follower(**overrides):
    """Build a fast deterministic follower for unit tests."""
    values = {
        'desired_agl': 1.5,
        'timeout': 0.5,
        'filter_alpha': 1.0,
        'max_target_rate': 1.0,
        'max_correction': 1.0,
        'min_tilt_cosine': 0.7,
        'sensor_offset': 0.0,
        'median_window': 1,
    }
    values.update(overrides)
    return TerrainFollower(**values)


def test_tilt_compensation_converts_slant_to_vertical_distance():
    """A 30 degree roll must multiply slant range by cos(30 degrees)."""
    quaternion = quaternion_from_roll(math.radians(30.0))
    assert body_vertical_cosine(quaternion) == pytest.approx(
        math.cos(math.radians(30.0)), abs=1.0e-6)


def test_target_rises_with_a_gentle_hill():
    """A 0.4 m terrain rise should raise local-Z target by 0.4 m."""
    control = follower()
    quaternion = upright_quaternion()
    assert control.ingest(1.5, quaternion, now=0.0)
    target, status = control.compute_target(1.5, 1.5, now=0.0)
    assert status == TerrainFollower.TRACKING
    assert target == pytest.approx(1.5)

    # Ground rises while the vehicle has not reacted yet: AGL falls to 1.1 m.
    assert control.ingest(1.1, quaternion, now=1.0)
    target, status = control.compute_target(1.5, 1.5, now=1.0)
    assert status == TerrainFollower.TRACKING
    assert target == pytest.approx(1.9)


def test_target_rate_is_limited():
    """A sudden surface change must not create a vertical setpoint jump."""
    control = follower(max_target_rate=0.25)
    quaternion = upright_quaternion()
    control.ingest(1.5, quaternion, now=0.0)
    control.compute_target(1.5, 1.5, now=0.0)
    control.ingest(0.8, quaternion, now=0.2)
    target, _ = control.compute_target(1.5, 1.5, now=0.2)
    assert target == pytest.approx(1.55)


def test_stale_range_freezes_last_target():
    """Range loss must hold the last local-Z target, never revert downward."""
    control = follower(timeout=0.2)
    quaternion = upright_quaternion()
    control.ingest(1.2, quaternion, now=0.0)
    target, _ = control.compute_target(1.5, 1.5, now=0.0)
    held, status = control.compute_target(1.6, 1.5, now=0.3)
    assert status == TerrainFollower.HOLD_RANGE_STALE
    assert held == pytest.approx(target)


def test_median_filter_rejects_one_short_ground_echo():
    """One isolated short return must not make the vehicle climb."""
    control = follower(median_window=5)
    quaternion = upright_quaternion()
    for index in range(4):
        control.ingest(1.5, quaternion, now=index * 0.02)
    control.ingest(0.2, quaternion, now=0.08)
    assert control.filtered_agl == pytest.approx(1.5)


def test_excessive_tilt_measurement_is_rejected():
    """A near-horizontal beam cannot be treated as vertical clearance."""
    control = follower(min_tilt_cosine=0.7)
    quaternion = quaternion_from_roll(math.radians(60.0))
    assert not control.ingest(1.5, quaternion, now=0.0)
    _, status = control.compute_target(1.5, 1.5, now=0.0)
    assert status == TerrainFollower.WAITING_FOR_RANGE
