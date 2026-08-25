"""Pure terrain-following math shared by ROS control and unit tests."""

from collections import deque
import math
from statistics import median


def clamp(value, lower, upper):
    """Return *value* limited to the inclusive ``[lower, upper]`` range."""
    return max(lower, min(upper, value))


def body_vertical_cosine(quaternion):
    """Return the world-Z component of the vehicle body-Z axis.

    A downward rangefinder measures a slant distance when the vehicle tilts.
    The vertical ground clearance is the slant distance multiplied by this
    cosine.  Yaw has no influence on the result.
    """
    norm = math.sqrt(
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + quaternion.w * quaternion.w
    )
    if norm <= 1.0e-9:
        return 0.0
    x = quaternion.x / norm
    y = quaternion.y / norm
    return 1.0 - 2.0 * (x * x + y * y)


class TerrainFollower:
    """Convert rangefinder AGL measurements into a smooth local-Z target.

    The EKF/local pose remains the navigation reference.  The rangefinder only
    moves the vertical target so the vehicle holds a requested clearance above
    the surface.  If measurements become stale, the last valid vertical target
    is frozen instead of falling back abruptly to the nominal local-Z target.
    """

    TRACKING = "TRACKING"
    WAITING_FOR_RANGE = "WAITING_FOR_RANGE"
    HOLD_RANGE_STALE = "HOLD_RANGE_STALE"

    def __init__(
        self,
        desired_agl,
        timeout,
        filter_alpha,
        max_target_rate,
        max_correction,
        min_tilt_cosine,
        sensor_offset=0.0,
        median_window=5,
    ):
        self.desired_agl = float(desired_agl)
        self.timeout = max(0.05, float(timeout))
        self.filter_alpha = clamp(float(filter_alpha), 0.01, 1.0)
        self.max_target_rate = max(0.01, float(max_target_rate))
        self.max_correction = max(0.01, float(max_correction))
        self.min_tilt_cosine = clamp(float(min_tilt_cosine), 0.05, 1.0)
        self.sensor_offset = float(sensor_offset)
        self.samples = deque(maxlen=max(1, int(median_window)))
        self.filtered_agl = None
        self.last_range_time = None
        self.target_z = None
        self.last_target_time = None
        self.status = self.WAITING_FOR_RANGE

    def reset_target(self):
        """Clear only the generated target while keeping the range filter warm."""
        self.target_z = None
        self.last_target_time = None

    def ingest(self, slant_range, quaternion, now):
        """Validate, tilt-correct and filter one rangefinder measurement."""
        if not math.isfinite(slant_range):
            return False
        cosine = body_vertical_cosine(quaternion)
        if cosine < self.min_tilt_cosine:
            return False
        vertical_agl = slant_range * cosine + self.sensor_offset
        if not math.isfinite(vertical_agl) or vertical_agl <= 0.0:
            return False
        now = float(now)
        if (self.last_range_time is not None
                and now - self.last_range_time > self.timeout):
            # Do not mix a new surface with samples captured before a dropout.
            self.samples.clear()
            self.filtered_agl = None
        self.samples.append(vertical_agl)
        robust_sample = float(median(self.samples))
        if self.filtered_agl is None:
            self.filtered_agl = robust_sample
        else:
            alpha = self.filter_alpha
            self.filtered_agl += alpha * (robust_sample - self.filtered_agl)
        self.last_range_time = now
        return True

    def range_is_fresh(self, now):
        """Return whether a filtered measurement exists and is not stale."""
        return (
            self.filtered_agl is not None
            and self.last_range_time is not None
            and float(now) - self.last_range_time <= self.timeout
        )

    def compute_target(self, local_z, nominal_z, now):
        """Return ``(target_z, status)`` for the current controller cycle."""
        now = float(now)
        if self.range_is_fresh(now):
            error = clamp(
                self.desired_agl - self.filtered_agl,
                -self.max_correction,
                self.max_correction,
            )
            raw_target = float(local_z) + error
            if self.target_z is None or self.last_target_time is None:
                self.target_z = raw_target
            else:
                dt = max(0.0, now - self.last_target_time)
                max_delta = self.max_target_rate * dt
                self.target_z += clamp(
                    raw_target - self.target_z, -max_delta, max_delta
                )
            self.last_target_time = now
            self.status = self.TRACKING
            return self.target_z, self.status

        if self.target_z is not None:
            self.status = self.HOLD_RANGE_STALE
            self.last_target_time = now
            return self.target_z, self.status

        self.status = self.WAITING_FOR_RANGE
        return float(nominal_z), self.status

    @property
    def agl_error(self):
        """Latest desired-minus-measured AGL error, or NaN before first data."""
        if self.filtered_agl is None:
            return math.nan
        return self.desired_agl - self.filtered_agl
