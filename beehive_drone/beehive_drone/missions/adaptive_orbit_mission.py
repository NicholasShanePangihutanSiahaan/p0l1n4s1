"""Opt-in real-stack strategy: require LiDAR before the existing pre-orbit phase."""

import math
import time
from copy import deepcopy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Range
from rclpy.qos import qos_profile_sensor_data
from beehive_drone.missions.basic_orbit import BasicOrbitMission
from beehive_drone.adaptive_orbit import orbit_target_z


class AdaptiveOrbitMission(BasicOrbitMission):
    def execute(self, fsm, active, elapsed, cx, cy):
        if not hasattr(self, "publisher"):
            if fsm.altitude_mode != "home_relative":
                raise ValueError(
                    "adaptive_orbit requires home_relative: fixed local Z during orbit"
                )
            self.result = None
            self.clearance = None
            self.held_since = None
            for name, value in dict(
                adaptive_orbit_fraction=0.5,
                adaptive_orbit_min_height=1.0,
                adaptive_orbit_max_height=4.0,
                lidar_verification_timeout=8.0,
                lidar_max_position_correction=0.6,
            ).items():
                if not fsm.has_parameter(name):
                    fsm.declare_parameter(name, value)
            self.publisher = fsm.create_publisher(
                Float64MultiArray, "/mission/lidar_candidate", 10
            )
            self.result_sub = fsm.create_subscription(
                Float64MultiArray, "/perception/lidar_verified_tree", self.receive, 10
            )
            self.range_sub = fsm.create_subscription(
                Range,
                "/mavros/rangefinder/rangefinder",
                self.range,
                qos_profile_sensor_data,
            )
        if fsm.state != "VERIFY_TREE":
            self.held_since = None
            return super().execute(fsm, active, elapsed, cx, cy)

        # Tahan posisi sambil meminta verifikasi LiDAR untuk kandidat terpilih.
        tree = fsm.target_tree
        if tree is None:
            return
        yaw = math.atan2(tree.y - cy, tree.x - cx)
        fsm.publish_goal(cx, cy, yaw)
        self.publisher.publish(
            Float64MultiArray(
                data=[float(tree.id), float(tree.x), float(tree.y), float(tree.z)]
            )
        )

        # Batasi waktu tunggu sebelum hasil LiDAR tersedia.
        timeout = float(fsm.get_parameter("lidar_verification_timeout").value)
        if elapsed > timeout:
            fsm.transition("ABORT")
            fsm.get_logger().error(
                "Adaptive orbit: LiDAR validation timed out; no orbit authorized."
            )
            return

        # Periksa kesegaran hasil, clearance, dan kestabilan pengamatan.
        now = time.monotonic()
        r = self.result
        fresh = r is not None and int(r[0]) == tree.id and now - r[-1] < 0.5
        clearance_ok = (
            self.clearance is not None
            and now - self.clearance[1] < 0.5
            and self.clearance[0] >= 0.8
        )
        if not fresh or not clearance_ok:
            self.held_since = None
            return
        _, x, y, base, height, radius, _ = r
        if math.hypot(x - tree.x, y - tree.y) > float(
            fsm.get_parameter("lidar_max_position_correction").value
        ):
            self.held_since = None
            return
        if self.held_since is None:
            self.held_since = now
        if now - self.held_since < 1.0:
            return

        # Kunci koordinat hasil koreksi dan hitung target altitude orbit.
        corrected = deepcopy(tree)
        corrected.x = float(x)
        corrected.y = float(y)
        corrected.z = float(base + height / 2)
        corrected.validated = True
        # LiDAR TF target frame and FC local coordinates must be aligned by the real localization stack.
        target = orbit_target_z(
            base,
            height,
            float(fsm.get_parameter("adaptive_orbit_fraction").value),
            float(fsm.get_parameter("adaptive_orbit_min_height").value),
            float(fsm.get_parameter("adaptive_orbit_max_height").value),
        )
        if (
            target - fsm.current_pose.pose.position.z > 1.5
            or target - fsm.current_pose.pose.position.z < -1.0
        ):
            fsm.transition("ABORT")
            fsm.get_logger().error(
                "Adaptive orbit: altitude change exceeds initial test bounds."
            )
            return

        # Teruskan ke fase penyesuaian altitude sebelum orbit.
        fsm.target_tree = corrected
        fsm.frozen_target_tree = deepcopy(corrected)
        fsm.tree_update_pub.publish(corrected)
        fsm.hold_x, fsm.hold_y, fsm.hold_yaw = cx, cy, yaw
        fsm.navigation_altitude = target
        fsm.transition("PRE_ORBIT_ALTITUDE")
        fsm.get_logger().info(
            f"LiDAR verified tree {tree.id}: visible height={height:.2f}, local Z target={target:.2f}"
        )

    def receive(self, msg):
        if (
            len(msg.data) == 6
            and all(math.isfinite(v) for v in msg.data)
            and msg.data[4] > 0.0
            and 0.1 <= msg.data[5] <= 0.65
        ):
            self.result = (*msg.data, time.monotonic())

    def range(self, msg):
        if (
            math.isfinite(msg.range)
            and max(0.08, msg.min_range) <= msg.range <= msg.max_range
        ):
            self.clearance = (msg.range, time.monotonic())
