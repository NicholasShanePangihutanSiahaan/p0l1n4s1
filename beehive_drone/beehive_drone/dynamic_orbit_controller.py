#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String

from beehive_drone.math_utils import quaternion_from_yaw, wrap_pi
from beehive_drone.mission_params import MissionConfig


class DynamicOrbitController(Node):
    """Produces a moving orbit setpoint and verifies real angular progress."""

    def __init__(self) -> None:
        super().__init__("dynamic_orbit_controller")

        defaults = {
            "world_frame": MissionConfig.WORLD_FRAME,
            "orbit_radius": MissionConfig.ORBIT_RADIUS,
            "orbit_altitude": MissionConfig.ORBIT_ALTITUDE,
            "orbit_velocity": MissionConfig.ORBIT_VELOCITY,
            "orbit_direction": MissionConfig.ORBIT_DIRECTION,
            "lookahead_distance": MissionConfig.ORBIT_LOOKAHEAD_DISTANCE,
            "radius_tolerance": MissionConfig.ORBIT_RADIUS_TOLERANCE,
            "completion_margin": MissionConfig.ORBIT_COMPLETION_MARGIN,
            "orbit_timeout_sec": MissionConfig.ORBIT_TIMEOUT_SEC,
            "yaw_offset": MissionConfig.YAW_OFFSET,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.orbit_radius = max(0.5, float(self.get_parameter("orbit_radius").value))
        self.orbit_altitude = float(self.get_parameter("orbit_altitude").value)
        self.orbit_velocity = max(0.1, float(self.get_parameter("orbit_velocity").value))
        self.direction = 1.0 if float(self.get_parameter("orbit_direction").value) >= 0.0 else -1.0
        self.lookahead_distance = max(
            0.2, float(self.get_parameter("lookahead_distance").value)
        )
        self.radius_tolerance = max(
            0.1, float(self.get_parameter("radius_tolerance").value)
        )
        self.completion_margin = max(
            0.0, float(self.get_parameter("completion_margin").value)
        )
        self.orbit_timeout_sec = float(self.get_parameter("orbit_timeout_sec").value)
        self.yaw_offset = float(self.get_parameter("yaw_offset").value)

        self.current_pose: Optional[PoseStamped] = None
        self.tree = Point()
        self.target_received_time: Optional[float] = None
        self.phase = "IDLE"
        self.phase_start_time = self.now_sec()
        self.radius_stable_since: Optional[float] = None
        self.last_angle: Optional[float] = None
        self.accumulated_angle = 0.0
        self.last_status = ""

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_callback, qos_sensor
        )
        self.create_subscription(Point, "/control/orbit_target", self.target_callback, 10)
        self.create_subscription(Bool, "/control/orbit_start", self.start_callback, 10)

        self.setpoint_pub = self.create_publisher(
            PoseStamped, "/control/dynamic_target", 10
        )
        self.status_pub = self.create_publisher(String, "/control/orbit_status", 10)
        self.progress_pub = self.create_publisher(
            Float32, "/control/orbit_progress", 10
        )

        self.create_timer(0.05, self.control_loop)
        self.publish_status("IDLE", force=True)
        self.get_logger().info("Dynamic Orbit Controller revisi aktif.")

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def target_callback(self, msg: Point) -> None:
        if not all(math.isfinite(float(v)) for v in (msg.x, msg.y, msg.z)):
            self.get_logger().warning("Target orbit invalid diabaikan.")
            return
        self.tree = msg
        self.target_received_time = self.now_sec()

    def start_callback(self, msg: Bool) -> None:
        if not msg.data:
            if self.phase != "IDLE":
                self.get_logger().info("Orbit dihentikan oleh mission state machine.")
            self.reset("IDLE")
            return

        if self.current_pose is None:
            self.publish_status("ORBIT_FAILED")
            return
        if self.target_received_time is None or self.now_sec() - self.target_received_time > 2.0:
            self.get_logger().error("Orbit ditolak: target belum tersedia atau stale.")
            self.publish_status("ORBIT_FAILED")
            return
        if self.phase == "IDLE":
            self.phase = "ALIGNING"
            self.phase_start_time = self.now_sec()
            self.radius_stable_since = None
            self.last_angle = None
            self.accumulated_angle = 0.0
            self.publish_status("ALIGNING")
            self.get_logger().info(
                f"Orbit dimulai pada ({self.tree.x:.2f}, {self.tree.y:.2f}), radius={self.orbit_radius:.2f}."
            )

    def reset(self, status: str) -> None:
        self.phase = "IDLE"
        self.phase_start_time = self.now_sec()
        self.radius_stable_since = None
        self.last_angle = None
        self.accumulated_angle = 0.0
        self.publish_status(status, force=True)
        self.publish_progress(0.0)

    def publish_status(self, status: str, force: bool = False) -> None:
        if not force and status == self.last_status:
            return
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.last_status = status

    def publish_progress(self, value: float) -> None:
        msg = Float32()
        msg.data = float(max(0.0, min(1.0, value)))
        self.progress_pub.publish(msg)

    def publish_target(self, angle: float, radius: float) -> None:
        target_x = self.tree.x + radius * math.cos(angle)
        target_y = self.tree.y + radius * math.sin(angle)
        yaw_to_tree = math.atan2(self.tree.y - target_y, self.tree.x - target_x)
        target_yaw = yaw_to_tree + self.direction * self.yaw_offset

        setpoint = PoseStamped()
        setpoint.header.frame_id = self.world_frame
        setpoint.header.stamp = self.get_clock().now().to_msg()
        setpoint.pose.position.x = float(target_x)
        setpoint.pose.position.y = float(target_y)
        setpoint.pose.position.z = float(self.orbit_altitude)
        setpoint.pose.orientation = quaternion_from_yaw(target_yaw)
        self.setpoint_pub.publish(setpoint)

    def control_loop(self) -> None:
        if self.phase == "IDLE" or self.current_pose is None:
            return

        now = self.now_sec()
        if now - self.phase_start_time > self.orbit_timeout_sec:
            self.get_logger().error("Orbit timeout.")
            self.reset("ORBIT_TIMEOUT")
            return

        px = float(self.current_pose.pose.position.x)
        py = float(self.current_pose.pose.position.y)
        dx = px - self.tree.x
        dy = py - self.tree.y
        current_radius = math.hypot(dx, dy)
        if current_radius < 0.05:
            current_angle = 0.0
        else:
            current_angle = math.atan2(dy, dx)

        if self.phase == "ALIGNING":
            self.publish_target(current_angle, self.orbit_radius)
            radius_ok = abs(current_radius - self.orbit_radius) <= self.radius_tolerance
            if radius_ok:
                if self.radius_stable_since is None:
                    self.radius_stable_since = now
                elif now - self.radius_stable_since >= 0.6:
                    self.phase = "ORBITING"
                    self.phase_start_time = now
                    self.last_angle = current_angle
                    self.accumulated_angle = 0.0
                    self.publish_status("IN_PROGRESS")
            else:
                self.radius_stable_since = None
            return

        if self.last_angle is not None:
            delta = wrap_pi(current_angle - self.last_angle)
            directed_delta = self.direction * delta
            # Ignore reverse motion and single-frame jumps.
            if 0.0 < directed_delta < 0.35 and abs(current_radius - self.orbit_radius) <= self.radius_tolerance:
                self.accumulated_angle += directed_delta
        self.last_angle = current_angle

        required_angle = max(0.1, 2.0 * math.pi - self.completion_margin)
        progress = self.accumulated_angle / required_angle
        self.publish_progress(progress)

        if self.accumulated_angle >= required_angle:
            self.get_logger().info("Orbit 360 derajat terverifikasi selesai.")
            self.reset("ORBIT_COMPLETED")
            return

        angular_speed = self.orbit_velocity / self.orbit_radius
        lookahead_angle = max(
            self.lookahead_distance / self.orbit_radius,
            angular_speed * 0.8,
        )
        target_angle = current_angle + self.direction * lookahead_angle
        self.publish_target(target_angle, self.orbit_radius)
        self.publish_status("IN_PROGRESS")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynamicOrbitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.reset("IDLE")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
