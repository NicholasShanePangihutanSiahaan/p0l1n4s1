#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from beehive_drone.math_utils import clamp, wrap_pi, yaw_from_quaternion
from beehive_drone.mission_params import MissionConfig


class VelocityController(Node):
    """Global ENU position-to-velocity controller with map and stale-data gates."""

    def __init__(self) -> None:
        super().__init__("velocity_controller")

        defaults = {
            "world_frame": MissionConfig.WORLD_FRAME,
            "kp_xy": MissionConfig.KP_XY,
            "kp_z": MissionConfig.KP_Z,
            "kp_yaw": MissionConfig.KP_YAW,
            "max_velocity_xy": MissionConfig.MAX_VELOCITY_XY,
            "max_velocity_z": MissionConfig.MAX_VELOCITY_Z,
            "max_velocity_yaw": MissionConfig.MAX_VELOCITY_YAW,
            "max_acceleration_xy": MissionConfig.MAX_ACCELERATION_XY,
            "max_acceleration_z": MissionConfig.MAX_ACCELERATION_Z,
            "goal_threshold_xy": MissionConfig.GOAL_THRESHOLD_XY,
            "goal_threshold_z": MissionConfig.GOAL_THRESHOLD_Z,
            "target_timeout_sec": MissionConfig.TARGET_TIMEOUT_SEC,
            "pose_timeout_sec": MissionConfig.POSE_TIMEOUT_SEC,
            "require_map_ready": MissionConfig.REQUIRE_TREE_MAP,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.kp_xy = float(self.get_parameter("kp_xy").value)
        self.kp_z = float(self.get_parameter("kp_z").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.max_velocity_xy = float(self.get_parameter("max_velocity_xy").value)
        self.max_velocity_z = float(self.get_parameter("max_velocity_z").value)
        self.max_velocity_yaw = float(self.get_parameter("max_velocity_yaw").value)
        self.max_acceleration_xy = float(
            self.get_parameter("max_acceleration_xy").value
        )
        self.max_acceleration_z = float(self.get_parameter("max_acceleration_z").value)
        self.goal_threshold_xy = float(self.get_parameter("goal_threshold_xy").value)
        self.goal_threshold_z = float(self.get_parameter("goal_threshold_z").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.pose_timeout_sec = float(self.get_parameter("pose_timeout_sec").value)
        self.require_map_ready = bool(self.get_parameter("require_map_ready").value)

        self.current_pose: Optional[PoseStamped] = None
        self.target_pose: Optional[PoseStamped] = None
        self.last_pose_time: Optional[float] = None
        self.last_target_time: Optional[float] = None
        self.last_loop_time = self.now_sec()
        self.last_vx = self.last_vy = self.last_vz = 0.0
        self.fsm_state = "INIT"
        self.map_ready = False
        self.last_warning_time = -1e9

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_callback, qos_sensor
        )
        self.create_subscription(
            PoseStamped, "/control/safe_target_pose", self.target_callback, 10
        )
        self.create_subscription(String, "/mission/fsm_state", self.state_callback, 10)
        self.create_subscription(Bool, "/map/trees_ready", self.map_ready_callback, qos_map)
        self.velocity_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            f"Velocity Controller safety revision aktif; frame={self.world_frame}."
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg
        self.last_pose_time = self.now_sec()

    def target_callback(self, msg: PoseStamped) -> None:
        self.target_pose = msg
        self.last_target_time = self.now_sec()

    def state_callback(self, msg: String) -> None:
        self.fsm_state = msg.data

    def map_ready_callback(self, msg: Bool) -> None:
        self.map_ready = bool(msg.data)

    @staticmethod
    def rate_limit(target: float, previous: float, max_delta: float) -> float:
        return previous + clamp(target - previous, -max_delta, max_delta)

    def publish_zero(self) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.world_frame
        self.velocity_pub.publish(cmd)
        self.last_vx = self.last_vy = self.last_vz = 0.0

    def control_loop(self) -> None:
        now = self.now_sec()
        dt = max(0.01, min(0.2, now - self.last_loop_time))
        self.last_loop_time = now

        no_velocity_states = {
            "INIT",
            "WAIT_CONNECTION",
            "PRESTREAM",
            "SET_MODE",
            "ARM",
            "TAKEOFF",
            "LAND",
            "WAIT_LANDED",
            "ABORTED",
            "DONE",
        }
        if self.fsm_state in no_velocity_states:
            return

        if (
            self.current_pose is None
            or self.target_pose is None
            or self.last_pose_time is None
            or self.last_target_time is None
            or now - self.last_pose_time > self.pose_timeout_sec
            or now - self.last_target_time > self.target_timeout_sec
        ):
            self.publish_zero()
            return

        map_required_states = {
            "APPROACH_TREE",
            "HOVER_BEFORE_ORBIT",
            "PREPARE_ORBIT",
            "WAIT_ORBIT",
        }
        if (
            self.require_map_ready
            and self.fsm_state in map_required_states
            and not self.map_ready
        ):
            self.publish_zero()
            return

        pose_frame = self.current_pose.header.frame_id.strip()
        target_frame = self.target_pose.header.frame_id.strip()
        if target_frame and target_frame != self.world_frame:
            if now - self.last_warning_time > 2.0:
                self.last_warning_time = now
                self.get_logger().error(
                    f"Target frame={target_frame}, expected={self.world_frame}; velocity=0."
                )
            self.publish_zero()
            return
        if pose_frame and pose_frame != self.world_frame:
            if now - self.last_warning_time > 5.0:
                self.last_warning_time = now
                self.get_logger().warning(
                    f"Pose frame={pose_frame}, configured world={self.world_frame}. "
                    "Pastikan keduanya numerik dan TF-nya konsisten."
                )

        current = self.current_pose.pose
        target = self.target_pose.pose
        ex = float(target.position.x - current.position.x)
        ey = float(target.position.y - current.position.y)
        ez = float(target.position.z - current.position.z)
        distance_xy = math.hypot(ex, ey)

        if distance_xy <= self.goal_threshold_xy:
            vx = vy = 0.0
        else:
            vx = self.kp_xy * ex
            vy = self.kp_xy * ey
            magnitude = math.hypot(vx, vy)
            if magnitude > self.max_velocity_xy:
                scale = self.max_velocity_xy / magnitude
                vx *= scale
                vy *= scale

        vz = 0.0 if abs(ez) <= self.goal_threshold_z else clamp(
            self.kp_z * ez, -self.max_velocity_z, self.max_velocity_z
        )

        max_delta_xy = self.max_acceleration_xy * dt
        max_delta_z = self.max_acceleration_z * dt
        vx = self.rate_limit(vx, self.last_vx, max_delta_xy)
        vy = self.rate_limit(vy, self.last_vy, max_delta_xy)
        vz = self.rate_limit(vz, self.last_vz, max_delta_z)

        current_yaw = yaw_from_quaternion(current.orientation)
        q = target.orientation
        q_norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        target_yaw = current_yaw if q_norm < 1e-6 else yaw_from_quaternion(q)
        yaw_error = wrap_pi(target_yaw - current_yaw)
        yaw_rate = clamp(
            self.kp_yaw * yaw_error,
            -self.max_velocity_yaw,
            self.max_velocity_yaw,
        )

        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.world_frame
        cmd.twist.linear.x = float(vx)
        cmd.twist.linear.y = float(vy)
        cmd.twist.linear.z = float(vz)
        cmd.twist.angular.z = float(yaw_rate)
        self.velocity_pub.publish(cmd)

        self.last_vx = vx
        self.last_vy = vy
        self.last_vz = vz


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
