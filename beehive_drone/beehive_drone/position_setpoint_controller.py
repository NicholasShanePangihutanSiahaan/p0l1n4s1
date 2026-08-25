#!/usr/bin/env python3
"""Safe, continuous position-setpoint adapter for MAVROS/ArduPilot."""

import math
import time
from copy import deepcopy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32, String

from beehive_drone.terrain_following import TerrainFollower


class PositionSetpointController(Node):
    def __init__(self):
        super().__init__('position_setpoint_controller')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('max_horizontal_step', 0.10)
        self.declare_parameter('max_vertical_step', 0.05)
        self.declare_parameter('target_timeout', 0.75)
        self.declare_parameter('output_frame', 'map')
        self.declare_parameter('terrain_following_enabled', False)
        self.declare_parameter(
            'rangefinder_topic', '/mavros/rangefinder/rangefinder')
        self.declare_parameter('desired_agl', 1.5)
        self.declare_parameter('rangefinder_timeout', 0.50)
        self.declare_parameter('rangefinder_min_valid', 0.08)
        self.declare_parameter('rangefinder_max_valid', 8.0)
        self.declare_parameter('rangefinder_altitude_offset', 0.0)
        self.declare_parameter('range_filter_alpha', 0.35)
        self.declare_parameter('range_median_window', 5)
        self.declare_parameter('minimum_tilt_cosine', 0.70)
        self.declare_parameter('max_terrain_correction', 1.0)
        self.declare_parameter('max_terrain_target_rate', 0.50)
        self.declare_parameter('hold_position_on_range_loss', True)
        self.max_xy = float(self.get_parameter('max_horizontal_step').value)
        self.max_z = float(self.get_parameter('max_vertical_step').value)
        self.timeout = float(self.get_parameter('target_timeout').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.terrain_enabled = bool(
            self.get_parameter('terrain_following_enabled').value)
        self.rangefinder_topic = str(
            self.get_parameter('rangefinder_topic').value)
        self.rangefinder_min = float(
            self.get_parameter('rangefinder_min_valid').value)
        self.rangefinder_max = float(
            self.get_parameter('rangefinder_max_valid').value)
        self.hold_on_range_loss = bool(
            self.get_parameter('hold_position_on_range_loss').value)
        self.pose = None
        self.target = None
        self.target_time = None
        self.enabled = False
        self.commanded_pose = None
        self.terrain_status = None
        self.terrain_follower = TerrainFollower(
            desired_agl=float(self.get_parameter('desired_agl').value),
            timeout=float(self.get_parameter('rangefinder_timeout').value),
            filter_alpha=float(self.get_parameter('range_filter_alpha').value),
            max_target_rate=float(
                self.get_parameter('max_terrain_target_rate').value),
            max_correction=float(
                self.get_parameter('max_terrain_correction').value),
            min_tilt_cosine=float(
                self.get_parameter('minimum_tilt_cosine').value),
            sensor_offset=float(
                self.get_parameter('rangefinder_altitude_offset').value),
            median_window=int(self.get_parameter('range_median_window').value),
        )
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos)
        self.create_subscription(PoseStamped, '/control/safe_target_pose', self.target_cb, 10)
        self.create_subscription(Bool, '/control/setpoint_enabled', self.enabled_cb, 10)
        self.create_subscription(
            Range, self.rangefinder_topic, self.rangefinder_cb, qos)
        self.pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        self.terrain_agl_pub = self.create_publisher(
            Float32, '/control/terrain/measured_agl', 10)
        self.terrain_error_pub = self.create_publisher(
            Float32, '/control/terrain/agl_error', 10)
        self.terrain_target_pub = self.create_publisher(
            Float32, '/control/terrain/target_z', 10)
        self.terrain_status_pub = self.create_publisher(
            String, '/control/terrain/status', 10)
        rate = max(10.0, float(self.get_parameter('publish_rate').value))
        self.create_timer(1.0 / rate, self.loop)
        mode = 'terrain-following' if self.terrain_enabled else 'fixed local-Z'
        self.get_logger().info(
            'Position setpoint controller aktif (continuous + slew limit, %s).'
            % mode)

    def pose_cb(self, msg):
        self.pose = msg

    def target_cb(self, msg):
        self.target = msg
        self.target_time = self.get_clock().now()

    def rangefinder_cb(self, msg):
        if not self.terrain_enabled or self.pose is None:
            return
        value = float(msg.range)
        lower = self.rangefinder_min
        upper = self.rangefinder_max
        if msg.min_range > 0.0:
            lower = max(lower, float(msg.min_range))
        if msg.max_range > 0.0:
            upper = min(upper, float(msg.max_range))
        if not math.isfinite(value) or value < lower or value > upper:
            return
        self.terrain_follower.ingest(
            value, self.pose.pose.orientation, time.monotonic())

    def enabled_cb(self, msg):
        if msg.data and not self.enabled and self.pose is not None:
            self.commanded_pose = deepcopy(self.pose)
            self.terrain_follower.reset_target()
        elif not msg.data:
            self.commanded_pose = None
            self.terrain_follower.reset_target()
        self.enabled = msg.data

    def publish_terrain_diagnostics(self, target_z, status):
        measured = Float32()
        measured.data = (
            float(self.terrain_follower.filtered_agl)
            if self.terrain_follower.filtered_agl is not None else float('nan'))
        error = Float32()
        error.data = float(self.terrain_follower.agl_error)
        target = Float32()
        target.data = float(target_z)
        status_msg = String()
        status_msg.data = status
        self.terrain_agl_pub.publish(measured)
        self.terrain_error_pub.publish(error)
        self.terrain_target_pub.publish(target)
        self.terrain_status_pub.publish(status_msg)
        if status != self.terrain_status:
            if status == TerrainFollower.TRACKING:
                self.get_logger().info('Terrain-following aktif; rangefinder valid.')
            else:
                self.get_logger().warning(
                    f'Terrain-following {status}; menahan posisi.')
            self.terrain_status = status

    @staticmethod
    def copy_pose(source, stamp, frame):
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = frame
        out.pose = source.pose
        return out

    def loop(self):
        if self.pose is None:
            return
        if not self.enabled:
            return
        now = self.get_clock().now()
        if self.commanded_pose is None:
            self.commanded_pose = deepcopy(self.pose)
        # Target kedaluwarsa berarti hold posisi saat ini, bukan meneruskan target lama.
        if self.target is None or self.target_time is None or \
                (now - self.target_time).nanoseconds * 1e-9 > self.timeout:
            self.commanded_pose = deepcopy(self.pose)
            self.pub.publish(self.copy_pose(self.commanded_pose, now.to_msg(), self.output_frame))
            return
        target_z = float(self.target.pose.position.z)
        if self.terrain_enabled:
            target_z, terrain_status = self.terrain_follower.compute_target(
                local_z=float(self.pose.pose.position.z),
                nominal_z=target_z,
                now=time.monotonic(),
            )
            self.publish_terrain_diagnostics(target_z, terrain_status)
            if (terrain_status != TerrainFollower.TRACKING
                    and self.hold_on_range_loss):
                self.commanded_pose = deepcopy(self.pose)
                self.commanded_pose.pose.orientation = deepcopy(
                    self.target.pose.orientation)
                self.pub.publish(self.copy_pose(
                    self.commanded_pose, now.to_msg(), self.output_frame))
                return
        # Slew setpoint internal, bukan membuat target selalu hanya beberapa cm
        # dari posisi aktual. Cara lama membuat position controller FC terus
        # melihat target dekat lalu mengerem pada approach, orbit, dan return.
        dx = self.target.pose.position.x - self.commanded_pose.pose.position.x
        dy = self.target.pose.position.y - self.commanded_pose.pose.position.y
        distance = math.hypot(dx, dy)
        if distance > self.max_xy:
            scale = self.max_xy / distance
            dx *= scale
            dy *= scale
        self.commanded_pose.pose.position.x += dx
        self.commanded_pose.pose.position.y += dy
        dz = target_z - self.commanded_pose.pose.position.z
        self.commanded_pose.pose.position.z += max(-self.max_z, min(self.max_z, dz))
        self.commanded_pose.pose.orientation = deepcopy(self.target.pose.orientation)
        self.pub.publish(self.copy_pose(self.commanded_pose, now.to_msg(), self.output_frame))


def main(args=None):
    rclpy.init(args=args)
    node = PositionSetpointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
