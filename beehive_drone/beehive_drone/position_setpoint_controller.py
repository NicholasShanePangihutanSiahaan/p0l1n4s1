#!/usr/bin/env python3
"""Safe, continuous position-setpoint adapter for MAVROS/ArduPilot."""

import math
import time
from copy import deepcopy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Range
from beehive_drone.terrain_following import TerrainFollower



class PositionSetpointController(Node):
    def __init__(self):
        super().__init__('position_setpoint_controller')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('max_horizontal_step', 0.10)
        self.declare_parameter('max_vertical_step', 0.05)
        self.declare_parameter('target_timeout', 0.75)
        self.declare_parameter('output_frame', 'map')
        self.declare_parameter('altitude_mode', 'home_relative')
        self.altitude_mode = str(self.get_parameter('altitude_mode').value)
        if self.altitude_mode not in ('home_relative', 'terrain_following'):
            raise ValueError('altitude_mode harus home_relative atau terrain_following')
        defaults = dict(rangefinder_topic='/mavros/rangefinder/rangefinder',
                        rangefinder_timeout=0.5, rangefinder_min_valid=0.08,
                        rangefinder_max_valid=8.0, rangefinder_altitude_offset=0.0,
                        range_filter_alpha=0.35, range_median_window=5,
                        minimum_tilt_cosine=0.7, max_terrain_correction=1.0,
                        max_terrain_target_rate=0.5)
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        value = lambda name: self.get_parameter(name).value
        self.terrain = TerrainFollower(
            timeout=value('rangefinder_timeout'), filter_alpha=value('range_filter_alpha'),
            max_target_rate=value('max_terrain_target_rate'),
            max_correction=value('max_terrain_correction'),
            min_tilt_cosine=value('minimum_tilt_cosine'),
            sensor_offset=value('rangefinder_altitude_offset'),
            median_window=value('range_median_window'))
        self.agl_target = None
        self.agl_target_time = None
        self.max_xy = float(self.get_parameter('max_horizontal_step').value)
        self.max_z = float(self.get_parameter('max_vertical_step').value)
        self.timeout = float(self.get_parameter('target_timeout').value)
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.pose = None
        self.target = None
        self.target_time = None
        self.enabled = False
        self.commanded_pose = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pose_cb, qos)
        self.create_subscription(PoseStamped, '/control/safe_target_pose', self.target_cb, 10)
        self.create_subscription(Bool, '/control/setpoint_enabled', self.enabled_cb, 10)
        if self.altitude_mode == 'terrain_following':
            self.create_subscription(Range, value('rangefinder_topic'), self.range_cb, qos)
            self.create_subscription(Float32, '/control/target_agl', self.agl_cb, 10)
        self.agl_pub = self.create_publisher(Float32, '/control/measured_agl', 10)
        self.pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        rate = max(10.0, float(self.get_parameter('publish_rate').value))
        self.create_timer(1.0 / rate, self.loop)
        self.get_logger().info(
            f'Position controller aktif: mode={self.altitude_mode}, target dari FSM.')

    def pose_cb(self, msg):
        self.pose = msg

    def target_cb(self, msg):
        self.target = msg
        self.target_time = self.get_clock().now()

    def agl_cb(self, msg):
        if math.isfinite(msg.data) and msg.data > 0.0:
            self.agl_target = float(msg.data)
            self.agl_target_time = time.monotonic()

    def range_cb(self, msg):
        if self.pose is None:
            return
        low = max(float(self.get_parameter('rangefinder_min_valid').value), msg.min_range)
        high = float(self.get_parameter('rangefinder_max_valid').value)
        if msg.max_range > 0.0:
            high = min(high, msg.max_range)
        if math.isfinite(msg.range) and low <= msg.range <= high:
            if self.terrain.ingest(msg.range, self.pose.pose.orientation, time.monotonic()):
                measured = Float32()
                measured.data = float(self.terrain.filtered_agl)
                self.agl_pub.publish(measured)

    def enabled_cb(self, msg):
        if msg.data and not self.enabled and self.pose is not None:
            self.commanded_pose = deepcopy(self.pose)
            self.terrain.reset_target()
        elif not msg.data:
            self.commanded_pose = None
        self.enabled = msg.data

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
        if self.altitude_mode == 'terrain_following':
            tick = time.monotonic()
            if (self.agl_target is None or self.agl_target_time is None
                    or tick - self.agl_target_time > self.timeout
                    or not self.terrain.range_is_fresh(tick)):
                # Freeze the commanded position on range/AGL-target loss.
                self.pub.publish(self.copy_pose(self.commanded_pose, now.to_msg(), self.output_frame))
                return
            target_z, _ = self.terrain.compute_target(
                self.pose.pose.position.z, target_z, tick, self.agl_target)
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
