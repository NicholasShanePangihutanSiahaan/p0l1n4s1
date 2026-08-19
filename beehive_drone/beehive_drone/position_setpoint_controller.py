#!/usr/bin/env python3
"""Safe, continuous position-setpoint adapter for MAVROS/ArduPilot."""

import math
from copy import deepcopy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class PositionSetpointController(Node):
    def __init__(self):
        super().__init__('position_setpoint_controller')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('max_horizontal_step', 0.10)
        self.declare_parameter('max_vertical_step', 0.05)
        self.declare_parameter('target_timeout', 0.75)
        self.declare_parameter('output_frame', 'map')
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
        self.pub = self.create_publisher(PoseStamped, '/mavros/setpoint_position/local', 10)
        rate = max(10.0, float(self.get_parameter('publish_rate').value))
        self.create_timer(1.0 / rate, self.loop)
        self.get_logger().info('Position setpoint controller aktif (continuous + slew limit).')

    def pose_cb(self, msg):
        self.pose = msg

    def target_cb(self, msg):
        self.target = msg
        self.target_time = self.get_clock().now()

    def enabled_cb(self, msg):
        if msg.data and not self.enabled and self.pose is not None:
            self.commanded_pose = deepcopy(self.pose)
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
        dz = self.target.pose.position.z - self.commanded_pose.pose.position.z
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
    node.destroy_node()
    rclpy.shutdown()
