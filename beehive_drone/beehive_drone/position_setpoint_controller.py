#!/usr/bin/env python3
"""Safe, continuous position-setpoint adapter for MAVROS/ArduPilot."""

import math
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
        # Target kedaluwarsa berarti hold posisi saat ini, bukan meneruskan target lama.
        if self.target is None or self.target_time is None or \
                (now - self.target_time).nanoseconds * 1e-9 > self.timeout:
            self.pub.publish(self.copy_pose(self.pose, now.to_msg(), self.output_frame))
            return
        out = self.copy_pose(self.target, now.to_msg(), self.output_frame)
        dx = self.target.pose.position.x - self.pose.pose.position.x
        dy = self.target.pose.position.y - self.pose.pose.position.y
        distance = math.hypot(dx, dy)
        if distance > self.max_xy:
            scale = self.max_xy / distance
            out.pose.position.x = self.pose.pose.position.x + dx * scale
            out.pose.position.y = self.pose.pose.position.y + dy * scale
        dz = self.target.pose.position.z - self.pose.pose.position.z
        out.pose.position.z = self.pose.pose.position.z + max(-self.max_z, min(self.max_z, dz))
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PositionSetpointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
