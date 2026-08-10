#!/usr/bin/env python3
"""Feed Gazebo odometry to ArduPilot EKF ExternalNav input (simulation only)."""

import copy
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class SimExternalOdometry(Node):
    def __init__(self):
        super().__init__('sim_external_odometry')
        self.declare_parameter('input_topic', '/simulation/ground_truth/odom')
        self.declare_parameter('output_topic', '/mavros/odometry/in')
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.pub = self.create_publisher(Odometry, output_topic, qos_profile_sensor_data)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/simulation/local_position/pose', qos_profile_sensor_data)
        self.create_subscription(Odometry, input_topic, self.callback, qos_profile_sensor_data)
        self.count = 0
        self.get_logger().info(f'ExternalNav simulation: {input_topic} -> {output_topic}')

    def callback(self, source):
        msg = copy.deepcopy(source)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        # Gazebo bridge sering memberi covariance nol. Berikan uncertainty realistis
        # agar EKF menerima measurement tanpa menganggapnya sempurna.
        msg.pose.covariance = [0.0] * 36
        msg.twist.covariance = [0.0] * 36
        for index, variance in ((0, 0.01), (7, 0.01), (14, 0.02),
                                (21, 0.02), (28, 0.02), (35, 0.03)):
            msg.pose.covariance[index] = variance
            msg.twist.covariance[index] = variance
        self.pub.publish(msg)
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.pose_pub.publish(pose)
        self.count += 1
        if self.count % 100 == 0:
            self.get_logger().info(
                f'ExternalNav aktif: x={msg.pose.pose.position.x:.2f}, '
                f'y={msg.pose.pose.position.y:.2f}, z={msg.pose.pose.position.z:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = SimExternalOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
