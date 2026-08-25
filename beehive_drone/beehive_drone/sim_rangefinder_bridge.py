#!/usr/bin/env python3
"""Convert the single-beam Gazebo LaserScan into a ROS Range message."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Range


class SimulationRangefinderBridge(Node):
    """Small simulation-only adapter; it never sends data to the real FC."""

    def __init__(self):
        super().__init__('sim_rangefinder_bridge')
        self.declare_parameter('input_topic', '/range')
        self.declare_parameter('output_topic', '/simulation/rangefinder')
        self.declare_parameter('frame_id', 'range_link')
        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.publisher = self.create_publisher(
            Range, output_topic, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, input_topic, self.callback, qos_profile_sensor_data)
        self.get_logger().info(
            f'Rangefinder simulasi: {input_topic} -> {output_topic}')

    def callback(self, scan):
        valid = [
            float(value) for value in scan.ranges
            if math.isfinite(value)
            and float(scan.range_min) <= value <= float(scan.range_max)
        ]
        measurement = min(valid) if valid else float('inf')
        msg = Range()
        msg.header = scan.header
        msg.header.frame_id = self.frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = max(
            abs(float(scan.angle_min)), abs(float(scan.angle_max)), 0.001)
        msg.min_range = float(scan.range_min)
        msg.max_range = float(scan.range_max)
        msg.range = measurement
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimulationRangefinderBridge()
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


if __name__ == '__main__':
    main()
