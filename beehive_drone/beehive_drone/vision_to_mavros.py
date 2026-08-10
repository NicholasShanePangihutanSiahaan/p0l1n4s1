#!/usr/bin/env python3
"""Forward ZED pose to the MAVROS vision-pose input for ArduPilot ExternalNav."""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class VisionToMavros(Node):
    def __init__(self):
        super().__init__('vision_to_mavros')
        self.declare_parameter('input_topic', '/zed/zed_node/pose')
        self.declare_parameter('output_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('replace_zero_timestamp', True)

        input_topic = str(self.get_parameter('input_topic').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.replace_zero_timestamp = bool(
            self.get_parameter('replace_zero_timestamp').value)
        self.publisher = self.create_publisher(PoseStamped, output_topic, 10)
        self.subscription = self.create_subscription(
            PoseStamped, input_topic, self.pose_callback, qos_profile_sensor_data)
        self.received = 0
        self.last_received = None
        self.create_timer(2.0, self.report)
        self.get_logger().info(
            f'Bridge VisualOdom aktif: {input_topic} -> {output_topic}')

    def pose_callback(self, msg):
        # MAVROS vision_pose menerima koordinat ROS ENU dan melakukan konversi
        # MAVLink/NED di plugin. Pertahankan timestamp pengukuran ZED bila valid.
        if self.replace_zero_timestamp and msg.header.stamp.sec == 0 and \
                msg.header.stamp.nanosec == 0:
            msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)
        self.received += 1
        self.last_received = self.get_clock().now()

    def report(self):
        if self.last_received is None:
            self.get_logger().warning('Belum menerima pose ZED.')
            return
        age = (self.get_clock().now() - self.last_received).nanoseconds * 1e-9
        if age > 1.0:
            self.get_logger().warning(f'Pose ZED stale: {age:.2f} s.')
        else:
            self.get_logger().info(f'VisualOdom diteruskan; total={self.received}, age={age:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = VisionToMavros()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
