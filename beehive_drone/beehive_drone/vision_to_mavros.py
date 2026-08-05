#!/usr/bin/env python3
"""Forward ZED VSLAM pose to MAVROS External Navigation input.

The coordinate convention conversion from ROS ENU to MAVLink/ArduPilot is
handled by the MAVROS vision_pose_estimate plugin. Camera mounting offsets and
EKF source selection must still be configured on the Pixhawk.
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class VisionToMavros(Node):
    def __init__(self) -> None:
        super().__init__("vision_to_mavros")

        self.declare_parameter("input_pose_topic", "/zed/zed_node/pose")
        self.declare_parameter("output_pose_topic", "/mavros/vision_pose/pose")
        self.declare_parameter("replace_timestamp", False)

        self.input_topic = str(self.get_parameter("input_pose_topic").value)
        self.output_topic = str(self.get_parameter("output_pose_topic").value)
        self.replace_timestamp = bool(self.get_parameter("replace_timestamp").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(PoseStamped, self.output_topic, 10)
        self.sub = self.create_subscription(
            PoseStamped, self.input_topic, self.pose_callback, qos
        )
        self.last_pose_time = None
        self.create_timer(1.0, self.health_check)
        self.get_logger().info(
            f"Jembatan VSLAM aktif: {self.input_topic} -> {self.output_topic}"
        )

    def pose_callback(self, msg: PoseStamped) -> None:
        out = PoseStamped()
        out.header = msg.header
        out.pose = msg.pose
        if self.replace_timestamp:
            out.header.stamp = self.get_clock().now().to_msg()
        self.last_pose_time = self.get_clock().now().nanoseconds * 1e-9
        self.pub.publish(out)

    def health_check(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_pose_time is None:
            self.get_logger().warning("Belum menerima pose VSLAM ZED.")
        elif now - self.last_pose_time > 1.0:
            self.get_logger().error(
                f"Pose VSLAM stale selama {now - self.last_pose_time:.1f} detik."
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionToMavros()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
