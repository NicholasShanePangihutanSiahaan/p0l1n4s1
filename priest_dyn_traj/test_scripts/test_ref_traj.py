#!/usr/bin/env python3
"""Publish a circular reference trajectory, dummy odometry, and test obstacles.

The FSM state is WAIT_ORBIT, which must be listed in the PRIEST node's
reference_trajectory_states parameter.

The script publishes:
- /mavros/local_position/odom
- /navigation/priest/trajectory/in
- /mission/fsm_state
- /ellipsoids
- /priest/test_obstacles_markers

Obstacle positions are regenerated near the circular trajectory every
--obstacle-period seconds.
"""

import argparse
import math
import random
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from pcl_cstm_msg.msg import AxisAlignedElipsoid, AxisAlignedElipsoidArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Circle radius in metres",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of circular path segments",
    )
    parser.add_argument(
        "--state",
        default="WAIT_ORBIT",
        help="FSM state listed in reference_trajectory_states",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.0,
        help="Trajectory and odometry republish interval in seconds",
    )

    parser.add_argument(
        "--obstacle-period",
        type=float,
        default=2.0,
        help="Time between random obstacle fields in seconds",
    )
    parser.add_argument(
        "--min-obstacles",
        type=int,
        default=5,
        help="Minimum generated obstacle count",
    )
    parser.add_argument(
        "--max-obstacles",
        type=int,
        default=10,
        help="Maximum generated obstacle count",
    )
    parser.add_argument(
        "--obstacle-radial-spread",
        type=float,
        default=0.75,
        help="Maximum radial offset from the circular trajectory in metres",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed for repeatable obstacle positions",
    )

    return parser.parse_known_args()


class CircularReferenceTest(Node):
    def __init__(self, cfg):
        super().__init__("priest_circular_reference_test")

        reliable = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.cfg = cfg
        self.random = random.Random(cfg.seed)
        self.last_report = 0.0
        self.obstacles = AxisAlignedElipsoidArray()

        self.path_pub = self.create_publisher(
            Path,
            "/navigation/priest/trajectory/in",
            reliable,
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            "/mavros/local_position/odom",
            qos_profile_sensor_data,
        )
        self.state_pub = self.create_publisher(
            String,
            "/mission/fsm_state",
            reliable,
        )
        self.obstacles_pub = self.create_publisher(
            AxisAlignedElipsoidArray,
            "/ellipsoids",
            qos_profile_sensor_data,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/priest/test_obstacles_markers",
            reliable,
        )

        self.create_subscription(
            Path,
            "/navigation/priest/trajectory/out",
            self.on_result,
            reliable,
        )

        self.create_timer(cfg.period, self.publish_test)
        self.create_timer(cfg.obstacle_period, self.publish_obstacles)

        self.publish_obstacles()
        self.publish_test()

    def publish_test(self):
        """Publish fixed odometry, circular reference path, and WAIT_ORBIT."""

        stamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "map"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.0

        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"

        # The circle starts at (0, 0, 0), because its center is (-radius, 0).
        circle_center_x = -self.cfg.radius
        circle_center_y = 0.0

        for i in range(self.cfg.samples + 1):
            theta = 2.0 * math.pi * i / self.cfg.samples

            pose = PoseStamped()
            pose.header = path.header

            pose.pose.position.x = (
                circle_center_x + self.cfg.radius * math.cos(theta)
            )
            pose.pose.position.y = (
                circle_center_y + self.cfg.radius * math.sin(theta)
            )
            pose.pose.position.z = 0.0

            yaw = theta + math.pi / 2.0
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            path.poses.append(pose)

        state = String()
        state.data = self.cfg.state

        self.odom_pub.publish(odom)
        self.path_pub.publish(path)
        self.state_pub.publish(state)

        # Republishing lets a newly started PRIEST node receive the current field.
        self.obstacles_pub.publish(self.obstacles)

    def publish_obstacles(self):
        """Generate random ellipsoids near the circular reference trajectory."""

        stamp = self.get_clock().now().to_msg()

        obstacle_count = self.random.randint(
            self.cfg.min_obstacles,
            self.cfg.max_obstacles,
        )

        obstacles = AxisAlignedElipsoidArray()
        obstacles.header.stamp = stamp
        obstacles.header.frame_id = "map"

        markers = MarkerArray()

        # Clear all markers from the previous obstacle field first.
        clear_markers = Marker()
        clear_markers.action = Marker.DELETEALL
        markers.markers.append(clear_markers)

        # This is the same center used by publish_test() for the path.
        circle_center_x = -self.cfg.radius
        circle_center_y = 0.0

        for index in range(obstacle_count):
            angle = self.random.uniform(0.0, 2.0 * math.pi)

            # Obstacles are distributed in a ring around the reference circle.
            radial_offset = self.random.uniform(
                -self.cfg.obstacle_radial_spread,
                self.cfg.obstacle_radial_spread,
            )
            distance_from_circle_center = max(
                0.05,
                self.cfg.radius + radial_offset,
            )

            center_x = (
                circle_center_x
                + distance_from_circle_center * math.cos(angle)
            )
            center_y = (
                circle_center_y
                + distance_from_circle_center * math.sin(angle)
            )
            center_z = self.random.uniform(0.2, 1.0)

            radius_x = self.random.uniform(0.15, 0.45)
            radius_y = self.random.uniform(0.15, 0.45)
            radius_z = self.random.uniform(0.20, 0.60)

            obstacle = AxisAlignedElipsoid()
            obstacle.center.x = center_x
            obstacle.center.y = center_y
            obstacle.center.z = center_z
            obstacle.radii.x = radius_x
            obstacle.radii.y = radius_y
            obstacle.radii.z = radius_z

            obstacles.elipsoids.append(obstacle)

            # RViz sphere: scale is diameter, not radius.
            marker = Marker()
            marker.header = obstacles.header
            marker.ns = "priest_test_obstacles"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = center_z
            marker.pose.orientation.w = 1.0

            marker.scale.x = 2.0 * radius_x
            marker.scale.y = 2.0 * radius_y
            marker.scale.z = 2.0 * radius_z

            marker.color.r = 1.0
            marker.color.g = 0.15
            marker.color.b = 0.05
            marker.color.a = 0.75

            markers.markers.append(marker)

        self.obstacles = obstacles
        
        origin_marker = Marker()
        origin_marker.header = obstacles.header
        origin_marker.ns = 'priest_test_origin'
        origin_marker.id = 0
        origin_marker.type = Marker.SPHERE
        origin_marker.action = Marker.ADD
        
        origin_marker.pose.position.x = 0.0
        origin_marker.pose.position.y = 0.0
        origin_marker.pose.position.z = 0.0
        origin_marker.pose.orientation.w = 0.2
        
        origin_marker.scale.x = 0.30
        origin_marker.scale.y = 0.30
        origin_marker.scale.z = 0.30
        
        origin_marker.color.r = 0.10
        origin_marker.color.g = 0.10
        origin_marker.color.b = 0.80
        origin_marker.color.a = 0.20
        
        markers.markers.append(origin_marker)
        
        self.obstacles_pub.publish(obstacles)
        self.marker_pub.publish(markers)

        self.get_logger().info(
            f"Published {obstacle_count} obstacles near the circular path"
        )

    def on_result(self, msg):
        now = time.monotonic()
        if now - self.last_report < 2.0:
            return

        self.last_report = now

        if msg.poses:
            end = msg.poses[-1].pose.position
            self.get_logger().info(
                f"PRIEST output: {len(msg.poses)} poses; endpoint "
                f"({end.x:.2f}, {end.y:.2f}, {end.z:.2f})"
            )
        else:
            self.get_logger().info(
                "PRIEST output is empty (waiting, stopped, or invalidated)"
            )


def main():
    cfg, ros_args = arguments()

    if cfg.radius <= 0 or not math.isfinite(cfg.radius) or cfg.samples < 12:
        raise SystemExit("Use --radius > 0 and --samples >= 12")

    if (
        cfg.period <= 0
        or cfg.obstacle_period <= 0
        or cfg.obstacle_radial_spread < 0
        or cfg.min_obstacles < 0
        or cfg.max_obstacles < cfg.min_obstacles
        or not cfg.state.strip()
    ):
        raise SystemExit(
            "Use positive periods, valid obstacle limits, and a nonempty state"
        )

    rclpy.init(args=ros_args)
    node = CircularReferenceTest(cfg)

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