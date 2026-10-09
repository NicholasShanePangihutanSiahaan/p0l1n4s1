#!/usr/bin/env python3
"""Publish synthetic inputs for testing PRIEST linear-goal mode.

Published topics:
    /mavros/local_position/odom
    /navigation/priest/linear_goal
    /mission/fsm_state
    /ellipsoids
    /priest/test_obstacles_markers

Subscribed topic:
    /navigation/priest/trajectory/out

The synthetic robot remains at (0, 0, 0), with zero velocity.
The default goal is (5, 0, 0).

Random obstacles are regenerated near the reference line every two seconds.
"""

import argparse
import math
import random
import time

import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry, Path
from pcl_cstm_msg.msg import (
    AxisAlignedElipsoid,
    AxisAlignedElipsoidArray,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--goal-x', type=float, default=5.0)
    parser.add_argument('--goal-y', type=float, default=0.0)
    parser.add_argument('--goal-z', type=float, default=0.0)

    parser.add_argument(
        '--state',
        default='LINEAR_GOAL',
        help='FSM state outside reference_trajectory_states',
    )
    parser.add_argument(
        '--period',
        type=float,
        default=0.1,
        help='Publishing period in seconds',
    )
    parser.add_argument(
        '--obstacle-period',
        type=float,
        default=2.0,
        help='Time between random obstacle updates',
    )
    parser.add_argument(
        '--min-obstacles',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--max-obstacles',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--obstacle-spread',
        type=float,
        default=0.60,
        help='Maximum perpendicular distance from the reference line',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional fixed random seed',
    )

    return parser.parse_known_args()


class LinearReferenceTest(Node):

    def __init__(self, config):
        super().__init__('priest_linear_reference_test')

        self.config = config
        self.last_report_time = 0.0
        self.random = random.Random(config.seed)

        # Stores the current obstacle field so it can be republished with the
        # high-frequency odometry and goal messages.
        self.obstacles = AxisAlignedElipsoidArray()

        reliable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ----------------------------------------------------------
        # Test input publishers
        # ----------------------------------------------------------

        self.odom_publisher = self.create_publisher(
            Odometry,
            '/mavros/local_position/odom',
            sensor_qos,
        )

        self.goal_publisher = self.create_publisher(
            Pose,
            '/navigation/priest/linear_goal',
            reliable_qos,
        )

        self.state_publisher = self.create_publisher(
            String,
            '/mission/fsm_state',
            reliable_qos,
        )

        self.obstacle_publisher = self.create_publisher(
            AxisAlignedElipsoidArray,
            '/ellipsoids',
            sensor_qos,
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/priest/test_obstacles_markers',
            reliable_qos,
        )

        # ----------------------------------------------------------
        # Planner result subscriber
        # ----------------------------------------------------------

        self.create_subscription(
            Path,
            '/navigation/priest/trajectory/out',
            self.on_planner_output,
            reliable_qos,
        )

        self.create_timer(
            self.config.period,
            self.publish_test_inputs,
        )

        self.create_timer(
            self.config.obstacle_period,
            self.publish_obstacles,
        )

        # Generate obstacles before publishing the first complete input set.
        self.publish_obstacles()
        self.publish_test_inputs()

        self.get_logger().info(
            'Publishing synthetic odometry at (0, 0, 0) and '
            f'linear goal at '
            f'({self.config.goal_x:.2f}, '
            f'{self.config.goal_y:.2f}, '
            f'{self.config.goal_z:.2f})'
        )

    def publish_test_inputs(self):
        """Publish odometry, linear goal, FSM state, and current obstacles."""

        stamp = self.get_clock().now().to_msg()

        # ----------------------------------------------------------
        # Synthetic odometry
        #
        # Position:    (0, 0, 0)
        # Velocity:    (0, 0, 0)
        # Orientation: identity quaternion
        # ----------------------------------------------------------

        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = 'map'
        odometry.child_frame_id = 'base_link'

        odometry.pose.pose.position.x = 0.0
        odometry.pose.pose.position.y = 0.0
        odometry.pose.pose.position.z = 0.0

        odometry.pose.pose.orientation.x = 0.0
        odometry.pose.pose.orientation.y = 0.0
        odometry.pose.pose.orientation.z = 0.0
        odometry.pose.pose.orientation.w = 1.0

        odometry.twist.twist.linear.x = 0.0
        odometry.twist.twist.linear.y = 0.0
        odometry.twist.twist.linear.z = 0.0

        odometry.twist.twist.angular.x = 0.0
        odometry.twist.twist.angular.y = 0.0
        odometry.twist.twist.angular.z = 0.0

        # ----------------------------------------------------------
        # Linear goal and FSM state
        # ----------------------------------------------------------

        goal = Pose()
        goal.position.x = self.config.goal_x
        goal.position.y = self.config.goal_y
        goal.position.z = self.config.goal_z
        goal.orientation.w = 1.0

        fsm_state = String()
        fsm_state.data = self.config.state

        # Keep the obstacle message fresh without generating new positions.
        self.obstacles.header.stamp = stamp
        self.obstacles.header.frame_id = 'map'

        self.odom_publisher.publish(odometry)
        self.goal_publisher.publish(goal)
        self.state_publisher.publish(fsm_state)
        self.obstacle_publisher.publish(self.obstacles)

    def publish_obstacles(self):
        """Generate random ellipsoids near the linear reference trajectory."""

        stamp = self.get_clock().now().to_msg()

        obstacle_count = self.random.randint(
            self.config.min_obstacles,
            self.config.max_obstacles,
        )

        goal = (
            self.config.goal_x,
            self.config.goal_y,
            self.config.goal_z,
        )

        path_length = math.sqrt(
            sum(coordinate * coordinate for coordinate in goal)
        )

        if path_length > 1e-9:
            direction = tuple(
                coordinate / path_length
                for coordinate in goal
            )
        else:
            direction = (1.0, 0.0, 0.0)

        # Construct two unit vectors perpendicular to the reference line.
        if abs(direction[2]) < 0.9:
            helper = (0.0, 0.0, 1.0)
        else:
            helper = (0.0, 1.0, 0.0)

        normal_a = (
            direction[1] * helper[2] - direction[2] * helper[1],
            direction[2] * helper[0] - direction[0] * helper[2],
            direction[0] * helper[1] - direction[1] * helper[0],
        )

        normal_length = math.sqrt(
            sum(component * component for component in normal_a)
        )

        normal_a = tuple(
            component / normal_length
            for component in normal_a
        )

        normal_b = (
            direction[1] * normal_a[2] - direction[2] * normal_a[1],
            direction[2] * normal_a[0] - direction[0] * normal_a[2],
            direction[0] * normal_a[1] - direction[1] * normal_a[0],
        )

        obstacles = AxisAlignedElipsoidArray()
        obstacles.header.stamp = stamp
        obstacles.header.frame_id = 'map'

        markers = MarkerArray()

        # Remove markers belonging to the previous randomized field.
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for index in range(obstacle_count):
            # Pick a position along the line without placing an obstacle
            # directly at the robot or exactly at the goal.
            along = self.random.uniform(0.15, 0.95)

            offset_a = self.random.uniform(
                -self.config.obstacle_spread,
                self.config.obstacle_spread,
            )
            offset_b = self.random.uniform(
                -self.config.obstacle_spread,
                self.config.obstacle_spread,
            )

            center = tuple(
                along * goal[axis]
                + offset_a * normal_a[axis]
                + offset_b * normal_b[axis]
                for axis in range(3)
            )

            radii = (
                self.random.uniform(0.15, 0.45),
                self.random.uniform(0.15, 0.45),
                self.random.uniform(0.20, 0.60),
            )

            # ------------------------------------------------------
            # PRIEST ellipsoid
            # ------------------------------------------------------

            obstacle = AxisAlignedElipsoid()

            obstacle.center.x = center[0]
            obstacle.center.y = center[1]
            obstacle.center.z = center[2]

            obstacle.radii.x = radii[0]
            obstacle.radii.y = radii[1]
            obstacle.radii.z = radii[2]

            obstacles.elipsoids.append(obstacle)

            # ------------------------------------------------------
            # Matching RViz marker
            # ------------------------------------------------------

            marker = Marker()
            marker.header = obstacles.header
            marker.ns = 'priest_test_obstacles'
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.w = 1.0

            # Marker scale uses diameter; the obstacle message uses radii.
            marker.scale.x = 2.0 * radii[0]
            marker.scale.y = 2.0 * radii[1]
            marker.scale.z = 2.0 * radii[2]

            marker.color.r = 1.0
            marker.color.g = 0.15
            marker.color.b = 0.05
            marker.color.a = 0.75

            markers.markers.append(marker)

        self.obstacles = obstacles
        
        goal_marker = Marker()
        goal_marker.header = obstacles.header
        goal_marker.ns = 'priest_test_goal'
        goal_marker.id = 0
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        
        goal_marker.pose.position.x = self.config.goal_x
        goal_marker.pose.position.y = self.config.goal_y
        goal_marker.pose.position.z = self.config.goal_z
        goal_marker.pose.orientation.w = 1.0
        
        goal_marker.scale.x = 0.30
        goal_marker.scale.y = 0.30
        goal_marker.scale.z = 0.30
        
        goal_marker.color.r = 0.10
        goal_marker.color.g = 1.00
        goal_marker.color.b = 0.10
        goal_marker.color.a = 0.2
        
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
        
        markers.markers.append(goal_marker)
        markers.markers.append(origin_marker)

        self.obstacle_publisher.publish(obstacles)
        self.marker_publisher.publish(markers)

        self.get_logger().info(
            f'Published {obstacle_count} random obstacles '
            'near the linear trajectory'
        )

    def on_planner_output(self, message):
        """Report the latest PRIEST output without flooding the terminal."""

        now = time.monotonic()

        if now - self.last_report_time < 2.0:
            return

        self.last_report_time = now

        if not message.poses:
            self.get_logger().info(
                'Received an empty PRIEST trajectory'
            )
            return

        endpoint = message.poses[-1].pose.position

        self.get_logger().info(
            f'Received {len(message.poses)} poses; '
            f'local endpoint = '
            f'({endpoint.x:.2f}, '
            f'{endpoint.y:.2f}, '
            f'{endpoint.z:.2f})'
        )


def main():
    config, ros_arguments = parse_arguments()

    if config.period <= 0.0 or config.obstacle_period <= 0.0:
        raise SystemExit(
            '--period and --obstacle-period must be greater than zero'
        )

    if (
        config.min_obstacles < 0
        or config.max_obstacles < config.min_obstacles
        or config.obstacle_spread < 0.0
    ):
        raise SystemExit(
            'Use valid obstacle limits and a nonnegative obstacle spread'
        )

    if not config.state.strip():
        raise SystemExit('--state must not be empty')

    goal_coordinates = (
        config.goal_x,
        config.goal_y,
        config.goal_z,
    )

    if not all(math.isfinite(value) for value in goal_coordinates):
        raise SystemExit('Goal coordinates must be finite')

    rclpy.init(args=ros_arguments)

    node = LinearReferenceTest(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()