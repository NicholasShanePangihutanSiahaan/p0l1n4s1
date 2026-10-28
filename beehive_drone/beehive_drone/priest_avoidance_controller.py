#!/usr/bin/env python3
"""Connect the proven mission goal interface to the optional PRIEST planner."""

import math
import time
from copy import deepcopy

import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Path
from pcl_cstm_msg.msg import (
    AxisAlignedElipsoid,
    AxisAlignedElipsoidArray,
    TrackedCylinderArray,
)
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


ORBIT_PATH_STATES = {'WAIT_ORBIT'}


def build_orbit_reference_path(center, current_pose, radius, point_count,
                               stamp=None, frame_id='map'):
    """Build the legacy CCW circular orbit as a closed reference Path."""
    count = max(16, int(point_count))
    radius = max(0.10, float(radius))
    start_angle = math.atan2(
        float(current_pose.position.y) - float(center.y),
        float(current_pose.position.x) - float(center.x))
    path = Path()
    path.header.frame_id = frame_id
    if stamp is not None:
        path.header.stamp = stamp
    for index in range(count + 1):
        angle = start_angle + 2.0 * math.pi * index / count
        waypoint = PoseStamped()
        waypoint.header.frame_id = frame_id
        if stamp is not None:
            waypoint.header.stamp = stamp
        waypoint.pose.position.x = float(center.x) + radius * math.cos(angle)
        waypoint.pose.position.y = float(center.y) + radius * math.sin(angle)
        waypoint.pose.position.z = float(center.z)
        path.poses.append(waypoint)
    return path


def select_lookahead_pose(path, current_pose, lookahead, progress=0.0):
    """Return (target, progress) using projection and forward arc length.

    Progress belongs to this path only and must reset when its geometry changes.
    Equal-distance projections prefer the earlier segment to avoid skipping ahead
    at a crossing. This geometric selector does not validate obstacle clearance.
    """
    if path is None or not path.poses:
        return None, 0.0

    def xyz(pose):
        return (float(pose.position.x), float(pose.position.y),
                float(pose.position.z))

    points = [xyz(stamped.pose) for stamped in path.poses]
    current = xyz(current_pose)
    if not all(math.isfinite(value) for point in points + [current]
               for value in point):
        return None, 0.0

    # Accumulate actual 3D distance, including vertical path motion.
    lengths = [math.dist(first, second)
               for first, second in zip(points, points[1:])]
    total = sum(lengths)
    progress = min(max(0.0, progress), total)
    best_distance_sq = math.inf
    projected_progress = progress
    segment_start = 0.0

    for index, length in enumerate(lengths):
        segment_end = segment_start + length
        if length > 1.0e-9 and segment_end >= progress:
            first, second = points[index], points[index + 1]
            direction = tuple(b - a for a, b in zip(first, second))
            fraction = sum((c - a) * d for c, a, d
                           in zip(current, first, direction)) / (length * length)
            # Restrict projection to the untraveled portion of this path.
            minimum = max(0.0, (progress - segment_start) / length)
            fraction = min(1.0, max(minimum, fraction))
            projected = tuple(a + fraction * d
                              for a, d in zip(first, direction))
            distance_sq = sum((c - q) ** 2
                              for c, q in zip(current, projected))
            if distance_sq < best_distance_sq - 1.0e-12:
                best_distance_sq = distance_sq
                projected_progress = segment_start + fraction * length
        segment_start = segment_end

    target_distance = min(total, projected_progress + max(0.0, lookahead))
    segment_start = 0.0
    for index, length in enumerate(lengths):
        if length > 1.0e-9 and target_distance <= segment_start + length:
            fraction = (target_distance - segment_start) / length
            target = deepcopy(path.poses[index + 1].pose)
            first, second = points[index], points[index + 1]
            target.position.x, target.position.y, target.position.z = (
                a + fraction * (b - a) for a, b in zip(first, second))
            # Mission yaw is applied by control_loop after selection.
            return target, projected_progress
        segment_start += length

    return deepcopy(path.poses[-1].pose), projected_progress


def cylinder_to_ellipsoid(tracked, minimum_radius, minimum_half_height):
    """Convert one tracked cylinder into the obstacle contract used by PRIEST."""
    cylinder = tracked.cylinder
    if not cylinder.is_valid:
        return None
    radius = max(minimum_radius, float(cylinder.radius))
    half_height = max(minimum_half_height, 0.5 * float(cylinder.height))
    values = (
        cylinder.pose.position.x,
        cylinder.pose.position.y,
        cylinder.pose.position.z,
        radius,
        half_height,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return None
    obstacle = AxisAlignedElipsoid()
    obstacle.center = deepcopy(cylinder.pose.position)
    obstacle.radii.x = radius
    obstacle.radii.y = radius
    obstacle.radii.z = half_height
    return obstacle


class PriestAvoidanceController(Node):
    """Feed goals/obstacles to PRIEST and expose one safe local target.

    Linear goals and the field-tested circular orbit are reference trajectories.
    PRIEST locally optimizes both against obstacles. Without a fresh optimized
    path the controller fails closed and holds the current vehicle position.
    """

    def __init__(self):
        super().__init__('priest_avoidance_controller')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('path_timeout', 5.0)
        self.declare_parameter('waypoint_lookahead', 0.60)
        self.declare_parameter('goal_republish_period', 0.50)
        self.declare_parameter('minimum_obstacle_radius', 0.10)
        self.declare_parameter('minimum_obstacle_half_height', 0.25)
        self.declare_parameter('orbit_radius', 2.7)
        self.declare_parameter('orbit_reference_points', 120)
        self.declare_parameter('reference_republish_period', 0.50)
        self.path_timeout = float(self.get_parameter('path_timeout').value)
        self.lookahead = float(self.get_parameter('waypoint_lookahead').value)
        self.goal_period = float(
            self.get_parameter('goal_republish_period').value)
        self.minimum_radius = float(
            self.get_parameter('minimum_obstacle_radius').value)
        self.minimum_half_height = float(
            self.get_parameter('minimum_obstacle_half_height').value)
        self.orbit_radius = float(self.get_parameter('orbit_radius').value)
        self.orbit_reference_points = int(
            self.get_parameter('orbit_reference_points').value)
        self.reference_period = float(
            self.get_parameter('reference_republish_period').value)

        self.current_pose = None
        self.fsm_goal = None
        self.orbit_goal = None
        self.orbit_center = None
        self.orbit_reference = None
        self.fsm_state = 'INIT'
        self.path = None
        self.path_received_at = None
        self.path_progress = 0.0
        self.path_geometry = None
        self.last_goal_sent_at = 0.0
        self.last_goal_xyz = None
        self.last_reference_sent_at = 0.0
        self.holding = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self.pose_callback, sensor_qos)
        self.create_subscription(
            PoseStamped, '/navigation/local_goal',
            self.fsm_goal_callback, 10)
        self.create_subscription(
            PoseStamped, '/control/dynamic_target',
            self.orbit_goal_callback, 10)
        self.create_subscription(
            Point, '/control/orbit_target',
            self.orbit_center_callback, 10)
        self.create_subscription(
            String, '/mission/fsm_state', self.state_callback, 10)
        self.create_subscription(
            Path, '/navigation/priest/trajectory/out',
            self.path_callback, reliable_qos)
        # self.create_subscription(
        #     TrackedCylinderArray, '/global_cylinders',
        #     self.obstacles_callback, sensor_qos)

        self.goal_publisher = self.create_publisher(
            Pose, '/navigation/priest/linear_goal', reliable_qos)
        self.reference_publisher = self.create_publisher(
            Path, '/navigation/priest/trajectory/in', reliable_qos)
        # self.obstacle_publisher = self.create_publisher(
        #     AxisAlignedElipsoidArray, '/ellipsoids', sensor_qos)
        self.safe_target_publisher = self.create_publisher(
            PoseStamped, '/control/safe_target_pose', 10)
        rate = max(5.0, float(self.get_parameter('publish_rate').value))
        self.create_timer(1.0 / rate, self.control_loop)
        self.get_logger().info(
            'PRIEST local trajectory optimizer aktif untuk linear dan orbit.')

    def pose_callback(self, message):
        self.current_pose = message

    def fsm_goal_callback(self, message):
        self.fsm_goal = message

    def orbit_goal_callback(self, message):
        self.orbit_goal = message

    def orbit_center_callback(self, message):
        self.orbit_center = deepcopy(message)
        self.orbit_reference = None

    def state_callback(self, message):
        if message.data != self.fsm_state:
            self.path = None
            self.path_received_at = None
            self.path_progress = 0.0
            self.path_geometry = None
            if message.data in ORBIT_PATH_STATES:
                self.orbit_reference = None
        self.fsm_state = message.data

    def path_callback(self, message):
        geometry = tuple((p.pose.position.x, p.pose.position.y, p.pose.position.z)
                         for p in message.poses)
        identity = (message.header.frame_id, geometry)
        if not message.poses or identity != self.path_geometry:
            self.path_progress = 0.0
        self.path_geometry = identity if message.poses else None
        self.path = message if message.poses else None
        self.path_received_at = time.monotonic()

    def obstacles_callback(self, message):
        output = AxisAlignedElipsoidArray()
        output.header = message.header
        for tracked in message.cylinders:
            obstacle = cylinder_to_ellipsoid(
                tracked, self.minimum_radius, self.minimum_half_height)
            if obstacle is not None:
                output.elipsoids.append(obstacle)
        self.obstacle_publisher.publish(output)

    def publish_linear_goal(self):
        if self.fsm_goal is None:
            return
        position = self.fsm_goal.pose.position
        xyz = (float(position.x), float(position.y), float(position.z))
        now = time.monotonic()
        changed = self.last_goal_xyz is None or any(
            abs(current - previous) > 1.0e-3
            for current, previous in zip(xyz, self.last_goal_xyz)
        )
        if changed or now - self.last_goal_sent_at >= self.goal_period:
            self.goal_publisher.publish(deepcopy(self.fsm_goal.pose))
            self.last_goal_xyz = xyz
            self.last_goal_sent_at = now

    def publish_target(self, pose):
        output = PoseStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = 'map'
        output.pose = deepcopy(pose)
        self.safe_target_publisher.publish(output)

    def publish_orbit_reference(self):
        if self.current_pose is None or self.orbit_center is None:
            return
        if self.orbit_reference is None:
            self.orbit_reference = build_orbit_reference_path(
                self.orbit_center,
                self.current_pose.pose,
                self.orbit_radius,
                self.orbit_reference_points,
                self.get_clock().now().to_msg(),
            )
        now = time.monotonic()
        if now - self.last_reference_sent_at >= self.reference_period:
            self.reference_publisher.publish(self.orbit_reference)
            self.last_reference_sent_at = now

    def fresh_selected_pose(self):
        path_fresh = (
            self.path is not None
            and self.path_received_at is not None
            and time.monotonic() - self.path_received_at <= self.path_timeout
        )
        if not path_fresh:
            return None
        selected, self.path_progress = select_lookahead_pose(
            self.path, self.current_pose.pose, self.lookahead,
            self.path_progress)
        return selected

    def control_loop(self):
        if self.current_pose is None:
            return
        if self.fsm_state in ORBIT_PATH_STATES:
            self.publish_orbit_reference()
            selected = self.fresh_selected_pose()
            if selected is None:
                selected = deepcopy(self.current_pose.pose)
                if not self.holding:
                    self.get_logger().warning(
                        'Trajectory orbit PRIEST belum fresh; hold posisi.')
                self.holding = True
            else:
                # Pertahankan yaw legacy yang selalu menghadap pusat pohon.
                if self.orbit_goal is not None:
                    selected.orientation = deepcopy(
                        self.orbit_goal.pose.orientation)
                self.holding = False
            self.publish_target(selected)
            return

        if self.fsm_goal is None:
            return
        self.publish_linear_goal()
        selected = self.fresh_selected_pose()
        if selected is None:
            selected = deepcopy(self.current_pose.pose)
            selected.orientation = deepcopy(self.fsm_goal.pose.orientation)
            if not self.holding:
                self.get_logger().warning(
                    'Belum ada trajectory PRIEST yang fresh; hold posisi.')
            self.holding = True
        else:
            selected.orientation = deepcopy(self.fsm_goal.pose.orientation)
            if self.holding:
                self.get_logger().info(
                    'Trajectory PRIEST diterima; navigasi dilanjutkan.')
            self.holding = False
        self.publish_target(selected)


def main(args=None):
    rclpy.init(args=args)
    node = PriestAvoidanceController()
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
