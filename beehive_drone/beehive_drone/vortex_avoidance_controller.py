#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32, String
from uav_interfaces.msg import TreeArray

from beehive_drone.mission_params import MissionConfig


class VortexAvoidanceController(Node):
    """Shapes mission targets while enforcing a keep-out radius around trunks."""

    def __init__(self) -> None:
        super().__init__("vortex_avoidance_controller")

        defaults = {
            "world_frame": MissionConfig.WORLD_FRAME,
            "influence_radius": MissionConfig.OBSTACLE_INFLUENCE_RADIUS,
            "hard_radius": MissionConfig.OBSTACLE_HARD_RADIUS,
            "active_tree_keepout_radius": MissionConfig.ACTIVE_TARGET_KEEP_OUT_RADIUS,
            "emergency_stop_radius": MissionConfig.EMERGENCY_STOP_RADIUS,
            "repulsive_gain": MissionConfig.REPULSIVE_GAIN,
            "vortex_gain": MissionConfig.VORTEX_GAIN,
            "attraction_gain": MissionConfig.ATTRACTION_GAIN,
            "max_target_shift": MissionConfig.MAX_TARGET_SHIFT,
            "goal_timeout_sec": MissionConfig.TARGET_TIMEOUT_SEC,
            "require_map_ready": MissionConfig.REQUIRE_TREE_MAP,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.influence_radius = float(self.get_parameter("influence_radius").value)
        self.hard_radius = float(self.get_parameter("hard_radius").value)
        self.active_tree_keepout_radius = float(
            self.get_parameter("active_tree_keepout_radius").value
        )
        self.emergency_stop_radius = float(
            self.get_parameter("emergency_stop_radius").value
        )
        self.repulsive_gain = float(self.get_parameter("repulsive_gain").value)
        self.vortex_gain = float(self.get_parameter("vortex_gain").value)
        self.attraction_gain = float(self.get_parameter("attraction_gain").value)
        self.max_target_shift = float(self.get_parameter("max_target_shift").value)
        self.goal_timeout_sec = float(self.get_parameter("goal_timeout_sec").value)
        self.require_map_ready = bool(self.get_parameter("require_map_ready").value)

        if self.influence_radius <= max(
            self.hard_radius, self.active_tree_keepout_radius
        ):
            raise ValueError("influence_radius harus lebih besar dari seluruh keep-out radius")

        self.current_pose: Optional[PoseStamped] = None
        self.fsm_goal: Optional[PoseStamped] = None
        self.orbit_goal: Optional[PoseStamped] = None
        self.fsm_goal_time: Optional[float] = None
        self.orbit_goal_time: Optional[float] = None
        self.fsm_state = "INIT"
        self.active_tree_id = -1
        self.trees = []
        self.map_ready = False
        self.last_warning_time = -1e9
        self.last_frame_warning = -1e9

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_callback, qos_sensor
        )
        self.create_subscription(String, "/mission/fsm_state", self.state_callback, 10)
        self.create_subscription(TreeArray, "/map/trees", self.tree_callback, qos_map)
        self.create_subscription(Bool, "/map/trees_ready", self.map_ready_callback, qos_map)
        self.create_subscription(
            PoseStamped, "/navigation/local_goal", self.fsm_goal_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/control/dynamic_target", self.orbit_goal_callback, 10
        )
        self.create_subscription(
            Int32, "/control/active_tree_id", self.active_tree_callback, 10
        )

        self.safe_target_pub = self.create_publisher(
            PoseStamped, "/control/safe_target_pose", 10
        )
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            "Vortex safety aktif; pohon target bebas diorbit di luar keep-out radius."
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg

    def state_callback(self, msg: String) -> None:
        self.fsm_state = msg.data

    def tree_callback(self, msg: TreeArray) -> None:
        self.trees = list(msg.trees)

    def map_ready_callback(self, msg: Bool) -> None:
        self.map_ready = bool(msg.data)

    def fsm_goal_callback(self, msg: PoseStamped) -> None:
        self.fsm_goal = msg
        self.fsm_goal_time = self.now_sec()

    def orbit_goal_callback(self, msg: PoseStamped) -> None:
        self.orbit_goal = msg
        self.orbit_goal_time = self.now_sec()

    def active_tree_callback(self, msg: Int32) -> None:
        self.active_tree_id = int(msg.data)

    def active_goal(self) -> Optional[PoseStamped]:
        now = self.now_sec()
        if self.fsm_state in {"PREPARE_ORBIT", "WAIT_ORBIT"}:
            if (
                self.orbit_goal is not None
                and self.orbit_goal_time is not None
                and now - self.orbit_goal_time <= self.goal_timeout_sec
            ):
                return self.orbit_goal
            return None
        if (
            self.fsm_goal is not None
            and self.fsm_goal_time is not None
            and now - self.fsm_goal_time <= self.goal_timeout_sec
        ):
            return self.fsm_goal
        return None

    def publish_hold(self) -> None:
        if self.current_pose is None:
            return
        safe = PoseStamped()
        safe.header.frame_id = self.world_frame
        safe.header.stamp = self.get_clock().now().to_msg()
        safe.pose = self.current_pose.pose
        self.safe_target_pub.publish(safe)

    def control_loop(self) -> None:
        if self.current_pose is None:
            return
        goal = self.active_goal()
        if goal is None:
            return

        if goal.header.frame_id and goal.header.frame_id != self.world_frame:
            now = self.now_sec()
            if now - self.last_frame_warning > 2.0:
                self.last_frame_warning = now
                self.get_logger().error(
                    f"Goal frame={goal.header.frame_id}, expected={self.world_frame}; HOLD."
                )
            self.publish_hold()
            return

        pass_through_states = {
            "SET_MODE",
            "ARM",
            "TAKEOFF",
            "HOLD",
            "SEARCH_TREE",
            "POST_ORBIT_HOVER",
            "HOVER_AT_PRE_ORBIT",
            "HOME_HOVER",
            "LAND",
            "WAIT_LANDED",
            "ABORTED",
            "DONE",
        }
        if self.fsm_state in pass_through_states:
            safe = PoseStamped()
            safe.header.frame_id = self.world_frame
            safe.header.stamp = self.get_clock().now().to_msg()
            safe.pose = goal.pose
            self.safe_target_pub.publish(safe)
            return

        map_required_states = {
            "APPROACH_TREE",
            "HOVER_BEFORE_ORBIT",
            "PREPARE_ORBIT",
            "WAIT_ORBIT",
        }
        if (
            self.require_map_ready
            and self.fsm_state in map_required_states
            and (not self.map_ready or not self.trees)
        ):
            self.publish_hold()
            return

        cx = float(self.current_pose.pose.position.x)
        cy = float(self.current_pose.pose.position.y)
        gx = float(goal.pose.position.x)
        gy = float(goal.pose.position.y)

        goal_dx = gx - cx
        goal_dy = gy - cy
        goal_distance = math.hypot(goal_dx, goal_dy)
        if goal_distance > 1e-6:
            goal_ux = goal_dx / goal_distance
            goal_uy = goal_dy / goal_distance
        else:
            goal_ux = goal_uy = 0.0

        field_x = self.attraction_gain * goal_ux
        field_y = self.attraction_gain * goal_uy
        closest_obstacle = float("inf")

        for tree in self.trees:
            is_active = int(tree.id) == self.active_tree_id
            dx = cx - float(tree.x)
            dy = cy - float(tree.y)
            distance = math.hypot(dx, dy)
            if distance <= 1e-4:
                continue

            # Pohon target boleh diorbit pada radius misi. Target aktif hanya
            # menghasilkan tolakan bila drone masuk ke keep-out radius; pohon
            # lain tetap menggunakan influence radius penuh.
            if is_active:
                closest_obstacle = min(closest_obstacle, distance)
                if distance >= self.active_tree_keepout_radius:
                    continue
                local_hard_radius = self.active_tree_keepout_radius
                local_influence_radius = self.active_tree_keepout_radius + 0.8
            else:
                local_hard_radius = self.hard_radius
                local_influence_radius = max(
                    self.influence_radius, local_hard_radius + 0.8
                )
                if distance >= local_influence_radius:
                    continue
                closest_obstacle = min(closest_obstacle, distance)

            ux = dx / distance
            uy = dy / distance
            effective_distance = max(distance, local_hard_radius * 0.35)
            repulsive = self.repulsive_gain * (
                (1.0 / effective_distance) - (1.0 / local_influence_radius)
            ) / (effective_distance * effective_distance)
            if distance < local_hard_radius:
                repulsive += 2.0 * self.repulsive_gain * (
                    local_hard_radius - distance
                ) / local_hard_radius

            field_x += repulsive * ux
            field_y += repulsive * uy

            if not is_active:
                tangent_a = (-uy, ux)
                tangent_b = (uy, -ux)
                dot_a = tangent_a[0] * goal_ux + tangent_a[1] * goal_uy
                dot_b = tangent_b[0] * goal_ux + tangent_b[1] * goal_uy
                tangent = tangent_a if dot_a >= dot_b else tangent_b
                vortex_weight = self.vortex_gain * (
                    local_influence_radius - distance
                ) / local_influence_radius
                field_x += vortex_weight * tangent[0]
                field_y += vortex_weight * tangent[1]

        if closest_obstacle < self.emergency_stop_radius:
            self.publish_hold()
            now = self.now_sec()
            if now - self.last_warning_time > 0.5:
                self.last_warning_time = now
                self.get_logger().error(
                    f"EMERGENCY HOLD: obstacle {closest_obstacle:.2f} m."
                )
            return

        field_magnitude = math.hypot(field_x, field_y)
        if field_magnitude > self.max_target_shift:
            scale = self.max_target_shift / field_magnitude
            field_x *= scale
            field_y *= scale

        if goal_distance < 0.05 and closest_obstacle == float("inf"):
            safe_x, safe_y = gx, gy
        else:
            safe_x = cx + field_x
            safe_y = cy + field_y

        safe = PoseStamped()
        safe.header.frame_id = self.world_frame
        safe.header.stamp = self.get_clock().now().to_msg()
        safe.pose.position.x = float(safe_x)
        safe.pose.position.y = float(safe_y)
        safe.pose.position.z = float(goal.pose.position.z)
        safe.pose.orientation = goal.pose.orientation
        self.safe_target_pub.publish(safe)

        now = self.now_sec()
        if closest_obstacle < self.hard_radius and now - self.last_warning_time > 1.0:
            self.last_warning_time = now
            self.get_logger().warning(
                f"Obstacle dekat: {closest_obstacle:.2f} m; target lokal digeser."
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VortexAvoidanceController()
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
