#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String
from uav_interfaces.msg import Tree, TreeArray

from beehive_drone.math_utils import quaternion_from_yaw, yaw_from_quaternion


class SimpleSingleTreeMission(Node):
    """Slim real-drone mission using ArduPilot's position controller directly.

    Nodes removed from the flight-control chain:
    - dynamic_orbit_controller
    - vortex_avoidance_controller
    - velocity_controller

    This node publishes one local position setpoint stream and contains the
    complete one-tree mission.  RC takeover remains latched and has priority.
    """

    MOVEMENT_STATES = {
        "HOLD",
        "SEARCH_TREE",
        "APPROACH_TREE",
        "HOVER_BEFORE_ORBIT",
        "ORBIT",
        "POST_ORBIT_HOVER",
        "RETURN_PRE_ORBIT",
        "RETURN_HOME",
        "HOME_HOVER",
    }

    TAKEOVER_STATES = {
        "ARM_SETTLE",
        "REQUEST_TAKEOFF",
        "WAIT_TAKEOFF_ACK",
        "TAKEOFF_CLIMB",
        *MOVEMENT_STATES,
        "LAND",
        "WAIT_LANDED",
        "TAKEOFF_ABORT",
    }

    RESULT_NAMES = {
        0: "ACCEPTED",
        1: "TEMPORARILY_REJECTED",
        2: "DENIED",
        3: "UNSUPPORTED",
        4: "FAILED",
        5: "IN_PROGRESS",
        6: "CANCELLED",
        255: "UNKNOWN",
    }

    def __init__(self) -> None:
        super().__init__("simple_single_tree_mission")

        defaults = {
            "world_frame": "odom",
            "flight_mode": "GUIDED",
            "flight_altitude": 2.0,
            "hold_after_takeoff": True,
            "arm_settle_sec": 1.5,
            "command_retry_sec": 1.0,
            "takeoff_ack_timeout_sec": 4.0,
            "takeoff_retry_delay_sec": 2.0,
            "takeoff_max_attempts": 3,
            "takeoff_progress_check_sec": 20.0,
            "takeoff_timeout_sec": 45.0,
            "min_takeoff_progress": 0.20,
            "pose_timeout_sec": 1.5,
            "enable_rc_takeover": True,
            "rc_takeover_confirm_sec": 0.30,
            "tree_search_timeout_sec": 35.0,
            "tree_search_radius": 18.0,
            "tree_min_confidence": 0.35,
            "approach_distance": 2.0,
            "position_tolerance": 0.35,
            "pre_orbit_hover_sec": 3.0,
            "post_orbit_hover_sec": 3.0,
            "home_hover_sec": 3.0,
            "orbit_radius": 2.0,
            "orbit_speed": 0.30,
            "orbit_direction": 1.0,
            "land_retry_sec": 5.0,
            "land_complete_altitude": 0.25,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.flight_mode = str(self.get_parameter("flight_mode").value).upper()
        self.flight_altitude = float(self.get_parameter("flight_altitude").value)
        self.hold_after_takeoff = bool(
            self.get_parameter("hold_after_takeoff").value
        )
        self.arm_settle_sec = float(self.get_parameter("arm_settle_sec").value)
        self.command_retry_sec = float(
            self.get_parameter("command_retry_sec").value
        )
        self.takeoff_ack_timeout_sec = float(
            self.get_parameter("takeoff_ack_timeout_sec").value
        )
        self.takeoff_retry_delay_sec = float(
            self.get_parameter("takeoff_retry_delay_sec").value
        )
        self.takeoff_max_attempts = int(
            self.get_parameter("takeoff_max_attempts").value
        )
        self.takeoff_progress_check_sec = float(
            self.get_parameter("takeoff_progress_check_sec").value
        )
        self.takeoff_timeout_sec = float(
            self.get_parameter("takeoff_timeout_sec").value
        )
        self.min_takeoff_progress = float(
            self.get_parameter("min_takeoff_progress").value
        )
        self.pose_timeout_sec = float(self.get_parameter("pose_timeout_sec").value)
        self.enable_rc_takeover = bool(
            self.get_parameter("enable_rc_takeover").value
        )
        self.rc_takeover_confirm_sec = float(
            self.get_parameter("rc_takeover_confirm_sec").value
        )
        self.tree_search_timeout_sec = float(
            self.get_parameter("tree_search_timeout_sec").value
        )
        self.tree_search_radius = float(
            self.get_parameter("tree_search_radius").value
        )
        self.tree_min_confidence = float(
            self.get_parameter("tree_min_confidence").value
        )
        self.approach_distance = float(
            self.get_parameter("approach_distance").value
        )
        self.position_tolerance = float(
            self.get_parameter("position_tolerance").value
        )
        self.pre_orbit_hover_sec = float(
            self.get_parameter("pre_orbit_hover_sec").value
        )
        self.post_orbit_hover_sec = float(
            self.get_parameter("post_orbit_hover_sec").value
        )
        self.home_hover_sec = float(
            self.get_parameter("home_hover_sec").value
        )
        self.orbit_radius = float(self.get_parameter("orbit_radius").value)
        self.orbit_speed = float(self.get_parameter("orbit_speed").value)
        self.orbit_direction = (
            1.0 if float(self.get_parameter("orbit_direction").value) >= 0.0 else -1.0
        )
        self.land_retry_sec = float(self.get_parameter("land_retry_sec").value)
        self.land_complete_altitude = float(
            self.get_parameter("land_complete_altitude").value
        )

        self.state = "WAIT_CONNECTION"
        self.state_enter_time = self.now_sec()
        self.last_command_time = -1e9
        self.last_land_command_time = -1e9

        self.connected = False
        self.armed = False
        self.current_mode = ""
        self.hovering = False
        self.altitude = 0.0
        self.current_pose: Optional[PoseStamped] = None
        self.last_pose_time: Optional[float] = None

        self.home_captured = False
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_yaw = 0.0
        self.target_z = self.flight_altitude

        self.target_pose: Optional[PoseStamped] = None
        self.target_tree: Optional[Tree] = None
        self.trees = []
        self.map_ready = False
        self.pre_orbit: Optional[Tuple[float, float, float]] = None
        self.orbit_start_angle = 0.0

        self.takeoff_attempts = 0
        self.takeoff_response_received = False
        self.takeoff_success = False
        self.takeoff_result = 255
        self.takeoff_start_z = 0.0
        self.takeoff_max_z = 0.0
        self.landed_state = 0
        self.relative_altitude = float("nan")
        self.last_abort_warning_time = -1e9

        self.mode_mismatch_since: Optional[float] = None
        self.pilot_override_latched = False

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
            PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_sensor
        )
        self.create_subscription(
            Bool, "/flight/telemetry/is_connected", self.connected_cb, 10
        )
        self.create_subscription(Bool, "/flight/telemetry/is_armed", self.armed_cb, 10)
        self.create_subscription(
            String, "/flight/telemetry/current_mode", self.mode_cb, 10
        )
        self.create_subscription(
            Bool, "/flight/telemetry/is_hovering", self.hover_cb, 10
        )
        self.create_subscription(
            Float32, "/flight/telemetry/altitude", self.altitude_cb, 10
        )
        self.create_subscription(
            Int32, "/flight/telemetry/landed_state", self.landed_state_cb, 10
        )
        self.create_subscription(
            Float32,
            "/flight/telemetry/relative_altitude",
            self.relative_altitude_cb,
            10,
        )
        self.create_subscription(
            Bool, "/flight/response/takeoff_success", self.takeoff_success_cb, 10
        )
        self.create_subscription(
            Int32, "/flight/response/takeoff_result", self.takeoff_result_cb, 10
        )
        self.create_subscription(TreeArray, "/map/trees", self.trees_cb, qos_map)
        self.create_subscription(
            Bool, "/map/trees_ready", self.map_ready_cb, qos_map
        )

        self.mode_pub = self.create_publisher(String, "/flight/cmd/set_mode", 10)
        self.arm_pub = self.create_publisher(Bool, "/flight/cmd/set_arm", 10)
        self.takeoff_pub = self.create_publisher(Float32, "/flight/cmd/takeoff", 10)
        self.land_pub = self.create_publisher(Bool, "/flight/cmd/land", 10)
        self.target_alt_pub = self.create_publisher(
            Float32, "/flight/target_altitude", 10
        )
        self.setpoint_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", 10
        )
        self.state_pub = self.create_publisher(String, "/mission/fsm_state", 10)
        self.override_pub = self.create_publisher(
            Bool, "/mission/pilot_override", 10
        )

        self.create_timer(0.05, self.loop)
        self.get_logger().info(
            "Simple Single Tree Mission aktif: direct position setpoint + RC takeover."
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_cb(self, msg: PoseStamped) -> None:
        self.current_pose = msg
        self.last_pose_time = self.now_sec()
        if not self.home_captured:
            self.home_x = float(msg.pose.position.x)
            self.home_y = float(msg.pose.position.y)
            self.home_z = float(msg.pose.position.z)
            self.home_yaw = yaw_from_quaternion(msg.pose.orientation)
            self.target_z = self.home_z + self.flight_altitude
            self.home_captured = True

    def connected_cb(self, msg: Bool) -> None:
        self.connected = bool(msg.data)

    def armed_cb(self, msg: Bool) -> None:
        self.armed = bool(msg.data)

    def mode_cb(self, msg: String) -> None:
        self.current_mode = msg.data.strip().upper()

    def hover_cb(self, msg: Bool) -> None:
        self.hovering = bool(msg.data)

    def altitude_cb(self, msg: Float32) -> None:
        self.altitude = float(msg.data)

    def landed_state_cb(self, msg: Int32) -> None:
        self.landed_state = int(msg.data)

    def relative_altitude_cb(self, msg: Float32) -> None:
        self.relative_altitude = float(msg.data)

    def takeoff_success_cb(self, msg: Bool) -> None:
        self.takeoff_success = bool(msg.data)
        self.takeoff_response_received = True

    def takeoff_result_cb(self, msg: Int32) -> None:
        self.takeoff_result = int(msg.data)

    def trees_cb(self, msg: TreeArray) -> None:
        self.trees = list(msg.trees)

    def map_ready_cb(self, msg: Bool) -> None:
        self.map_ready = bool(msg.data)

    def pose_fresh(self) -> bool:
        return (
            self.last_pose_time is not None
            and self.now_sec() - self.last_pose_time <= self.pose_timeout_sec
        )

    def transition(self, new_state: str, reason: str = "") -> None:
        if new_state == self.state:
            return
        old = self.state
        self.state = new_state
        self.state_enter_time = self.now_sec()
        self.last_command_time = -1e9
        text = f"{old} -> {new_state}"
        if reason:
            text += f" | {reason}"
        self.get_logger().info(text)

    def command_due(self, period: Optional[float] = None) -> bool:
        interval = self.command_retry_sec if period is None else period
        now = self.now_sec()
        if now - self.last_command_time >= interval:
            self.last_command_time = now
            return True
        return False

    def publish_state(self) -> None:
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

        override = Bool()
        override.data = self.pilot_override_latched
        self.override_pub.publish(override)

    def send_bool(self, publisher, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        publisher.publish(msg)

    def publish_target_altitude(self) -> None:
        msg = Float32()
        msg.data = float(self.target_z)
        self.target_alt_pub.publish(msg)

    def set_target(self, x: float, y: float, z: float, yaw: float) -> None:
        target = PoseStamped()
        target.header.frame_id = self.world_frame
        target.pose.position.x = float(x)
        target.pose.position.y = float(y)
        target.pose.position.z = float(z)
        target.pose.orientation = quaternion_from_yaw(yaw)
        self.target_pose = target

    def publish_setpoint(self) -> None:
        if self.target_pose is None or self.pilot_override_latched:
            return
        msg = PoseStamped()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = self.target_pose.pose
        self.setpoint_pub.publish(msg)
        self.publish_target_altitude()

    def distance_to_target(self) -> float:
        if self.current_pose is None or self.target_pose is None:
            return float("inf")
        dx = self.target_pose.pose.position.x - self.current_pose.pose.position.x
        dy = self.target_pose.pose.position.y - self.current_pose.pose.position.y
        dz = self.target_pose.pose.position.z - self.current_pose.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def hold_home(self) -> None:
        self.set_target(
            self.home_x, self.home_y, self.target_z, self.home_yaw
        )

    def check_remote_takeover(self, now: float) -> bool:
        if self.pilot_override_latched:
            return True
        if not self.enable_rc_takeover or not self.armed:
            self.mode_mismatch_since = None
            return False
        if self.state not in self.TAKEOVER_STATES:
            self.mode_mismatch_since = None
            return False

        allowed = {self.flight_mode}
        if self.state in {"LAND", "WAIT_LANDED"}:
            allowed.add("LAND")

        if self.current_mode and self.current_mode not in allowed:
            if self.mode_mismatch_since is None:
                self.mode_mismatch_since = now
                self.get_logger().warning(
                    f"Mode berubah ke {self.current_mode}; setpoint dihentikan segera."
                )
            if now - self.mode_mismatch_since >= self.rc_takeover_confirm_sec:
                self.pilot_override_latched = True
                self.target_pose = None
                self.transition(
                    "PILOT_OVERRIDE",
                    f"Pilot takeover terkonfirmasi pada mode {self.current_mode}",
                )
            return True

        self.mode_mismatch_since = None
        return False

    def valid_tree(self, tree: Tree) -> bool:
        return (
            not bool(tree.inspected)
            and float(tree.confidence) >= self.tree_min_confidence
            and all(math.isfinite(float(v)) for v in (tree.x, tree.y, tree.z))
        )

    def nearest_tree(self) -> Optional[Tree]:
        if self.current_pose is None or not self.map_ready:
            return None
        px = float(self.current_pose.pose.position.x)
        py = float(self.current_pose.pose.position.y)
        choices = []
        for tree in self.trees:
            if not self.valid_tree(tree):
                continue
            distance = math.hypot(float(tree.x) - px, float(tree.y) - py)
            if distance <= self.tree_search_radius:
                choices.append((distance, int(tree.id), tree))
        return min(choices, default=(0.0, 0, None))[2]

    def compute_pre_orbit(self, tree: Tree) -> Tuple[float, float, float]:
        assert self.current_pose is not None
        px = float(self.current_pose.pose.position.x)
        py = float(self.current_pose.pose.position.y)
        dx = px - float(tree.x)
        dy = py - float(tree.y)
        norm = math.hypot(dx, dy)
        if norm < 0.1:
            dx, dy, norm = -1.0, 0.0, 1.0
        x = float(tree.x) + self.approach_distance * dx / norm
        y = float(tree.y) + self.approach_distance * dy / norm
        yaw = math.atan2(float(tree.y) - y, float(tree.x) - x)
        return x, y, yaw

    def request_takeoff(self) -> None:
        self.takeoff_attempts += 1
        self.takeoff_response_received = False
        self.takeoff_success = False
        self.takeoff_result = 255

        msg = Float32()
        # CommandTOL/NAV_TAKEOFF receives the requested takeoff altitude.  Keep
        # this identical to the known-working pratesting node.
        msg.data = float(self.flight_altitude)
        self.takeoff_pub.publish(msg)
        self.get_logger().info(
            f"Permintaan takeoff #{self.takeoff_attempts}: {self.flight_altitude:.2f} m"
        )

    def begin_takeoff_abort(self, reason: str) -> None:
        self.target_pose = None
        self.transition("TAKEOFF_ABORT", reason)

    def loop(self) -> None:
        self.publish_state()
        now = self.now_sec()

        if self.check_remote_takeover(now):
            if self.state == "PILOT_OVERRIDE" and not self.armed:
                self.transition("DONE", "Drone sudah disarm oleh pilot")
            return

        if self.current_pose is None:
            return

        if self.state in self.MOVEMENT_STATES:
            if not self.connected or not self.pose_fresh():
                # Do not invent new commands when local pose is stale.  Keep the
                # last setpoint unpublished so pilot takeover remains clean.
                return
            self.publish_setpoint()

        if self.state == "WAIT_CONNECTION":
            if self.connected and self.pose_fresh() and self.home_captured:
                self.transition("SET_MODE", "MAVROS dan local pose tersedia")

        elif self.state == "SET_MODE":
            if self.current_mode == self.flight_mode:
                self.transition("ARM")
            elif self.command_due():
                msg = String()
                msg.data = self.flight_mode
                self.mode_pub.publish(msg)

        elif self.state == "ARM":
            if self.armed:
                # Re-capture home immediately after arming so the mission uses
                # the same local origin as the actual flight attempt.
                pose = self.current_pose.pose
                self.home_x = float(pose.position.x)
                self.home_y = float(pose.position.y)
                self.home_z = float(pose.position.z)
                self.home_yaw = yaw_from_quaternion(pose.orientation)
                self.target_z = self.home_z + self.flight_altitude
                self.transition("ARM_SETTLE", "Armed; menunggu FCU stabil")
            elif self.command_due():
                self.send_bool(self.arm_pub, True)

        elif self.state == "ARM_SETTLE":
            if not self.armed:
                self.transition("ARM", "Status armed hilang")
            elif now - self.state_enter_time >= self.arm_settle_sec:
                self.transition("REQUEST_TAKEOFF")

        elif self.state == "REQUEST_TAKEOFF":
            self.request_takeoff()
            self.transition("WAIT_TAKEOFF_ACK")

        elif self.state == "WAIT_TAKEOFF_ACK":
            elapsed = now - self.state_enter_time
            if self.takeoff_response_received:
                result_name = self.RESULT_NAMES.get(self.takeoff_result, "UNKNOWN")
                # ACCEPTED and IN_PROGRESS both mean the FCU has started handling
                # the command.  Do not resend NAV_TAKEOFF in that condition.
                if self.takeoff_success or self.takeoff_result in (0, 5):
                    self.takeoff_start_z = self.altitude
                    self.takeoff_max_z = self.altitude
                    self.publish_target_altitude()
                    self.transition(
                        "TAKEOFF_CLIMB",
                        f"Autopilot menerima takeoff: {self.takeoff_result} {result_name}",
                    )
                # Only a temporary rejection or unknown/timeout condition is
                # worth retrying. DENIED/UNSUPPORTED/FAILED should stop immediately.
                elif (
                    self.takeoff_result in (1, 255)
                    and self.takeoff_attempts < self.takeoff_max_attempts
                ):
                    self.transition(
                        "TAKEOFF_RETRY_WAIT",
                        f"Takeoff ditolak sementara: {self.takeoff_result} {result_name}",
                    )
                else:
                    self.begin_takeoff_abort(
                        f"Takeoff ditolak: {self.takeoff_result} {result_name}; "
                        f"percobaan={self.takeoff_attempts}"
                    )
            elif elapsed >= self.takeoff_ack_timeout_sec:
                if self.takeoff_attempts < self.takeoff_max_attempts:
                    self.transition("TAKEOFF_RETRY_WAIT", "ACK takeoff timeout")
                else:
                    self.begin_takeoff_abort("ACK takeoff tidak diterima")

        elif self.state == "TAKEOFF_RETRY_WAIT":
            if now - self.state_enter_time >= self.takeoff_retry_delay_sec:
                self.transition("REQUEST_TAKEOFF")

        elif self.state == "TAKEOFF_CLIMB":
            self.publish_target_altitude()
            self.takeoff_max_z = max(self.takeoff_max_z, self.altitude)
            elapsed = now - self.state_enter_time
            climb = self.takeoff_max_z - self.takeoff_start_z

            if self.hovering:
                self.hold_home()
                if self.hold_after_takeoff:
                    self.transition("HOLD", "Hover 2 m stabil")
                else:
                    self.transition("SEARCH_TREE", "Hover stabil; mencari pohon")
            elif (
                elapsed >= self.takeoff_progress_check_sec
                and climb < self.min_takeoff_progress
            ):
                self.begin_takeoff_abort(
                    f"Takeoff diterima tetapi tidak naik; max_climb={climb:.2f} m"
                )
            elif elapsed >= self.takeoff_timeout_sec:
                self.begin_takeoff_abort(
                    f"Takeoff timeout; max_climb={climb:.2f} m"
                )

        elif self.state == "TAKEOFF_ABORT":
            rise = self.altitude - self.home_z
            rel_alt_ok = (
                math.isfinite(self.relative_altitude)
                and abs(self.relative_altitude) <= 0.20
            )
            confirmed_on_ground = self.landed_state == 1
            confirmed_airborne = self.landed_state in (2, 3, 4) or rise > 0.30

            if not self.armed:
                self.transition("DONE", "Takeoff gagal dan kendaraan disarm")
            elif confirmed_on_ground and rel_alt_ok and abs(rise) <= 0.20:
                # Never disarm from local-Z alone. Require the FCU land detector,
                # relative altitude, and local pose to agree that it is on ground.
                if self.command_due(1.0):
                    self.send_bool(self.arm_pub, False)
                    self.get_logger().warning(
                        "Takeoff gagal dan FCU mengonfirmasi ON_GROUND; meminta DISARM."
                    )
            elif confirmed_airborne:
                self.transition("LAND", "Takeoff gagal setelah kendaraan terangkat")
            elif now - self.last_abort_warning_time >= 2.0:
                self.last_abort_warning_time = now
                self.get_logger().warning(
                    "Takeoff gagal tetapi status darat/udara belum pasti. "
                    "Tidak melakukan auto-disarm; pilot dapat takeover lewat RC."
                )

        elif self.state == "HOLD":
            self.hold_home()

        elif self.state == "SEARCH_TREE":
            self.hold_home()
            tree = self.nearest_tree()
            if tree is not None:
                self.target_tree = tree
                self.pre_orbit = self.compute_pre_orbit(tree)
                x, y, yaw = self.pre_orbit
                self.set_target(x, y, self.target_z, yaw)
                self.transition(
                    "APPROACH_TREE",
                    f"Target pohon ID={tree.id}; menuju jarak {self.approach_distance:.1f} m",
                )
            elif now - self.state_enter_time >= self.tree_search_timeout_sec:
                self.transition("HOME_HOVER", "Pohon tidak ditemukan")

        elif self.state == "APPROACH_TREE":
            if self.target_tree is None or self.pre_orbit is None:
                self.transition("RETURN_HOME", "Target pohon tidak tersedia")
            elif self.distance_to_target() <= self.position_tolerance:
                self.transition("HOVER_BEFORE_ORBIT", "Titik sebelum orbit tercapai")

        elif self.state == "HOVER_BEFORE_ORBIT":
            if now - self.state_enter_time >= self.pre_orbit_hover_sec:
                assert self.target_tree is not None
                assert self.current_pose is not None
                dx = float(self.current_pose.pose.position.x) - float(self.target_tree.x)
                dy = float(self.current_pose.pose.position.y) - float(self.target_tree.y)
                self.orbit_start_angle = math.atan2(dy, dx)
                self.transition("ORBIT", "Memulai satu putaran")

        elif self.state == "ORBIT":
            if self.target_tree is None:
                self.transition("RETURN_HOME", "Target pohon hilang")
            else:
                elapsed = now - self.state_enter_time
                angular_speed = max(0.02, self.orbit_speed / self.orbit_radius)
                travelled = min(2.0 * math.pi, elapsed * angular_speed)
                theta = self.orbit_start_angle + self.orbit_direction * travelled
                tx = float(self.target_tree.x) + self.orbit_radius * math.cos(theta)
                ty = float(self.target_tree.y) + self.orbit_radius * math.sin(theta)
                yaw = math.atan2(float(self.target_tree.y) - ty, float(self.target_tree.x) - tx)
                self.set_target(tx, ty, self.target_z, yaw)
                if travelled >= 2.0 * math.pi:
                    self.transition("POST_ORBIT_HOVER", "Orbit 360 derajat selesai")

        elif self.state == "POST_ORBIT_HOVER":
            if now - self.state_enter_time >= self.post_orbit_hover_sec:
                if self.pre_orbit is None:
                    self.transition("RETURN_HOME")
                else:
                    x, y, yaw = self.pre_orbit
                    self.set_target(x, y, self.target_z, yaw)
                    self.transition("RETURN_PRE_ORBIT")

        elif self.state == "RETURN_PRE_ORBIT":
            if self.distance_to_target() <= self.position_tolerance:
                self.hold_home()
                self.transition("RETURN_HOME", "Kembali ke titik sebelum orbit")

        elif self.state == "RETURN_HOME":
            self.hold_home()
            if self.distance_to_target() <= self.position_tolerance:
                self.transition("HOME_HOVER", "Kembali ke titik takeoff")

        elif self.state == "HOME_HOVER":
            self.hold_home()
            if now - self.state_enter_time >= self.home_hover_sec:
                self.target_pose = None
                self.transition("LAND")

        elif self.state == "LAND":
            self.target_pose = None
            if not self.armed:
                self.transition("DONE", "Sudah disarm")
            else:
                if now - self.last_land_command_time >= self.land_retry_sec:
                    self.last_land_command_time = now
                    self.send_bool(self.land_pub, True)
                self.transition("WAIT_LANDED")

        elif self.state == "WAIT_LANDED":
            self.target_pose = None
            if not self.armed or self.landed_state == 1:
                self.transition("DONE", "Pendaratan selesai")
            elif now - self.last_land_command_time >= self.land_retry_sec:
                self.last_land_command_time = now
                self.send_bool(self.land_pub, True)

        elif self.state == "PILOT_OVERRIDE":
            self.target_pose = None
            if not self.armed:
                self.transition("DONE", "Drone disarm setelah takeover")

        elif self.state == "DONE":
            self.target_pose = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleSingleTreeMission()
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
