#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from beehive_drone.mission_params import MissionConfig
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool, String, Float32
from uav_interfaces.msg import TreeArray, Tree


def euler_to_quaternion(roll, pitch, yaw):
    """Konversi sudut Euler menjadi quaternion ROS."""
    qx = (
        math.sin(roll / 2.0) * math.cos(pitch / 2.0) * math.cos(yaw / 2.0)
        - math.cos(roll / 2.0) * math.sin(pitch / 2.0) * math.sin(yaw / 2.0)
    )
    qy = (
        math.cos(roll / 2.0) * math.sin(pitch / 2.0) * math.cos(yaw / 2.0)
        + math.sin(roll / 2.0) * math.cos(pitch / 2.0) * math.sin(yaw / 2.0)
    )
    qz = (
        math.cos(roll / 2.0) * math.cos(pitch / 2.0) * math.sin(yaw / 2.0)
        - math.sin(roll / 2.0) * math.sin(pitch / 2.0) * math.cos(yaw / 2.0)
    )
    qw = (
        math.cos(roll / 2.0) * math.cos(pitch / 2.0) * math.cos(yaw / 2.0)
        + math.sin(roll / 2.0) * math.sin(pitch / 2.0) * math.sin(yaw / 2.0)
    )
    return qx, qy, qz, qw


def quaternion_to_yaw(qx, qy, qz, qw):
    """Konversi quaternion ROS menjadi yaw."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class MissionStateMachine(Node):
    """
    FSM pengujian satu pohon.

    Urutan misi:
        INIT
        -> WAIT_GUIDED
        -> WAIT_ARM
        -> WAIT_TAKEOFF
        -> TREE_SEARCH_HOVER
        -> APPROACH_TREE
        -> PRE_ORBIT_HOVER
        -> START_ORBIT
        -> WAIT_ORBIT
        -> POST_ORBIT_HOVER
        -> LANDING
        -> DONE
    """

    def __init__(self):
        super().__init__("mission_state_machine")

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

        # ==================================================
        # Parameter misi satu pohon
        # ==================================================
        self.flight_altitude = MissionConfig.FLIGHT_ALTITUDE
        self.approach_safe_dist = MissionConfig.APPROACH_SAFE_DIST

        self.tree_check_hover_duration = 4.0
        self.pre_orbit_hover_duration = 4.0
        self.post_orbit_hover_duration = 4.0

        self.max_tree_search_distance = 15.0
        self.approach_goal_tolerance = 0.6
        self.safe_distance_tolerance = 0.5
        self.land_command_period = 2.0

        # ==================================================
        # State dan data misi
        # ==================================================
        self.state = "INIT"
        self.state_enter_time = self.get_clock().now()
        self.retry_counter = 0

        self.current_pose = None
        self.trees = []
        self.target_tree = None
        self.orbit_status = "IDLE"

        self.is_armed = False
        self.current_mode = ""
        self.is_hovering = False

        # Titik hover setelah takeoff.
        self.search_hold_x = 0.0
        self.search_hold_y = 0.0
        self.search_hold_yaw = 0.0

        # Kandidat pohon harus tersedia selama beberapa detik sebelum dipilih.
        self.candidate_tree_id = None
        self.candidate_since = None

        # Titik approach dibuat tetap agar tidak bergeser setiap siklus.
        self.approach_x = 0.0
        self.approach_y = 0.0
        self.approach_yaw = 0.0

        # Pusat orbit dikunci setelah verifikasi.
        self.orbit_x = 0.0
        self.orbit_y = 0.0
        self.orbit_z = 0.0

        # Titik hover setelah orbit.
        self.post_hold_x = 0.0
        self.post_hold_y = 0.0
        self.post_hold_yaw = 0.0

        self.last_land_command_time = None

        # ==================================================
        # Subscriber
        # ==================================================
        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.pose_cb,
            qos_sensor,
        )
        self.create_subscription(
            String,
            "/control/orbit_status",
            self.orbit_status_cb,
            10,
        )
        self.create_subscription(
            TreeArray,
            "/map/trees",
            self.tree_cb,
            qos_map,
        )

        self.create_subscription(
            Bool,
            "/flight/telemetry/is_armed",
            self.arm_cb,
            10,
        )
        self.create_subscription(
            String,
            "/flight/telemetry/current_mode",
            self.mode_cb,
            10,
        )
        self.create_subscription(
            Bool,
            "/flight/telemetry/is_hovering",
            self.hover_cb,
            10,
        )

        # ==================================================
        # Publisher
        # ==================================================
        self.cmd_mode_pub = self.create_publisher(
            String, "/flight/cmd/set_mode", 10
        )
        self.cmd_arm_pub = self.create_publisher(
            Bool, "/flight/cmd/set_arm", 10
        )
        self.cmd_takeoff_pub = self.create_publisher(
            Float32, "/flight/cmd/takeoff", 10
        )
        self.cmd_land_pub = self.create_publisher(
            Bool, "/flight/cmd/land", 10
        )

        self.orbit_start_pub = self.create_publisher(
            Bool, "/control/orbit_start", 10
        )
        self.orbit_target_pub = self.create_publisher(
            Point, "/control/orbit_target", 10
        )

        self.local_goal_pub = self.create_publisher(
            PoseStamped, "/navigation/local_goal", 10
        )
        self.fsm_status_pub = self.create_publisher(
            String, "/mission/fsm_state", 10
        )
        self.tree_update_pub = self.create_publisher(
            Tree, "/map/tree_update", 10
        )

        # FSM 10 Hz.
        self.timer = self.create_timer(0.1, self.fsm_loop)

        self.get_logger().info(
            "Mission State Machine satu pohon siap: "
            "takeoff -> cek pohon -> approach -> orbit -> land"
        )

    # ==================================================
    # Callback
    # ==================================================
    def pose_cb(self, msg):
        self.current_pose = msg

    def orbit_status_cb(self, msg):
        self.orbit_status = msg.data

    def tree_cb(self, msg):
        self.trees = msg.trees

    def arm_cb(self, msg):
        self.is_armed = msg.data

    def mode_cb(self, msg):
        self.current_mode = msg.data

    def hover_cb(self, msg):
        self.is_hovering = msg.data

    # ==================================================
    # Helper
    # ==================================================
    def transition(self, new_state, log_message=None):
        self.state = new_state
        self.state_enter_time = self.get_clock().now()
        self.retry_counter = 0

        if log_message:
            self.get_logger().info(log_message)

    def state_elapsed(self):
        duration = self.get_clock().now() - self.state_enter_time
        return duration.nanoseconds / 1e9

    def current_yaw(self):
        if self.current_pose is None:
            return 0.0

        q = self.current_pose.pose.orientation
        return quaternion_to_yaw(q.x, q.y, q.z, q.w)

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def get_tree_by_id(self, tree_id):
        for tree in self.trees:
            if tree.id == tree_id:
                return tree
        return None

    def find_nearest_uninspected_tree(self):
        if self.current_pose is None:
            return None

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        best_tree = None
        best_distance = float("inf")

        for tree in self.trees:
            if tree.inspected:
                continue

            dist = self.distance(cx, cy, tree.x, tree.y)
            if dist < best_distance and dist <= self.max_tree_search_distance:
                best_distance = dist
                best_tree = tree

        return best_tree

    def calculate_approach_goal(self, tree):
        """Hitung titik berhenti pada jarak aman di depan pohon."""
        if self.current_pose is None:
            return False

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        dx = tree.x - cx
        dy = tree.y - cy
        distance_to_tree = math.hypot(dx, dy)

        if distance_to_tree < 0.1:
            self.get_logger().warning(
                "Posisi drone terlalu dekat dengan pusat pohon; approach dibatalkan."
            )
            return False

        unit_x = dx / distance_to_tree
        unit_y = dy / distance_to_tree

        self.approach_x = tree.x - self.approach_safe_dist * unit_x
        self.approach_y = tree.y - self.approach_safe_dist * unit_y
        self.approach_yaw = math.atan2(dy, dx)
        return True

    def publish_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = float(self.flight_altitude)

        qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw

        self.local_goal_pub.publish(goal)

    def publish_orbit_target(self):
        target = Point()
        target.x = float(self.orbit_x)
        target.y = float(self.orbit_y)
        target.z = float(self.orbit_z)
        self.orbit_target_pub.publish(target)

    def publish_orbit_start(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.orbit_start_pub.publish(msg)

    def mark_target_tree_inspected(self):
        if self.target_tree is None:
            return

        latest_tree = self.get_tree_by_id(self.target_tree.id)
        source_tree = latest_tree if latest_tree is not None else self.target_tree

        update = Tree()
        update.id = source_tree.id
        update.x = source_tree.x
        update.y = source_tree.y
        update.z = source_tree.z
        update.confidence = source_tree.confidence
        update.inspected = True
        self.tree_update_pub.publish(update)

        self.get_logger().info(
            f"Pohon ID:{source_tree.id} ditandai selesai diinspeksi."
        )

    # ==================================================
    # FSM utama
    # ==================================================
    def fsm_loop(self):
        if self.current_pose is None:
            return

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        cz = self.current_pose.pose.position.z

        status_msg = String()
        status_msg.data = self.state
        self.fsm_status_pub.publish(status_msg)

        # --------------------------------------------------
        # 1. PRE-FLIGHT
        # --------------------------------------------------
        if self.state == "INIT":
            # Pastikan orbit tidak aktif dari eksekusi sebelumnya.
            self.publish_orbit_start(False)

            mode_msg = String()
            mode_msg.data = "GUIDED"
            self.cmd_mode_pub.publish(mode_msg)

            self.transition(
                "WAIT_GUIDED",
                "Meminta transisi ke mode GUIDED...",
            )

        elif self.state == "WAIT_GUIDED":
            if self.current_mode == "GUIDED":
                arm_msg = Bool()
                arm_msg.data = True
                self.cmd_arm_pub.publish(arm_msg)

                self.transition(
                    "WAIT_ARM",
                    "Mode GUIDED aktif. Meminta arming motor...",
                )
            else:
                self.retry_counter += 1
                if self.retry_counter >= 20:
                    mode_msg = String()
                    mode_msg.data = "GUIDED"
                    self.cmd_mode_pub.publish(mode_msg)
                    self.retry_counter = 0

        elif self.state == "WAIT_ARM":
            if self.is_armed:
                takeoff_msg = Float32()
                takeoff_msg.data = float(self.flight_altitude)
                self.cmd_takeoff_pub.publish(takeoff_msg)

                self.transition(
                    "WAIT_TAKEOFF",
                    f"Armed. Takeoff ke {self.flight_altitude:.1f} meter...",
                )
            else:
                self.retry_counter += 1
                if self.retry_counter >= 20:
                    arm_msg = Bool()
                    arm_msg.data = True
                    self.cmd_arm_pub.publish(arm_msg)
                    self.retry_counter = 0
                    self.get_logger().info("Mengulangi permintaan arming...")

        elif self.state == "WAIT_TAKEOFF":
            if self.is_hovering:
                self.search_hold_x = cx
                self.search_hold_y = cy
                self.search_hold_yaw = self.current_yaw()

                self.candidate_tree_id = None
                self.candidate_since = None

                self.transition(
                    "TREE_SEARCH_HOVER",
                    "Takeoff selesai. Hovering sambil memeriksa satu pohon.",
                )

        # --------------------------------------------------
        # 2. HOVER DAN PILIH SATU POHON
        # --------------------------------------------------
        elif self.state == "TREE_SEARCH_HOVER":
            self.publish_goal(
                self.search_hold_x,
                self.search_hold_y,
                self.search_hold_yaw,
            )

            candidate = self.find_nearest_uninspected_tree()

            if candidate is None:
                self.candidate_tree_id = None
                self.candidate_since = None
                return

            if candidate.id != self.candidate_tree_id:
                self.candidate_tree_id = candidate.id
                self.candidate_since = self.get_clock().now()
                self.get_logger().info(
                    f"Kandidat pohon ID:{candidate.id} ditemukan. "
                    "Menunggu posisi stabil..."
                )
                return

            stable_duration = (
                self.get_clock().now() - self.candidate_since
            ).nanoseconds / 1e9

            if stable_duration >= self.tree_check_hover_duration:
                self.target_tree = candidate

                if not self.calculate_approach_goal(candidate):
                    self.candidate_tree_id = None
                    self.candidate_since = None
                    return

                self.transition(
                    "APPROACH_TREE",
                    f"Pohon ID:{candidate.id} dipilih. Mendekati titik aman di "
                    f"({self.approach_x:.2f}, {self.approach_y:.2f}).",
                )

        # --------------------------------------------------
        # 3. APPROACH KE TITIK AMAN
        # --------------------------------------------------
        elif self.state == "APPROACH_TREE":
            if self.target_tree is None:
                self.transition(
                    "TREE_SEARCH_HOVER",
                    "Target pohon kosong. Kembali memeriksa peta.",
                )
                return

            latest_tree = self.get_tree_by_id(self.target_tree.id)
            if latest_tree is None:
                self.target_tree = None
                self.candidate_tree_id = None
                self.candidate_since = None
                self.transition(
                    "TREE_SEARCH_HOVER",
                    "Target pohon hilang dari peta. Kembali hovering.",
                )
                return

            self.target_tree = latest_tree
            target_yaw = math.atan2(latest_tree.y - cy, latest_tree.x - cx)
            self.publish_goal(self.approach_x, self.approach_y, target_yaw)

            distance_to_approach = self.distance(
                cx, cy, self.approach_x, self.approach_y
            )

            if distance_to_approach <= self.approach_goal_tolerance:
                self.transition(
                    "PRE_ORBIT_HOVER",
                    "Titik approach tercapai. Hovering untuk memastikan jarak aman.",
                )

        # --------------------------------------------------
        # 4. HOVER DAN VERIFIKASI SEBELUM ORBIT
        # --------------------------------------------------
        elif self.state == "PRE_ORBIT_HOVER":
            if self.target_tree is None:
                self.transition("TREE_SEARCH_HOVER")
                return

            latest_tree = self.get_tree_by_id(self.target_tree.id)
            if latest_tree is None:
                self.target_tree = None
                self.transition(
                    "TREE_SEARCH_HOVER",
                    "Pohon hilang saat verifikasi. Kembali hovering.",
                )
                return

            self.target_tree = latest_tree
            target_yaw = math.atan2(latest_tree.y - cy, latest_tree.x - cx)
            self.publish_goal(self.approach_x, self.approach_y, target_yaw)

            if self.state_elapsed() < self.pre_orbit_hover_duration:
                return

            actual_distance = self.distance(
                cx, cy, latest_tree.x, latest_tree.y
            )
            min_safe = max(
                0.5,
                self.approach_safe_dist - self.safe_distance_tolerance,
            )
            max_safe = (
                self.approach_safe_dist + self.safe_distance_tolerance
            )

            if min_safe <= actual_distance <= max_safe:
                self.orbit_x = latest_tree.x
                self.orbit_y = latest_tree.y
                self.orbit_z = latest_tree.z
                self.orbit_status = "IDLE"

                self.transition(
                    "START_ORBIT",
                    f"Posisi aman: jarak ke pohon {actual_distance:.2f} meter. "
                    "Memulai orbit.",
                )
            else:
                self.get_logger().warning(
                    f"Jarak verifikasi {actual_distance:.2f} meter tidak sesuai "
                    f"rentang {min_safe:.2f}-{max_safe:.2f} meter. "
                    "Mengulangi approach."
                )

                if self.calculate_approach_goal(latest_tree):
                    self.transition("APPROACH_TREE")
                else:
                    self.target_tree = None
                    self.transition("TREE_SEARCH_HOVER")

        # --------------------------------------------------
        # 5. ORBIT SATU POHON
        # --------------------------------------------------
        elif self.state == "START_ORBIT":
            # Publikasikan target dan perintah beberapa kali sampai controller
            # mengonfirmasi bahwa orbit sedang berjalan.
            self.publish_orbit_target()
            self.publish_orbit_start(True)

            if self.orbit_status == "IN_PROGRESS":
                self.transition(
                    "WAIT_ORBIT",
                    "Orbit aktif. Menunggu satu putaran selesai.",
                )
            elif self.state_elapsed() > 2.0:
                # Tetap lanjut ke WAIT_ORBIT; perintah start masih akan dikirim
                # berulang di state tersebut.
                self.transition(
                    "WAIT_ORBIT",
                    "Menunggu status orbit selesai.",
                )

        elif self.state == "WAIT_ORBIT":
            # Menjaga target dan perintah orbit tetap tersedia.
            self.publish_orbit_target()
            self.publish_orbit_start(True)

            if self.orbit_status == "ORBIT_COMPLETED":
                self.publish_orbit_start(False)
                self.mark_target_tree_inspected()

                self.post_hold_x = cx
                self.post_hold_y = cy
                self.post_hold_yaw = self.current_yaw()

                self.transition(
                    "POST_ORBIT_HOVER",
                    "Orbit selesai. Hovering sebelum landing.",
                )

        # --------------------------------------------------
        # 6. HOVER SETELAH ORBIT
        # --------------------------------------------------
        elif self.state == "POST_ORBIT_HOVER":
            self.publish_orbit_start(False)
            self.publish_goal(
                self.post_hold_x,
                self.post_hold_y,
                self.post_hold_yaw,
            )

            if self.state_elapsed() >= self.post_orbit_hover_duration:
                self.last_land_command_time = None
                self.transition(
                    "LANDING",
                    "Hover akhir selesai. Memulai landing.",
                )

        # --------------------------------------------------
        # 7. LANDING
        # --------------------------------------------------
        elif self.state == "LANDING":
            self.publish_orbit_start(False)

            now_sec = self.get_clock().now().nanoseconds / 1e9
            if (
                self.last_land_command_time is None
                or now_sec - self.last_land_command_time
                >= self.land_command_period
            ):
                land_msg = Bool()
                land_msg.data = True
                self.cmd_land_pub.publish(land_msg)
                self.last_land_command_time = now_sec
                self.get_logger().info("Perintah landing dikirim.")

            # Selesai jika autopilot sudah disarm atau posisi lokal mendekati tanah.
            if not self.is_armed or (
                self.state_elapsed() > 2.0 and cz <= 0.20
            ):
                self.transition(
                    "DONE",
                    "Drone telah mendarat. Misi satu pohon selesai.",
                )

        elif self.state == "DONE":
            self.publish_orbit_start(False)


def main(args=None):
    rclpy.init(args=args)
    node = MissionStateMachine()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
