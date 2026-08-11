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
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return qx, qy, qz, qw

def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class MissionStateMachine(Node):
    def __init__(self):
        super().__init__("mission_state_machine")

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ==========================================
        # Parameter Strategi Kebun 
        # ==========================================
        self.explore_speed = MissionConfig.EXPLORE_SPEED  # Kecepatan menyusuri lorong (m/s)          
        self.crab_speed = MissionConfig.CRAB_SPEED
        self.end_of_row_dist = MissionConfig.END_OF_ROW_DIST
        self.end_of_farm_dist = MissionConfig.END_OF_FARM_DIST
        self.approach_safe_dist = MissionConfig.APPROACH_SAFE_DIST
        self.flight_altitude = MissionConfig.FLIGHT_ALTITUDE
        self.declare_parameter('flight_altitude', self.flight_altitude)
        self.declare_parameter('approach_distance', self.approach_safe_dist)
        self.declare_parameter('tree_distance_tolerance', 0.5)
        self.declare_parameter('approach_goal_tolerance', 0.15)
        self.declare_parameter('verification_retry_limit', 3)
        self.declare_parameter('require_safety_monitor', True)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('state_timeout', 120.0)
        self.declare_parameter('pose_timeout', 1.0)
        self.declare_parameter('post_takeoff_hover_time', 2.0)
        self.flight_altitude = float(self.get_parameter('flight_altitude').value)
        self.approach_safe_dist = float(
            self.get_parameter('approach_distance').value)
        self.tree_distance_tolerance = float(
            self.get_parameter('tree_distance_tolerance').value)
        self.approach_goal_tolerance = float(
            self.get_parameter('approach_goal_tolerance').value)
        self.verification_retry_limit = int(
            self.get_parameter('verification_retry_limit').value)
        self.require_safety = bool(self.get_parameter('require_safety_monitor').value)
        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.state_timeout = float(self.get_parameter('state_timeout').value)
        self.pose_timeout = float(self.get_parameter('pose_timeout').value)
        self.post_takeoff_hover_time = float(
            self.get_parameter('post_takeoff_hover_time').value)

        # ==========================================
        # Variabel State & Navigasi
        # ==========================================
        self.state = "WAIT_START"
        self.state_since = self.get_clock().now()
        self.start_requested = self.auto_start
        self.safety_ok = False
        self.safety_reason = 'watchdog_not_ready'
        self.last_pose_time = None
        self.retry_counter = 0
        self.verification_retries = 0
        self.hover_timer = 0
        self.orbit_status = "IDLE"
        self.current_pose = None
        self.home_pose = None
        self.hold_x = 0.0
        self.hold_y = 0.0
        self.hold_yaw = 0.0
        self.navigation_altitude = None
        self.trees = []
        self.target_tree = None

        # Variabel Telemetri Penerbangan (Dari Flight Manager)
        self.is_armed = False
        self.current_mode = ""
        self.is_hovering = False

        self.explore_dir_x = 1.0          
        self.explore_dir_y = 1.0          
        
        self.last_tree_x = 0.0
        self.last_tree_y = 0.0
        self.crab_start_y = 0.0
        
        self.spin_accumulated = 0.0
        self.last_yaw = 0.0

        # ==========================================
        # Subscriber
        # ==========================================
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_sensor)
        self.orbit_status_sub = self.create_subscription(String, "/control/orbit_status", self.orbit_status_cb, 10)
        self.tree_sub = self.create_subscription(TreeArray, "/map/trees", self.tree_cb, qos_map)

        # Telemetri dari Flight Manager
        self.telemetry_arm_sub = self.create_subscription(Bool, "/flight/telemetry/is_armed", self.arm_cb, 10)
        self.telemetry_mode_sub = self.create_subscription(String, "/flight/telemetry/current_mode", self.mode_cb, 10)
        self.telemetry_hover_sub = self.create_subscription(Bool, "/flight/telemetry/is_hovering", self.hover_cb, 10)
        self.create_subscription(Bool, '/mission/start', self.start_cb, 10)
        self.create_subscription(Bool, '/mission/safety_ok', self.safety_cb, 10)
        self.create_subscription(String, '/mission/safety_reason', self.safety_reason_cb, 10)

        # ==========================================
        # Publisher
        # ==========================================
        # Command ke Flight Manager
        self.cmd_mode_pub = self.create_publisher(String, "/flight/cmd/set_mode", 10)
        self.cmd_arm_pub = self.create_publisher(Bool, "/flight/cmd/set_arm", 10)
        self.cmd_takeoff_pub = self.create_publisher(Float32, "/flight/cmd/takeoff", 10)
        self.cmd_land_pub = self.create_publisher(Bool, "/flight/cmd/land", 10)
        
        # Command ke Dynamic Orbit Controller
        self.orbit_start_pub = self.create_publisher(Bool, "/control/orbit_start", 10)
        self.orbit_target_pub = self.create_publisher(Point, "/control/orbit_target", 10)
        
        # Command navigasi lokal
        self.local_goal_pub = self.create_publisher(PoseStamped, "/navigation/local_goal", 10)
        self.fsm_status_pub = self.create_publisher(String, "/mission/fsm_state", 10)
        # Publisher untuk memperbarui status pohon ke Tree Mapper
        self.tree_update_pub = self.create_publisher(Tree, "/map/tree_update", 10)
        self.setpoint_enable_pub = self.create_publisher(Bool, '/control/setpoint_enabled', 10)

        # Timer FSM berjalan pada 10 Hz
        self.timer = self.create_timer(0.1, self.fsm_loop)
        self.get_logger().info("Mission State Machine (The Brain) Siap!")

    # --- Callbacks Sensor & Status ---
    def pose_cb(self, msg):
        self.current_pose = msg
        self.last_pose_time = self.get_clock().now()
        # Pose pertama sebelum takeoff menjadi home dinamis. Salin nilainya,
        # jangan simpan referensi message yang akan terus diperbarui.
        if self.home_pose is None and self.state in ("WAIT_START", "INIT", "WAIT_GUIDED", "WAIT_ARM"):
            self.home_pose = (
                msg.pose.position.x,
                msg.pose.position.y,
                quaternion_to_yaw(msg.pose.orientation)
            )
    def orbit_status_cb(self, msg): self.orbit_status = msg.data
    def tree_cb(self, msg): self.trees = msg.trees
    
    # --- Callbacks Telemetri ---
    def arm_cb(self, msg): self.is_armed = msg.data
    def mode_cb(self, msg): self.current_mode = msg.data
    def hover_cb(self, msg): self.is_hovering = msg.data
    def start_cb(self, msg): self.start_requested = msg.data
    def safety_cb(self, msg): self.safety_ok = msg.data
    def safety_reason_cb(self, msg): self.safety_reason = msg.data

    def transition(self, state):
        self.state = state
        self.state_since = self.get_clock().now()

    def publish_setpoint_enabled(self, enabled):
        msg = Bool(); msg.data = enabled
        self.setpoint_enable_pub.publish(msg)

    # --- Helper Functions ---
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def current_yaw(self):
        return quaternion_to_yaw(self.current_pose.pose.orientation)

    def find_uninspected_tree(self):
        if self.current_pose is None: return None
        cx, cy = self.current_pose.pose.position.x, self.current_pose.pose.position.y
        
        best_tree = None
        min_dist = float('inf')

        for tree in self.trees:
            if not tree.inspected:
                dist = self.distance(cx, cy, tree.x, tree.y)
                is_ahead = (tree.x - cx) * self.explore_dir_x >= -1.0
                if is_ahead and dist < min_dist and dist < 15.0: 
                    min_dist = dist
                    best_tree = tree
        return best_tree

    def publish_goal(self, x, y, yaw):
        goal = PoseStamped()
        goal.header.frame_id = "odom"
        goal.header.stamp = self.get_clock().now().to_msg()
        
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        # flight_altitude adalah tinggi terhadap home untuk CommandTOL. Setelah
        # takeoff, pertahankan koordinat Z lokal yang benar-benar dicapai FC.
        goal.pose.position.z = (
            self.navigation_altitude
            if self.navigation_altitude is not None else self.flight_altitude
        )
        
        qx, qy, qz, qw = euler_to_quaternion(0, 0, yaw)
        goal.pose.orientation.x = qx
        goal.pose.orientation.y = qy
        goal.pose.orientation.z = qz
        goal.pose.orientation.w = qw
        
        self.local_goal_pub.publish(goal)

    # ==========================================
    # LOGIKA STATE MACHINE UTAMA
    # ==========================================
    def fsm_loop(self):
        if self.current_pose is None:
            return

        active = self.state not in ('WAIT_START', 'DONE', 'ABORT', 'MANUAL_OVERRIDE')
        navigation_states = (
            'POST_TAKEOFF_HOVER',
            'EXPLORE_ROW', 'APPROACH_TREE', 'VERIFY_TREE', 'START_ORBIT',
            'WAIT_ORBIT', 'POST_ORBIT_HOVER', 'ALIGN_HOME', 'END_OF_ROW',
            'CRAB_SCAN', 'RETURN_TO_HOME', 'HOME_HOVER', 'FINAL_SPIN')
        self.publish_setpoint_enabled(self.state in navigation_states)
        pose_age = float('inf') if self.last_pose_time is None else \
            (self.get_clock().now() - self.last_pose_time).nanoseconds * 1e-9
        if active and pose_age > self.pose_timeout:
            self.transition('ABORT')
            self.get_logger().error('ABORT: local pose kedaluwarsa.')
        if active and self.require_safety and not self.safety_ok:
            self.transition('ABORT')
            self.get_logger().error(f'ABORT watchdog: {self.safety_reason}')
        if active and self.is_armed and self.current_mode not in ('GUIDED', ''):
            self.transition('MANUAL_OVERRIDE')
            self.get_logger().warning(f'Manual takeover terdeteksi: mode={self.current_mode}')
        elapsed = (self.get_clock().now() - self.state_since).nanoseconds * 1e-9
        timeout_exempt = ('WAIT_START', 'EXPLORE_ROW', 'WAIT_ORBIT', 'LANDING', 'DONE',
                          'ABORT', 'MANUAL_OVERRIDE')
        if active and self.state not in timeout_exempt and elapsed > self.state_timeout:
            self.transition('ABORT')
            self.get_logger().error(f'ABORT: timeout state setelah {elapsed:.1f}s.')

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        msg = String(); msg.data = self.state
        self.fsm_status_pub.publish(msg)

        # --- FASE PRE-FLIGHT ---
        if self.state == 'WAIT_START':
            if self.start_requested and (self.safety_ok or not self.require_safety):
                # Capture terakhir tepat sebelum arm sebagai titik takeoff lokal.
                self.home_pose = (cx, cy, self.current_yaw())
                self.transition('INIT')
                self.get_logger().info('Start diterima dan sensor sehat; memulai preflight.')

        elif self.state == "INIT":
            mode_msg = String(); mode_msg.data = "GUIDED"
            self.cmd_mode_pub.publish(mode_msg)
            self.transition("WAIT_GUIDED")
            self.retry_counter = 0
            self.get_logger().info("Meminta transisi ke mode GUIDED...")

        elif self.state == "WAIT_GUIDED":
            if self.current_mode == "GUIDED":
                arm_msg = Bool(); arm_msg.data = True
                self.cmd_arm_pub.publish(arm_msg)
                self.transition("WAIT_ARM")
                self.retry_counter = 0
                self.get_logger().info("Mode GUIDED aktif. Meminta Arming Motor...")
            else:
                self.retry_counter += 1
                if self.retry_counter > 20:  # Ulangi perintah setiap 2 detik (20 x 0.1s)
                    mode_msg = String(); mode_msg.data = "GUIDED"
                    self.cmd_mode_pub.publish(mode_msg)
                    self.retry_counter = 0

        elif self.state == "WAIT_ARM":
            if self.is_armed:
                takeoff_msg = Float32(); takeoff_msg.data = self.flight_altitude
                self.cmd_takeoff_pub.publish(takeoff_msg)
                self.transition("WAIT_TAKEOFF")
                self.retry_counter = 0
                self.get_logger().info(f"Motor Bersenjata (Armed). Takeoff ke ketinggian {self.flight_altitude}m...")
            else:
                self.retry_counter += 1
                if self.retry_counter > 20:  # Ulangi perintah setiap 2 detik
                    arm_msg = Bool(); arm_msg.data = True
                    self.cmd_arm_pub.publish(arm_msg)
                    self.get_logger().info("Mencoba Arming ulang... (Menunggu Pre-arm good dari ArduPilot)")
                    self.retry_counter = 0

        elif self.state == "WAIT_TAKEOFF":
            if self.is_hovering:
                self.hold_x = cx
                self.hold_y = cy
                self.hold_yaw = self.current_yaw()
                self.navigation_altitude = self.current_pose.pose.position.z
                self.last_tree_x = cx
                self.last_tree_y = cy
                self.transition("POST_TAKEOFF_HOVER")
                self.get_logger().info(
                    "Altitude takeoff tercapai. Menahan posisi selama "
                    f"{self.post_takeoff_hover_time:.1f} detik.")

        elif self.state == "POST_TAKEOFF_HOVER":
            self.publish_goal(self.hold_x, self.hold_y, self.hold_yaw)
            if elapsed >= self.post_takeoff_hover_time:
                self.transition("EXPLORE_ROW")
                self.get_logger().info(
                    "Hover pasca-takeoff selesai. Mulai EXPLORE_ROW "
                    "(mencari pohon).")

        # --- FASE MISI UTAMA ---
        elif self.state == "EXPLORE_ROW":
            self.target_tree = self.find_uninspected_tree()
            
            if self.target_tree is not None:
                self.verification_retries = 0
                self.transition("APPROACH_TREE")
                self.get_logger().info(f"Pohon ditemukan di ({self.target_tree.x:.1f}, {self.target_tree.y:.1f})")
            else:
                target_x = cx + (self.explore_speed * self.explore_dir_x)
                target_yaw = 0.0 if self.explore_dir_x > 0 else math.pi
                self.publish_goal(target_x, cy, target_yaw)

                dist_from_last = self.distance(cx, cy, self.last_tree_x, self.last_tree_y)
                if dist_from_last > self.end_of_row_dist:
                    self.transition("END_OF_ROW")
                    self.get_logger().info("Lorong Habis. Bersiap pindah lorong.")

        elif self.state == "APPROACH_TREE":
            # Hitung sudut arah (yaw) dari drone menuju pohon
            target_yaw = math.atan2(self.target_tree.y - cy, self.target_tree.x - cx)
            
            # 1. Kalkulasi TITIK PENGEREMAN (2 meter di depan pohon)
            stop_x = self.target_tree.x - (self.approach_safe_dist * math.cos(target_yaw))
            stop_y = self.target_tree.y - (self.approach_safe_dist * math.sin(target_yaw))
            
            # 2. Hitung jarak drone ke TITIK PENGEREMAN (bukan ke pohon)
            dist_to_stop = self.distance(cx, cy, stop_x, stop_y)
            
            # Arrival tolerance harus lebih kecil daripada toleransi verifikasi.
            # Nilai lama 0.6 m membuat drone mulai verifikasi terlalu jauh.
            if dist_to_stop > self.approach_goal_tolerance:
                self.publish_goal(stop_x, stop_y, target_yaw)
            else:
                self.transition("VERIFY_TREE")
                self.hover_timer = 0
                self.get_logger().info("Titik pengereman tercapai. Hovering 4 detik untuk stabilisasi...")
        
        elif self.state == "VERIFY_TREE":
            # 1. Tahan posisi (Hovering) menghadap arah pohon target
            target_yaw = math.atan2(self.target_tree.y - cy, self.target_tree.x - cx)
            self.publish_goal(cx, cy, target_yaw)
            
            self.hover_timer += 1
            
            # Setelah 40 siklus (4 detik hovering stabil)
            if self.hover_timer >= 40:
                
                # --- CARI POHON BERDASARKAN ID ASLI SECARA KETAT ---
                target_matched_tree = None
                for tree in self.trees:
                    if tree.id == self.target_tree.id:
                        target_matched_tree = tree
                        break
                
                # Cek 1: Apakah ID pohon tersebut masih ada di database mapper?
                if target_matched_tree is not None:
                    
                    # Cek 2: HITUNG JARAK RIILL AKTUAL DARI DRONE KE POHON TERSEBUT
                    actual_dist_to_tree = self.distance(cx, cy, target_matched_tree.x, target_matched_tree.y)
                    
                    min_verify = max(
                        0.1, self.approach_safe_dist - self.tree_distance_tolerance)
                    max_verify = (
                        self.approach_safe_dist + self.tree_distance_tolerance)
                    if min_verify <= actual_dist_to_tree <= max_verify:
                        self.target_tree = target_matched_tree  
                        self.transition("START_ORBIT")
                        self.get_logger().info(f"Verifikasi sukses! Pohon ID:{target_matched_tree.id} valid di jarak {actual_dist_to_tree:.2f}m. Memulai orbit.")
                    else:
                        self.verification_retries += 1
                        if self.verification_retries <= self.verification_retry_limit:
                            self.target_tree = target_matched_tree
                            self.hover_timer = 0
                            self.transition("APPROACH_TREE")
                            self.get_logger().warning(
                                f"Pohon ID:{target_matched_tree.id} di {actual_dist_to_tree:.2f}m, "
                                f"di luar rentang {min_verify:.2f}..{max_verify:.2f}m; "
                                f"koreksi approach {self.verification_retries}/"
                                f"{self.verification_retry_limit}.")
                        else:
                            self.get_logger().warning(
                                f"Pohon ID:{target_matched_tree.id} gagal verifikasi "
                                f"{self.verification_retries} kali; dihapus.")
                            update_msg = Tree()
                            update_msg.id = target_matched_tree.id
                            update_msg.confidence = -1.0
                            self.tree_update_pub.publish(update_msg)
                            self.target_tree = None
                            self.transition("EXPLORE_ROW")
                        
                else:
                    self.get_logger().warn("Pohon Hantu hilang dari peta saat hovering! Membatalkan orbit.")
                    if self.target_tree is not None:
                        update_msg = Tree()
                        update_msg.id = self.target_tree.id
                        update_msg.confidence = -1.0 
                        self.tree_update_pub.publish(update_msg)
                    
                    self.target_tree = None
                    self.transition("EXPLORE_ROW")
                    
        elif self.state == "START_ORBIT":
            target_msg = Point()
            target_msg.x = self.target_tree.x
            target_msg.y = self.target_tree.y
            # Kirim Z lokal misi, bukan tinggi/geometri pusat pohon.
            target_msg.z = float(self.navigation_altitude)
            self.orbit_target_pub.publish(target_msg)
            
            start_msg = Bool(); start_msg.data = True
            self.orbit_start_pub.publish(start_msg)
            self.transition("WAIT_ORBIT")

        elif self.state == "WAIT_ORBIT":
            if self.orbit_status == "ORBIT_COMPLETED":
                # 1. Matikan perintah orbit
                stop_msg = Bool(); stop_msg.data = False
                self.orbit_start_pub.publish(stop_msg)
                
                # 2. UPDATE MAPPER: Tandai pohon ini SUDAH DIINSPEKSI
                if self.target_tree is not None:
                    update_msg = Tree()
                    update_msg.id = self.target_tree.id
                    update_msg.x = self.target_tree.x
                    update_msg.y = self.target_tree.y
                    update_msg.z = self.target_tree.z
                    update_msg.confidence = self.target_tree.confidence
                    
                    # INI KUNCI UTAMANYA:
                    update_msg.inspected = True 
                    
                    self.tree_update_pub.publish(update_msg)
                    self.get_logger().info(f"Pohon ID:{self.target_tree.id} ditandai SELESAI (Inspected).")

                # Misi hanya menginspeksi satu pohon. Tahan posisi akhir orbit
                # sebelum menghadap dan kembali ke titik takeoff.
                self.hold_x = cx
                self.hold_y = cy
                self.hold_yaw = self.current_yaw()
                self.hover_timer = 0
                self.target_tree = None
                self.transition("POST_ORBIT_HOVER")
                self.get_logger().info(
                    "Orbit satu pohon selesai. Hover sebelum kembali ke titik takeoff."
                )
            elif self.orbit_status.startswith('ORBIT_FAILED'):
                self.transition('ABORT')
                self.get_logger().error(f'ABORT: {self.orbit_status}')

        elif self.state == "POST_ORBIT_HOVER":
            self.publish_goal(self.hold_x, self.hold_y, self.hold_yaw)
            self.hover_timer = self.hover_timer + 1 if self.is_hovering else 0

            required_ticks = int(MissionConfig.POST_ORBIT_HOVER_TIME / 0.1)
            if self.hover_timer >= required_ticks:
                self.hover_timer = 0
                self.transition("ALIGN_HOME")
                self.get_logger().info("Hover stabil. Menyesuaikan yaw menuju home.")

        elif self.state == "ALIGN_HOME":
            if self.home_pose is None:
                self.get_logger().error("Home belum tersimpan; menahan posisi untuk keselamatan.")
                self.publish_goal(self.hold_x, self.hold_y, self.hold_yaw)
                return

            home_x, home_y, _ = self.home_pose
            yaw_to_home = math.atan2(home_y - cy, home_x - cx)
            self.publish_goal(self.hold_x, self.hold_y, yaw_to_home)

            yaw_error = abs(self.normalize_angle(yaw_to_home - self.current_yaw()))
            aligned = yaw_error <= MissionConfig.HOME_YAW_TOLERANCE
            self.hover_timer = self.hover_timer + 1 if aligned and self.is_hovering else 0

            required_ticks = int(MissionConfig.HOME_ALIGN_TIME / 0.1)
            if self.hover_timer >= required_ticks:
                self.hover_timer = 0
                self.transition("RETURN_TO_HOME")
                self.get_logger().info("Arah ke home stabil. Mulai kembali ke titik takeoff.")

        elif self.state == "END_OF_ROW":
            retreat_x = self.last_tree_x - (self.approach_safe_dist * self.explore_dir_x)
            target_yaw = 0.0 if self.explore_dir_x > 0 else math.pi
            
            self.publish_goal(retreat_x, self.last_tree_y, target_yaw)
            
            if abs(cx - retreat_x) < 0.5:
                self.explore_dir_x *= -1.0 
                self.crab_start_y = cy
                self.transition("CRAB_SCAN")
                self.get_logger().info("Mundur selesai. Memulai Crab Scan 90 derajat.")

        elif self.state == "CRAB_SCAN":
            target_y = cy + (self.crab_speed * self.explore_dir_y)
            target_yaw = 0.0 if self.explore_dir_x > 0 else math.pi
            self.publish_goal(cx, target_y, target_yaw)
            
            self.target_tree = self.find_uninspected_tree()
            if self.target_tree is not None:
                self.last_tree_y = self.target_tree.y
                self.transition("APPROACH_TREE")
                self.get_logger().info("Lorong baru ditemukan!")
            else:
                if abs(cy - self.crab_start_y) > self.end_of_farm_dist:
                    self.transition("RETURN_TO_HOME")
                    self.get_logger().info("Lahan habis. Cari jalur untuk pulang (RTH).")

        elif self.state == "RETURN_TO_HOME":
            if self.home_pose is None:
                return

            home_x, home_y, home_yaw = self.home_pose
            target_yaw = math.atan2(home_y - cy, home_x - cx)
            self.publish_goal(home_x, home_y, target_yaw)

            if self.distance(cx, cy, home_x, home_y) < MissionConfig.HOME_POSITION_TOLERANCE:
                self.hold_yaw = home_yaw
                self.hover_timer = 0
                self.transition("HOME_HOVER")
                self.get_logger().info("Tiba di titik takeoff. Hover sebelum landing.")

        elif self.state == "HOME_HOVER":
            home_x, home_y, _ = self.home_pose
            self.publish_goal(home_x, home_y, self.hold_yaw)
            self.hover_timer = self.hover_timer + 1 if self.is_hovering else 0

            required_ticks = int(MissionConfig.HOME_HOVER_TIME / 0.1)
            if self.hover_timer >= required_ticks:
                self.transition("LANDING")
                self.get_logger().info("Hover home selesai. Memulai pendaratan.")

        elif self.state == "FINAL_SPIN":
            qx = self.current_pose.pose.orientation.x
            qy = self.current_pose.pose.orientation.y
            qz = self.current_pose.pose.orientation.z
            qw = self.current_pose.pose.orientation.w
            current_yaw = math.atan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
            
            delta = current_yaw - self.last_yaw
            if delta > math.pi: delta -= 2 * math.pi
            elif delta < -math.pi: delta += 2 * math.pi
            
            self.spin_accumulated += abs(delta)
            self.last_yaw = current_yaw
            
            self.target_tree = self.find_uninspected_tree()
            if self.target_tree is not None:
                self.transition("APPROACH_TREE")
                self.get_logger().info("Pohon terlewat ditemukan saat Final Spin!")
                return
                
            if self.spin_accumulated >= 2 * math.pi:
                self.transition("LANDING")
                self.get_logger().info("Area bersih. Memulai Pendaratan.")
            else:
                target_yaw = current_yaw + 0.2
                self.publish_goal(cx, cy, target_yaw)

        elif self.state == "LANDING":
            land_msg = Bool(); land_msg.data = True
            self.cmd_land_pub.publish(land_msg)
            # Ulangi command sampai kendaraan benar-benar turun atau disarm,
            # agar satu pesan yang hilang tidak menggagalkan landing.
            if not self.is_armed or self.current_pose.pose.position.z < 0.20:
                self.transition("DONE")
                self.get_logger().info("Landing selesai. Misi DONE.")
            
        elif self.state == "DONE":
            self.publish_setpoint_enabled(False)

        elif self.state == 'ABORT':
            self.publish_setpoint_enabled(False)
            stop = Bool(); stop.data = False
            self.orbit_start_pub.publish(stop)
            # BRAKE meminta flight controller menghentikan kendaraan saat estimasi masih tersedia.
            mode = String(); mode.data = 'BRAKE'
            self.cmd_mode_pub.publish(mode)

        elif self.state == 'MANUAL_OVERRIDE':
            # Tidak mengirim setpoint/mode apa pun; pilot RC memegang kendali penuh.
            self.publish_setpoint_enabled(False)

def main(args=None):
    rclpy.init(args=args)
    node = MissionStateMachine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
