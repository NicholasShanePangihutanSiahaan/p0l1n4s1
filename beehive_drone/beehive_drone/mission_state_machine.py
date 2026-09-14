#!/usr/bin/env python3

import math
from copy import deepcopy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from beehive_drone.mission_params import MissionConfig
from beehive_drone.missions import MISSION_STRATEGIES
from geometry_msgs.msg import PoseStamped, Point, Pose
from std_msgs.msg import Bool, String, Float32
from uav_interfaces.msg import TreeArray, Tree, ActiveTree

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


def landing_command_due(current_mode, last_command_age, retry_interval):
    """Return whether the LAND request may be sent without flooding MAVROS."""
    return current_mode != 'LAND' and last_command_age >= retry_interval


def yaw_aligned(current_yaw, target_yaw, tolerance):
    """Return true when the shortest yaw error is within tolerance."""
    return abs(math.atan2(
        math.sin(target_yaw - current_yaw),
        math.cos(target_yaw - current_yaw))) <= tolerance

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
        # Strategy Registry Lookup
        # ==========================================
        self.declare_parameter('mission_type', 'flower_mission')
        mission_type = str(self.get_parameter('mission_type').value)
        
        strategy_cls = MISSION_STRATEGIES.get(mission_type)
        if strategy_cls is None:
            valid_keys = list(MISSION_STRATEGIES.keys())
            self.get_logger().error(
                f"Invalid mission_type: '{mission_type}'. Registered strategies: {valid_keys}"
            )
            raise ValueError(f"Unknown mission strategy '{mission_type}'")

        self.mission_strategy = strategy_cls()
        self.get_logger().info(f"Loaded mission strategy: '{mission_type}'")

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
        self.declare_parameter('orbit_radius', 3.0)
        self.declare_parameter('post_takeoff_hover_time', 2.0)
        self.declare_parameter('require_vision_before_start', False)
        self.declare_parameter(
            'vision_pose_topic', '/mavros/vision_pose/pose')
        self.declare_parameter('vision_pose_timeout', 1.0)
        self.declare_parameter('vision_warmup_time', 5.0)
        self.declare_parameter('landing_retry_interval', 1.0)
        self.declare_parameter('align_yaw_tolerance_degrees', 7.5)
        self.declare_parameter('align_yaw_hold_time', 1.5)
        self.declare_parameter('require_frame_alignment', False)
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
        self.require_vision_before_start = bool(
            self.get_parameter('require_vision_before_start').value)
        self.vision_pose_topic = str(
            self.get_parameter('vision_pose_topic').value)
        self.vision_pose_timeout = max(
            0.1, float(self.get_parameter('vision_pose_timeout').value))
        self.vision_warmup_time = max(
            0.0, float(self.get_parameter('vision_warmup_time').value))
        self.landing_retry_interval = max(
            0.5, float(self.get_parameter('landing_retry_interval').value))
        self.align_yaw_tolerance = math.radians(max(
            1.0, float(self.get_parameter(
                'align_yaw_tolerance_degrees').value)))
        self.align_yaw_hold_time = max(
            0.5, float(self.get_parameter('align_yaw_hold_time').value))
        self.require_frame_alignment = bool(
            self.get_parameter('require_frame_alignment').value)
        self.orbit_radius = float(self.get_parameter('orbit_radius').value)

        # ==========================================
        # Variabel State & Navigasi
        # ==========================================
        self.previous_state = None
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
        self.abort_command_sent = False
        self.last_landing_command_time = None
        self.trees = []
        self.target_tree = None
        self.frozen_target_tree = None
        self.align_yaw_since = None
        self.first_vision_time = None
        self.last_vision_time = None
        self.last_vision_wait_log = None
        self.frame_alignment_ready = not self.require_frame_alignment
        
        # Variabel deteksi bunga
        self.receiving_flower_pose = False
        self.done_receiving_flower_pose = False # memastikan cuman sekali terima data flower untuk sekali orbit
        self.flower_pose = None

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
        self.flower_sub = self.create_subscription(Pose, "/mission/current_flower", self.flower_cb, 10)
        self.orbit_status_sub = self.create_subscription(String, "/control/orbit_status", self.orbit_status_cb, 10)
        self.tree_sub = self.create_subscription(TreeArray, "/map/trees", self.tree_cb, qos_map)
        self.create_subscription(
            PoseStamped, self.vision_pose_topic, self.vision_pose_cb, qos_sensor)

        # Telemetri dari Flight Manager
        self.telemetry_arm_sub = self.create_subscription(Bool, "/flight/telemetry/is_armed", self.arm_cb, 10)
        self.telemetry_mode_sub = self.create_subscription(String, "/flight/telemetry/current_mode", self.mode_cb, 10)
        self.telemetry_hover_sub = self.create_subscription(Bool, "/flight/telemetry/is_hovering", self.hover_cb, 10)
        self.create_subscription(Bool, '/mission/start', self.start_cb, 10)
        self.create_subscription(Bool, '/mission/safety_ok', self.safety_cb, 10)
        self.create_subscription(String, '/mission/safety_reason', self.safety_reason_cb, 10)
        self.create_subscription(Bool, '/alignment/ready', self.alignment_cb, 10)

        # ==========================================
        # Publisher
        # ==========================================
        
        # Pohon saat ini
        self.active_tree_pub = self.create_publisher(ActiveTree, "/mission/current_tree", 10)
        
        # Command ke Flight Manager
        self.cmd_mode_pub = self.create_publisher(String, "/flight/cmd/set_mode", 10)
        self.cmd_arm_pub = self.create_publisher(Bool, "/flight/cmd/set_arm", 10)
        self.cmd_takeoff_pub = self.create_publisher(Float32, "/flight/cmd/takeoff", 10)
        self.cmd_land_pub = self.create_publisher(Bool, "/flight/cmd/land", 10)
        
        # Command ke Dynamic Orbit Controller
        self.orbit_start_pub = self.create_publisher(Bool, "/control/orbit_start", 10)
        self.orbit_pause_pub = self.create_publisher(Bool, "/control/orbit_pause", 10)
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
    def alignment_cb(self, msg): self.frame_alignment_ready = bool(msg.data)
    
    def flower_cb(self,msg):
        if self.done_receiving_flower_pose == False:
            self.receiving_flower_pose = True
            self.flower_pose = deepcopy(msg)

    def vision_pose_cb(self, _msg):
        now = self.get_clock().now()
        if self.last_vision_time is None or (
                now - self.last_vision_time).nanoseconds * 1e-9 > \
                self.vision_pose_timeout:
            self.first_vision_time = now
        self.last_vision_time = now

    def vision_ready(self):
        if not self.require_vision_before_start:
            return True
        if self.first_vision_time is None or self.last_vision_time is None:
            return False
        now = self.get_clock().now()
        age = (now - self.last_vision_time).nanoseconds * 1e-9
        warmup = (now - self.first_vision_time).nanoseconds * 1e-9
        return age <= self.vision_pose_timeout and warmup >= self.vision_warmup_time

    def log_vision_wait(self):
        now = self.get_clock().now()
        if self.last_vision_wait_log is not None and (
                now - self.last_vision_wait_log).nanoseconds * 1e-9 < 2.0:
            return
        self.last_vision_wait_log = now
        if self.last_vision_time is None:
            detail = f'belum ada data {self.vision_pose_topic}'
        else:
            age = (now - self.last_vision_time).nanoseconds * 1e-9
            warmup = (
                (now - self.first_vision_time).nanoseconds * 1e-9
                if self.first_vision_time is not None else 0.0)
            detail = (
                f'age={age:.2f}s, warm-up={warmup:.1f}/'
                f'{self.vision_warmup_time:.1f}s')
        self.get_logger().warning(
            f'WAIT_START: menunggu vision bridge stabil ({detail}).')
    
    # --- Callbacks Telemetri ---
    def arm_cb(self, msg): self.is_armed = msg.data
    def mode_cb(self, msg): self.current_mode = msg.data
    def hover_cb(self, msg): self.is_hovering = msg.data
    def start_cb(self, msg): self.start_requested = msg.data
    def safety_cb(self, msg): self.safety_ok = msg.data
    def safety_reason_cb(self, msg): self.safety_reason = msg.data

    def transition(self, state):
        self.previous_state = self.state
        self.state = state
        self.state_since = self.get_clock().now()
        if state == 'ABORT':
            self.abort_command_sent = False
        if state == 'LANDING':
            self.last_landing_command_time = None
        if state != 'ALIGN_TO_TREE':
            self.align_yaw_since = None

    def publish_setpoint_enabled(self, enabled):
        msg = Bool(); msg.data = enabled
        self.setpoint_enable_pub.publish(msg)

    # --- Helper Functions ---
    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))
    
    def pub_current_tree(self, active_tree, is_orbiting=False):
        current_tree_orb = ActiveTree()
        current_tree_orb.is_currenty_orbiting = is_orbiting
        current_tree_orb.tree = deepcopy(active_tree)

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

        active = self.state not in ('WAIT_START', 'DONE', 'ABORT', 'MANUAL_OVERRIDE', 'MANUAL_SPRAY')
        self.publish_setpoint_enabled(self.state in self.mission_strategy.navigation_states)
        pose_age = float('inf') if self.last_pose_time is None else \
            (self.get_clock().now() - self.last_pose_time).nanoseconds * 1e-9
        if active and pose_age > self.pose_timeout:
            self.transition('ABORT')
            self.get_logger().error('ABORT: local pose kedaluwarsa.')
        if active and self.require_safety and not self.safety_ok:
            self.transition('ABORT')
            self.get_logger().error(f'ABORT watchdog: {self.safety_reason}')
        if active and self.require_frame_alignment and not self.frame_alignment_ready:
            self.transition('ABORT')
            self.get_logger().error('ABORT: alignment ZED-FC hilang/reset.')
        # ArduPilot berpindah dari GUIDED ke LAND setelah command landing. LAND
        # adalah bagian normal misi, bukan takeover pilot.
        expected_land_mode = self.state == 'LANDING' and self.current_mode == 'LAND'
        if active and self.is_armed and not expected_land_mode and \
                self.current_mode not in ('GUIDED', ''):
            if self.state == 'FLOWER_ALIGNED':
                # Drone operator spray secara manual
                self.transition('MANUAL_SPRAY')
                self.get_logger().info(f'Switch mode terdeteksi di FLOWER_ALIGNED! Masuk ke MANUAL_SPRAY (mode={self.current_mode})')
            else:
                # Drone operator override
                self.transition('MANUAL_OVERRIDE')
                self.get_logger().warning(f'Manual takeover terdeteksi: mode={self.current_mode}')
        elif self.state == 'MANUAL_SPRAY' and self.current_mode == 'GUIDED':
            self.transition('ALIGN_TO_LAST_ORBIT')
            self.get_logger().info('Mode kembali ke GUIDED dari MANUAL_SPRAY! Berpindah ke ALIGN_TO_LAST_ORBIT.')
        elapsed = (self.get_clock().now() - self.state_since).nanoseconds * 1e-9
        
        if active and self.state not in self.mission_strategy.timeout_exempt_states and elapsed > self.state_timeout:
            self.transition('ABORT')
            self.get_logger().error(f'ABORT: timeout state setelah {elapsed:.1f}s.')

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y

        msg = String(); msg.data = self.state
        self.fsm_status_pub.publish(msg)
        
        if active and self.state in ('ALIGN_TO_TREE', 'APPROACH_TREE', 'VERIFY_TREE', 'START_ORBIT', 'WAIT_ORBIT'):
            self.pub_current_tree(self.target_tree, True)
        else:
            self.pub_current_tree(None, False)
        
        self.mission_strategy.execute(self,active,elapsed,cx,cy)

def main(args=None):
    rclpy.init(args=args)
    node = MissionStateMachine()
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

if __name__ == "__main__":
    main()