#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from beehive_drone.mission_params import MissionConfig
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String
from uav_interfaces.msg import TreeArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class VortexAvoidanceController(Node):
    def __init__(self):
        super().__init__("vortex_avoidance_controller")

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
        # Parameter Medan Pusaran (Vortex & Potential Field)
        # ==========================================
        self.safety_radius = MissionConfig.SAFETY_RADIUS       # Batas aman drone bereaksi (meter)
        self.repulsive_gain = MissionConfig.REPULSIVE_GAIN      # Kekuatan gaya tolak menjauhi halangan
        self.vortex_gain = MissionConfig.VORTEX_GAIN         # Kekuatan gaya geser/meliuk (tangensial)
        self.attraction_gain = MissionConfig.ATTRACTION_GAIN     # Tarikan ke tujuan asli
        
        self.max_shift = MissionConfig.MAX_SHIFT           # Batas maksimal pergeseran vektor (m)

        # ==========================================
        # Variabel State
        # ==========================================
        self.current_pose = None
        self.fsm_state = "INIT"
        self.trees = []                # Database obstacle statis
        self.branches = []             # (Opsional) Ranting dinamis dari deteksi depth
        
        self.fsm_goal = None
        self.orbit_goal = None

        # ==========================================
        # Subscriber
        # ==========================================
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_sensor)
        self.fsm_state_sub = self.create_subscription(String, "/mission/fsm_state", self.fsm_state_cb, 10)
        self.tree_sub = self.create_subscription(TreeArray, "/map/trees", self.tree_cb, qos_map)
        
        # Menerima "niat/target" dari FSM atau Orbit
        self.fsm_goal_sub = self.create_subscription(PoseStamped, "/navigation/local_goal", self.fsm_goal_cb, 10)
        self.orbit_goal_sub = self.create_subscription(PoseStamped, "/control/dynamic_target", self.orbit_goal_cb, 10)
        
        # (Opsional) Subscriber untuk deteksi ranting melintang dari kamera ZED 2i
        # self.branch_sub = self.create_subscription(Point, "/perception/obstacle_point", self.branch_cb, 10)

        # ==========================================
        # Publisher
        # ==========================================
        # Mengirim Point target yang SUDAH AMAN ke velocity_controller.py
        self.safe_target_pub = self.create_publisher(PoseStamped, "/control/safe_target_pose", 10)

        # Timer berjalan pada 20 Hz (sinkron dengan kontroler gerak)
        self.timer = self.create_timer(0.05, self.vortex_control_loop)

        self.get_logger().info("Vortex Avoidance (Potential Field) Controller Siap!")

    # --- Callbacks ---
    def pose_cb(self, msg): self.current_pose = msg
    def fsm_state_cb(self, msg): self.fsm_state = msg.data
    def tree_cb(self, msg): self.trees = msg.trees
    def fsm_goal_cb(self, msg): self.fsm_goal = msg
    def orbit_goal_cb(self, msg): self.orbit_goal = msg

    # --- Logika Matematika Vortex ---
    def vortex_control_loop(self):
        if self.current_pose is None:
            return

        cx = self.current_pose.pose.position.x
        cy = self.current_pose.pose.position.y
        cz = self.current_pose.pose.position.z

        # 1. Tentukan Goal Mana yang Sedang Aktif
        active_goal = None
        if self.fsm_state in ["START_ORBIT", "WAIT_ORBIT"]:
            active_goal = self.orbit_goal
        else:
            active_goal = self.fsm_goal

        if active_goal is None:
            return

        target_x = active_goal.pose.position.x
        target_y = active_goal.pose.position.y
        target_z = active_goal.pose.position.z

        # 2. Vektor Tarikan (Attractive Vector) ke Tujuan Asli
        # Dihitung relatif terhadap posisi drone saat ini
        dist_to_goal = math.sqrt((target_x - cx)**2 + (target_y - cy)**2)
        if dist_to_goal > 0.1:
            # NORMALISASI VEKTOR: Tarikan selalu konstan (kekuatannya = attraction_gain)
            # Tidak peduli sejauh apa pun targetnya, tarikan tidak akan pernah membesar.
            att_dx = ((target_x - cx) / dist_to_goal) * self.attraction_gain
            att_dy = ((target_y - cy) / dist_to_goal) * self.attraction_gain
        else:
            att_dx = 0.0
            att_dy = 0.0
        # 3. Kalkulasi Vektor Penolakan (Repulsive) & Pusaran (Vortex)
        rep_dx = 0.0
        rep_dy = 0.0
        vort_dx = 0.0
        vort_dy = 0.0

        for tree in self.trees:
            dx_obs = cx - tree.x  # Arah dari rintangan KE drone
            dy_obs = cy - tree.y
            
            dist_to_obs = math.sqrt(dx_obs**2 + dy_obs**2)

            if dist_to_obs < self.safety_radius and dist_to_obs > 0:
                self.get_logger().warn(f"SAFETY TRIGGERED! Menghindari objek pada jarak {dist_to_obs:.2f}m")
                
                # Jarak penetrasi ke dalam zona bahaya
                penetration = self.safety_radius - dist_to_obs
                
                # Sudut tolakan
                push_angle = math.atan2(dy_obs, dx_obs)

                # A. Gaya Tolak Normal (Repulsive Force) - Mencegah tabrakan frontal
                force = penetration * self.repulsive_gain
                rep_dx += math.cos(push_angle) * force
                rep_dy += math.sin(push_angle) * force

                # B. Gaya Tangensial (Vortex Force) - Membanting kemudi menyamping
                # Pusaran sudut: -90 derajat (Clockwise) agar mengalir di sekitar objek
                vortex_angle = push_angle - (math.pi / 2.0)
                v_force = penetration * self.vortex_gain
                
                vort_dx += math.cos(vortex_angle) * v_force
                vort_dy += math.sin(vortex_angle) * v_force

        # 4. Resultan Vektor Akhir
        final_dx = att_dx + rep_dx + vort_dx
        final_dy = att_dy + rep_dy + vort_dy

        # Membatasi pergeseran target ekstrem agar drone tidak terguling
        shift_mag = math.sqrt(final_dx**2 + final_dy**2)
        if shift_mag > self.max_shift:
            final_dx = (final_dx / shift_mag) * self.max_shift
            final_dy = (final_dy / shift_mag) * self.max_shift

        safe_pose = PoseStamped()
        safe_pose.header.frame_id = "odom"
        safe_pose.header.stamp = self.get_clock().now().to_msg()

        # 1. Memasukkan Posisi (X,Y,Z) yang sudah aman dari gaya tolak
        safe_pose.pose.position.x = cx + final_dx
        safe_pose.pose.position.y = cy + final_dy
        safe_pose.pose.position.z = float(target_z)

        # 2. Meneruskan Orientasi/Sudut Kamera ASLI dari FSM atau Orbit
        safe_pose.pose.orientation = active_goal.pose.orientation

        self.safe_target_pub.publish(safe_pose)

def main(args=None):
    rclpy.init(args=args)
    node = VortexAvoidanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()