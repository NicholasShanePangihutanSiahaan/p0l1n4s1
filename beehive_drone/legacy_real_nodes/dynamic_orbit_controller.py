#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from beehive_drone.mission_params import MissionConfig
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def euler_to_quaternion(roll, pitch, yaw):
    """Fungsi pembantu untuk konversi sudut Euler ke Quaternion ROS"""
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return qx, qy, qz, qw

class DynamicOrbitController(Node):
    def __init__(self):
        super().__init__("dynamic_orbit_controller")

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==========================================
        # Parameter Orbit Dinamis
        # ==========================================
        self.orbit_radius = MissionConfig.ORBIT_RADIUS       # Jarak ideal dari pohon (meter)
        self.orbit_altitude = MissionConfig.ORBIT_ALTITUDE     # Ketinggian orbit standar (meter)
        self.orbit_velocity = MissionConfig.ORBIT_VELOCITY     # Kecepatan rotasi angular ekuivalen linear (m/s)
        self.yaw_offset = MissionConfig.YAW_OFFSET # Offset 45 derajat (0.785 radian)

        # ==========================================
        # Variabel State
        # ==========================================
        self.is_orbiting = False
        self.tree_x = 0.0
        self.tree_y = 0.0
        self.tree_z = 0.0
        
        self.current_pose = None
        self.last_angle = None
        self.accumulated_angle = 0.0

        # ==========================================
        # Subscriber
        # ==========================================
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            "/mavros/local_position/pose", 
            self.pose_callback, 
            qos_sensor
        )
        self.target_sub = self.create_subscription(
            Point, 
            "/control/orbit_target", 
            self.target_callback, 
            10
        )
        self.start_sub = self.create_subscription(
            Bool, 
            "/control/orbit_start", 
            self.start_callback, 
            10
        )

        # ==========================================
        # Publisher
        # ==========================================
        # Mengirim setpoint ke lapisan Vortex Avoidance (bukan langsung ke MAVROS)
        self.setpoint_pub = self.create_publisher(
            PoseStamped, 
            "/control/dynamic_target", 
            10
        )
        self.status_pub = self.create_publisher(
            String, 
            "/control/orbit_status", 
            10
        )

        # Loop kontrol berjalan pada 20Hz (0.05 detik)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Dynamic Orbit Controller (45-Deg Yaw Offset) Aktif.")

    def pose_callback(self, msg):
        self.current_pose = msg

    def target_callback(self, msg):
        self.tree_x = msg.x
        self.tree_y = msg.y
        self.tree_z = msg.z

    def start_callback(self, msg):
        if msg.data and not self.is_orbiting:
            self.is_orbiting = True
            self.accumulated_angle = 0.0
            self.last_angle = None
            self.get_logger().info(f"Memulai orbit pada pohon di ({self.tree_x:.2f}, {self.tree_y:.2f})")
        elif not msg.data:
            self.is_orbiting = False
            self.get_logger().info("Orbit dibatalkan oleh State Machine.")

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def control_loop(self):
        if not self.is_orbiting or self.current_pose is None:
            return

        # 1. Kalkulasi posisi drone terhadap pusat pohon
        dx = self.current_pose.pose.position.x - self.tree_x
        dy = self.current_pose.pose.position.y - self.tree_y
        
        current_angle = math.atan2(dy, dx)
        current_r = math.sqrt(dx*dx + dy*dy)

        # 2. Akumulasi total sudut rotasi (mencapai 360 derajat / 2 PI)
        if self.last_angle is not None:
            delta = current_angle - self.last_angle
            # Normalisasi perbedaan sudut agar tetap dalam rentang -PI hingga PI
            if delta > math.pi: 
                delta -= 2 * math.pi
            elif delta < -math.pi: 
                delta += 2 * math.pi
            
            # Hanya catat progres putaran jika drone sudah berada di radius yang mendekati target
            if abs(current_r - self.orbit_radius) < 1.0:
                self.accumulated_angle += abs(delta)

        self.last_angle = current_angle

        # 3. Pengecekan status selesai
        if self.accumulated_angle >= 2 * math.pi:
            self.is_orbiting = False
            self.publish_status("ORBIT_COMPLETED")
            self.get_logger().info("Orbit 360 derajat selesai.")
            return
        else:
            self.publish_status("IN_PROGRESS")

        # 4. Kalkulasi Setpoint Target Posisi (Translasi)
        # Gunakan konsep "Carrot on a Stick" (Umpan Jauh).
        # Target harus dilempar cukup jauh ke depan agar tidak terkena auto-rem (threshold 0.5m) dari velocity_controller.
        lookahead_distance = 1.5 # meter di depan lintasan
        lookahead_angle = lookahead_distance / self.orbit_radius
        
        target_angle = current_angle + lookahead_angle # Bergerak CCW (Berlawanan arah jarum jam)

        # Koreksi Spiral: Jika drone terlempar menjauh, tarik kembali perlahan (P-Controller kecil)
        target_r = current_r + (self.orbit_radius - current_r) * 0.15
        target_r = max(1.5, target_r) # Batas minimum aman agar tidak menabrak inti pohon

        target_x = self.tree_x + target_r * math.cos(target_angle)
        target_y = self.tree_y + target_r * math.sin(target_angle)

        # 5. Kalkulasi Orientasi Kamera 45 Derajat (Yaw Offset)
        # Sudut murni jika kamera melihat tepat ke titik pusat pohon:
        yaw_to_tree = math.atan2(self.tree_y - target_y, self.tree_x - target_x)
        
        # Karena kita bergerak CCW, untuk melihat sedikit ke depan lintasan sambil mengawasi pohon,
        # kita menggeser kamera sebesar -45 derajat dari titik pusat.
        target_yaw = yaw_to_tree - self.yaw_offset

        qx, qy, qz, qw = euler_to_quaternion(0, 0, target_yaw)

        # 6. Publikasikan target dinamis
        sp = PoseStamped()
        sp.header.frame_id = "odom"
        sp.header.stamp = self.get_clock().now().to_msg()
        
        sp.pose.position.x = target_x
        sp.pose.position.y = target_y
        sp.pose.position.z = self.orbit_altitude
        
        sp.pose.orientation.x = qx
        sp.pose.orientation.y = qy
        sp.pose.orientation.z = qz
        sp.pose.orientation.w = qw

        self.setpoint_pub.publish(sp)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicOrbitController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()