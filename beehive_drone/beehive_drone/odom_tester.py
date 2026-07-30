#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Bool, Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def euler_to_quaternion(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return qx, qy, qz, qw

class OdomTester(Node):
    def __init__(self):
        super().__init__("odom_tester")

        # ==========================================
        # Konfigurasi Misi Uji
        # ==========================================
        self.alt_target = 2.0      # Ketinggian 2 meter
        self.hover_duration = 5.0  # Waktu jeda di setiap titik (5 detik)
        self.dist_tolerance = 0.3  # Toleransi jarak sampai titik (meter)
        
        self.current_pose = None
        self.start_x = 0.0
        self.start_y = 0.0
        
        self.step = "INIT"
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "odom"
        
        self.hover_start_time = 0.0
        self.orbit_start_time = 0.0  # Timer khusus untuk Orbit
        self.retry_counter = 0

        # Variabel Telemetri
        self.is_armed = False
        self.current_mode = ""
        self.is_hovering = False

        # ==========================================
        # ROS 2 Interfaces
        # ==========================================
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_sensor)
        self.setpoint_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

        self.telemetry_arm_sub = self.create_subscription(Bool, "/flight/telemetry/is_armed", self.arm_cb, 10)
        self.telemetry_mode_sub = self.create_subscription(String, "/flight/telemetry/current_mode", self.mode_cb, 10)
        self.telemetry_hover_sub = self.create_subscription(Bool, "/flight/telemetry/is_hovering", self.hover_cb, 10)

        self.cmd_mode_pub = self.create_publisher(String, "/flight/cmd/set_mode", 10)
        self.cmd_arm_pub = self.create_publisher(Bool, "/flight/cmd/set_arm", 10)
        self.cmd_takeoff_pub = self.create_publisher(Float32, "/flight/cmd/takeoff", 10)
        self.cmd_land_pub = self.create_publisher(Bool, "/flight/cmd/land", 10)

        # Loop Kontrol Utama (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Odom Tester (Manuver Orbit) Aktif.")

    # --- Callbacks ---
    def pose_cb(self, msg): self.current_pose = msg
    def arm_cb(self, msg): self.is_armed = msg.data
    def mode_cb(self, msg): self.current_mode = msg.data
    def hover_cb(self, msg): self.is_hovering = msg.data

    def distance_to_target(self):
        if not self.current_pose: return float('inf')
        dx = self.target_pose.pose.position.x - self.current_pose.pose.position.x
        dy = self.target_pose.pose.position.y - self.current_pose.pose.position.y
        dz = self.target_pose.pose.position.z - self.current_pose.pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def trigger_hover(self, next_step_name):
        self.get_logger().info(f"Titik tercapai. Menahan posisi {self.hover_duration} detik...")
        self.hover_start_time = time.time()
        self.next_step_after_hover = next_step_name
        self.step = "HOVERING"

    def control_loop(self):
        if self.current_pose is None:
            return

        # FITUR SAFETY (PENGAMBILALIHAN MANUAL OLEH PILOT)
        if self.step not in ["INIT", "DONE", "MANUAL_OVERRIDE"]:
            if self.current_mode not in ["GUIDED", ""]:
                self.get_logger().warn(f"PENGAMBILALIHAN RC DETEKSI! Mode: {self.current_mode}.")
                self.step = "MANUAL_OVERRIDE"

        if self.step == "MANUAL_OVERRIDE":
            return

        # Publikasikan setpoint konstan kecuali saat di tanah / sedang mendarat
        if self.step not in ["INIT", "ARMING", "TAKEOFF", "WAIT_TAKEOFF", "LAND", "DONE", "MANUAL_OVERRIDE"]:
            self.target_pose.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(self.target_pose)

        # ========================================================
        # STATE MACHINE (FSM)
        # ========================================================
        if self.step == "INIT":
            if self.retry_counter % 20 == 0:
                mode_msg = String(); mode_msg.data = "GUIDED"
                self.cmd_mode_pub.publish(mode_msg)
            if self.current_mode == "GUIDED":
                self.step = "ARMING"
                self.retry_counter = 0
            self.retry_counter += 1

        elif self.step == "ARMING":
            if self.retry_counter % 20 == 0:
                arm_msg = Bool(); arm_msg.data = True
                self.cmd_arm_pub.publish(arm_msg)
            if self.is_armed:
                self.get_logger().info("Armed! Bersiap Takeoff...")
                
                self.start_pose = self.current_pose.pose
                self.start_x = self.start_pose.position.x
                self.start_y = self.start_pose.position.y
                
                self.target_pose.pose.position.x = self.start_x
                self.target_pose.pose.position.y = self.start_y
                self.target_pose.pose.position.z = self.alt_target
                self.target_pose.pose.orientation = self.start_pose.orientation
                
                self.step = "TAKEOFF"
                self.retry_counter = 0
            self.retry_counter += 1

        elif self.step == "TAKEOFF":
            takeoff_msg = Float32(); takeoff_msg.data = self.alt_target
            self.cmd_takeoff_pub.publish(takeoff_msg)
            self.get_logger().info(f"Takeoff ke {self.alt_target}m...")
            self.step = "WAIT_TAKEOFF"

        elif self.step == "WAIT_TAKEOFF":
            if self.is_hovering:
                self.trigger_hover("KIRI_2M")

        elif self.step == "HOVERING":
            if time.time() - self.hover_start_time > self.hover_duration:
                self.step = self.next_step_after_hover
                self.get_logger().info(f"Mulai manuver: {self.step}")

        # --- BLOK MANUVER LURUS ---
        elif self.step == "KIRI_2M":
            self.target_pose.pose.position.x = self.start_x
            self.target_pose.pose.position.y = self.start_y + 2.0
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("DEPAN_1M")

        elif self.step == "DEPAN_1M":
            self.target_pose.pose.position.x = self.start_x + 1.0
            self.target_pose.pose.position.y = self.start_y + 2.0
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("MULAI_ORBIT")

        # === BLOK ORBIT (LINGKARAN PENUH) ===
        elif self.step == "MULAI_ORBIT":
            self.get_logger().info("Memulai Orbit Radius 0.5m...")
            self.orbit_start_time = time.time()
            self.step = "ORBIT"
            
        elif self.step == "ORBIT":
            radius = 0.5
            # Titik Pusat Lingkaran (0.5 meter di depan drone saat ini)
            cx = self.start_x + 1.5
            cy = self.start_y + 2.0
            
            elapsed = time.time() - self.orbit_start_time
            angular_speed = 0.5  # Kecepatan putar (Radian per detik). ~12.5 detik utk 1 putaran penuh.
            
            # Dimulai dari Pi (180 derajat) agar titik awal orbit mulus tanpa lompat posisi
            theta = math.pi + (elapsed * angular_speed) 
            
            # Hitung koordinat X dan Y pada lingkaran
            orbit_x = cx + radius * math.cos(theta)
            orbit_y = cy + radius * math.sin(theta)
            
            # Hitung arah kepala drone (Yaw) agar kamera selalu menatap ke pusat lingkaran
            orbit_yaw = math.atan2(cy - orbit_y, cx - orbit_x)
            
            self.target_pose.pose.position.x = orbit_x
            self.target_pose.pose.position.y = orbit_y
            
            # Masukkan nilai Yaw baru ke Quaternion
            qx, qy, qz, qw = euler_to_quaternion(0, 0, orbit_yaw)
            self.target_pose.pose.orientation.x = qx
            self.target_pose.pose.orientation.y = qy
            self.target_pose.pose.orientation.z = qz
            self.target_pose.pose.orientation.w = qw
            
            # Cek apakah sudah berputar sejauh 360 derajat (2 * Pi)
            if elapsed * angular_speed >= 2 * math.pi:
                self.get_logger().info("Orbit 360 derajat selesai!")
                
                # Kunci posisinya tepat di titik asal sebelum mulai orbit
                self.target_pose.pose.position.x = self.start_x + 1.0
                self.target_pose.pose.position.y = self.start_y + 2.0
                
                # Kembalikan hadap kepala drone (Yaw) ke arah semula
                self.target_pose.pose.orientation = self.start_pose.orientation
                
                # Jeda sejenak sebelum mundur
                self.trigger_hover("BELAKANG_1M")
        # ====================================

        elif self.step == "BELAKANG_1M":
            self.target_pose.pose.position.x = self.start_x
            self.target_pose.pose.position.y = self.start_y + 2.0
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("KANAN_2M")
                
        elif self.step == "KANAN_2M":
            self.target_pose.pose.position.x = self.start_x
            self.target_pose.pose.position.y = self.start_y
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("LAND")

        elif self.step == "LAND":
            land_msg = Bool(); land_msg.data = True
            self.cmd_land_pub.publish(land_msg)
            self.step = "DONE"
            self.get_logger().info("Pendaratan dikirim. Misi Selesai.")

def main(args=None):
    rclpy.init(args=args)
    node = OdomTester()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()