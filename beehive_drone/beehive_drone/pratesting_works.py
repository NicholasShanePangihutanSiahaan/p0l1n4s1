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
        self.alt_target = 2.0      # Ketinggian diubah menjadi 2 meter
        self.hover_duration = 5.0  # Waktu hovering diubah menjadi 5 detik
        self.dist_tolerance = 0.3 
        
        self.current_pose = None
        self.start_x = 0.0
        self.start_y = 0.0
        
        self.step = "INIT"
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "odom"
        
        self.hover_start_time = 0.0
        self.retry_counter = 0

        # Variabel Telemetri dari Flight Manager
        self.is_armed = False
        self.current_mode = ""
        self.is_hovering = False

        # ==========================================
        # ROS 2 Interfaces (Menggunakan Flight Manager)
        # ==========================================
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # MAVROS Pose (Hanya untuk baca koordinat aktual & kirim Setpoint navigasi)
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_sensor)
        self.setpoint_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", 10)

        # TELEMETRI SUBSCRIBER (Dari Flight Manager)
        self.telemetry_arm_sub = self.create_subscription(Bool, "/flight/telemetry/is_armed", self.arm_cb, 10)
        self.telemetry_mode_sub = self.create_subscription(String, "/flight/telemetry/current_mode", self.mode_cb, 10)
        self.telemetry_hover_sub = self.create_subscription(Bool, "/flight/telemetry/is_hovering", self.hover_cb, 10)

        # COMMAND PUBLISHER (Ke Flight Manager)
        self.cmd_mode_pub = self.create_publisher(String, "/flight/cmd/set_mode", 10)
        self.cmd_arm_pub = self.create_publisher(Bool, "/flight/cmd/set_arm", 10)
        self.cmd_takeoff_pub = self.create_publisher(Float32, "/flight/cmd/takeoff", 10)
        self.cmd_land_pub = self.create_publisher(Bool, "/flight/cmd/land", 10)

        # Loop Kontrol Utama (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Odom Tester (via Flight Manager) Aktif.")

    # --- Callbacks ---
    def pose_cb(self, msg): self.current_pose = msg
    def arm_cb(self, msg): self.is_armed = msg.data
    def mode_cb(self, msg): self.current_mode = msg.data
    def hover_cb(self, msg): self.is_hovering = msg.data

    def set_target(self, x, y, z, yaw=0.0):
        self.target_pose.pose.position.x = float(x)
        self.target_pose.pose.position.y = float(y)
        self.target_pose.pose.position.z = float(z)
        
        qx, qy, qz, qw = euler_to_quaternion(0, 0, yaw)
        self.target_pose.pose.orientation.x = qx
        self.target_pose.pose.orientation.y = qy
        self.target_pose.pose.orientation.z = qz
        self.target_pose.pose.orientation.w = qw

    def trigger_hover(self, next_step_name):
        self.get_logger().info(f"Titik tercapai. Hovering {self.hover_duration} detik...")
        self.hover_start_time = time.time()
        self.next_step_after_hover = next_step_name
        self.step = "HOVERING"

    def control_loop(self):
        if self.current_pose is None:
            return
        if self.step not in ["INIT", "DONE", "MANUAL_OVERRIDE"]:
            if self.current_mode not in ["GUIDED", ""]:
                self.get_logger().warn(f"PENGAMBILALIHAN RC DETEKSI! Mode berubah ke {self.current_mode}.")
                self.get_logger().warn("Misi Jetson DIBATALKAN. Kendali 100% di tangan Pilot.")
                self.step = "MANUAL_OVERRIDE"

        # Jika pilot sudah mengambil alih, hentikan semua perintah Jetson
        if self.step == "MANUAL_OVERRIDE":
            return

        # 1. PERBAIKAN: Hentikan pengiriman setpoint saat LAND dan DONE
        if self.step not in ["INIT", "ARMING", "TAKEOFF", "WAIT_TAKEOFF", "LAND", "DONE"]:
            self.target_pose.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(self.target_pose)

        if self.step == "INIT":
            if self.retry_counter % 20 == 0:
                mode_msg = String(); mode_msg.data = "GUIDED"
                self.cmd_mode_pub.publish(mode_msg)
                self.get_logger().info("Meminta mode GUIDED via Flight Manager...")
                
            if self.current_mode == "GUIDED":
                self.step = "ARMING"
                self.retry_counter = 0
            self.retry_counter += 1

        elif self.step == "ARMING":
            if self.retry_counter % 20 == 0:
                arm_msg = Bool(); arm_msg.data = True
                self.cmd_arm_pub.publish(arm_msg)
                self.get_logger().info("Meminta Arming via Flight Manager...")
                
            if self.is_armed:
                self.get_logger().info("Armed! Bersiap Takeoff...")
                
                # 2. PERBAIKAN: Simpan seluruh Pose saat Arming (termasuk Yaw/Orientasi asli)
                self.start_pose = self.current_pose.pose
                self.start_x = self.start_pose.position.x
                self.start_y = self.start_pose.position.y
                
                # Menggunakan orientasi asli agar drone tidak mendadak berputar
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
            self.get_logger().info(f"Eksekusi Takeoff ke {self.alt_target}m...")
            self.step = "WAIT_TAKEOFF"

        elif self.step == "WAIT_TAKEOFF":
            if self.is_hovering:
                self.get_logger().info("Hovering stabil pasca-takeoff tercapai.")
                # Target tidak perlu diubah lagi karena sudah di-set saat ARMING
                self.trigger_hover("LAND")

        elif self.step == "HOVERING":
            if time.time() - self.hover_start_time > self.hover_duration:
                self.step = self.next_step_after_hover
                self.get_logger().info(f"Mulai manuver: {self.step}")

        elif self.step == "LAND":
            # Setpoint_pub sudah tidak aktif di sini karena pengecualian di baris teratas
            land_msg = Bool(); land_msg.data = True
            self.cmd_land_pub.publish(land_msg)
            self.step = "DONE"
            self.get_logger().info("Mode pendaratan dikirim via Flight Manager. Uji coba SELESAI.")

def main(args=None):
    rclpy.init(args=args)
    node = OdomTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()