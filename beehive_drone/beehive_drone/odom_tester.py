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
        self.alt_target = 3.0
        self.hover_duration = 4.0 
        self.dist_tolerance = 0.3 
        
        self.current_pose = None
        self.start_x = 0.0
        self.start_y = 0.0
        
        self.step = "INIT"
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "odom"
        
        self.hover_start_time = 0.0
        self.orbit_start_time = 0.0
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

    def distance_to_target(self):
        if not self.current_pose: return float('inf')
        dx = self.target_pose.pose.position.x - self.current_pose.pose.position.x
        dy = self.target_pose.pose.position.y - self.current_pose.pose.position.y
        dz = self.target_pose.pose.position.z - self.current_pose.pose.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

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

        # WAJIB: Selalu publikasikan setpoint MAVROS 20Hz agar mode GUIDED tidak terputus.
        if self.step not in ["INIT", "ARMING", "TAKEOFF", "WAIT_TAKEOFF"]:
            self.target_pose.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_pub.publish(self.target_pose)

        # Logic Berdasarkan Sekuens FSM
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
                self.start_x = self.current_pose.pose.position.x
                self.start_y = self.current_pose.pose.position.y
                
                # Kunci setpoint ke Z=3.0m agar takeoff tidak dilawan oleh perintah Z=0
                self.set_target(self.start_x, self.start_y, self.alt_target)
                self.step = "TAKEOFF"
                self.retry_counter = 0
            self.retry_counter += 1

        elif self.step == "TAKEOFF":
            takeoff_msg = Float32(); takeoff_msg.data = self.alt_target
            self.cmd_takeoff_pub.publish(takeoff_msg)
            self.get_logger().info(f"Eksekusi Takeoff ke {self.alt_target}m...")
            self.step = "WAIT_TAKEOFF"

        elif self.step == "WAIT_TAKEOFF":
            # Memanfaatkan logika Hover cerdas milik Flight Manager
            if self.is_hovering:
                self.get_logger().info("Hovering stabil pasca-takeoff tercapai.")
                self.set_target(self.start_x, self.start_y, self.alt_target)
                self.trigger_hover("MAJU_3M")

        elif self.step == "HOVERING":
            if time.time() - self.hover_start_time > self.hover_duration:
                self.step = self.next_step_after_hover
                self.get_logger().info(f"Mulai manuver: {self.step}")

        elif self.step == "MAJU_3M":
            self.set_target(self.start_x + 3.0, self.start_y, self.alt_target)
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("KIRI_3M")

        elif self.step == "KIRI_3M":
            self.set_target(self.start_x + 3.0, self.start_y + 3.0, self.alt_target)
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("KANAN_3M")

        elif self.step == "KANAN_3M":
            self.set_target(self.start_x + 3.0, self.start_y - 3.0, self.alt_target)
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("MUNDUR_3M")

        elif self.step == "MUNDUR_3M":
            self.set_target(self.start_x, self.start_y, self.alt_target)
            if self.distance_to_target() < self.dist_tolerance:
                self.trigger_hover("MAJU_LAGI_3M")

        elif self.step == "MAJU_LAGI_3M":
            self.set_target(self.start_x + 3.0, self.start_y, self.alt_target)
            if self.distance_to_target() < self.dist_tolerance:
                self.get_logger().info("Titik tengah tercapai. Memulai manuver Orbit...")
                self.orbit_start_time = time.time()
                self.step = "ORBIT_TEST"

        elif self.step == "ORBIT_TEST":
            radius = 2.0
            cx = self.start_x + 3.0
            cy = self.start_y + radius 
            
            elapsed = time.time() - self.orbit_start_time
            angular_speed = 0.5 
            theta = -math.pi/2 + (elapsed * angular_speed)
            
            orbit_x = cx + radius * math.cos(theta)
            orbit_y = cy + radius * math.sin(theta)
            orbit_yaw = math.atan2(cy - orbit_y, cx - orbit_x)
            
            self.set_target(orbit_x, orbit_y, self.alt_target, yaw=orbit_yaw)
            
            if elapsed * angular_speed >= 2 * math.pi:
                self.get_logger().info("Orbit 360 derajat selesai. Meminta Pendaratan...")
                self.step = "LAND"

        elif self.step == "LAND":
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