#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from beehive_drone.mission_params import MissionConfig
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def quaternion_to_yaw(qx, qy, qz, qw):
    """Konversi Quaternion ke sudut Yaw (Euler)"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class VelocityController(Node):
    def __init__(self):
        super().__init__("velocity_controller")

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==================================================
        # Parameter Proportional Gain (Translasi & Rotasi)
        # ==================================================
        self.kp_xy = MissionConfig.KP_XY            # Diperbesar sedikit agar lebih responsif
        self.kp_z = MissionConfig.KP_Z
        self.kp_yaw = MissionConfig.KP_YAW           # Gain untuk rotasi Yaw (Heading)

        # ==================================================
        # Parameter Limit Kecepatan (Maksimum)
        # ==================================================
        self.max_velocity_xy = MissionConfig.MAX_VELOCITY_XY  # m/s
        self.max_velocity_z = MissionConfig.MAX_VELOCITY_Z   # m/s
        self.max_velocity_yaw = MissionConfig.MAX_VELOCITY_YAW # rad/s (Sekitar 30 derajat per detik)

        self.goal_threshold = MissionConfig.GOAL_THRESHOLD   # meter (Jarak dianggap sampai)

        # ==================================================
        # State
        # ==================================================
        self.current_pose = None
        self.target_pose = None

        # ==================================================
        # Subscriber & Publisher
        # ==================================================
        # Membaca posisi UAV saat ini
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.pose_callback,
            qos_sensor
        )

        # Membaca target AMAN dari Vortex Avoidance (Sekarang membaca PoseStamped, bukan Point)
        self.target_sub = self.create_subscription(
            PoseStamped,
            "/control/safe_target_pose",
            self.target_callback,
            10
        )

        # Mengirim Twist (Kecepatan) ke MAVROS
        self.velocity_pub = self.create_publisher(
            TwistStamped,
            "/mavros/setpoint_velocity/cmd_vel",
            10
        )

        # Loop berjalan pada 20Hz (0.05 detik)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Velocity Controller (Dengan Yaw/Heading Control) Aktif")

    def pose_callback(self, msg):
        self.current_pose = msg

    def target_callback(self, msg):
        self.target_pose = msg

    def limit(self, value, maximum):
        """Membatasi nilai agar tidak melebihi kecepatan maksimum"""
        if value > maximum: return maximum
        if value < -maximum: return -maximum
        return value

    def control_loop(self):
        if self.current_pose is None or self.target_pose is None:
            return
        if self.current_pose.pose.position.z < 1.5:
            return

        # ==================================================
        # 1. Kalkulasi Error Posisi GLOBAL (X, Y, Z)
        # ==================================================
        ex = self.target_pose.pose.position.x - self.current_pose.pose.position.x
        ey = self.target_pose.pose.position.y - self.current_pose.pose.position.y
        ez = self.target_pose.pose.position.z - self.current_pose.pose.position.z
        
        distance = math.sqrt(ex*ex + ey*ey + ez*ez)

        # ==================================================
        # 2. Kalkulasi Error Sudut (Yaw)
        # ==================================================
        curr_qx = self.current_pose.pose.orientation.x
        curr_qy = self.current_pose.pose.orientation.y
        curr_qz = self.current_pose.pose.orientation.z
        curr_qw = self.current_pose.pose.orientation.w
        current_yaw = quaternion_to_yaw(curr_qx, curr_qy, curr_qz, curr_qw)

        tgt_qx = self.target_pose.pose.orientation.x
        tgt_qy = self.target_pose.pose.orientation.y
        tgt_qz = self.target_pose.pose.orientation.z
        tgt_qw = self.target_pose.pose.orientation.w
        target_yaw = quaternion_to_yaw(tgt_qx, tgt_qy, tgt_qz, tgt_qw)

        e_yaw = target_yaw - current_yaw
        while e_yaw > math.pi: e_yaw -= 2.0 * math.pi
        while e_yaw < -math.pi: e_yaw += 2.0 * math.pi

        # ==================================================
        # 3. Meracik Komando Kecepatan (Twist)
        # ==================================================
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        # MAVROS ArduPilot mengevaluasi ini sebagai Global ENU
        cmd.header.frame_id = "odom" 

        if distance < self.goal_threshold:
            cmd.twist.linear.x = 0.0
            cmd.twist.linear.y = 0.0
            cmd.twist.linear.z = 0.0
        else:
            # Hitung kecepatan mentah
            vx = self.kp_xy * ex
            vy = self.kp_xy * ey
            
            # NORMALISASI VEKTOR (Wajib untuk Orbit)
            # Ini memastikan arah vektor (sudut panah) tidak rusak meskipun kecepatannya dibatasi
            v_mag = math.sqrt(vx**2 + vy**2)
            if v_mag > self.max_velocity_xy:
                scale = self.max_velocity_xy / v_mag
                vx *= scale
                vy *= scale
                
            cmd.twist.linear.x = float(vx)
            cmd.twist.linear.y = float(vy)
            cmd.twist.linear.z = float(self.limit(self.kp_z * ez, self.max_velocity_z))

        # Kontrol Rotasi (Angular Yaw)
        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = float(self.limit(self.kp_yaw * e_yaw, self.max_velocity_yaw))

        self.velocity_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()