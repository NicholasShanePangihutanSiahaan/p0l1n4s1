#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def quaternion_to_yaw(qx, qy, qz, qw):
    """Konversi Quaternion ke sudut Yaw"""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

class TreeLocalizer(Node):
    def __init__(self):
        super().__init__("tree_localizer")

        # ==========================================
        # Camera Intrinsic (ZED 2i)
        # ==========================================
        self.fx = 381.3611502479812
        self.fy = 381.3611502479812
        self.cx = 320.0
        self.cy = 240.0

        # ==========================================
        # Variabel Pose Drone
        # ==========================================
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_yaw = 0.0
        self.have_pose = False

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ==========================================
        # Subscriber & Publisher
        # ==========================================
        # DENGARKAN POSE DRONE UNTUK TITIK ACUAN
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.pose_callback,
            qos_sensor
        )

        self.pixel_sub = self.create_subscription(
            Point,
            "/perception/tree_pixel",
            self.pixel_callback,
            10
        )

        # PUBLIKASIKAN KOORDINAT DUNIA (WORLD FRAME)
        self.pub = self.create_publisher(
            Point,
            "/perception/tree_position_camera",
            10
        )

        self.get_logger().info("Tree Localizer (World Frame) Started")

    def pose_callback(self, msg):
        self.drone_x = msg.pose.position.x
        self.drone_y = msg.pose.position.y
        self.drone_z = msg.pose.position.z
        
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.drone_yaw = quaternion_to_yaw(qx, qy, qz, qw)
        
        self.have_pose = True

    def pixel_callback(self, msg):
        if not self.have_pose:
            return

        u = msg.x
        v = msg.y
        Z_cam = msg.z  # Ini adalah kedalaman asli dari sensor (misal 7.15 meter)

        if Z_cam <= 0.0:
            return

        # 1. Proyeksi Relatif ke Frame Kamera
        X_cam = (u - self.cx) * Z_cam / self.fx
        Y_cam = (v - self.cy) * Z_cam / self.fy

        # 2. Transformasi ke Orientasi Fisik Drone (base_link)
        # Asumsi standar ROS: Kamera menghadap sumbu X drone
        # Kamera Z (Maju) = Drone X (Maju)
        # Kamera X (Kanan) = Drone -Y (Kanan)
        bl_x = Z_cam
        bl_y = -X_cam

        # 3. Transformasi Rotasi ke World Frame (Odometry) menggunakan Yaw Drone
        yaw = self.drone_yaw
        world_x = self.drone_x + (bl_x * math.cos(yaw)) - (bl_y * math.sin(yaw))
        world_y = self.drone_y + (bl_x * math.sin(yaw)) + (bl_y * math.cos(yaw))
        
        # Kalkulasi estimasi tinggi pohon
        world_z = self.drone_z - Y_cam 

        # 4. Kirim Titik yang Sudah Akurat ke Tree Mapper
        point = Point()
        point.x = world_x
        point.y = world_y
        point.z = world_z

        self.pub.publish(point)

        self.get_logger().info(
            f"Proyeksi Pohon -> World X: {world_x:.2f}m, World Y: {world_y:.2f}m (Jarak Aktual: {Z_cam:.2f}m)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TreeLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()