#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped
from zed_msgs.msg import ObjectsStamped  # Import pesan khusus ZED
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
        # Parameter Intrinsik Dihapus!
        # ZED SDK sudah menangani proyeksi spasial 3D.
        # ==========================================

        # Variabel Pose Drone
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
        # Dengarkan pose drone untuk titik acuan World Frame
        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.pose_callback,
            qos_sensor
        )

        # SUBSCRIBER BARU: Mendengarkan bounding box & posisi spasial AI ZED
        self.obj_sub = self.create_subscription(
            ObjectsStamped,
            "/zed/zed_node/obj_det/objects",
            self.objects_callback,
            10
        )

        # Publikasikan koordinat dunia ke Tree Mapper
        # Kita tetap menggunakan Point agar kompatibel dengan tree_mapper.py lama
        self.pub = self.create_publisher(
            Point,
            "/perception/tree_position_camera",
            10
        )

        self.get_logger().info("Tree Localizer AI (World Frame) Started")

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

    def objects_callback(self, msg: ObjectsStamped):
        if not self.have_pose or not msg.objects:
            return

        # ZED dapat mendeteksi beberapa pohon sekaligus dalam satu frame,
        # kita iterasikan semuanya dan kirim ke Tree Mapper.
        for obj in msg.objects:
            # Opsional: Abaikan deteksi dengan confidence di bawah threshold
            if obj.confidence < 40.0:
                continue

            pos = obj.position
            
            # ZED mengembalikan [X, Y, Z] dalam Optical Frame (Kamera)
            X_cam = float(pos[0]) # Kanan/Kiri
            Y_cam = float(pos[1]) # Atas/Bawah
            Z_cam = float(pos[2]) # Maju/Mundur (Kedalaman)

            # Filter validasi jika ZED gagal menghitung depth pada piksel tersebut
            if math.isnan(X_cam) or math.isnan(Y_cam) or math.isnan(Z_cam) or Z_cam <= 0.0:
                continue

            # 1. Transformasi ke Orientasi Fisik Drone (base_link)
            # Asumsi standar ROS: Kamera menghadap sumbu X drone
            # Kamera Z (Maju) = Drone X (Maju)
            # Kamera X (Kanan) = Drone -Y (Kanan)
            bl_x = Z_cam
            bl_y = -X_cam

            # 2. Transformasi Rotasi ke World Frame (Odometry) menggunakan Yaw Drone
            yaw = self.drone_yaw
            world_x = self.drone_x + (bl_x * math.cos(yaw)) - (bl_y * math.sin(yaw))
            world_y = self.drone_y + (bl_x * math.sin(yaw)) + (bl_y * math.cos(yaw))
            
            # Kalkulasi estimasi tinggi/posisi Z pohon di dunia
            world_z = self.drone_z - Y_cam 

            # 3. Kirim Titik yang Sudah Akurat ke Tree Mapper
            point = Point()
            point.x = world_x
            point.y = world_y
            point.z = world_z

            self.pub.publish(point)

            self.get_logger().info(
                f"Pohon [{obj.label}] -> W_X: {world_x:.2f}m, W_Y: {world_y:.2f}m (Jarak: {Z_cam:.2f}m, Conf: {obj.confidence:.1f}%)"
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