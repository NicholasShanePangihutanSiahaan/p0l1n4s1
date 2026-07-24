#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import cv2
import numpy as np
import message_filters

class TreeDetector(Node):

    def __init__(self):
        super().__init__("tree_detector")
        self.bridge = CvBridge()

        # Threshold HSV (Pastikan ini sudah dikalibrasi untuk warna pohon)
        self.lower = np.array([5, 60, 20], dtype=np.uint8)
        self.upper = np.array([25, 255, 180], dtype=np.uint8)

        self.min_area = 250
        self.min_aspect_ratio = 2.0
        self.kernel = np.ones((5, 5), np.uint8)
        self.show_debug = True

        # ---------- ROS Publishers ----------
        self.pixel_pub = self.create_publisher(Point, "/perception/tree_pixel", 10)
        self.detect_pub = self.create_publisher(Bool, "/perception/tree_detected", 10)

        # ---------- ROS Subscribers (Message Filters) ----------
        self.rgb_sub = message_filters.Subscriber(self, Image, "/zed/zed_node/rgb/image_rect_color")
        self.depth_sub = message_filters.Subscriber(self, Image, "/zed/zed_node/depth/depth_registered")

        # Sinkronisasi Waktu
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.05)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("Tree Detector Started - Menunggu Gambar Tersinkronisasi...")

    # INI ADALAH FUNGSI GABUNGANNYA
    def sync_callback(self, rgb_msg, depth_msg):
        
        # 1. Konversi kedua message secara bersamaan (Waktu dijamin sama!)
        image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        # 2. Proses Deteksi Gambar (RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w == 0:
                continue

            aspect = h / float(w)
            if aspect < self.min_aspect_ratio:
                continue

            score = area * aspect
            if score > best_score:
                best_score = score
                best = (x, y, w, h, area, aspect)

        detected = Bool()

        # Jika pohon tidak terdeteksi
        if best is None:
            detected.data = False
            self.detect_pub.publish(detected)
            if self.show_debug:
                cv2.imshow("mask", mask)
                cv2.imshow("tree_detector", image)
                cv2.waitKey(1)
            return

        # Jika pohon terdeteksi
        x, y, w, h, area, aspect = best
        
        center_x = float(x + w / 2.0)
        center_y = float(y + h / 2.0)

        # 3. Hitung Depth (Lempar array depth_image ke fungsi calculate_depth)
        depth = self.calculate_depth(x, y, w, h, depth_image)

        if depth is None:
            detected.data = False
            self.detect_pub.publish(detected)
            return
        
        # 4. Publikasikan Data Titik Pusat dan Kedalaman
        point = Point()
        point.x = center_x
        point.y = center_y
        point.z = depth
        self.pixel_pub.publish(point)

        detected.data = True
        self.detect_pub.publish(detected)
        
        self.get_logger().info(
            f"Tree detected | Area={area:.1f} depth={depth:.2f} m Center=({center_x:.1f},{center_y:.1f})"
        )

        # 5. Visualisasi (Debug)
        if self.show_debug:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.circle(image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            cv2.putText(image, f"{depth:.2f} m", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("mask", mask)
            cv2.imshow("tree_detector", image)
            cv2.waitKey(1)

    # Tambahkan parameter depth_image pada fungsi ini
    def calculate_depth(self, x, y, w, h, depth_image):
        margin_x = int(w * 0.40)
        margin_y = int(h * 0.40)

        x1 = max(0, x + margin_x)
        y1 = max(0, y + margin_y)
        x2 = min(depth_image.shape[1], x + w - margin_x)
        y2 = min(depth_image.shape[0], y + h - margin_y)

        roi = depth_image[y1:y2, x1:x2]

        if x2 <= x1 or y2 <= y1:
            return None

        # Saring nilai depth yang valid
        roi = roi[np.isfinite(roi)]
        roi = roi[(roi > 0.2) & (roi < 20.0)]

        if roi.size == 0:
            return None

        # Ambil median agar tidak terpengaruh titik outlier (misal daun terbang)
        depth = np.median(roi)
        return float(depth)


def main(args=None):
    rclpy.init(args=args)
    node = TreeDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()