#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class VisionToMavros(Node):
    def __init__(self):
        super().__init__('vision_to_mavros')

        # 1. Menerima data dari ZED (VSLAM)
        self.sub = self.create_subscription(
            PoseStamped,
            '/zed/zed_node/pose',
            self.pose_callback,
            10
        )

        # 2. Mengirim data ke ArduPilot (Pixhawk) via MAVROS
        self.pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        self.get_logger().info("Jembatan VSLAM (ZED) ke MAVROS (Pixhawk) telah AKTIF!")

    def pose_callback(self, msg):
        # Meneruskan data mentah. 
        # Jika kelak diperlukan penyelarasan waktu (timestamp) 
        # atau rotasi sumbu orientasi, logikanya bisa ditambahkan di sini.
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionToMavros()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
        
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()