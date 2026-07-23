#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point

from uav_interfaces.msg import Tree
from uav_interfaces.msg import TreeArray
from beehive_drone.mission_params import MissionConfig
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy

class TreeMapper(Node):

    def __init__(self):
        super().__init__("tree_mapper")

        ##################################################
        # Parameters
        ##################################################
        self.frame_id = "odom"

        # maksimum jarak agar dianggap pohon yang sama
        self.merge_distance = MissionConfig.TREE_MERGE_DISTANCE

        # confidence model
        self.max_confidence = MissionConfig.TREE_MAX_CONFIDENCE
        self.new_tree_confidence = MissionConfig.TREE_NEW_CONFIDENCE
        self.confidence_increment = MissionConfig.TREE_CONFIDENCE_INCREMENT
        self.confidence_decay = MissionConfig.TREE_CONFIDENCE_DECAY

        # waktu hilang sebelum confidence turun
        self.timeout = MissionConfig.TREE_TIMEOUT

        ##################################################
        # Database
        ##################################################
        self.tree_database = {}
        self.next_tree_id = 1

        ##################################################
        # Subscribers
        ##################################################
        # hasil deteksi perception
        self.sub = self.create_subscription(
            Point,
            "/perception/tree_position_camera",
            self.tree_callback,
            10
        )

        # hasil inspeksi
        self.create_subscription(
            Tree,
            "/map/tree_update",
            self.tree_update_callback,
            10
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        ##################################################
        # Publishers
        ##################################################
        self.tree_pub = self.create_publisher(
            TreeArray,
            "/map/trees",
            map_qos
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/tree_markers",
            10
        )

        ##################################################
        # Timer
        ##################################################
        self.timer = self.create_timer(
            5.0,
            self.update_confidence
        )

        self.get_logger().info("Tree Mapper Started")


    ##################################################
    # Receive tree detection
    ##################################################
    def tree_callback(self,msg):
        x = msg.x
        y = msg.y
        z = msg.z

        if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
            self.get_logger().warning("Invalid tree position ignored")
            return

        nearest_id = None
        nearest_distance = float("inf")

        ##################################################
        # Search nearest tree
        ##################################################
        for tree_id,tree in self.tree_database.items():
            distance = math.sqrt(
                (x-tree["x"])**2 +
                (y-tree["y"])**2
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_id = tree_id

        ##################################################
        # Existing tree
        ##################################################
        if nearest_id is not None and nearest_distance < self.merge_distance:
            tree = self.tree_database[nearest_id]
            tree["count"] += 1

            alpha = 1.0 / tree["count"]

            # update posisi
            tree["x"] += alpha * (x-tree["x"])
            tree["y"] += alpha * (y-tree["y"])
            tree["z"] += alpha * (z-tree["z"])

            ##################################################
            # Confidence hanya naik jika belum inspected
            ##################################################
            if not tree["inspected"]:
                tree["confidence"] = min(
                    tree["confidence"] + self.confidence_increment,
                    self.max_confidence
                )

            tree["last_seen"] = time.time()
            self.get_logger().debug(f"Tree {nearest_id} updated")

        ##################################################
        # New tree
        ##################################################
        else:
            tree_id = self.next_tree_id
            self.tree_database[tree_id] = {
                "id":tree_id,
                "x":x,
                "y":y,
                "z":z,
                "confidence": self.new_tree_confidence,
                "count":1,
                "inspected":False,
                "last_seen": time.time()
            }
            self.next_tree_id += 1
            self.get_logger().info(f"New tree {tree_id} ({x:.2f},{y:.2f},{z:.2f})")

        self.publish_tree()
        self.publish_marker()


    ##################################################
    # Receive inspection result
    ##################################################
    def tree_update_callback(self,msg):
        
        ##################################################
        # 1. CEK SINYAL PEMUSNAHAN DARI FSM
        ##################################################
        if msg.confidence == -1.0:
            if msg.id in self.tree_database:
                del self.tree_database[msg.id]
                self.get_logger().info(f"Pohon Hantu ID:{msg.id} resmi DIHAPUS dari database peta.")
                
                # Update visualisasi untuk segera membuang pohon yang dihapus
                self.publish_tree()
                self.publish_marker()
            return
        ##################################################

        ##################################################
        # 2. LOGIKA UPDATE NORMAL
        ##################################################
        if msg.id not in self.tree_database:
            self.get_logger().warning(f"Unknown tree ID {msg.id}")
            return

        tree = self.tree_database[msg.id]

        # Update full information
        tree["x"] = msg.x
        tree["y"] = msg.y
        tree["z"] = msg.z
        tree["confidence"] = msg.confidence
        tree["inspected"] = msg.inspected
        tree["last_seen"] = time.time()

        self.get_logger().info(f"Tree {msg.id} inspected={msg.inspected}")

        self.publish_tree()
        self.publish_marker()


    ##################################################
    # Confidence aging
    ##################################################
    def update_confidence(self):
        now = time.time()

        for tree in self.tree_database.values():
            ##################################################
            # Jangan decay pohon selesai inspeksi
            ##################################################
            if tree["inspected"]:
                continue

            elapsed = now - tree["last_seen"]

            if elapsed > self.timeout:
                tree["confidence"] -= self.confidence_decay
                if tree["confidence"] < 0:
                    tree["confidence"] = 0.0

        self.publish_tree()
        self.publish_marker()


    ##################################################
    # Publish TreeArray
    ##################################################
    def publish_tree(self):
        msg = TreeArray()

        for tree in self.tree_database.values():
            t = Tree()
            t.id = tree["id"]
            t.x = tree["x"]
            t.y = tree["y"]
            t.z = tree["z"]
            t.confidence = tree["confidence"]
            t.inspected = tree["inspected"]

            msg.trees.append(t)

        self.tree_pub.publish(msg)


    ##################################################
    # RVIZ Marker
    ##################################################
    def publish_marker(self):
        markers = MarkerArray()

        # Membersihkan teks lama di RVIZ sebelum mempublikasikan ulang
        # Ini mencegah teks pohon hantu tertinggal di layar setelah dihapus
        delete_all_marker = Marker()
        delete_all_marker.action = Marker.DELETEALL
        markers.markers.append(delete_all_marker)

        ##################################################
        # Sphere marker
        ##################################################
        sphere = Marker()
        sphere.header.frame_id = self.frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = "trees"
        sphere.id = 0
        sphere.type = Marker.SPHERE_LIST
        sphere.action = Marker.ADD

        sphere.scale.x = 0.5
        sphere.scale.y = 0.5
        sphere.scale.z = 0.5
        sphere.pose.orientation.w = 1.0

        for tree in self.tree_database.values():
            p = Point()
            p.x = tree["x"]
            p.y = tree["y"]
            p.z = tree["z"]
            sphere.points.append(p)

        markers.markers.append(sphere)

        ##################################################
        # Text marker
        ##################################################
        marker_id = 1000

        for tree in self.tree_database.values():
            text = Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "tree_id"
            text.id = marker_id
            marker_id += 1

            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = tree["x"]
            text.pose.position.y = tree["y"]
            text.pose.position.z = 1.5
            text.pose.orientation.w = 1.0
            text.scale.z = 0.5

            status = "DONE" if tree["inspected"] else "NEW"
            text.text = f"ID:{tree['id']} C:{tree['confidence']:.2f} {status}"

            markers.markers.append(text)

        self.marker_pub.publish(markers)

##################################################
def main(args=None):
    rclpy.init(args=args)
    node = TreeMapper()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()