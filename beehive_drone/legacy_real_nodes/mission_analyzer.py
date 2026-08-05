#!/usr/bin/env python3

import math
import time
import csv
from datetime import datetime

import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy
)

from geometry_msgs.msg import (
    PoseStamped,
    PoseArray,
    TwistStamped  # [TAMBAHAN] Untuk membaca Vx, Vy, Vz
)

# [TAMBAHAN] Untuk membaca Airspeed, Groundspeed, Heading, Throttle
from mavros_msgs.msg import VfrHud 

from std_msgs.msg import String

from uav_interfaces.msg import (
    Tree,
    TreeArray
)


class MissionAnalyzer(Node):

    def __init__(self):

        super().__init__(
            "mission_analyzer"
        )

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.start_time = time.time()

        self.current_status = "INIT"

        self.status_history = []

        self.have_pose = False

        self.uav_x = 0.0
        self.uav_y = 0.0
        self.uav_z = 0.0

        # [TAMBAHAN] Variabel status terkini
        self.uav_vx = 0.0
        self.uav_vy = 0.0
        self.uav_vz = 0.0
        self.uav_heading = 0.0
        self.uav_airspeed = 0.0
        self.uav_groundspeed = 0.0
        self.uav_throttle = 0.0

        self.path_x = []
        self.path_y = []
        self.path_z = []
        
        # [TAMBAHAN] Array untuk menyimpan jejak telemetri ke CSV
        self.path_vx = []
        self.path_vy = []
        self.path_vz = []
        self.path_heading = []
        self.path_airspeed = []
        self.path_groundspeed = []
        self.path_throttle = []

        self.path_time = []

        self.total_distance = 0.0

        self.trees = []

        self.total_tree = 0

        self.detected_tree = 0

        self.validated_tree = 0

        self.inspected_tree = 0

        self.coverage = 0.0

        self.previous_pose = None
        
        self.completed_waypoints = 0

        self.current_state = "INIT"

        self.current_follower = "WAITING"

        self.follower_history = []

        self.completed_trajectory = 0

        self.current_waypoints = None

        self.total_waypoints = 0

        self.waypoint_history = []

        self.average_speed = 0.0

        plt.ion()

        self.figure = plt.figure(
            figsize=(12,10)
        )

        self.axis = self.figure.add_subplot(111)

        self.create_subscription(
            PoseStamped,
            "/mavros/local_position/pose",
            self.pose_callback,
            qos_sensor
        )

        # [TAMBAHAN] Subscriber untuk Local Velocity (M/S)
        self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self.vel_callback,
            qos_sensor
        )

        # [TAMBAHAN] Subscriber untuk HUD (Angin, Heading, Airspeed)
        self.create_subscription(
            VfrHud,
            "/mavros/vfr_hud",
            self.vfr_callback,
            qos_sensor
        )

        self.create_subscription(
            TreeArray,
            "/map/trees",
            self.tree_callback,
            qos_map
        )

        self.create_subscription(
            String,
            "/mission/status",
            self.mission_callback,
            10
        )

        self.create_subscription(
            String,
            "/navigation/follower_status",
            self.follower_callback,
            10
        )

        self.create_subscription(
            PoseArray,
            "/mission/inspection_waypoints",
            self.waypoint_callback,
            10
        )

        self.timer = self.create_timer(
            0.5,
            self.update
        )

        self.get_logger().info(
            "Mission Analyzer Started (With Full Telemetry Logging)"
        )

    # [TAMBAHAN] Callback Kecepatan Lokal
    def vel_callback(self, msg):
        self.uav_vx = msg.twist.linear.x
        self.uav_vy = msg.twist.linear.y
        self.uav_vz = msg.twist.linear.z

    # [TAMBAHAN] Callback VFR HUD (Heading, Airspeed)
    def vfr_callback(self, msg):
        self.uav_heading = msg.heading
        self.uav_airspeed = msg.airspeed
        self.uav_groundspeed = msg.groundspeed
        self.uav_throttle = msg.throttle

    def pose_callback(self, msg):

        self.uav_x = msg.pose.position.x
        self.uav_y = msg.pose.position.y
        self.uav_z = msg.pose.position.z

        self.have_pose = True

        self.path_x.append(self.uav_x)
        self.path_y.append(self.uav_y)
        self.path_z.append(self.uav_z)
        
        # [TAMBAHAN] Sinkronisasi: Saat pose dicatat, catat juga status telemetri saat itu
        self.path_vx.append(self.uav_vx)
        self.path_vy.append(self.uav_vy)
        self.path_vz.append(self.uav_vz)
        self.path_heading.append(self.uav_heading)
        self.path_airspeed.append(self.uav_airspeed)
        self.path_groundspeed.append(self.uav_groundspeed)
        self.path_throttle.append(self.uav_throttle)

        elapsed = time.time() - self.start_time

        self.path_time.append(elapsed)

        if self.previous_pose is not None:

            dx = msg.pose.position.x - self.previous_pose.position.x
            dy = msg.pose.position.y - self.previous_pose.position.y
            dz = msg.pose.position.z - self.previous_pose.position.z

            self.total_distance += math.sqrt(
                dx*dx + dy*dy + dz*dz
            )

        self.previous_pose = msg.pose

        if elapsed > 0.0:

            self.average_speed = (
                self.total_distance /
                elapsed
            )

    def mission_callback(self, msg):

        status = msg.data

        if status == self.current_status:
            return

        self.current_status = status

        now = self.get_clock().now().nanoseconds / 1e9

        self.status_history.append(
            (now, status)
        )

        self.get_logger().info(
            f"[MISSION] {status}"
        )

    def tree_callback(self, msg):

        self.trees = msg.trees
        self.total_tree = len(msg.trees)

        self.inspected_tree = sum(
            1 for t in msg.trees
            if t.inspected
        )

        self.validated_tree = sum(
            1 for t in msg.trees
            if t.validated
        )

        if self.total_tree > 0:
            self.coverage = (self.inspected_tree / self.total_tree) * 100.0
        else:
            self.coverage = 0.0

    def follower_callback(self, msg):

        status = msg.data

        if status == self.current_follower:
            return

        self.current_follower = status

        timestamp = (
            time.time() -
            self.start_time
        )

        self.follower_history.append(
            (
                timestamp,
                status
            )
        )

        if status == "WAYPOINT_REACHED":
            self.completed_waypoints += 1

        if status == "TRAJECTORY_COMPLETED":
            self.completed_trajectory += 1

        self.get_logger().info(
            f"[FOLLOWER] {status}"
        )

    def update(self):
        self.calculate_statistics()
        self.draw_map()

    def waypoint_callback(self, msg):

        self.current_waypoints = msg
        self.total_waypoints = len(msg.poses)

        timestamp = (
            time.time() -
            self.start_time
        )

        waypoints = []
        for pose in msg.poses:
            waypoints.append(
                (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                )
            )

        self.waypoint_history.append(
            (
                timestamp,
                waypoints
            )
        )

        self.get_logger().info(
            f"[WAYPOINT] Received {self.total_waypoints} waypoint(s)"
        )

    def calculate_statistics(self):

        if self.start_time is None:
            duration = 0.0
        else:
            duration = time.time() - self.start_time

        distance = self.total_distance

        if duration > 0.0:
            avg_speed = distance / duration
        else:
            avg_speed = 0.0

        total_tree = len(self.trees)
        inspected_tree = 0
        validated_tree = 0

        for tree in self.trees:
            if tree.inspected:
                inspected_tree += 1
            if tree.validated:
                validated_tree += 1

        if total_tree > 0:
            coverage = (inspected_tree / total_tree) * 100.0
        else:
            coverage = 0.0

        total_waypoint = self.total_waypoints
        finished_waypoint = self.completed_waypoints

        self.get_logger().info("========================================")
        self.get_logger().info("MISSION SUMMARY")
        self.get_logger().info("========================================")
        self.get_logger().info(f"Mission Time      : {duration:.1f} s")
        self.get_logger().info(f"Travel Distance   : {distance:.2f} m")
        self.get_logger().info(f"Average Speed     : {avg_speed:.2f} m/s")
        self.get_logger().info(f"Trees Detected    : {total_tree}")
        self.get_logger().info(f"Trees Inspected   : {inspected_tree}")
        self.get_logger().info(f"Trees Validated   : {validated_tree}")
        self.get_logger().info(f"Coverage          : {coverage:.1f} %")
        self.get_logger().info(f"Waypoints Planned : {total_waypoint}")
        self.get_logger().info(f"Waypoints Passed  : {finished_waypoint}")
        self.get_logger().info(f"Trajectory Finish : {self.completed_trajectory}")
        self.get_logger().info("========================================")

    def draw_map(self):

        self.axis.clear()

        self.axis.set_title(
            "UAV Mission Analysis",
            fontsize=16,
            fontweight="bold"
        )

        self.axis.set_xlabel("X (m)")
        self.axis.set_ylabel("Y (m)")
        self.axis.grid(True)
        self.axis.set_aspect("equal", adjustable="box")

        if len(self.path_x) > 1:
            self.axis.plot(
                self.path_x,
                self.path_y,
                color="blue",
                linewidth=2,
                label="Trajectory"
            )

        if self.have_pose:
            self.axis.scatter(
                self.uav_x,
                self.uav_y,
                color="red",
                s=120,
                marker="o",
                label="UAV"
            )
            
        self.axis.scatter(
            0,
            0,
            marker="*",
            s=180,
            color="gold",
            label="Home"
        )
        
        for tree in self.trees:

            if tree.inspected:
                color = "green"
            elif tree.validated:
                color = "orange"
            else:
                color = "black"

            self.axis.scatter(
                tree.x,
                tree.y,
                color=color,
                s=60
            )

            self.axis.text(
                tree.x,
                tree.y + 0.3,
                str(tree.id),
                fontsize=8
            )

        if self.current_waypoints is not None:

            orbit_x = []
            orbit_y = []

            for pose in self.current_waypoints.poses:
                orbit_x.append(pose.position.x)
                orbit_y.append(pose.position.y)

            if len(orbit_x) > 0:
                orbit_x.append(orbit_x[0])
                orbit_y.append(orbit_y[0])

                self.axis.plot(
                    orbit_x,
                    orbit_y,
                    "--",
                    color="purple",
                    linewidth=2,
                    label="Inspection Orbit"
                )

        mission_time = time.time() - self.start_time

        info = (
            f"State : {self.current_status}\n"
            f"Follower : {self.current_follower}\n"
            f"Mission Time : {mission_time:.1f} s\n"
            f"Distance : {self.total_distance:.1f} m\n"
            f"Coverage : {self.coverage:.1f}%\n"
            f"Trees : {self.inspected_tree}/{self.total_tree}"
        )

        self.axis.text(
            0.02,
            0.98,
            info,
            transform=self.axis.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                facecolor="white",
                alpha=0.8
            )
        )

        self.axis.legend()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def save_csv(self):

        filename = (
            "mission_result_" +
            time.strftime("%Y%m%d_%H%M%S") +
            ".csv"
        )

        with open(filename, "w", newline="") as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(["Mission Summary"])
            writer.writerow(["Mission Time (s)", time.time() - self.start_time])
            writer.writerow(["Travel Distance (m)", self.total_distance])
            writer.writerow(["Total Tree", self.total_tree])
            writer.writerow(["Inspected Tree", self.inspected_tree])
            writer.writerow(["Validated Tree", self.validated_tree])
            writer.writerow(["Coverage (%)", self.coverage])
            writer.writerow([])

            writer.writerow(["Trajectory"])
            
            # [REVISI PENTING] Penambahan Kolom Analisis Lengkap
            writer.writerow([
                "Index", "Time (s)", "X", "Y", "Z (Altitude)", 
                "Vx", "Vy", "Vz", "Heading (deg)", 
                "Airspeed", "Groundspeed", "Throttle (%)"
            ])

            # [REVISI PENTING] Mencetak semua data ke CSV
            for i in range(len(self.path_x)):
                writer.writerow([
                    i,
                    self.path_time[i],
                    self.path_x[i],
                    self.path_y[i],
                    self.path_z[i],
                    self.path_vx[i],
                    self.path_vy[i],
                    self.path_vz[i],
                    self.path_heading[i],
                    self.path_airspeed[i],
                    self.path_groundspeed[i],
                    self.path_throttle[i]
                ])

            writer.writerow([])
            writer.writerow(["Trees"])
            writer.writerow([
                "ID", "X", "Y", "Confidence", "Validated", "Inspected"
            ])

            for tree in self.trees:
                writer.writerow([
                    tree.id,
                    tree.x,
                    tree.y,
                    tree.confidence,
                    tree.validated,
                    tree.inspected
                ])

        self.get_logger().info(
            f"CSV saved : {filename}"
        )

    def save_png(self):

        filename = (
            "mission_map_" +
            time.strftime("%Y%m%d_%H%M%S") +
            ".png"
        )

        self.figure.savefig(
            filename,
            dpi=300,
            bbox_inches="tight"
        )

        self.get_logger().info(
            f"Map saved : {filename}"
        )

    def shutdown(self):

        self.get_logger().info("Saving mission result...")

        self.calculate_statistics()
        self.draw_map()
        self.save_csv()
        self.save_png()

        plt.close(self.figure)

        self.get_logger().info("Mission Analyzer Shutdown")

def main(args=None):

    rclpy.init(args=args)
    node = MissionAnalyzer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
    
if __name__ == "__main__":
    main()