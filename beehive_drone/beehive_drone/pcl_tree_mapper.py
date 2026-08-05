#!/usr/bin/env python3

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped
from pcl_cstm_msg.msg import TrackedCylinderArray
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformException, TransformListener
from uav_interfaces.msg import Tree, TreeArray
from visualization_msgs.msg import Marker, MarkerArray

from beehive_drone.mission_params import MissionConfig


@dataclass
class TreeRecord:
    tree_id: int
    x: float
    y: float
    z: float
    confidence: float
    validated: bool
    inspected: bool
    last_seen: float
    seen_count: int = 1
    missed_count: int = 0
    radius: float = 0.0
    height: float = 0.0
    source: str = "pcl"


class PclTreeMapper(Node):
    """Converts tracked PCL cylinders into one safe, frame-consistent tree map."""

    def __init__(self) -> None:
        super().__init__("pcl_tree_mapper")

        defaults = {
            "world_frame": MissionConfig.WORLD_FRAME,
            "pcl_source_frame": MissionConfig.PCL_FRAME,
            "pcl_topic": MissionConfig.PCL_TOPIC,
            "min_seen_count": MissionConfig.PCL_MIN_SEEN_COUNT,
            "max_missed_count": MissionConfig.PCL_MAX_MISSED_COUNT,
            "min_confidence": MissionConfig.PCL_MIN_CONFIDENCE,
            "min_radius": MissionConfig.PCL_MIN_RADIUS,
            "max_radius": MissionConfig.PCL_MAX_RADIUS,
            "min_height": MissionConfig.PCL_MIN_HEIGHT,
            "max_height": MissionConfig.PCL_MAX_HEIGHT,
            "tree_stale_sec": MissionConfig.TREE_STALE_SEC,
            "pcl_stream_timeout_sec": MissionConfig.PCL_STREAM_TIMEOUT_SEC,
            "min_ready_trees": MissionConfig.MIN_READY_TREES,
            "reject_hold_sec": MissionConfig.TREE_REJECT_HOLD_SEC,
            "fallback_merge_distance": MissionConfig.TREE_FALLBACK_MERGE_DISTANCE,
            "enable_fallback_points": False,
            "allow_identity_frame_relabel": MissionConfig.ALLOW_IDENTITY_FRAME_RELABEL,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.pcl_source_frame = str(self.get_parameter("pcl_source_frame").value)
        self.pcl_topic = str(self.get_parameter("pcl_topic").value)
        self.min_seen_count = int(self.get_parameter("min_seen_count").value)
        self.max_missed_count = int(self.get_parameter("max_missed_count").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.min_radius = float(self.get_parameter("min_radius").value)
        self.max_radius = float(self.get_parameter("max_radius").value)
        self.min_height = float(self.get_parameter("min_height").value)
        self.max_height = float(self.get_parameter("max_height").value)
        self.tree_stale_sec = float(self.get_parameter("tree_stale_sec").value)
        self.pcl_stream_timeout_sec = float(
            self.get_parameter("pcl_stream_timeout_sec").value
        )
        self.min_ready_trees = int(self.get_parameter("min_ready_trees").value)
        self.reject_hold_sec = float(self.get_parameter("reject_hold_sec").value)
        self.fallback_merge_distance = float(
            self.get_parameter("fallback_merge_distance").value
        )
        self.enable_fallback_points = bool(
            self.get_parameter("enable_fallback_points").value
        )
        self.allow_identity_frame_relabel = bool(
            self.get_parameter("allow_identity_frame_relabel").value
        )

        self.tree_database: Dict[int, TreeRecord] = {}
        self.rejected_until: Dict[int, float] = {}
        self.next_fallback_id = 100_000
        self.last_pcl_msg_time: Optional[float] = None
        self.last_transform_warning = -1e9
        self.last_ready_state: Optional[bool] = None

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            TrackedCylinderArray, self.pcl_topic, self.pcl_callback, qos_sensor
        )
        self.create_subscription(Tree, "/map/tree_update", self.tree_update_callback, 10)
        if self.enable_fallback_points:
            self.create_subscription(
                PointStamped,
                "/perception/tree_position_world",
                self.fallback_point_callback,
                qos_sensor,
            )

        self.tree_pub = self.create_publisher(TreeArray, "/map/trees", qos_map)
        self.marker_pub = self.create_publisher(MarkerArray, "/tree_markers", 10)
        self.ready_pub = self.create_publisher(Bool, "/map/trees_ready", qos_map)
        self.count_pub = self.create_publisher(Int32, "/map/tree_count", qos_map)
        self.create_timer(0.5, self.housekeeping)

        self.get_logger().info(
            "PCL Tree Mapper safety revision aktif; "
            f"input={self.pcl_topic}, source={self.pcl_source_frame}, "
            f"world={self.world_frame}, stale={self.tree_stale_sec:.1f}s."
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _finite(*values: float) -> bool:
        return all(math.isfinite(float(v)) for v in values)

    @staticmethod
    def _rotate_by_quaternion(
        x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float
    ) -> Tuple[float, float, float]:
        # Quaternion-vector multiplication written explicitly to avoid an
        # additional tf2_geometry_msgs runtime dependency.
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)
        rx = x + qw * tx + (qy * tz - qz * ty)
        ry = y + qw * ty + (qz * tx - qx * tz)
        rz = z + qw * tz + (qx * ty - qy * tx)
        return rx, ry, rz

    def transform_point(
        self, x: float, y: float, z: float, source_frame: str
    ) -> Optional[Tuple[float, float, float]]:
        source = source_frame.strip() or self.pcl_source_frame
        if source == self.world_frame:
            return float(x), float(y), float(z)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                source,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            rx, ry, rz = self._rotate_by_quaternion(
                float(x), float(y), float(z), q.x, q.y, q.z, q.w
            )
            return rx + t.x, ry + t.y, rz + t.z
        except TransformException as exc:
            now = self.now_sec()
            if self.allow_identity_frame_relabel:
                if now - self.last_transform_warning > 5.0:
                    self.last_transform_warning = now
                    self.get_logger().warning(
                        f"TF {source}->{self.world_frame} belum tersedia; "
                        "simulasi memakai identity relabel. Untuk hardware nyata, "
                        "set allow_identity_frame_relabel=false dan sediakan TF valid. "
                        f"Detail: {exc}"
                    )
                return float(x), float(y), float(z)

            if now - self.last_transform_warning > 2.0:
                self.last_transform_warning = now
                self.get_logger().error(
                    f"Cylinder diabaikan karena TF {source}->{self.world_frame} gagal: {exc}"
                )
            return None

    def _accepted_cylinder(self, tracked) -> bool:
        cylinder = tracked.cylinder
        if tracked.id < 0:
            return False
        if tracked.seen_count < self.min_seen_count:
            return False
        if tracked.missed_count > self.max_missed_count:
            return False
        if not cylinder.is_valid:
            return False
        if cylinder.confidence < self.min_confidence:
            return False
        if not (self.min_radius <= cylinder.radius <= self.max_radius):
            return False
        if not (self.min_height <= cylinder.height <= self.max_height):
            return False
        p = cylinder.pose.position
        return self._finite(p.x, p.y, p.z)

    def pcl_callback(self, msg: TrackedCylinderArray) -> None:
        now = self.now_sec()
        self.last_pcl_msg_time = now
        accepted_ids = set()

        for tracked in msg.cylinders:
            tree_id = int(tracked.id)
            if self.rejected_until.get(tree_id, 0.0) > now:
                continue
            if not self._accepted_cylinder(tracked):
                continue

            cylinder = tracked.cylinder
            source_frame = (
                cylinder.header.frame_id
                or msg.header.frame_id
                or self.pcl_source_frame
            )
            p = cylinder.pose.position
            transformed = self.transform_point(p.x, p.y, p.z, source_frame)
            if transformed is None:
                continue
            x, y, z = transformed
            accepted_ids.add(tree_id)
            confidence = max(0.0, min(1.0, float(cylinder.confidence)))
            existing = self.tree_database.get(tree_id)

            if existing is None:
                self.tree_database[tree_id] = TreeRecord(
                    tree_id=tree_id,
                    x=x,
                    y=y,
                    z=z,
                    confidence=confidence,
                    validated=True,
                    inspected=False,
                    last_seen=now,
                    seen_count=int(tracked.seen_count),
                    missed_count=int(tracked.missed_count),
                    radius=float(cylinder.radius),
                    height=float(cylinder.height),
                )
                self.get_logger().info(
                    f"Pohon PCL baru ID={tree_id} di ({x:.2f}, {y:.2f}) "
                    f"frame={self.world_frame}."
                )
            else:
                alpha = 0.30
                existing.x = (1.0 - alpha) * existing.x + alpha * x
                existing.y = (1.0 - alpha) * existing.y + alpha * y
                existing.z = (1.0 - alpha) * existing.z + alpha * z
                existing.confidence = max(existing.confidence, confidence)
                existing.validated = True
                existing.last_seen = now
                existing.seen_count = int(tracked.seen_count)
                existing.missed_count = int(tracked.missed_count)
                existing.radius = float(cylinder.radius)
                existing.height = float(cylinder.height)

        for tree_id, record in self.tree_database.items():
            if record.source == "pcl" and tree_id not in accepted_ids:
                record.missed_count += 1

        self.publish_all()

    def fallback_point_callback(self, msg: PointStamped) -> None:
        p = msg.point
        if not self._finite(p.x, p.y, p.z):
            return
        transformed = self.transform_point(
            p.x, p.y, p.z, msg.header.frame_id or self.world_frame
        )
        if transformed is None:
            return
        x, y, z = transformed
        now = self.now_sec()

        nearest: Optional[TreeRecord] = None
        nearest_distance = float("inf")
        for record in self.tree_database.values():
            distance = math.hypot(record.x - x, record.y - y)
            if distance < nearest_distance:
                nearest = record
                nearest_distance = distance

        if nearest is not None and nearest_distance <= self.fallback_merge_distance:
            nearest.seen_count += 1
            alpha = min(0.5, 1.0 / max(2, nearest.seen_count))
            nearest.x += alpha * (x - nearest.x)
            nearest.y += alpha * (y - nearest.y)
            nearest.z += alpha * (z - nearest.z)
            nearest.confidence = min(1.0, nearest.confidence + 0.08)
            nearest.validated = nearest.seen_count >= 3
            nearest.last_seen = now
        else:
            tree_id = self.next_fallback_id
            self.next_fallback_id += 1
            self.tree_database[tree_id] = TreeRecord(
                tree_id=tree_id,
                x=x,
                y=y,
                z=z,
                confidence=0.25,
                validated=False,
                inspected=False,
                last_seen=now,
                source="fallback",
            )
        self.publish_all()

    def tree_update_callback(self, msg: Tree) -> None:
        tree_id = int(msg.id)
        now = self.now_sec()

        if float(msg.confidence) < 0.0:
            self.tree_database.pop(tree_id, None)
            self.rejected_until[tree_id] = now + self.reject_hold_sec
            self.get_logger().warning(
                f"Pohon ID={tree_id} ditolak selama {self.reject_hold_sec:.1f} detik."
            )
            self.publish_all()
            return

        record = self.tree_database.get(tree_id)
        if record is None:
            self.get_logger().warning(f"Update untuk ID pohon tidak dikenal: {tree_id}.")
            return

        if bool(msg.inspected):
            record.inspected = True
        if hasattr(msg, "validated") and bool(msg.validated):
            record.validated = True
        if 0.0 <= float(msg.confidence) <= 1.0:
            record.confidence = max(record.confidence, float(msg.confidence))
        if self._finite(msg.x, msg.y, msg.z) and not (
            msg.x == 0.0 and msg.y == 0.0 and msg.z == 0.0
        ):
            record.x = float(msg.x)
            record.y = float(msg.y)
            record.z = float(msg.z)
        # Mission updates must not make a stale sensor map appear fresh.
        self.publish_all()

    def pcl_stream_fresh(self) -> bool:
        return (
            self.last_pcl_msg_time is not None
            and self.now_sec() - self.last_pcl_msg_time <= self.pcl_stream_timeout_sec
        )

    def map_ready(self) -> bool:
        valid_count = sum(
            record.validated for record in self.tree_database.values()
        )
        return self.pcl_stream_fresh() and valid_count >= self.min_ready_trees

    def housekeeping(self) -> None:
        now = self.now_sec()
        for tree_id in [
            tree_id
            for tree_id, expiry in self.rejected_until.items()
            if expiry <= now
        ]:
            self.rejected_until.pop(tree_id, None)

        # Do not erase the complete map just because the PCL pipeline is late.
        # Readiness becomes false and the mission holds position instead.
        if self.pcl_stream_fresh():
            stale_ids = [
                tree_id
                for tree_id, record in self.tree_database.items()
                if not record.inspected and now - record.last_seen > self.tree_stale_sec
            ]
            for tree_id in stale_ids:
                self.tree_database.pop(tree_id, None)
                self.get_logger().warning(
                    f"Pohon stale ID={tree_id} dihapus setelah {self.tree_stale_sec:.1f}s."
                )

        self.publish_all()

    @staticmethod
    def _to_tree_msg(record: TreeRecord) -> Tree:
        tree = Tree()
        tree.id = int(record.tree_id)
        tree.x = float(record.x)
        tree.y = float(record.y)
        tree.z = float(record.z)
        tree.confidence = float(record.confidence)
        tree.inspected = bool(record.inspected)
        if hasattr(tree, "validated"):
            tree.validated = bool(record.validated)
        return tree

    def publish_all(self) -> None:
        array = TreeArray()
        for tree_id in sorted(self.tree_database):
            array.trees.append(self._to_tree_msg(self.tree_database[tree_id]))
        self.tree_pub.publish(array)

        ready = self.map_ready()
        ready_msg = Bool()
        ready_msg.data = ready
        self.ready_pub.publish(ready_msg)

        count_msg = Int32()
        count_msg.data = len(array.trees)
        self.count_pub.publish(count_msg)

        if ready != self.last_ready_state:
            self.last_ready_state = ready

            message = (
                f"Map ready={ready}; trees={len(array.trees)}; "
                f"pcl_fresh={self.pcl_stream_fresh()}."
            )

            # Gunakan baris pemanggilan berbeda untuk setiap severity.
            # rclpy Humble tidak mengizinkan severity berubah pada call site
            # yang sama ketika metode logger dipilih secara dinamis.
            if ready:
                self.get_logger().info(message)
            else:
                self.get_logger().warning(message)

        self.publish_markers()

    def publish_markers(self) -> None:
        marker_array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        marker_id = 0
        for record in self.tree_database.values():
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "pcl_trees"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = record.x
            marker.pose.position.y = record.y
            marker.pose.position.z = max(0.5, record.z)
            marker.pose.orientation.w = 1.0
            diameter = max(0.25, 2.0 * record.radius)
            marker.scale.x = diameter
            marker.scale.y = diameter
            marker.scale.z = max(1.0, record.height)
            marker.color.a = 0.85
            if record.inspected:
                marker.color.g = 1.0
            elif record.validated:
                marker.color.r = 1.0
                marker.color.g = 0.65
            else:
                marker.color.r = 0.7
                marker.color.g = 0.7
                marker.color.b = 0.7
            marker_array.markers.append(marker)

            text = Marker()
            text.header = marker.header
            text.ns = "pcl_tree_labels"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = record.x
            text.pose.position.y = record.y
            text.pose.position.z = max(2.0, record.z + record.height * 0.5 + 0.5)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.45
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            status = "DONE" if record.inspected else "READY"
            text.text = f"ID:{record.tree_id} {status} C:{record.confidence:.2f}"
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PclTreeMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
