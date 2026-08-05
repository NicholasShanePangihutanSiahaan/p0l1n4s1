#!/usr/bin/env python3

import csv
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import VfrHud
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String
from uav_interfaces.msg import TreeArray


TRAJECTORY_FIELDS = [
    "wall_time_iso",
    "ros_time_sec",
    "elapsed_sec",
    "x_m",
    "y_m",
    "z_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "yaw_deg",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "speed_xy_mps",
    "speed_3d_mps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
    "accel_xy_mps2",
    "accel_3d_mps2",
    "step_distance_m",
    "total_distance_m",
    "home_x_m",
    "home_y_m",
    "home_z_m",
    "distance_to_home_2d_m",
    "distance_to_home_3d_m",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_error_x_m",
    "target_error_y_m",
    "target_error_z_m",
    "distance_to_target_2d_m",
    "distance_to_target_3d_m",
    "heading_deg",
    "airspeed_mps",
    "groundspeed_mps",
    "throttle_pct",
    "fsm_state",
    "mission_status",
    "orbit_status",
    "orbit_progress",
    "orbit_progress_pct",
    "active_tree_id",
    "active_tree_x_m",
    "active_tree_y_m",
    "active_tree_z_m",
    "active_tree_confidence",
    "distance_to_active_tree_2d_m",
    "distance_to_active_tree_3d_m",
    "nearest_tree_id",
    "nearest_tree_distance_2d_m",
    "tree_total",
    "tree_validated",
    "tree_inspected",
    "coverage_pct",
    "map_ready",
    "connected",
    "armed",
    "hovering",
    "flight_mode",
    "telemetry_altitude_m",
]

METRIC_FIELDS = [
    "wall_time_iso",
    "elapsed_sec",
    "fsm_state",
    "mission_status",
    "orbit_status",
    "orbit_progress_pct",
    "x_m",
    "y_m",
    "z_m",
    "speed_3d_mps",
    "total_distance_m",
    "distance_to_home_2d_m",
    "active_tree_id",
    "distance_to_active_tree_2d_m",
    "tree_total",
    "tree_validated",
    "tree_inspected",
    "coverage_pct",
    "max_altitude_m",
    "max_speed_3d_mps",
    "max_accel_3d_mps2",
    "minimum_active_tree_distance_2d_m",
    "connected",
    "armed",
    "hovering",
    "flight_mode",
]


class MissionAnalyzer(Node):
    """Mission logger with immediate CSV output, autosave, and 2D/3D maps."""

    def __init__(self) -> None:
        super().__init__("mission_analyzer")

        self.declare_parameter("output_dir", "~/beehive_mission_results")
        self.declare_parameter("sample_period_sec", 0.2)
        self.declare_parameter("summary_period_sec", 5.0)
        self.declare_parameter("autosave_period_sec", 10.0)
        self.declare_parameter("enable_live_plot", False)
        self.declare_parameter("save_2d_png", True)
        self.declare_parameter("save_3d_png", True)
        self.declare_parameter("save_png_during_mission", True)
        self.declare_parameter("plot_dpi", 180)
        self.declare_parameter("max_plot_samples", 20000)

        root_dir = Path(
            os.path.expanduser(str(self.get_parameter("output_dir").value))
        )
        root_dir.mkdir(parents=True, exist_ok=True)
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = root_dir / f"mission_{session_stamp}_{os.getpid()}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.sample_period_sec = max(
            0.02, float(self.get_parameter("sample_period_sec").value)
        )
        self.summary_period_sec = max(
            0.5, float(self.get_parameter("summary_period_sec").value)
        )
        self.autosave_period_sec = max(
            1.0, float(self.get_parameter("autosave_period_sec").value)
        )
        self.enable_live_plot = bool(self.get_parameter("enable_live_plot").value)
        self.save_2d_png = bool(self.get_parameter("save_2d_png").value)
        self.save_3d_png = bool(self.get_parameter("save_3d_png").value)
        self.save_png_during_mission = bool(
            self.get_parameter("save_png_during_mission").value
        )
        self.plot_dpi = max(72, int(self.get_parameter("plot_dpi").value))
        self.max_plot_samples = max(
            100, int(self.get_parameter("max_plot_samples").value)
        )

        self.start_time = self.now_sec()
        self.start_wall_time = self.wall_time_iso()
        self.last_sample_time = -1e9
        self.last_summary_time = -1e9
        self.last_autosave_time = -1e9
        self.current_state = "INIT"
        self.previous_state = ""
        self.mission_status = ""
        self.orbit_status = "IDLE"
        self.orbit_progress = 0.0
        self.current_pose: Optional[PoseStamped] = None
        self.safe_target: Optional[PoseStamped] = None
        self.current_velocity = (0.0, 0.0, 0.0)
        self.previous_velocity: Optional[Tuple[float, float, float]] = None
        self.previous_velocity_time: Optional[float] = None
        self.current_acceleration = (0.0, 0.0, 0.0)
        self.heading = 0.0
        self.airspeed = 0.0
        self.groundspeed = 0.0
        self.throttle = 0.0
        self.telemetry_altitude = math.nan
        self.connected = False
        self.armed = False
        self.hovering = False
        self.map_ready = False
        self.flight_mode = ""
        self.active_tree_id = -1
        self.trees = []
        self.rows = []
        self.state_events = []
        self.total_distance = 0.0
        self.previous_position: Optional[Tuple[float, float, float]] = None
        self.home_position: Optional[Tuple[float, float, float]] = None
        self.max_altitude = -math.inf
        self.max_speed_3d = 0.0
        self.max_accel_3d = 0.0
        self.minimum_active_tree_distance_2d = math.inf
        self.final_saved = False
        self.plt = None
        self.live_figure = None
        self.live_axis = None

        self.trajectory_path = self.run_dir / "trajectory_detailed.csv"
        self.states_path = self.run_dir / "state_events.csv"
        self.metrics_path = self.run_dir / "mission_metrics.csv"
        self.trees_path = self.run_dir / "trees_latest.csv"
        self.state_durations_path = self.run_dir / "state_durations.csv"
        self.summary_path = self.run_dir / "mission_summary.csv"
        self.map_2d_path = self.run_dir / "map_2d.png"
        self.map_3d_path = self.run_dir / "map_3d.png"

        self.trajectory_handle = self.trajectory_path.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.trajectory_writer = csv.DictWriter(
            self.trajectory_handle, fieldnames=TRAJECTORY_FIELDS
        )
        self.trajectory_writer.writeheader()
        self.trajectory_handle.flush()

        self.states_handle = self.states_path.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.states_writer = csv.DictWriter(
            self.states_handle,
            fieldnames=["wall_time_iso", "elapsed_sec", "previous_state", "state"],
        )
        self.states_writer.writeheader()
        self.states_handle.flush()

        self.metrics_handle = self.metrics_path.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self.metrics_writer = csv.DictWriter(
            self.metrics_handle, fieldnames=METRIC_FIELDS
        )
        self.metrics_writer.writeheader()
        self.metrics_handle.flush()

        self._append_state_event("", "INIT")

        if self.enable_live_plot:
            try:
                self._ensure_matplotlib(live=True)
                self.plt.ion()
                self.live_figure, self.live_axis = self.plt.subplots(figsize=(10, 8))
            except Exception as exc:  # noqa: BLE001
                self.enable_live_plot = False
                self.get_logger().warning(f"Live plot dinonaktifkan: {exc}")

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self.pose_callback, qos_sensor
        )
        self.create_subscription(
            TwistStamped,
            "/mavros/local_position/velocity_local",
            self.velocity_callback,
            qos_sensor,
        )
        self.create_subscription(VfrHud, "/mavros/vfr_hud", self.vfr_callback, qos_sensor)
        self.create_subscription(TreeArray, "/map/trees", self.tree_callback, qos_map)
        self.create_subscription(Bool, "/map/trees_ready", self.map_ready_callback, qos_map)
        self.create_subscription(String, "/mission/fsm_state", self.state_callback, 10)
        self.create_subscription(String, "/mission/status", self.status_callback, 10)
        self.create_subscription(
            String, "/control/orbit_status", self.orbit_status_callback, 10
        )
        self.create_subscription(
            Float32, "/control/orbit_progress", self.orbit_progress_callback, 10
        )
        self.create_subscription(
            Int32, "/control/active_tree_id", self.active_tree_callback, 10
        )
        self.create_subscription(
            PoseStamped, "/control/safe_target_pose", self.safe_target_callback, 10
        )
        self.create_subscription(
            Bool, "/flight/telemetry/is_connected", self.connected_callback, 10
        )
        self.create_subscription(
            Bool, "/flight/telemetry/is_armed", self.armed_callback, 10
        )
        self.create_subscription(
            Bool, "/flight/telemetry/is_hovering", self.hovering_callback, 10
        )
        self.create_subscription(
            String, "/flight/telemetry/current_mode", self.mode_callback, 10
        )
        self.create_subscription(
            Float32, "/flight/telemetry/altitude", self.altitude_callback, 10
        )

        self.create_timer(0.2, self.update)
        self.get_logger().info(
            "Mission Analyzer aktif; CSV dibuat langsung dan autosave aktif. "
            f"Output sesi={self.run_dir}"
        )
        self.get_logger().info(
            f"Trajectory CSV={self.trajectory_path}; 3D map={self.map_3d_path}"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def wall_time_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

    @staticmethod
    def finite_or_blank(value: float):
        return value if math.isfinite(value) else ""

    @staticmethod
    def quaternion_to_rpy(msg: PoseStamped) -> Tuple[float, float, float]:
        q = msg.pose.orientation
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def pose_callback(self, msg: PoseStamped) -> None:
        self.current_pose = msg
        now = self.now_sec()
        if now - self.last_sample_time < self.sample_period_sec:
            return
        self.last_sample_time = now

        p = msg.pose.position
        current = (float(p.x), float(p.y), float(p.z))
        if self.home_position is None:
            self.home_position = current

        step = 0.0
        if self.previous_position is not None:
            dx = current[0] - self.previous_position[0]
            dy = current[1] - self.previous_position[1]
            dz = current[2] - self.previous_position[2]
            candidate = math.sqrt(dx * dx + dy * dy + dz * dz)
            if candidate < 3.0:
                step = candidate
                self.total_distance += step
        self.previous_position = current

        vx, vy, vz = self.current_velocity
        speed_xy = math.hypot(vx, vy)
        speed_3d = math.sqrt(vx * vx + vy * vy + vz * vz)
        ax, ay, az = self.current_acceleration
        accel_xy = math.hypot(ax, ay)
        accel_3d = math.sqrt(ax * ax + ay * ay + az * az)
        roll, pitch, yaw = self.quaternion_to_rpy(msg)

        self.max_altitude = max(self.max_altitude, current[2])
        self.max_speed_3d = max(self.max_speed_3d, speed_3d)
        self.max_accel_3d = max(self.max_accel_3d, accel_3d)

        home = self.home_position
        home_dx = current[0] - home[0]
        home_dy = current[1] - home[1]
        home_dz = current[2] - home[2]
        distance_home_2d = math.hypot(home_dx, home_dy)
        distance_home_3d = math.sqrt(home_dx * home_dx + home_dy * home_dy + home_dz * home_dz)

        target_values = self.target_values(current)
        active_values = self.active_tree_values(current)
        nearest_id, nearest_distance = self.nearest_tree(current)
        total, validated, inspected = self.tree_counts()
        coverage = 100.0 * inspected / total if total else 0.0

        active_distance_2d = active_values[4]
        if math.isfinite(active_distance_2d):
            self.minimum_active_tree_distance_2d = min(
                self.minimum_active_tree_distance_2d, active_distance_2d
            )

        row = {
            "wall_time_iso": self.wall_time_iso(),
            "ros_time_sec": now,
            "elapsed_sec": now - self.start_time,
            "x_m": current[0],
            "y_m": current[1],
            "z_m": current[2],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "yaw_deg": math.degrees(yaw),
            "vx_mps": vx,
            "vy_mps": vy,
            "vz_mps": vz,
            "speed_xy_mps": speed_xy,
            "speed_3d_mps": speed_3d,
            "ax_mps2": ax,
            "ay_mps2": ay,
            "az_mps2": az,
            "accel_xy_mps2": accel_xy,
            "accel_3d_mps2": accel_3d,
            "step_distance_m": step,
            "total_distance_m": self.total_distance,
            "home_x_m": home[0],
            "home_y_m": home[1],
            "home_z_m": home[2],
            "distance_to_home_2d_m": distance_home_2d,
            "distance_to_home_3d_m": distance_home_3d,
            "target_x_m": target_values[0],
            "target_y_m": target_values[1],
            "target_z_m": target_values[2],
            "target_error_x_m": target_values[3],
            "target_error_y_m": target_values[4],
            "target_error_z_m": target_values[5],
            "distance_to_target_2d_m": target_values[6],
            "distance_to_target_3d_m": target_values[7],
            "heading_deg": self.heading,
            "airspeed_mps": self.airspeed,
            "groundspeed_mps": self.groundspeed,
            "throttle_pct": self.throttle,
            "fsm_state": self.current_state,
            "mission_status": self.mission_status,
            "orbit_status": self.orbit_status,
            "orbit_progress": self.orbit_progress,
            "orbit_progress_pct": 100.0 * self.orbit_progress,
            "active_tree_id": self.active_tree_id,
            "active_tree_x_m": active_values[0],
            "active_tree_y_m": active_values[1],
            "active_tree_z_m": active_values[2],
            "active_tree_confidence": active_values[3],
            "distance_to_active_tree_2d_m": active_values[4],
            "distance_to_active_tree_3d_m": active_values[5],
            "nearest_tree_id": nearest_id,
            "nearest_tree_distance_2d_m": nearest_distance,
            "tree_total": total,
            "tree_validated": validated,
            "tree_inspected": inspected,
            "coverage_pct": coverage,
            "map_ready": self.map_ready,
            "connected": self.connected,
            "armed": self.armed,
            "hovering": self.hovering,
            "flight_mode": self.flight_mode,
            "telemetry_altitude_m": self.finite_or_blank(self.telemetry_altitude),
        }

        self.rows.append(row)
        if len(self.rows) > self.max_plot_samples:
            self.rows = self.rows[-self.max_plot_samples :]
        self.trajectory_writer.writerow(row)
        self.trajectory_handle.flush()

    def velocity_callback(self, msg: TwistStamped) -> None:
        now = self.now_sec()
        v = msg.twist.linear
        current = (float(v.x), float(v.y), float(v.z))
        if self.previous_velocity is not None and self.previous_velocity_time is not None:
            dt = now - self.previous_velocity_time
            if 1e-3 <= dt <= 2.0:
                self.current_acceleration = (
                    (current[0] - self.previous_velocity[0]) / dt,
                    (current[1] - self.previous_velocity[1]) / dt,
                    (current[2] - self.previous_velocity[2]) / dt,
                )
        self.current_velocity = current
        self.previous_velocity = current
        self.previous_velocity_time = now

    def vfr_callback(self, msg: VfrHud) -> None:
        self.heading = float(msg.heading)
        self.airspeed = float(msg.airspeed)
        self.groundspeed = float(msg.groundspeed)
        self.throttle = float(msg.throttle)

    def tree_callback(self, msg: TreeArray) -> None:
        self.trees = list(msg.trees)

    def map_ready_callback(self, msg: Bool) -> None:
        self.map_ready = bool(msg.data)

    def state_callback(self, msg: String) -> None:
        if msg.data == self.current_state:
            return
        previous = self.current_state
        self.previous_state = previous
        self.current_state = msg.data
        self._append_state_event(previous, msg.data)
        self.save_checkpoint(final=msg.data == "DONE", reason=f"state={msg.data}")

    def status_callback(self, msg: String) -> None:
        self.mission_status = msg.data

    def orbit_status_callback(self, msg: String) -> None:
        self.orbit_status = msg.data

    def orbit_progress_callback(self, msg: Float32) -> None:
        self.orbit_progress = max(0.0, min(1.0, float(msg.data)))

    def active_tree_callback(self, msg: Int32) -> None:
        self.active_tree_id = int(msg.data)

    def safe_target_callback(self, msg: PoseStamped) -> None:
        self.safe_target = msg

    def connected_callback(self, msg: Bool) -> None:
        self.connected = bool(msg.data)

    def armed_callback(self, msg: Bool) -> None:
        self.armed = bool(msg.data)

    def hovering_callback(self, msg: Bool) -> None:
        self.hovering = bool(msg.data)

    def mode_callback(self, msg: String) -> None:
        self.flight_mode = msg.data

    def altitude_callback(self, msg: Float32) -> None:
        self.telemetry_altitude = float(msg.data)

    def _append_state_event(self, previous: str, state: str) -> None:
        event = {
            "wall_time_iso": self.wall_time_iso(),
            "elapsed_sec": self.now_sec() - self.start_time,
            "previous_state": previous,
            "state": state,
        }
        self.state_events.append(event)
        self.states_writer.writerow(event)
        self.states_handle.flush()

    def tree_counts(self) -> Tuple[int, int, int]:
        total = len(self.trees)
        inspected = sum(bool(getattr(tree, "inspected", False)) for tree in self.trees)
        validated = sum(
            bool(
                getattr(
                    tree,
                    "validated",
                    float(getattr(tree, "confidence", 0.0)) >= 0.35,
                )
            )
            for tree in self.trees
        )
        return total, validated, inspected

    def find_tree(self, tree_id: int):
        return next(
            (tree for tree in self.trees if int(getattr(tree, "id", -1)) == tree_id),
            None,
        )

    def active_tree_values(
        self, current: Tuple[float, float, float]
    ) -> Tuple[object, object, object, object, float, float]:
        tree = self.find_tree(self.active_tree_id)
        if tree is None:
            return "", "", "", "", math.nan, math.nan
        tx = float(tree.x)
        ty = float(tree.y)
        tz = float(tree.z)
        dx = current[0] - tx
        dy = current[1] - ty
        dz = current[2] - tz
        return (
            tx,
            ty,
            tz,
            float(getattr(tree, "confidence", 0.0)),
            math.hypot(dx, dy),
            math.sqrt(dx * dx + dy * dy + dz * dz),
        )

    def nearest_tree(self, current: Tuple[float, float, float]) -> Tuple[object, object]:
        best_id = ""
        best_distance = math.inf
        for tree in self.trees:
            distance = math.hypot(current[0] - float(tree.x), current[1] - float(tree.y))
            if distance < best_distance:
                best_distance = distance
                best_id = int(tree.id)
        return best_id, self.finite_or_blank(best_distance)

    def target_values(
        self, current: Tuple[float, float, float]
    ) -> Tuple[object, object, object, object, object, object, object, object]:
        if self.safe_target is None:
            return "", "", "", "", "", "", "", ""
        p = self.safe_target.pose.position
        tx, ty, tz = float(p.x), float(p.y), float(p.z)
        ex = tx - current[0]
        ey = ty - current[1]
        ez = tz - current[2]
        return tx, ty, tz, ex, ey, ez, math.hypot(ex, ey), math.sqrt(ex * ex + ey * ey + ez * ez)

    def update(self) -> None:
        now = self.now_sec()
        if now - self.last_summary_time >= self.summary_period_sec:
            self.last_summary_time = now
            self.log_summary_and_metric(now)

        if now - self.last_autosave_time >= self.autosave_period_sec:
            self.last_autosave_time = now
            self.save_checkpoint(final=False, reason="autosave")

        if self.enable_live_plot:
            self.draw_live_map()

    def current_metrics(self, now: Optional[float] = None) -> Dict[str, object]:
        now = self.now_sec() if now is None else now
        total, validated, inspected = self.tree_counts()
        coverage = 100.0 * inspected / total if total else 0.0
        x = y = z = speed = distance_home = math.nan
        active_distance = math.nan
        if self.rows:
            last = self.rows[-1]
            x = float(last["x_m"])
            y = float(last["y_m"])
            z = float(last["z_m"])
            speed = float(last["speed_3d_mps"])
            distance_home = float(last["distance_to_home_2d_m"])
            active_raw = last["distance_to_active_tree_2d_m"]
            if active_raw != "":
                active_distance = float(active_raw)

        return {
            "wall_time_iso": self.wall_time_iso(),
            "elapsed_sec": now - self.start_time,
            "fsm_state": self.current_state,
            "mission_status": self.mission_status,
            "orbit_status": self.orbit_status,
            "orbit_progress_pct": 100.0 * self.orbit_progress,
            "x_m": self.finite_or_blank(x),
            "y_m": self.finite_or_blank(y),
            "z_m": self.finite_or_blank(z),
            "speed_3d_mps": self.finite_or_blank(speed),
            "total_distance_m": self.total_distance,
            "distance_to_home_2d_m": self.finite_or_blank(distance_home),
            "active_tree_id": self.active_tree_id,
            "distance_to_active_tree_2d_m": self.finite_or_blank(active_distance),
            "tree_total": total,
            "tree_validated": validated,
            "tree_inspected": inspected,
            "coverage_pct": coverage,
            "max_altitude_m": self.finite_or_blank(self.max_altitude),
            "max_speed_3d_mps": self.max_speed_3d,
            "max_accel_3d_mps2": self.max_accel_3d,
            "minimum_active_tree_distance_2d_m": self.finite_or_blank(
                self.minimum_active_tree_distance_2d
            ),
            "connected": self.connected,
            "armed": self.armed,
            "hovering": self.hovering,
            "flight_mode": self.flight_mode,
        }

    def log_summary_and_metric(self, now: float) -> None:
        metrics = self.current_metrics(now)
        self.metrics_writer.writerow(metrics)
        self.metrics_handle.flush()
        self.get_logger().info(
            f"State={self.current_state} | t={float(metrics['elapsed_sec']):.1f}s | "
            f"distance={self.total_distance:.1f}m | "
            f"trees={metrics['tree_inspected']}/{metrics['tree_total']} | "
            f"validated={metrics['tree_validated']} | "
            f"orbit={float(metrics['orbit_progress_pct']):.1f}% | "
            f"armed={self.armed} | hovering={self.hovering}"
        )

    def _ensure_matplotlib(self, live: bool = False) -> None:
        if self.plt is not None:
            return
        import matplotlib

        if not live:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt

    def draw_live_map(self) -> None:
        if self.live_axis is None or self.live_figure is None:
            return
        try:
            self._draw_2d(self.live_axis)
            self.live_figure.canvas.draw_idle()
            self.live_figure.canvas.flush_events()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"Live plot gagal: {exc}")
            self.enable_live_plot = False

    def _draw_2d(self, axis) -> None:
        axis.clear()
        axis.set_title("UAV Mission Map — Top View")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.grid(True)
        axis.set_aspect("equal", adjustable="box")

        if len(self.rows) > 1:
            axis.plot(
                [float(row["x_m"]) for row in self.rows],
                [float(row["y_m"]) for row in self.rows],
                linewidth=1.5,
                label="Trajectory",
            )
        if self.current_pose is not None:
            p = self.current_pose.pose.position
            axis.scatter(p.x, p.y, s=80, marker="o", label="UAV")
        if self.home_position is not None:
            axis.scatter(
                self.home_position[0],
                self.home_position[1],
                marker="*",
                s=140,
                label="Takeoff/Home",
            )

        for tree in self.trees:
            inspected = bool(getattr(tree, "inspected", False))
            marker = "x" if inspected else "o"
            size = 70 if int(tree.id) == self.active_tree_id else 45
            axis.scatter(float(tree.x), float(tree.y), s=size, marker=marker)
            axis.text(float(tree.x), float(tree.y) + 0.25, str(tree.id), fontsize=8)

        total, validated, inspected = self.tree_counts()
        axis.text(
            0.02,
            0.98,
            f"State: {self.current_state}\nTrees: {inspected}/{total}\n"
            f"Validated: {validated}\nDistance: {self.total_distance:.1f} m\n"
            f"Orbit: {100.0 * self.orbit_progress:.1f}%",
            transform=axis.transAxes,
            verticalalignment="top",
        )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")

    def _draw_3d(self, axis) -> None:
        axis.set_title("UAV Mission Map — 3D Trajectory")
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.grid(True)

        if len(self.rows) > 1:
            axis.plot(
                [float(row["x_m"]) for row in self.rows],
                [float(row["y_m"]) for row in self.rows],
                [float(row["z_m"]) for row in self.rows],
                linewidth=1.5,
                label="Trajectory",
            )
        if self.current_pose is not None:
            p = self.current_pose.pose.position
            axis.scatter(p.x, p.y, p.z, s=70, marker="o", label="UAV")
        if self.home_position is not None:
            axis.scatter(
                self.home_position[0],
                self.home_position[1],
                self.home_position[2],
                marker="*",
                s=130,
                label="Takeoff/Home",
            )

        for tree in self.trees:
            tx, ty, tz = float(tree.x), float(tree.y), float(tree.z)
            marker = "x" if bool(getattr(tree, "inspected", False)) else "o"
            size = 70 if int(tree.id) == self.active_tree_id else 45
            axis.scatter(tx, ty, tz, s=size, marker=marker)
            axis.text(tx, ty, tz + 0.15, str(tree.id), fontsize=8)

        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best")
        try:
            axis.set_box_aspect((1.0, 1.0, 0.6))
        except Exception:  # noqa: BLE001
            pass

    def save_plots(self) -> None:
        if not self.rows or not (self.save_2d_png or self.save_3d_png):
            return
        self._ensure_matplotlib(live=self.enable_live_plot)

        if self.save_2d_png:
            figure, axis = self.plt.subplots(figsize=(10, 8))
            self._draw_2d(axis)
            figure.savefig(self.map_2d_path, dpi=self.plot_dpi, bbox_inches="tight")
            self.plt.close(figure)

        if self.save_3d_png:
            figure = self.plt.figure(figsize=(11, 9))
            axis = figure.add_subplot(111, projection="3d")
            self._draw_3d(axis)
            figure.savefig(self.map_3d_path, dpi=self.plot_dpi, bbox_inches="tight")
            self.plt.close(figure)

    def write_tree_snapshot(self) -> None:
        temporary = self.trees_path.with_suffix(".tmp")
        current = None
        if self.current_pose is not None:
            p = self.current_pose.pose.position
            current = (float(p.x), float(p.y), float(p.z))
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "id",
                "x_m",
                "y_m",
                "z_m",
                "confidence",
                "validated",
                "inspected",
                "is_active",
                "distance_from_uav_2d_m",
                "distance_from_uav_3d_m",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for tree in sorted(self.trees, key=lambda item: int(item.id)):
                d2 = d3 = ""
                if current is not None:
                    dx = current[0] - float(tree.x)
                    dy = current[1] - float(tree.y)
                    dz = current[2] - float(tree.z)
                    d2 = math.hypot(dx, dy)
                    d3 = math.sqrt(dx * dx + dy * dy + dz * dz)
                writer.writerow(
                    {
                        "id": int(tree.id),
                        "x_m": float(tree.x),
                        "y_m": float(tree.y),
                        "z_m": float(tree.z),
                        "confidence": float(getattr(tree, "confidence", 0.0)),
                        "validated": bool(
                            getattr(
                                tree,
                                "validated",
                                float(getattr(tree, "confidence", 0.0)) >= 0.35,
                            )
                        ),
                        "inspected": bool(getattr(tree, "inspected", False)),
                        "is_active": int(tree.id) == self.active_tree_id,
                        "distance_from_uav_2d_m": d2,
                        "distance_from_uav_3d_m": d3,
                    }
                )
        temporary.replace(self.trees_path)

    def state_duration_rows(self, end_elapsed: float) -> Iterable[Dict[str, object]]:
        if not self.state_events:
            return []
        rows = []
        for index, event in enumerate(self.state_events):
            start = float(event["elapsed_sec"])
            end = (
                float(self.state_events[index + 1]["elapsed_sec"])
                if index + 1 < len(self.state_events)
                else end_elapsed
            )
            rows.append(
                {
                    "state": event["state"],
                    "start_elapsed_sec": start,
                    "end_elapsed_sec": end,
                    "duration_sec": max(0.0, end - start),
                }
            )
        return rows

    def write_state_durations(self, end_elapsed: float) -> None:
        temporary = self.state_durations_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            fields = ["state", "start_elapsed_sec", "end_elapsed_sec", "duration_sec"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.state_duration_rows(end_elapsed))
        temporary.replace(self.state_durations_path)

    def write_summary(self, final: bool, reason: str) -> None:
        metrics = self.current_metrics()
        final_x = final_y = final_z = final_home_distance = ""
        if self.rows:
            last = self.rows[-1]
            final_x = last["x_m"]
            final_y = last["y_m"]
            final_z = last["z_m"]
            final_home_distance = last["distance_to_home_2d_m"]
        summary = {
            "start_wall_time_iso": self.start_wall_time,
            "last_save_wall_time_iso": self.wall_time_iso(),
            "save_reason": reason,
            "is_final": final,
            "duration_sec": metrics["elapsed_sec"],
            "final_state": self.current_state,
            "mission_status": self.mission_status,
            "orbit_status": self.orbit_status,
            "orbit_progress_pct": metrics["orbit_progress_pct"],
            "sample_count": len(self.rows),
            "total_distance_m": self.total_distance,
            "max_altitude_m": self.finite_or_blank(self.max_altitude),
            "max_speed_3d_mps": self.max_speed_3d,
            "max_accel_3d_mps2": self.max_accel_3d,
            "minimum_active_tree_distance_2d_m": self.finite_or_blank(
                self.minimum_active_tree_distance_2d
            ),
            "tree_total": metrics["tree_total"],
            "tree_validated": metrics["tree_validated"],
            "tree_inspected": metrics["tree_inspected"],
            "coverage_pct": metrics["coverage_pct"],
            "final_x_m": final_x,
            "final_y_m": final_y,
            "final_z_m": final_z,
            "final_distance_to_home_2d_m": final_home_distance,
            "connected": self.connected,
            "armed": self.armed,
            "hovering": self.hovering,
            "flight_mode": self.flight_mode,
            "output_directory": str(self.run_dir),
        }
        temporary = self.summary_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)
        temporary.replace(self.summary_path)

    def save_checkpoint(self, final: bool, reason: str) -> None:
        if final and self.final_saved:
            return
        try:
            self.trajectory_handle.flush()
            self.states_handle.flush()
            self.metrics_handle.flush()
            self.write_tree_snapshot()
            elapsed = self.now_sec() - self.start_time
            self.write_state_durations(elapsed)
            self.write_summary(final=final, reason=reason)
            if final or self.save_png_during_mission:
                self.save_plots()
            if final:
                self.final_saved = True
                self.get_logger().info(
                    f"Hasil akhir misi tersimpan di {self.run_dir}"
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Gagal menyimpan hasil analyzer: {exc}")

    def close_files(self) -> None:
        for handle in (
            self.trajectory_handle,
            self.states_handle,
            self.metrics_handle,
        ):
            try:
                if not handle.closed:
                    handle.flush()
                    handle.close()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self) -> None:
        self.save_checkpoint(final=True, reason="node_shutdown")
        self.close_files()
        if self.live_figure is not None and self.plt is not None:
            self.plt.close(self.live_figure)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionAnalyzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
