#!/usr/bin/env python3
"""Mission telemetry recorder and headless 2D/3D report generator."""

import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, TwistStamped
from mavros_msgs.msg import ExtendedState, State, StatusText
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import FluidPressure, Range
from std_msgs.msg import Bool, Float32, Float64, String
from uav_interfaces.msg import TreeArray


class MissionAnalyzer(Node):
    def __init__(self):
        super().__init__('mission_analyzer')

        self.declare_parameter('output_directory', '~/beehive_mission_reports')
        self.declare_parameter('sample_period', 0.20)
        self.declare_parameter('console_period', 2.0)
        self.declare_parameter('autosave_period', 10.0)
        self.declare_parameter('map_frame', 'odom')
        self.declare_parameter('expected_takeoff_altitude', 1.5)
        self.declare_parameter('topic_stale_threshold', 1.5)
        self.declare_parameter('altitude_jump_threshold', 0.50)
        self.declare_parameter('altitude_disagreement_threshold', 0.75)

        root = Path(os.path.expanduser(str(
            self.get_parameter('output_directory').value))).resolve()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = root / f'mission_{stamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.map_frame = str(self.get_parameter('map_frame').value)
        self.sample_period = max(0.05, float(self.get_parameter('sample_period').value))
        self.console_period = max(0.5, float(self.get_parameter('console_period').value))
        self.autosave_period = max(2.0, float(self.get_parameter('autosave_period').value))
        self.expected_takeoff_altitude = float(self.get_parameter('expected_takeoff_altitude').value)
        self.topic_stale_threshold = max(0.1, float(self.get_parameter('topic_stale_threshold').value))
        self.altitude_jump_threshold = max(0.05, float(self.get_parameter('altitude_jump_threshold').value))
        self.altitude_disagreement_threshold = max(0.05, float(
            self.get_parameter('altitude_disagreement_threshold').value))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.start_wall = time.time()
        self.latest_pose = None
        self.latest_pose_stamp = None
        self.home = None
        self.trees = {}
        self.waypoints = []
        self.samples = []
        self.state_events = []
        self.tree_snapshots = []
        self.current_state = 'WAITING'
        self.previous_sample = None
        self.total_distance_3d = 0.0
        self.total_distance_xy = 0.0
        self.max_speed = 0.0
        self.max_altitude = -math.inf
        self.min_altitude = math.inf
        self.final_saved = False

        # Snapshot diagnostik pasif. Analyzer tidak pernah mengirim command atau
        # setpoint; nilai yang belum pernah diterima disimpan sebagai NaN.
        self.diagnostics = {
            'connected': False, 'armed': False, 'flight_mode': '',
            'landed_state': -1, 'rangefinder_m': math.nan,
            'relative_alt_m': math.nan, 'local_vz_mps': math.nan,
            'static_pressure_pa': math.nan,
            'zed_z_m': math.nan, 'vision_z_m': math.nan,
            'telemetry_altitude_m': math.nan, 'target_altitude_m': math.nan,
            'setpoint_z_m': math.nan, 'is_hovering': False,
        }
        self.diagnostic_samples = []
        self.fc_messages = []
        self.diagnostic_events = []
        self.topic_last_seen = {}
        self.altitude_baselines = {}
        self.previous_diagnostic = None
        self.last_event_time = {}

        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self.pose_callback, sensor_qos)
        self.create_subscription(TreeArray, '/map/trees', self.tree_callback, map_qos)
        self.create_subscription(String, '/mission/fsm_state', self.state_callback, 10)
        self.create_subscription(PoseArray, '/mission/inspection_waypoints',
                                 self.waypoint_callback, 10)
        self.create_subscription(State, '/mavros/state', self.mavros_state_callback, 10)
        self.create_subscription(ExtendedState, '/mavros/extended_state',
                                 self.extended_state_callback, 10)
        self.create_subscription(Range, '/mavros/rangefinder/rangefinder',
                                 self.rangefinder_callback, sensor_qos)
        self.create_subscription(FluidPressure, '/mavros/imu/static_pressure',
                                 self.pressure_callback, sensor_qos)
        self.create_subscription(Float64, '/mavros/global_position/rel_alt',
                                 self.relative_alt_callback, sensor_qos)
        self.create_subscription(TwistStamped, '/mavros/local_position/velocity_local',
                                 self.velocity_callback, sensor_qos)
        self.create_subscription(PoseStamped, '/mavros/vision_pose/pose',
                                 self.vision_pose_callback, sensor_qos)
        self.create_subscription(PoseStamped, '/zed/zed_node/pose',
                                 self.zed_pose_callback, sensor_qos)
        self.create_subscription(Float32, '/flight/telemetry/altitude',
                                 self.telemetry_altitude_callback, 10)
        self.create_subscription(Float32, '/flight/target_altitude',
                                 self.target_altitude_callback, 10)
        # Mission utama mengirim target takeoff pada topic command ini. Dengarkan
        # juga topic tersebut agar target/error tetap tercatat tanpa perlu
        # mengubah flight_manager.
        self.create_subscription(Float32, '/flight/cmd/takeoff',
                                 self.target_altitude_callback, 10)
        self.create_subscription(Bool, '/flight/telemetry/is_hovering',
                                 self.hover_callback, 10)
        self.create_subscription(PoseStamped, '/mavros/setpoint_position/local',
                                 self.setpoint_callback, 10)
        self.create_subscription(StatusText, '/mavros/statustext/recv',
                                 self.statustext_callback, 10)
        self.create_timer(self.sample_period, self.sample)
        self.create_timer(self.console_period, self.print_status)
        self.create_timer(self.autosave_period, self.autosave)

        self.get_logger().info(
            f'Mission Analyzer aktif; laporan: {self.output_dir}')

    def elapsed(self):
        return time.time() - self.start_wall

    def mark_received(self, name):
        self.topic_last_seen[name] = self.elapsed()

    def topic_age(self, name, now):
        stamp = self.topic_last_seen.get(name)
        return now - stamp if stamp is not None else math.nan

    def add_diagnostic_event(self, event, value=math.nan, detail='', cooldown=1.0):
        now = self.elapsed()
        if now - self.last_event_time.get(event, -math.inf) < cooldown:
            return
        self.last_event_time[event] = now
        self.diagnostic_events.append({
            'time_s': now, 'state': self.current_state, 'event': event,
            'value': value, 'detail': detail})

    @staticmethod
    def yaw_from_quaternion(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def pose_callback(self, msg):
        self.mark_received('local_pose')
        self.latest_pose = msg.pose
        self.latest_pose_stamp = msg.header.stamp
        if self.home is None:
            p = msg.pose.position
            self.home = (float(p.x), float(p.y), float(p.z))

    def state_callback(self, msg):
        if msg.data == self.current_state:
            return
        self.current_state = msg.data
        self.state_events.append({'time_s': self.elapsed(), 'state': msg.data})
        self.get_logger().info(f'[STATE] {msg.data}')
        if msg.data in ('DONE', 'ABORT', 'MANUAL_OVERRIDE'):
            self.save_all(final=True)

    def tree_callback(self, msg):
        now = self.elapsed()
        updated = {}
        for tree in msg.trees:
            item = {
                'id': int(tree.id), 'x': float(tree.x), 'y': float(tree.y),
                'z': float(tree.z), 'confidence': float(tree.confidence),
                'inspected': bool(tree.inspected), 'validated': bool(tree.validated),
                'orbit_count': int(tree.orbit_count), 'time_s': now,
            }
            updated[item['id']] = item
        self.trees = updated
        self.tree_snapshots.extend(updated.values())

    def waypoint_callback(self, msg):
        self.waypoints = [(float(p.position.x), float(p.position.y),
                           float(p.position.z)) for p in msg.poses]

    def mavros_state_callback(self, msg):
        self.mark_received('mavros_state')
        self.diagnostics['connected'] = bool(msg.connected)
        self.diagnostics['armed'] = bool(msg.armed)
        self.diagnostics['flight_mode'] = str(msg.mode)

    def extended_state_callback(self, msg):
        self.mark_received('extended_state')
        self.diagnostics['landed_state'] = int(msg.landed_state)

    def rangefinder_callback(self, msg):
        self.mark_received('rangefinder')
        self.diagnostics['rangefinder_m'] = float(msg.range)

    def pressure_callback(self, msg):
        self.mark_received('static_pressure')
        self.diagnostics['static_pressure_pa'] = float(msg.fluid_pressure)

    def relative_alt_callback(self, msg):
        self.mark_received('relative_alt')
        self.diagnostics['relative_alt_m'] = float(msg.data)

    def velocity_callback(self, msg):
        self.mark_received('local_velocity')
        self.diagnostics['local_vz_mps'] = float(msg.twist.linear.z)

    def vision_pose_callback(self, msg):
        self.mark_received('vision_pose')
        self.diagnostics['vision_z_m'] = float(msg.pose.position.z)

    def zed_pose_callback(self, msg):
        self.mark_received('zed_pose')
        self.diagnostics['zed_z_m'] = float(msg.pose.position.z)

    def telemetry_altitude_callback(self, msg):
        self.mark_received('telemetry_altitude')
        self.diagnostics['telemetry_altitude_m'] = float(msg.data)

    def target_altitude_callback(self, msg):
        self.mark_received('target_altitude')
        self.diagnostics['target_altitude_m'] = float(msg.data)

    def hover_callback(self, msg):
        self.mark_received('hover')
        self.diagnostics['is_hovering'] = bool(msg.data)

    def setpoint_callback(self, msg):
        self.mark_received('setpoint')
        self.diagnostics['setpoint_z_m'] = float(msg.pose.position.z)

    def statustext_callback(self, msg):
        self.fc_messages.append({
            'time_s': self.elapsed(), 'severity': int(msg.severity),
            'text': str(msg.text),
        })

    def sample(self):
        if self.latest_pose is None:
            return
        now = self.elapsed()
        p = self.latest_pose.position
        yaw = self.yaw_from_quaternion(self.latest_pose.orientation)
        speed = 0.0
        if self.previous_sample is not None:
            dt = now - self.previous_sample['time_s']
            dx = p.x - self.previous_sample['x']
            dy = p.y - self.previous_sample['y']
            dz = p.z - self.previous_sample['z']
            dxy = math.hypot(dx, dy)
            d3 = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.total_distance_xy += dxy
            self.total_distance_3d += d3
            if dt > 1e-3:
                speed = d3 / dt
                self.max_speed = max(self.max_speed, speed)
        row = {
            'time_s': now, 'x': float(p.x), 'y': float(p.y), 'z': float(p.z),
            'yaw_deg': math.degrees(yaw), 'speed_mps': speed,
            'state': self.current_state,
        }
        self.samples.append(row)
        diag = {
            'time_s': now, 'state': self.current_state,
            'local_z_m': float(p.z), **self.diagnostics,
        }
        sources = {
            'relative_alt_m': diag['relative_alt_m'],
            'local_z_m': diag['local_z_m'],
            'zed_z_m': diag['zed_z_m'],
            'vision_z_m': diag['vision_z_m'],
        }
        for key, value in sources.items():
            if key not in self.altitude_baselines and math.isfinite(value):
                self.altitude_baselines[key] = value
            baseline = self.altitude_baselines.get(key, math.nan)
            diag[key.replace('_m', '_from_home_m')] = (
                value - baseline if math.isfinite(value) and math.isfinite(baseline)
                else math.nan)
        diag['expected_takeoff_altitude_m'] = self.expected_takeoff_altitude
        diag['static_pressure_hpa'] = diag['static_pressure_pa'] / 100.0
        tracked_topics = (
            'local_pose', 'mavros_state', 'extended_state', 'rangefinder',
            'static_pressure', 'relative_alt', 'local_velocity', 'vision_pose',
            'zed_pose', 'telemetry_altitude', 'target_altitude', 'hover', 'setpoint')
        for name in tracked_topics:
            age = self.topic_age(name, now)
            diag[f'{name}_received'] = name in self.topic_last_seen
            diag[f'{name}_age_s'] = age
            diag[f'{name}_stale'] = (
                not math.isfinite(age) or age > self.topic_stale_threshold)
        target = diag['target_altitude_m']
        rel_alt = diag['relative_alt_m']
        setpoint_z = diag['setpoint_z_m']
        diag['relative_alt_error_m'] = (
            target - rel_alt if math.isfinite(target) and math.isfinite(rel_alt)
            else math.nan)
        diag['local_z_error_m'] = (
            setpoint_z - float(p.z) if math.isfinite(setpoint_z) else math.nan)
        diag['rel_local_home_disagreement_m'] = (
            diag['relative_alt_from_home_m'] - diag['local_z_from_home_m']
            if math.isfinite(diag['relative_alt_from_home_m']) else math.nan)
        diag['rel_vision_home_disagreement_m'] = (
            diag['relative_alt_from_home_m'] - diag['vision_z_from_home_m']
            if math.isfinite(diag['relative_alt_from_home_m']) and
            math.isfinite(diag['vision_z_from_home_m']) else math.nan)
        if self.previous_diagnostic is not None:
            for key in ('relative_alt_m', 'local_z_m', 'vision_z_m',
                        'static_pressure_hpa'):
                old = self.previous_diagnostic.get(key, math.nan)
                new = diag.get(key, math.nan)
                delta = new - old if math.isfinite(old) and math.isfinite(new) else math.nan
                threshold = 0.20 if key == 'static_pressure_hpa' else self.altitude_jump_threshold
                if math.isfinite(delta) and abs(delta) >= threshold:
                    self.add_diagnostic_event(
                        f'{key}_jump', delta, f'{old:.3f} -> {new:.3f}', cooldown=0.0)
        disagreement = diag['rel_local_home_disagreement_m']
        if (math.isfinite(disagreement) and
                abs(disagreement) >= self.altitude_disagreement_threshold):
            self.add_diagnostic_event(
                'relative_local_altitude_disagreement', disagreement,
                'relative_alt_from_home - local_z_from_home')
        self.previous_diagnostic = dict(diag)
        self.diagnostic_samples.append(diag)
        self.previous_sample = row
        self.max_altitude = max(self.max_altitude, float(p.z))
        self.min_altitude = min(self.min_altitude, float(p.z))

    def statistics(self):
        duration = self.elapsed()
        inspected = sum(t['inspected'] for t in self.trees.values())
        validated = sum(t['validated'] for t in self.trees.values())
        count = len(self.trees)
        avg_speed = self.total_distance_3d / duration if duration > 0.0 else 0.0
        current = self.samples[-1] if self.samples else None
        return {
            'duration_s': duration,
            'state': self.current_state,
            'frame': self.map_frame,
            'current_drone': current,
            'home': self.home,
            'distance_xy_m': self.total_distance_xy,
            'distance_3d_m': self.total_distance_3d,
            'average_speed_mps': avg_speed,
            'maximum_speed_mps': self.max_speed,
            'minimum_altitude_m': None if self.min_altitude == math.inf else self.min_altitude,
            'maximum_altitude_m': None if self.max_altitude == -math.inf else self.max_altitude,
            'trees_detected': count,
            'trees_validated': validated,
            'trees_inspected': inspected,
            'coverage_percent': 100.0 * inspected / count if count else 0.0,
            'sample_count': len(self.samples),
        }

    def print_status(self):
        stat = self.statistics()
        pose = stat['current_drone']
        if pose is None:
            self.get_logger().info(f'[ANALYZER] state={self.current_state}; menunggu pose')
            return
        tree_text = ', '.join(
            f"ID{t['id']}=({t['x']:.2f},{t['y']:.2f},{t['z']:.2f})"
            for t in sorted(self.trees.values(), key=lambda item: item['id'])) or '-'
        self.get_logger().info(
            f"[ANALYZER] state={self.current_state} | drone=({pose['x']:.2f},"
            f"{pose['y']:.2f},{pose['z']:.2f}) m | yaw={pose['yaw_deg']:.1f} deg | "
            f"v={pose['speed_mps']:.2f} m/s | jarak={self.total_distance_3d:.2f} m | "
            f"pohon: {tree_text}")

    def autosave(self):
        self.save_data_files()
        self.draw_maps()

    @staticmethod
    def write_csv(path, rows, fields):
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def save_data_files(self):
        self.write_csv(self.output_dir / 'drone_trajectory.csv', self.samples,
                       ['time_s', 'x', 'y', 'z', 'yaw_deg', 'speed_mps', 'state'])
        self.write_csv(self.output_dir / 'tree_map.csv',
                       sorted(self.trees.values(), key=lambda item: item['id']),
                       ['id', 'x', 'y', 'z', 'confidence', 'inspected',
                        'validated', 'orbit_count', 'time_s'])
        self.write_csv(self.output_dir / 'state_history.csv', self.state_events,
                       ['time_s', 'state'])
        diagnostic_fields = [
            'time_s', 'state', 'connected', 'armed', 'flight_mode',
            'landed_state', 'rangefinder_m', 'static_pressure_pa',
            'static_pressure_hpa', 'relative_alt_m', 'local_z_m',
            'local_vz_mps', 'zed_z_m', 'vision_z_m',
            'telemetry_altitude_m', 'target_altitude_m', 'setpoint_z_m',
            'expected_takeoff_altitude_m', 'is_hovering',
            'relative_alt_from_home_m', 'local_z_from_home_m',
            'zed_z_from_home_m', 'vision_z_from_home_m',
            'relative_alt_error_m', 'local_z_error_m',
            'rel_local_home_disagreement_m', 'rel_vision_home_disagreement_m',
        ]
        for name in (
                'local_pose', 'mavros_state', 'extended_state', 'rangefinder',
                'static_pressure', 'relative_alt', 'local_velocity', 'vision_pose',
                'zed_pose', 'telemetry_altitude', 'target_altitude', 'hover',
                'setpoint'):
            diagnostic_fields.extend(
                [f'{name}_received', f'{name}_age_s', f'{name}_stale'])
        self.write_csv(self.output_dir / 'altitude_diagnostics.csv',
                       self.diagnostic_samples, diagnostic_fields)
        self.write_csv(self.output_dir / 'fc_messages.csv', self.fc_messages,
                       ['time_s', 'severity', 'text'])
        self.write_csv(self.output_dir / 'diagnostic_events.csv',
                       self.diagnostic_events,
                       ['time_s', 'state', 'event', 'value', 'detail'])
        report = self.statistics()
        report['trees'] = sorted(self.trees.values(), key=lambda item: item['id'])
        report['state_history'] = self.state_events
        with (self.output_dir / 'mission_summary.json').open(
                'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2)

    def tree_color(self, tree):
        if tree['inspected']:
            return 'green'
        if tree['validated']:
            return 'orange'
        return 'black'

    def draw_maps(self):
        if not self.samples:
            return
        xs = [s['x'] for s in self.samples]
        ys = [s['y'] for s in self.samples]
        zs = [s['z'] for s in self.samples]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(xs, ys, color='royalblue', linewidth=1.5, label='Lintasan drone')
        ax.scatter(xs[-1], ys[-1], color='red', s=80, label='Drone')
        if self.home:
            ax.scatter(self.home[0], self.home[1], marker='*', color='gold',
                       edgecolor='black', s=180, label='Takeoff/Home')
        for tree in sorted(self.trees.values(), key=lambda item: item['id']):
            ax.scatter(tree['x'], tree['y'], color=self.tree_color(tree), s=80)
            ax.annotate(f"T{tree['id']}\n({tree['x']:.2f}, {tree['y']:.2f})",
                        (tree['x'], tree['y']), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)
        if self.waypoints:
            wx = [p[0] for p in self.waypoints]
            wy = [p[1] for p in self.waypoints]
            ax.plot(wx, wy, '--', color='purple', linewidth=1, label='Waypoint')
        ax.set(title=f'Peta Misi 2D — {self.current_state}', xlabel='X (m)', ylabel='Y (m)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'map_2d.png', dpi=180)
        plt.close(fig)

        fig = plt.figure(figsize=(11, 8))
        ax3 = fig.add_subplot(111, projection='3d')
        ax3.plot(xs, ys, zs, color='royalblue', linewidth=1.5, label='Lintasan drone')
        ax3.scatter(xs[-1], ys[-1], zs[-1], color='red', s=60, label='Drone')
        if self.home:
            ax3.scatter(*self.home, marker='*', color='gold', s=150, label='Takeoff/Home')
        for tree in sorted(self.trees.values(), key=lambda item: item['id']):
            ax3.scatter(tree['x'], tree['y'], tree['z'], color=self.tree_color(tree), s=60)
            ax3.text(tree['x'], tree['y'], tree['z'], f" T{tree['id']}", fontsize=8)
        ax3.set(title=f'Peta Misi 3D — {self.current_state}',
                xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
        ax3.legend(loc='best')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'map_3d.png', dpi=180)
        plt.close(fig)

        fig, (alt_ax, speed_ax) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        ts = [s['time_s'] for s in self.samples]
        alt_ax.plot(ts, zs, color='darkgreen')
        alt_ax.set_ylabel('Ketinggian Z (m)')
        alt_ax.grid(True, alpha=0.3)
        speed_ax.plot(ts, [s['speed_mps'] for s in self.samples], color='darkorange')
        speed_ax.set(xlabel='Waktu (s)', ylabel='Kecepatan (m/s)')
        speed_ax.grid(True, alpha=0.3)
        fig.suptitle('Profil Penerbangan')
        fig.tight_layout()
        fig.savefig(self.output_dir / 'flight_profile.png', dpi=180)
        plt.close(fig)

        if self.diagnostic_samples:
            ds = self.diagnostic_samples
            dts = [item['time_s'] for item in ds]
            fig, (alt_ax, err_ax, vz_ax) = plt.subplots(
                3, 1, figsize=(12, 10), sharex=True)
            series = (
                ('rangefinder_m', 'Rangefinder', 'tab:purple'),
                ('relative_alt_m', 'Relative altitude', 'tab:blue'),
                ('local_z_m', 'Local pose Z', 'tab:green'),
                ('zed_z_m', 'ZED Z', 'tab:orange'),
                ('vision_z_m', 'Vision pose Z', 'tab:brown'),
                ('target_altitude_m', 'Target altitude', 'tab:red'),
                ('setpoint_z_m', 'Setpoint Z', 'tab:pink'),
                ('expected_takeoff_altitude_m', 'Expected takeoff', 'black'),
            )
            for key, label, color in series:
                alt_ax.plot(dts, [item[key] for item in ds],
                            label=label, color=color, linewidth=1.2)
            alt_ax.set_ylabel('Altitude / Z (m)')
            alt_ax.grid(True, alpha=0.3)
            alt_ax.legend(loc='best', ncol=2, fontsize=8)
            err_ax.plot(dts, [item['relative_alt_error_m'] for item in ds],
                        label='Target - rel_alt', color='tab:red')
            err_ax.plot(dts, [item['local_z_error_m'] for item in ds],
                        label='Setpoint Z - local Z', color='tab:cyan')
            err_ax.axhline(0.0, color='black', linewidth=0.7)
            err_ax.set_ylabel('Error (m)')
            err_ax.grid(True, alpha=0.3)
            err_ax.legend(loc='best')
            vz_ax.plot(dts, [item['local_vz_mps'] for item in ds],
                       color='tab:orange', label='Local Vz')
            vz_ax.axhline(0.0, color='black', linewidth=0.7)
            vz_ax.set(xlabel='Waktu (s)', ylabel='Vz (m/s)')
            vz_ax.grid(True, alpha=0.3)
            vz_ax.legend(loc='best')
            fig.suptitle('Diagnostik Altitude dan Estimasi Vertikal')
            fig.tight_layout()
            fig.savefig(self.output_dir / 'altitude_diagnostics.png', dpi=180)
            plt.close(fig)

            fig, pressure_ax = plt.subplots(figsize=(12, 5))
            pressure_ax.plot(dts, [item['static_pressure_hpa'] for item in ds],
                             color='tab:blue', label='Static pressure')
            pressure_ax.set(xlabel='Waktu (s)', ylabel='Tekanan (hPa)',
                            title='Tekanan Barometer FC')
            pressure_ax.grid(True, alpha=0.3)
            pressure_ax.legend(loc='best')
            fig.tight_layout()
            fig.savefig(self.output_dir / 'barometer_pressure.png', dpi=180)
            plt.close(fig)

    def save_all(self, final=False):
        self.save_data_files()
        self.draw_maps()
        if final and not self.final_saved:
            self.final_saved = True
            stat = self.statistics()
            self.get_logger().info(
                f"Laporan akhir tersimpan: {self.output_dir} | "
                f"durasi={stat['duration_s']:.1f}s, jarak={stat['distance_3d_m']:.2f}m, "
                f"pohon={stat['trees_detected']}, inspected={stat['trees_inspected']}")

    def shutdown(self):
        self.save_all(final=True)


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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
