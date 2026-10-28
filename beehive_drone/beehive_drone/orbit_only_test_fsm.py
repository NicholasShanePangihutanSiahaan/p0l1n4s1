#!/usr/bin/env python3
"""Minimal field-test FSM: GUIDED, arm, takeoff, one orbit, then land."""

import time

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String


class OrbitOnlyTestFsm(Node):
    """Run one synthetic orbit without tree perception or a tree map."""

    def __init__(self):
        super().__init__('orbit_only_test_fsm')
        self.declare_parameter('auto_start', False)
        self.declare_parameter('takeoff_altitude', 1.5)
        self.declare_parameter('orbit_center_offset_x', 2.0)
        self.declare_parameter('orbit_center_offset_y', 0.0)
        self.declare_parameter('command_retry_period', 1.0)
        self.declare_parameter('takeoff_timeout', 45.0)
        self.declare_parameter('orbit_start_delay', 2.0)
        self.declare_parameter('landing_timeout', 60.0)

        self.auto_start = bool(self.get_parameter('auto_start').value)
        self.takeoff_altitude = float(
            self.get_parameter('takeoff_altitude').value)
        self.center_dx = float(
            self.get_parameter('orbit_center_offset_x').value)
        self.center_dy = float(
            self.get_parameter('orbit_center_offset_y').value)
        self.retry_period = max(
            0.2, float(self.get_parameter('command_retry_period').value))
        self.takeoff_timeout = max(
            5.0, float(self.get_parameter('takeoff_timeout').value))
        self.orbit_start_delay = max(
            0.0, float(self.get_parameter('orbit_start_delay').value))
        self.landing_timeout = max(
            10.0, float(self.get_parameter('landing_timeout').value))

        self.state = 'WAIT_START'
        self.state_since = time.monotonic()
        self.last_command = 0.0
        self.start_requested = self.auto_start
        self.pose = None
        self.home = None
        self.mode = ''
        self.armed = False
        self.hovering = False
        self.orbit_status = 'IDLE'
        self.orbit_center = None
        self.guided_was_acquired = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            PoseStamped, '/mavros/local_position/pose',
            self.pose_callback, sensor_qos)
        self.create_subscription(
            String, '/flight/telemetry/current_mode',
            self.mode_callback, 10)
        self.create_subscription(
            Bool, '/flight/telemetry/is_armed',
            self.arm_callback, 10)
        self.create_subscription(
            Bool, '/flight/telemetry/is_hovering',
            self.hover_callback, 10)
        self.create_subscription(
            String, '/control/orbit_status',
            self.orbit_status_callback, 10)
        self.create_subscription(
            Bool, '/mission/start', self.start_callback, 10)

        self.mode_pub = self.create_publisher(
            String, '/flight/cmd/set_mode', 10)
        self.arm_pub = self.create_publisher(
            Bool, '/flight/cmd/set_arm', 10)
        self.takeoff_pub = self.create_publisher(
            Float32, '/flight/cmd/takeoff', 10)
        self.land_pub = self.create_publisher(
            Bool, '/flight/cmd/land', 10)
        self.orbit_target_pub = self.create_publisher(
            Point, '/control/orbit_target', 10)
        self.orbit_start_pub = self.create_publisher(
            Bool, '/control/orbit_start', 10)
        self.setpoint_enable_pub = self.create_publisher(
            Bool, '/control/setpoint_enabled', 10)
        self.state_pub = self.create_publisher(
            String, '/mission/fsm_state', 10)

        self.create_timer(0.1, self.tick)
        self.get_logger().info(
            'Orbit-only test siap. Kirim true ke /mission/start untuk mulai.')

    def pose_callback(self, message):
        self.pose = message
        if self.home is None and self.state in (
                'WAIT_START', 'WAIT_GUIDED', 'WAIT_ARM'):
            self.home = (
                float(message.pose.position.x),
                float(message.pose.position.y),
            )

    def mode_callback(self, message):
        self.mode = message.data

    def arm_callback(self, message):
        self.armed = bool(message.data)

    def hover_callback(self, message):
        self.hovering = bool(message.data)

    def orbit_status_callback(self, message):
        self.orbit_status = message.data

    def start_callback(self, message):
        if message.data:
            self.start_requested = True

    def transition(self, state):
        if state == self.state:
            return
        self.get_logger().info(f'{self.state} -> {state}')
        self.state = state
        self.state_since = time.monotonic()
        self.last_command = 0.0

    def elapsed(self):
        return time.monotonic() - self.state_since

    def command_due(self):
        now = time.monotonic()
        if now - self.last_command < self.retry_period:
            return False
        self.last_command = now
        return True

    def publish_bool(self, publisher, value):
        message = Bool()
        message.data = bool(value)
        publisher.publish(message)

    def publish_state(self):
        message = String()
        message.data = self.state
        self.state_pub.publish(message)

    def publish_orbit_target(self):
        if self.orbit_center is None:
            return
        target = Point()
        target.x, target.y = self.orbit_center
        target.z = self.takeoff_altitude
        self.orbit_target_pub.publish(target)

    def request_land(self, reason):
        self.get_logger().warning(reason)
        self.publish_bool(self.orbit_start_pub, False)
        self.transition('LANDING')

    def tick(self):
        self.publish_state()
        self.publish_bool(
            self.setpoint_enable_pub,
            self.state in ('START_ORBIT', 'WAIT_ORBIT'))

        active = self.state not in (
            'WAIT_START', 'WAIT_GUIDED', 'DONE', 'MANUAL_OVERRIDE')
        if (active and self.guided_was_acquired
                and self.mode not in ('', 'GUIDED', 'LAND')):
            self.publish_bool(self.orbit_start_pub, False)
            self.transition('MANUAL_OVERRIDE')
            self.get_logger().warning(
                f'Pengambilalihan terdeteksi: mode={self.mode}.')
            return

        if self.state == 'WAIT_START':
            if self.start_requested and self.pose is not None:
                self.transition('WAIT_GUIDED')

        elif self.state == 'WAIT_GUIDED':
            if self.mode == 'GUIDED':
                self.guided_was_acquired = True
                self.transition('WAIT_ARM')
            elif self.command_due():
                message = String()
                message.data = 'GUIDED'
                self.mode_pub.publish(message)

        elif self.state == 'WAIT_ARM':
            if self.armed:
                self.transition('WAIT_TAKEOFF')
            elif self.command_due():
                self.publish_bool(self.arm_pub, True)

        elif self.state == 'WAIT_TAKEOFF':
            if self.hovering:
                self.transition('ORBIT_DELAY')
            elif self.elapsed() > self.takeoff_timeout:
                self.request_land('Takeoff timeout; meminta LAND.')
            elif self.command_due():
                message = Float32()
                message.data = self.takeoff_altitude
                self.takeoff_pub.publish(message)

        elif self.state == 'ORBIT_DELAY':
            if self.home is None:
                self.request_land('Home pose tidak tersedia; meminta LAND.')
            elif self.elapsed() >= self.orbit_start_delay:
                self.orbit_center = (
                    self.home[0] + self.center_dx,
                    self.home[1] + self.center_dy,
                )
                self.transition('START_ORBIT')

        elif self.state == 'START_ORBIT':
            self.publish_orbit_target()
            # Beri waktu target mencapai controller sebelum sinyal start;
            # kedua pesan memakai topic berbeda sehingga urutan DDS tidak
            # boleh diasumsikan.
            if self.elapsed() < 0.5:
                return
            self.publish_bool(self.orbit_start_pub, True)
            self.get_logger().info(
                'Orbit sintetis dimulai: pusat=(%.2f, %.2f), altitude=%.2f'
                % (*self.orbit_center, self.takeoff_altitude))
            self.transition('WAIT_ORBIT')

        elif self.state == 'WAIT_ORBIT':
            self.publish_orbit_target()
            if self.orbit_status == 'ORBIT_COMPLETED':
                self.publish_bool(self.orbit_start_pub, False)
                self.transition('LANDING')
            elif self.orbit_status.startswith('ORBIT_FAILED'):
                self.request_land(
                    f'{self.orbit_status}; meminta LAND.')

        elif self.state == 'LANDING':
            self.publish_bool(self.orbit_start_pub, False)
            if not self.armed:
                self.transition('DONE')
            elif self.elapsed() > self.landing_timeout:
                self.get_logger().error(
                    'Landing timeout. Ambil alih dengan RC.')
                self.transition('MANUAL_OVERRIDE')
            elif self.command_due() and self.mode != 'LAND':
                self.publish_bool(self.land_pub, True)

        elif self.state in ('DONE', 'MANUAL_OVERRIDE'):
            self.publish_bool(self.setpoint_enable_pub, False)


def main(args=None):
    rclpy.init(args=args)
    node = OrbitOnlyTestFsm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
