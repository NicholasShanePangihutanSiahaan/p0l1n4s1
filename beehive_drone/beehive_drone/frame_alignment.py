#!/usr/bin/env python3
"""Lock a yaw/translation transform from the raw ZED map to MAVROS local."""

import math
from collections import deque
from copy import deepcopy

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from mavros_msgs.msg import State
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32
from tf2_ros import StaticTransformBroadcaster


def wrap_angle(angle):
    """Wrap radians to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    """Extract ENU yaw from a geometry quaternion."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def yaw_quaternion(yaw):
    """Return an (x, y, z, w) quaternion containing yaw only."""
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def multiply_quaternion(left, right):
    """Multiply two (x, y, z, w) quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def circular_mean(values):
    """Return the circular mean of an iterable of radians."""
    return math.atan2(
        sum(math.sin(value) for value in values),
        sum(math.cos(value) for value in values),
    )


def transform_xy(x, y, yaw, tx, ty):
    """Rotate XY by yaw, then apply translation."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * x - sine * y + tx,
        sine * x + cosine * y + ty,
    )


class ZedFrameAlignment(Node):
    """Calibrate while disarmed, lock once, and publish aligned ZED poses."""

    def __init__(self):
        super().__init__('zed_frame_alignment')
        defaults = {
            'zed_pose_topic': '/zed/zed_node/pose',
            'fc_pose_topic': '/mavros/local_position/pose',
            'state_topic': '/mavros/state',
            'output_topic': '/zed/aligned_pose',
            'output_frame': 'map',
            'raw_frame': 'zed_map_raw',
            'minimum_samples': 50,
            'maximum_position_step': 0.02,
            'maximum_yaw_step_degrees': 1.0,
            'maximum_pair_age': 0.15,
            'publish_before_lock': False,
            'reset_position_jump': 1.0,
            'reset_yaw_jump_degrees': 30.0,
            'use_fixed_yaw_offset': False,
            'fixed_yaw_offset_degrees': 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.output_frame = str(self.get_parameter('output_frame').value)
        self.raw_frame = str(self.get_parameter('raw_frame').value)
        self.minimum_samples = max(
            10, int(self.get_parameter('minimum_samples').value))
        self.max_position_step = max(
            0.001, float(self.get_parameter('maximum_position_step').value))
        self.max_yaw_step = math.radians(max(
            0.1, float(self.get_parameter('maximum_yaw_step_degrees').value)))
        self.max_pair_age = max(
            0.02, float(self.get_parameter('maximum_pair_age').value))
        self.publish_before_lock = bool(
            self.get_parameter('publish_before_lock').value)
        self.reset_position_jump = max(
            0.1, float(self.get_parameter('reset_position_jump').value))
        self.reset_yaw_jump = math.radians(max(
            5.0, float(self.get_parameter('reset_yaw_jump_degrees').value)))
        self.use_fixed_yaw_offset = bool(
            self.get_parameter('use_fixed_yaw_offset').value)
        self.fixed_yaw_offset = math.radians(float(
            self.get_parameter('fixed_yaw_offset_degrees').value))

        self.zed_pose = None
        self.fc_pose = None
        self.zed_time = None
        self.fc_time = None
        self.armed = False
        self.last_pair = None
        self.samples = deque(maxlen=self.minimum_samples)
        self.locked = False
        self.faulted = False
        self.previous_zed_pose = None
        self.yaw_offset = 0.0
        self.tx = self.ty = self.tz = 0.0

        self.publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter('output_topic').value), 10)
        self.ready_publisher = self.create_publisher(
            Bool, '/alignment/ready', 10)
        self.offset_publisher = self.create_publisher(
            Float32, '/alignment/yaw_offset_deg', 10)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.create_subscription(
            PoseStamped, str(self.get_parameter('zed_pose_topic').value),
            self.zed_callback, qos_profile_sensor_data)
        self.create_subscription(
            PoseStamped, str(self.get_parameter('fc_pose_topic').value),
            self.fc_callback, qos_profile_sensor_data)
        self.create_subscription(
            State, str(self.get_parameter('state_topic').value),
            self.state_callback, 10)
        self.create_timer(0.1, self.calibrate)
        self.create_timer(1.0, self.report)

    def state_callback(self, msg):
        self.armed = bool(msg.armed)

    def fc_callback(self, msg):
        self.fc_pose = msg
        self.fc_time = self.get_clock().now()

    def zed_callback(self, msg):
        if self.locked and self.previous_zed_pose is not None:
            position_jump = math.sqrt(
                (msg.pose.position.x - self.previous_zed_pose.pose.position.x) ** 2 +
                (msg.pose.position.y - self.previous_zed_pose.pose.position.y) ** 2 +
                (msg.pose.position.z - self.previous_zed_pose.pose.position.z) ** 2)
            yaw_jump = abs(wrap_angle(
                yaw_from_quaternion(msg.pose.orientation) -
                yaw_from_quaternion(self.previous_zed_pose.pose.orientation)))
            if position_jump > self.reset_position_jump or \
                    yaw_jump > self.reset_yaw_jump:
                self.faulted = True
                self.get_logger().error(
                    'ZED tracking jump terdeteksi '
                    f'(pos={position_jump:.2f} m, yaw={math.degrees(yaw_jump):.1f} deg). '
                    'Pose alignment dihentikan; land/abort dan restart seluruh sesi.')
        self.previous_zed_pose = deepcopy(msg)
        self.zed_pose = msg
        self.zed_time = self.get_clock().now()
        if self.locked and not self.faulted:
            self.publisher.publish(self.aligned_pose(msg))
        elif self.publish_before_lock:
            self.publisher.publish(msg)

    def pair_is_fresh(self):
        if self.zed_time is None or self.fc_time is None:
            return False
        return abs((self.zed_time - self.fc_time).nanoseconds) * 1e-9 <= \
            self.max_pair_age

    def stationary(self, current):
        if self.last_pair is None:
            return True
        zed, fc = current
        old_zed, old_fc = self.last_pair
        for new, old in ((zed, old_zed), (fc, old_fc)):
            distance = math.hypot(
                new.pose.position.x - old.pose.position.x,
                new.pose.position.y - old.pose.position.y)
            yaw_step = abs(wrap_angle(
                yaw_from_quaternion(new.pose.orientation) -
                yaw_from_quaternion(old.pose.orientation)))
            if distance > self.max_position_step or yaw_step > self.max_yaw_step:
                return False
        return True

    def calibrate(self):
        if self.locked or self.armed or not self.pair_is_fresh():
            return
        current = (deepcopy(self.zed_pose), deepcopy(self.fc_pose))
        if not self.stationary(current):
            self.samples.clear()
            self.last_pair = current
            return
        self.last_pair = current
        zed, fc = current
        yaw_offset = wrap_angle(
            yaw_from_quaternion(fc.pose.orientation) -
            yaw_from_quaternion(zed.pose.orientation))
        self.samples.append((zed, fc, yaw_offset))
        if len(self.samples) < self.minimum_samples:
            return

        self.yaw_offset = (
            self.fixed_yaw_offset if self.use_fixed_yaw_offset else
            circular_mean(sample[2] for sample in self.samples))
        translations = []
        for zed_sample, fc_sample, _ in self.samples:
            rotated_x, rotated_y = transform_xy(
                zed_sample.pose.position.x, zed_sample.pose.position.y,
                self.yaw_offset, 0.0, 0.0)
            translations.append((
                fc_sample.pose.position.x - rotated_x,
                fc_sample.pose.position.y - rotated_y,
                fc_sample.pose.position.z - zed_sample.pose.position.z,
            ))
        count = float(len(translations))
        self.tx = sum(value[0] for value in translations) / count
        self.ty = sum(value[1] for value in translations) / count
        self.tz = sum(value[2] for value in translations) / count
        self.locked = True
        self.publish_transform()
        self.get_logger().warning(
            'Alignment LOCKED untuk sesi ini: yaw_offset='
            f'{math.degrees(self.yaw_offset):.2f} deg, '
            f't=({self.tx:.3f},{self.ty:.3f},{self.tz:.3f}) m. '
            'Restart node jika tracking ZED reset.')

    def aligned_pose(self, source):
        out = deepcopy(source)
        out.header.frame_id = self.output_frame
        out.pose.position.x, out.pose.position.y = transform_xy(
            source.pose.position.x, source.pose.position.y,
            self.yaw_offset, self.tx, self.ty)
        out.pose.position.z = source.pose.position.z + self.tz
        source_q = (
            source.pose.orientation.x, source.pose.orientation.y,
            source.pose.orientation.z, source.pose.orientation.w)
        q = multiply_quaternion(yaw_quaternion(self.yaw_offset), source_q)
        out.pose.orientation.x, out.pose.orientation.y = q[0], q[1]
        out.pose.orientation.z, out.pose.orientation.w = q[2], q[3]
        return out

    def publish_transform(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.output_frame
        transform.child_frame_id = self.raw_frame
        transform.transform.translation.x = self.tx
        transform.transform.translation.y = self.ty
        transform.transform.translation.z = self.tz
        q = yaw_quaternion(self.yaw_offset)
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(transform)

    def report(self):
        ready = Bool(); ready.data = self.locked and not self.faulted
        self.ready_publisher.publish(ready)
        offset = Float32(); offset.data = math.degrees(self.yaw_offset)
        self.offset_publisher.publish(offset)
        if not self.locked:
            self.get_logger().info(
                f'Alignment menunggu drone diam/disarmed: '
                f'{len(self.samples)}/{self.minimum_samples} sampel.')


def main(args=None):
    rclpy.init(args=args)
    node = ZedFrameAlignment()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

