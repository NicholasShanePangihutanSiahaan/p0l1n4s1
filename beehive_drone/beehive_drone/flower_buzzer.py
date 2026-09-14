#!/usr/bin/env python3

"""Sound a buzzer once when a palm-oil flower is detected by ZED."""

from typing import Iterable

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import SetBool
from zed_msgs.msg import ObjectsStamped


DEFAULT_FLOWER_LABELS = (
    'male-flowers',
    'pre-receptive-female',
    'receptive-female',
    'post-receptive-female',
)


def normalize_label(label: str) -> str:
    """Normalize labels emitted by different ZED wrapper versions."""
    return label.strip().lower()


def contains_flower(objects: Iterable, labels: set[str],
                    minimum_confidence: float) -> bool:
    """Return whether a usable detection belongs to a flower class."""
    return any(
        normalize_label(obj.label) in labels
        and float(obj.confidence) >= minimum_confidence
        for obj in objects
    )


class FlowerBuzzer(Node):
    """Bridge raw ZED detections to the tested pb_sprayer GPIO service."""

    def __init__(self):
        super().__init__('flower_buzzer')

        self.declare_parameter(
            'objects_topic', '/zed/zed_node/obj_det/objects')
        self.declare_parameter('buzzer_service', '/spray')
        self.declare_parameter('flower_labels', list(DEFAULT_FLOWER_LABELS))
        self.declare_parameter('minimum_confidence', 25.0)
        self.declare_parameter('beep_duration', 0.5)
        self.declare_parameter('detection_clear_time', 1.0)
        self.declare_parameter('retrigger_cooldown', 2.0)

        objects_topic = str(self.get_parameter('objects_topic').value)
        buzzer_service = str(self.get_parameter('buzzer_service').value)
        self.flower_labels = {
            normalize_label(value)
            for value in self.get_parameter('flower_labels').value
        }
        self.minimum_confidence = max(
            0.0, float(self.get_parameter('minimum_confidence').value))
        self.beep_duration = max(
            0.05, float(self.get_parameter('beep_duration').value))
        self.detection_clear_time = max(
            0.1, float(self.get_parameter('detection_clear_time').value))
        self.retrigger_cooldown = max(
            0.0, float(self.get_parameter('retrigger_cooldown').value))

        self.detection_active = False
        self.last_detection_time = None
        self.last_trigger_time = None
        self.pending_activation = False
        self.buzzer_active = False
        self.off_deadline = None
        self.last_service_warning = None

        self.buzzer_client = self.create_client(SetBool, buzzer_service)
        self.objects_sub = self.create_subscription(
            ObjectsStamped, objects_topic, self.objects_callback,
            qos_profile_sensor_data)
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            f'Flower buzzer ready: topic={objects_topic}, '
            f'service={buzzer_service}, labels={sorted(self.flower_labels)}')

    def objects_callback(self, msg: ObjectsStamped) -> None:
        if not contains_flower(
                msg.objects, self.flower_labels, self.minimum_confidence):
            return

        now = self.get_clock().now()
        self.last_detection_time = now
        if self.detection_active:
            return

        self.detection_active = True
        since_trigger = (
            float('inf') if self.last_trigger_time is None else
            (now - self.last_trigger_time).nanoseconds * 1e-9)
        if since_trigger >= self.retrigger_cooldown:
            self.pending_activation = True

    def call_buzzer(self, enabled: bool) -> bool:
        if not self.buzzer_client.service_is_ready():
            now = self.get_clock().now()
            warning_age = (
                float('inf') if self.last_service_warning is None else
                (now - self.last_service_warning).nanoseconds * 1e-9)
            if warning_age >= 2.0:
                self.get_logger().warning(
                    'Service buzzer tidak tersedia; jalankan pb_sprayer.')
                self.last_service_warning = now
            return False

        request = SetBool.Request()
        request.data = enabled
        future = self.buzzer_client.call_async(request)
        future.add_done_callback(self.service_response_callback)
        return True

    def service_response_callback(self, future) -> None:
        try:
            response = future.result()
            if not response.success:
                self.get_logger().error(
                    f'Perintah buzzer gagal: {response.message}')
        except Exception as exc:  # ROS service failures are runtime-specific.
            self.get_logger().error(f'Pemanggilan service buzzer gagal: {exc}')

    def control_loop(self) -> None:
        now = self.get_clock().now()

        if self.detection_active and self.last_detection_time is not None:
            unseen_for = (
                now - self.last_detection_time).nanoseconds * 1e-9
            if unseen_for >= self.detection_clear_time:
                self.detection_active = False

        if self.pending_activation and not self.buzzer_active:
            if self.call_buzzer(True):
                self.pending_activation = False
                self.buzzer_active = True
                self.last_trigger_time = now
                self.off_deadline = now + Duration(seconds=self.beep_duration)
                self.get_logger().info('Bunga sawit terdeteksi: buzzer ON.')

        if self.buzzer_active and now >= self.off_deadline:
            if self.call_buzzer(False):
                self.buzzer_active = False
                self.off_deadline = None
                self.get_logger().info('Buzzer OFF.')


def main(args=None):
    rclpy.init(args=args)
    node = FlowerBuzzer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.buzzer_active:
            node.call_buzzer(False)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
