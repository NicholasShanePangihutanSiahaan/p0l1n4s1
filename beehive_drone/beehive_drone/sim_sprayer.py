#!/usr/bin/env python3
"""Simulation-only replacement for the physical ``/spray`` service."""

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class SimSprayer(Node):
    """Acknowledge sprayer commands and expose their state in the ROS log."""

    def __init__(self):
        super().__init__('sim_sprayer')
        self.enabled = False
        self.service = self.create_service(SetBool, 'spray', self.set_spray)
        self.get_logger().info('Mock sprayer siap pada service /spray')

    def set_spray(self, request, response):
        self.enabled = bool(request.data)
        response.success = True
        response.message = 'sprayer ON' if self.enabled else 'sprayer OFF'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SimSprayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
