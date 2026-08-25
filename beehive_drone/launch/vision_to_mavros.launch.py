"""Start only the ZED-to-MAVROS external-navigation bridge."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('beehive_drone'), 'config', 'real.yaml')
    input_topic = LaunchConfiguration('input_topic')
    output_topic = LaunchConfiguration('output_topic')

    return LaunchDescription([
        DeclareLaunchArgument(
            'input_topic', default_value='/zed/zed_node/pose'),
        DeclareLaunchArgument(
            'output_topic', default_value='/mavros/vision_pose/pose'),
        Node(
            package='beehive_drone',
            executable='vision_to_mavros',
            name='vision_to_mavros',
            parameters=[config, {
                'input_topic': input_topic,
                'output_topic': output_topic,
            }],
            output='screen',
        ),
    ])
