"""Start only the ZED-to-MAVROS external-navigation bridge."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('beehive_drone'), 'config', 'real.yaml')
    input_topic = LaunchConfiguration('input_topic')
    raw_input_topic = LaunchConfiguration('raw_input_topic')
    output_topic = LaunchConfiguration('output_topic')
    use_alignment = LaunchConfiguration('use_alignment')
    use_fixed_yaw_offset = LaunchConfiguration('use_fixed_yaw_offset')
    fixed_yaw_offset_degrees = LaunchConfiguration('fixed_yaw_offset_degrees')

    return LaunchDescription([
        DeclareLaunchArgument(
            'raw_input_topic', default_value='/zed/zed_node/pose'),
        DeclareLaunchArgument(
            'input_topic', default_value='/zed/aligned_pose'),
        DeclareLaunchArgument(
            'use_alignment', default_value='true'),
        DeclareLaunchArgument(
            'use_fixed_yaw_offset', default_value='false'),
        DeclareLaunchArgument(
            'fixed_yaw_offset_degrees', default_value='0.0'),
        DeclareLaunchArgument(
            'output_topic', default_value='/mavros/vision_pose/pose'),
        # Node(
        #     package='beehive_drone',
        #     executable='zed_frame_alignment',
        #     name='zed_frame_alignment',
        #     condition=IfCondition(use_alignment),
        #     parameters=[config, {
        #         'zed_pose_topic': raw_input_topic,
        #         'use_fixed_yaw_offset': ParameterValue(
        #             use_fixed_yaw_offset, value_type=bool),
        #         'fixed_yaw_offset_degrees': ParameterValue(
        #             fixed_yaw_offset_degrees, value_type=float),
        #     }],
        #     output='screen',
        # ),
        Node(
            package='beehive_drone',
            executable='vision_to_mavros',
            name='vision_to_mavros',
            parameters=[config, {
                'input_topic': raw_input_topic,
                'output_topic': output_topic,
            }],
            output='screen',
        ),
    ])