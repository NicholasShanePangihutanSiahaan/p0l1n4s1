"""Start only the ZED-to-MAVROS external-navigation bridge."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('beehive_drone'), 'config', 'real.yaml')
    input_topic = LaunchConfiguration('input_topic')
    raw_input_topic = LaunchConfiguration('raw_input_topic')
    output_topic = LaunchConfiguration('output_topic')
    use_alignment = LaunchConfiguration('use_alignment')
    use_fixed_yaw_offset = LaunchConfiguration('use_fixed_yaw_offset')
    fixed_yaw_offset_degrees = LaunchConfiguration('fixed_yaw_offset_degrees')
    xacro_path = LaunchConfiguration('xacro_path')
    
    default_xacro_path = os.path.join(
        get_package_share_directory('beehive_drone'),
        'urdf',
        'beehive_drone.urdf.xacro',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'raw_input_topic', default_value='/zed/zed_node/odom'),
        DeclareLaunchArgument(
            'xacro_drone', default_value=default_xacro_path),
        DeclareLaunchArgument(
            'input_topic', default_value='/zed/aligned_pose'),
        DeclareLaunchArgument(
            'use_alignment', default_value='false'),
        DeclareLaunchArgument(
            'use_fixed_yaw_offset', default_value='false'),
        DeclareLaunchArgument(
            'fixed_yaw_offset_degrees', default_value='0.0'),
        DeclareLaunchArgument(
            'output_topic', default_value='/mavros/vision_pose/pose'),
        Node(
            package='beehive_drone',
            executable='zed_frame_alignment',
            name='zed_frame_alignment',
            condition=IfCondition(use_alignment),
            parameters=[config, {
                'zed_pose_topic': raw_input_topic,
                'use_fixed_yaw_offset': ParameterValue(
                    use_fixed_yaw_offset, value_type=bool),
                'fixed_yaw_offset_degrees': ParameterValue(
                    fixed_yaw_offset_degrees, value_type=float),
            }],
            output='screen',
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom_tf",
            arguments=[
                "--x", "0.0",
                "--y", "0.0",
                "--z", "0.0",
                "--roll", "0.0",
                "--pitch", "0.0",
                "--yaw", "0.0",
                "--frame-id", "map",
                "--child-frame-id", "odom",
            ],
        ),
        Node(
            package='beehive_drone',
            executable='vision_to_mavros',
            name='vision_to_mavros',
            condition=IfCondition(use_alignment),
            parameters=[config, {
                'input_topic': input_topic,
                'output_topic': output_topic,
            }],
            output='screen',
        ),
        Node(
            package='beehive_drone',
            executable='vision_to_mavros',
            name='vision_to_mavros',
            condition=UnlessCondition(use_alignment),
            parameters=[config, {
                'input_topic': raw_input_topic,
                'output_topic': output_topic,
            }],
            output='screen',
        ),
        IncludeLaunchDescription( # ZED2i Package
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('zed_wrapper'),
                'launch',
                'zed_camera.launch.py'
            )
        ),
        launch_arguments={
            'camera_name': 'zed2i',
            'camera_model': 'zed2i',
            'publish_urdf': 'true',
            'xacro_path': xacro_path,
            'publish_tf': 'false'
        }.items()
        )
    ])
