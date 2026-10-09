"""Run the production pollination stack with Gazebo sensor adapters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    beehive_share = get_package_share_directory('beehive_drone')
    pcl_share = get_package_share_directory('point-cloud-test')

    arguments = [
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('mission_type', default_value='basic_orbit'),
        DeclareLaunchArgument('max_trees', default_value='2'),
        DeclareLaunchArgument(
            'tree_positions',
            default_value='7.0,0.0;14.0,0.0;21.0,0.0'),
        DeclareLaunchArgument('tree_x', default_value='7.0'),
        DeclareLaunchArgument('tree_y', default_value='0.0'),
        DeclareLaunchArgument('tree_ground_z', default_value='0.0'),
        DeclareLaunchArgument('expected_tree_count', default_value='2'),
        DeclareLaunchArgument('camera_x', default_value='0.14'),
        DeclareLaunchArgument('camera_y', default_value='0.06'),
        DeclareLaunchArgument('camera_z', default_value='0.02'),
        DeclareLaunchArgument('camera_roll', default_value='0.0'),
        DeclareLaunchArgument('camera_pitch', default_value='0.0'),
        DeclareLaunchArgument('camera_yaw', default_value='0.0'),
        DeclareLaunchArgument('position_noise_stddev', default_value='0.0'),
        DeclareLaunchArgument('dropout_every_n', default_value='0'),
        DeclareLaunchArgument(
            'report_output_directory',
            default_value='~/beehive_mission_reports/gazebo'),
    ]

    adapter = Node(
        package='beehive_drone', executable='sim_zed_adapter',
        name='sim_zed_adapter', output='screen', parameters=[{
            'tree_x': typed('tree_x', float),
            'tree_y': typed('tree_y', float),
            'tree_ground_z': typed('tree_ground_z', float),
            'tree_positions': LaunchConfiguration('tree_positions'),
            'camera_x': typed('camera_x', float),
            'camera_y': typed('camera_y', float),
            'camera_z': typed('camera_z', float),
            'camera_roll': typed('camera_roll', float),
            'camera_pitch': typed('camera_pitch', float),
            'camera_yaw': typed('camera_yaw', float),
            'position_noise_stddev': typed('position_noise_stddev', float),
            'dropout_every_n': typed('dropout_every_n', int),
        }])

    range_adapter = Node(
        package='beehive_drone', executable='sim_rangefinder_bridge',
        name='sim_rangefinder_to_mavros', output='screen', parameters=[{
            'input_topic': '/range',
            'output_topic': '/mavros/rangefinder/rangefinder',
            'frame_id': 'range_link',
        }])

    validator = Node(
        package='beehive_drone', executable='real_stack_sim_validator',
        name='real_stack_sim_validator', output='screen', parameters=[{
            'expected_tree_x': typed('tree_x', float),
            'expected_tree_y': typed('tree_y', float),
            'expected_tree_count': typed('expected_tree_count', int),
        }])

    return LaunchDescription(arguments + [
        LogInfo(msg=(
            'Gazebo stack memakai algoritma polinasi dan adapter sensor '
            'sintetis. Jangan jalankan pb_sprayer atau perception stack lain.')),
        Node(
            package='beehive_drone', executable='sim_sprayer',
            name='sim_sprayer', output='screen'),
        adapter,
        range_adapter,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                beehive_share, 'launch', 'vision_to_mavros.launch.py')),
            launch_arguments={
                'use_fixed_yaw_offset': 'true',
                'fixed_yaw_offset_degrees': '0.0',
            }.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                pcl_share, 'launch', 'bb_proc_node.launch.py')),
            launch_arguments={'pose_topic': '/zed/aligned_pose'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                beehive_share, 'launch', 'real_mission.launch.py')),
            launch_arguments={
                'auto_start': LaunchConfiguration('auto_start'),
                'mission_type': LaunchConfiguration('mission_type'),
                'max_trees': LaunchConfiguration('max_trees'),
                'enable_flower_detection': 'false',
                'record_data': 'false',
                'analyzer_output_directory':
                    LaunchConfiguration('report_output_directory'),
            }.items()),
        validator,
    ])
