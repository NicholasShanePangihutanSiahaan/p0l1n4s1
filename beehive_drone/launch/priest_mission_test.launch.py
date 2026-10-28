"""Run the production mission launch with PRIEST enabled for integration tests."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    mission_launch = os.path.join(
        get_package_share_directory('beehive_drone'),
        'launch', 'real_mission.launch.py')
    default_config = os.path.join(
            get_package_share_directory('beehive_drone'), 'config', 'sim.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('mission_type', default_value='basic_orbit'),
        DeclareLaunchArgument('mission_mode', default_value='single_tree'),
        DeclareLaunchArgument('max_trees', default_value='0'),
        DeclareLaunchArgument('altitude_source', default_value='local_position'),
        DeclareLaunchArgument(
            'enable_flower_detection', default_value='false'),
        DeclareLaunchArgument(
            'priest_delay', default_value='15.0',
            description='Delay untuk JIT Compilation dari node PRIEST.')
        ,
        DeclareLaunchArgument('config_file', default_value=default_config),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(mission_launch),
            launch_arguments={
                'auto_start': LaunchConfiguration('auto_start'),
                'mission_type': LaunchConfiguration('mission_type'),
                'mission_mode': LaunchConfiguration('mission_mode'),
                'max_trees': LaunchConfiguration('max_trees'),
                'use_priest': 'true',
                'enable_flower_detection':
                    LaunchConfiguration('enable_flower_detection'),
                'record_data': 'false',
                'altitude_source': LaunchConfiguration('altitude_source'),
                'priest_delay': LaunchConfiguration('priest_delay'),
                'config_file': LaunchConfiguration('config_file'),
            }.items()),
    ])
