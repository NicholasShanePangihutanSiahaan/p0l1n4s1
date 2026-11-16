"""Opt-in adaptive orbit using real localization; no simulation pose publishers."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share=get_package_share_directory('beehive_drone')
    config=os.path.join(share,'config','real_adaptive.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('lidar_topic',description='Actual PointCloud2 topic from the real LiDAR driver'),
        DeclareLaunchArgument('auto_start',default_value='false'),
        Node(package='beehive_drone',executable='lidar_tree_verifier',parameters=[config,{'lidar_topic':LaunchConfiguration('lidar_topic')}],output='screen'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share,'launch','real_mission.launch.py')),launch_arguments={'config_file':config,'mission_type':'adaptive_orbit','mission_mode':'single_tree','max_trees':'1','auto_start':LaunchConfiguration('auto_start'),'record_data':'true'}.items())
    ])
