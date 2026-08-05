#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("beehive_drone")
    config_file = os.path.join(share, "config", "mission_real_pcl.yaml")
    cloud = LaunchConfiguration("point_cloud_topic")
    odom = LaunchConfiguration("odom_topic")

    return LaunchDescription([
        DeclareLaunchArgument(
            "point_cloud_topic",
            default_value="/zed/zed_node/point_cloud/cloud_registered",
        ),
        DeclareLaunchArgument("odom_topic", default_value="/mavros/odometry/out"),
        Node(
            package="point-cloud-test",
            executable="pcl_proc_node",
            name="pcl_proc_node",
            output="screen",
            parameters=[config_file],
            remappings=[
                ("/input_cloud", cloud),
                ("/odom", odom),
                ("/output_cloud", "/perception/pcl/filtered_cloud"),
                ("/clusters", "/perception/pcl/clusters"),
                ("/cylinders", "/perception/pcl/cylinders"),
                ("/global/cylinders", "/perception/pcl/tracked_cylinders"),
            ],
        ),
        Node(
            package="beehive_drone",
            executable="pcl_tree_mapper",
            name="pcl_tree_mapper",
            output="screen",
            parameters=[config_file],
        ),
    ])
