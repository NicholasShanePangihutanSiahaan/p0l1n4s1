#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("beehive_drone")
    config_file = os.path.join(share, "config", "mission_real_pcl.yaml")

    use_vslam_bridge = LaunchConfiguration("use_vslam_bridge")
    use_pcl = LaunchConfiguration("use_pcl")
    use_analyzer = LaunchConfiguration("use_analyzer")
    hold_after_takeoff = LaunchConfiguration("hold_after_takeoff")
    point_cloud_topic = LaunchConfiguration("point_cloud_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    zed_pose_topic = LaunchConfiguration("zed_pose_topic")

    return LaunchDescription([
        DeclareLaunchArgument("use_vslam_bridge", default_value="true"),
        DeclareLaunchArgument("use_pcl", default_value="true"),
        DeclareLaunchArgument("use_analyzer", default_value="true"),
        # Safety default: first launch only arms/takes off/holds. Set false after bench tests.
        DeclareLaunchArgument("hold_after_takeoff", default_value="true"),
        DeclareLaunchArgument(
            "point_cloud_topic",
            default_value="/zed/zed_node/point_cloud/cloud_registered",
        ),
        DeclareLaunchArgument(
            "zed_pose_topic", default_value="/zed/zed_node/pose"
        ),
        DeclareLaunchArgument(
            "odom_topic", default_value="/mavros/odometry/out"
        ),

        Node(
            package="beehive_drone",
            executable="vision_to_mavros",
            name="vision_to_mavros",
            output="screen",
            condition=IfCondition(use_vslam_bridge),
            parameters=[
                config_file,
                {"input_pose_topic": zed_pose_topic},
            ],
        ),
        Node(
            package="point-cloud-test",
            executable="pcl_proc_node",
            name="pcl_proc_node",
            output="screen",
            condition=IfCondition(use_pcl),
            parameters=[config_file],
            remappings=[
                ("/input_cloud", point_cloud_topic),
                ("/odom", odom_topic),
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
            condition=IfCondition(use_pcl),
            parameters=[config_file],
        ),
        Node(
            package="beehive_drone",
            executable="flight_manager",
            name="flight_manager",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="beehive_drone",
            executable="mission_state_machine",
            name="mission_state_machine",
            output="screen",
            parameters=[
                config_file,
                {
                    "hold_after_takeoff": ParameterValue(
                        hold_after_takeoff, value_type=bool
                    )
                },
            ],
        ),
        Node(
            package="beehive_drone",
            executable="dynamic_orbit_controller",
            name="dynamic_orbit_controller",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="beehive_drone",
            executable="vortex_avoidance_controller",
            name="vortex_avoidance_controller",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="beehive_drone",
            executable="velocity_controller",
            name="velocity_controller",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="beehive_drone",
            executable="mission_analyzer",
            name="mission_analyzer",
            output="screen",
            condition=IfCondition(use_analyzer),
            parameters=[config_file],
        ),
    ])
