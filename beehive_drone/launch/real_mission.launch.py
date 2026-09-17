"""Real mission nodes.

MAVROS, ZED wrapper, ``vision_to_mavros``, and ``bb_proc_node.launch.py`` must
already be healthy. Perception is intentionally kept out of this launch so the
mission never starts a second publisher on ``/global_cylinders``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config_path = os.path.join(
        get_package_share_directory("beehive_drone"), "config", "real.yaml"
    )
    auto_start = LaunchConfiguration("auto_start")
    analyzer_output_directory = LaunchConfiguration("analyzer_output_directory")
    mission_type = LaunchConfiguration("mission_type")
    config_file = LaunchConfiguration("config_file")
    record_data = LaunchConfiguration("record_data")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config_path,
                description="Path to custom YAML parameter configuration file.",
            ),
            DeclareLaunchArgument(
                "auto_start",
                default_value="false",
                description=(
                    "true: otomatis GUIDED/arm/takeoff setelah local pose "
                    "tersedia. Aktifkan hanya setelah MAVROS, ZED, dan vision "
                    "bridge sehat."
                ),
            ),
            DeclareLaunchArgument(
                "analyzer_output_directory",
                default_value="~/beehive_mission_reports/real",
                description="Mission analyzer output directory.",
            ),
            DeclareLaunchArgument(
                "record_data",
                default_value="true",
                description="Set to true to enable shell script data recording process.",
            ),
            DeclareLaunchArgument(
                "mission_type",
                default_value="basic_orbit",
                description=(
                    "Strategi misi yang akan dijalankan (misal: basic_orbit). "
                    "Terdaftar di beehive_drone.missions.MISSION_STRATEGIES."
                ),
            ),
            # /global_cylinders berasal dari bb_pcl_proc_node yang dijalankan
            # terpisah setelah ZED object detection sehat.
            Node(
                package="beehive_drone",
                executable="tree_mapper",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="vortex_avoidance_controller",
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="dynamic_orbit_controller",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="position_setpoint_controller",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="flight_manager",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="mission_safety_monitor",
                parameters=[config_file],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="mission_analyzer",
                condition=IfCondition(record_data),
                parameters=[
                    config_file,
                    {"output_directory": analyzer_output_directory},
                ],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="mission_state_machine",
                parameters=[
                    config_file,
                    {"auto_start": auto_start, "mission_type": mission_type},
                ],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="detect_flower",
                parameters=[config_file],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    "/bin/bash",
                    "/home/palmbee1/DTETI-WS/data/record_scripts.sh",
                    "/home/palmbee1/DTETI-WS/data/",
                ],
                output="screen",
                condition=IfCondition(record_data),
            ),
        ]
    )
