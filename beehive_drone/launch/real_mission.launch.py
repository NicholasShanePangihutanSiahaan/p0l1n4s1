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
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config_path = os.path.join(
        get_package_share_directory("beehive_drone"), "config", "real.yaml"
    )
    auto_start = LaunchConfiguration("auto_start")
    analyzer_output_directory = LaunchConfiguration("analyzer_output_directory")
    mission_type = LaunchConfiguration("mission_type")
    mission_mode = LaunchConfiguration("mission_mode")
    max_trees = LaunchConfiguration("max_trees")
    require_tree_ahead = LaunchConfiguration("require_tree_ahead")
    config_file = LaunchConfiguration("config_file")
    record_data = LaunchConfiguration("record_data")
    enable_flower_detection = LaunchConfiguration("enable_flower_detection")
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
            DeclareLaunchArgument(
                "mission_mode",
                default_value="single_tree",
                choices=["single_tree", "multi_tree"],
                description=(
                    "single_tree: satu pohon lalu pulang; multi_tree: lanjut "
                    "memilih pohon berikutnya sampai max_trees tercapai."
                ),
            ),
            DeclareLaunchArgument(
                "enable_flower_detection",
                default_value="true",
                description=(
                    "Jalankan detect_flower; dapat dimatikan untuk simulasi "
                    "pohon tanpa objek bunga sintetis."
                ),
            ),
            DeclareLaunchArgument(
                "max_trees",
                default_value="0",
                description=(
                    "Batas pohon yang diproses; 0 berarti lanjut sampai "
                    "eksplorasi selesai."
                ),
            ),
            DeclareLaunchArgument(
                "require_tree_ahead", default_value="true",
                description="Filter pohon nyata ke arah eksplorasi; target virtual selalu dikecualikan.",
            ),
            DeclareLaunchArgument(
                "virtual_tree_position_mode", default_value="home_relative",
                choices=["toward_home", "home_relative", "map"],
            ),
            DeclareLaunchArgument("virtual_tree_position_x", default_value="6.0"),
            DeclareLaunchArgument("virtual_tree_position_y", default_value="3.0"),
            DeclareLaunchArgument(
                "virtual_tree_offset_toward_home", default_value="6.0"),
            DeclareLaunchArgument("virtual_tree_id", default_value="9001"),
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
                    {
                        "auto_start": ParameterValue(
                            auto_start, value_type=bool
                        ),
                        "mission_type": mission_type,
                        "mission_mode": mission_mode,
                        "max_trees": ParameterValue(
                            max_trees, value_type=int
                        ),
                        "require_tree_ahead": ParameterValue(
                            require_tree_ahead, value_type=bool
                        ),
                        "virtual_tree_position_mode": LaunchConfiguration(
                            "virtual_tree_position_mode"
                        ),
                        "virtual_tree_position_x": ParameterValue(
                            LaunchConfiguration("virtual_tree_position_x"),
                            value_type=float,
                        ),
                        "virtual_tree_position_y": ParameterValue(
                            LaunchConfiguration("virtual_tree_position_y"),
                            value_type=float,
                        ),
                        "virtual_tree_offset_toward_home": ParameterValue(
                            LaunchConfiguration(
                                "virtual_tree_offset_toward_home"
                            ), value_type=float,
                        ),
                        "virtual_tree_id": ParameterValue(
                            LaunchConfiguration("virtual_tree_id"),
                            value_type=int,
                        ),
                    },
                ],
                output="screen",
            ),
            Node(
                package="beehive_drone",
                executable="detect_flower",
                condition=IfCondition(enable_flower_detection),
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
