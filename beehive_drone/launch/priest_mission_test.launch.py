"""Real mission nodes.

MAVROS, ZED wrapper, ``vision_to_mavros``, and ``bb_proc_node.launch.py`` must
already be healthy. Perception is intentionally kept out of this launch so the
mission never starts a second publisher on ``/global_cylinders``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
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
    config_file = LaunchConfiguration("config_file")
    record_data = LaunchConfiguration("record_data")
    priest_delay = LaunchConfiguration("priest_delay")

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
                "priest_delay",
                default_value="10.0",
                description=("Delay PRIEST sebab optimisasi JAX"),
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
            # /global_cylinders berasal dari bb_pcl_proc_node yang dijalankan
            # terpisah setelah ZED object detection sehat.
            Node(
                package="priest_dyn_traj",
                executable="priest_dyn_node",
                name="priest_dyn_node",
                parameters=[config_file],
                output="screen",
                emulate_tty=True,
            ),
            TimerAction(
                period=priest_delay,
                actions=[
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
                                "max_trees": ParameterValue(max_trees, value_type=int),
                            },
                        ],
                        output="screen",
                    ),
                ],
                output="screen",
            ),
        ]
    )
