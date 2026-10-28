"""Launch the field-tested mission stack with optional PRIEST avoidance."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('beehive_drone'), 'config', 'real.yaml')
    config = LaunchConfiguration('config_file')
    auto_start = LaunchConfiguration('auto_start')
    mission_type = LaunchConfiguration('mission_type')
    mission_mode = LaunchConfiguration('mission_mode')
    max_trees = LaunchConfiguration('max_trees')
    use_priest = LaunchConfiguration('use_priest')
    record_data = LaunchConfiguration('record_data')
    enable_flower = LaunchConfiguration('enable_flower_detection')
    altitude_source = LaunchConfiguration('altitude_source')
    hover_tolerance = LaunchConfiguration('hover_tolerance')
    priest_delay = LaunchConfiguration('priest_delay')

    arguments = [
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument(
            'mission_type', default_value='basic_orbit',
            description='Registered beehive_drone mission strategy.'),
        DeclareLaunchArgument(
            'mission_mode', default_value='single_tree',
            choices=['single_tree', 'multi_tree']),
        DeclareLaunchArgument(
            'priest_delay', default_value='10.0',
            description='Delay untuk JIT Compilation dari node PRIEST.'),
        DeclareLaunchArgument(
            'max_trees', default_value='1',
            description='Tree limit in multi_tree mode; 0 is unlimited.'),
        DeclareLaunchArgument(
            'use_priest', default_value='false',
            description=(
                'false preserves branch_wafi vortex behavior; true routes '
                'non-orbit goals through PRIEST.')),
        DeclareLaunchArgument(
            'enable_flower_detection', default_value='true'),
        DeclareLaunchArgument(
            'altitude_source', default_value='rangefinder',
            choices=['local_position', 'relative_alt', 'rangefinder'],
            description='Sumber altitude untuk konfirmasi takeoff.'),
        DeclareLaunchArgument(
            'hover_tolerance', default_value='0.2',
            description='Toleransi konfirmasi ketinggian takeoff.'),
        DeclareLaunchArgument('record_data', default_value='false'),
        DeclareLaunchArgument(
            'analyzer_output_directory',
            default_value='~/beehive_mission_reports/real'),
        DeclareLaunchArgument(
            'record_script',
            default_value='/home/palmbee1/DTETI-WS/data/record_scripts.sh'),
        DeclareLaunchArgument(
            'record_output_directory',
            default_value='/home/palmbee1/DTETI-WS/data/'),
    ]

    arguments.extend([
        DeclareLaunchArgument('require_tree_ahead', default_value='true'),
        DeclareLaunchArgument('virtual_tree_position_mode', default_value='home_relative',
                              choices=['toward_home', 'home_relative', 'map']),
        DeclareLaunchArgument('virtual_tree_position_x', default_value='6.0'),
        DeclareLaunchArgument('virtual_tree_position_y', default_value='3.0'),
        DeclareLaunchArgument('virtual_tree_offset_toward_home', default_value='6.0'),
        DeclareLaunchArgument('virtual_tree_id', default_value='9001'),
    ])

    priest_node = [
        Node(
            package='priest_dyn_traj', executable='priest_dyn_node',
            name='priest_dyn_node', condition=IfCondition(use_priest),
            parameters=[config], output='screen', emulate_tty=True),
    ]


    nodes = [
        Node(
            package='beehive_drone', executable='tree_mapper',
            parameters=[config], output='screen'),
        Node(
            package='beehive_drone', executable='vortex_avoidance_controller',
            condition=UnlessCondition(use_priest), output='screen'),
        Node(
            package='beehive_drone', executable='priest_avoidance_controller',
            condition=IfCondition(use_priest), parameters=[config],
            output='screen'),
        Node(
            package='beehive_drone', executable='dynamic_orbit_controller',
            parameters=[config], output='screen'),
        Node(
            package='beehive_drone', executable='position_setpoint_controller',
            parameters=[config], output='screen'),
        Node(
            package='beehive_drone', executable='flight_manager',
            parameters=[config, {
                'runtime_altitude_source': altitude_source,
                'runtime_hover_tolerance': ParameterValue(
                    hover_tolerance, value_type=float),
            }], output='screen'),
        Node(
            package='beehive_drone', executable='mission_safety_monitor',
            parameters=[config], output='screen'),
        Node(
            package='beehive_drone', executable='mission_analyzer',
            condition=IfCondition(record_data),
            parameters=[config, {
                'output_directory':
                    LaunchConfiguration('analyzer_output_directory')}],
            output='screen'),
        # Exactly one FSM owns mode/arm/takeoff/navigation commands.
        Node(
            package='beehive_drone', executable='mission_state_machine',
            parameters=[config, {
                'runtime_auto_start': auto_start,
                'require_tree_ahead': ParameterValue(
                    LaunchConfiguration('require_tree_ahead'), value_type=bool),
                'virtual_tree_position_mode': LaunchConfiguration('virtual_tree_position_mode'),
                'virtual_tree_position_x': ParameterValue(
                    LaunchConfiguration('virtual_tree_position_x'), value_type=float),
                'virtual_tree_position_y': ParameterValue(
                    LaunchConfiguration('virtual_tree_position_y'), value_type=float),
                'virtual_tree_offset_toward_home': ParameterValue(
                    LaunchConfiguration('virtual_tree_offset_toward_home'), value_type=float),
                'virtual_tree_id': ParameterValue(
                    LaunchConfiguration('virtual_tree_id'), value_type=int),
                'runtime_mission_type': mission_type,
                'runtime_mission_mode': mission_mode,
                'runtime_max_trees': ParameterValue(
                    max_trees, value_type=int),
            }],
            output='screen'),
        Node(
            package='beehive_drone', executable='detect_flower',
            condition=IfCondition(enable_flower), parameters=[config],
            output='screen'),
        ExecuteProcess(
            cmd=[
                '/bin/bash', LaunchConfiguration('record_script'),
                LaunchConfiguration('record_output_directory')],
            condition=IfCondition(record_data), output='screen'),
    ]

    delayed_nodes = TimerAction(
        period=priest_delay,
        actions=nodes,
        condition=IfCondition(use_priest)
    )

    immediate_nodes = GroupAction(
        actions=nodes,
        condition=UnlessCondition(use_priest),
    )
    return LaunchDescription(
        arguments
        + priest_node
        + [
            immediate_nodes,
            delayed_nodes,
        ]
    )
