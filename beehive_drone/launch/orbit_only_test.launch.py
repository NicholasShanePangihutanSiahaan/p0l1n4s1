"""Field test: GUIDED -> arm -> takeoff -> synthetic orbit -> landing."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def typed(name, value_type):
    return ParameterValue(LaunchConfiguration(name), value_type=value_type)


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('takeoff_altitude', default_value='1.5'),
        DeclareLaunchArgument('altitude_source', default_value='rangefinder'),
        DeclareLaunchArgument('hover_tolerance', default_value='0.25'),
        DeclareLaunchArgument('orbit_radius', default_value='2.0'),
        DeclareLaunchArgument('orbit_velocity', default_value='0.5'),
        DeclareLaunchArgument('orbit_timeout', default_value='120.0'),
        DeclareLaunchArgument(
            'orbit_center_offset_x', default_value='2.0',
            description='Pusat orbit relatif terhadap X posisi takeoff.'),
        DeclareLaunchArgument(
            'orbit_center_offset_y', default_value='0.0',
            description='Pusat orbit relatif terhadap Y posisi takeoff.'),
    ]

    nodes = [
        Node(
            package='beehive_drone', executable='flight_manager',
            parameters=[{
                'altitude_source': LaunchConfiguration('altitude_source'),
                'hover_tolerance': typed('hover_tolerance', float),
            }], output='screen'),
        Node(
            package='beehive_drone', executable='dynamic_orbit_controller',
            parameters=[{
                'orbit_radius': typed('orbit_radius', float),
                'orbit_altitude': typed('takeoff_altitude', float),
                'orbit_velocity': typed('orbit_velocity', float),
                'orbit_timeout': typed('orbit_timeout', float),
            }], output='screen'),
        Node(
            package='beehive_drone', executable='vortex_avoidance_controller',
            output='screen'),
        Node(
            package='beehive_drone', executable='position_setpoint_controller',
            output='screen'),
        Node(
            package='beehive_drone', executable='orbit_only_test_fsm',
            parameters=[{
                'auto_start': typed('auto_start', bool),
                'takeoff_altitude': typed('takeoff_altitude', float),
                'orbit_center_offset_x': typed(
                    'orbit_center_offset_x', float),
                'orbit_center_offset_y': typed(
                    'orbit_center_offset_y', float),
            }], output='screen'),
    ]
    return LaunchDescription(arguments + nodes)
