"""Real ZED2i + Pixhawk 6C launch. MAVROS and ZED wrapper must already run."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory('beehive_drone'), 'config', 'real.yaml')
    cloud = LaunchConfiguration('pointcloud_topic')
    odom = LaunchConfiguration('odometry_topic')
    auto_start = LaunchConfiguration('auto_start')
    return LaunchDescription([
        DeclareLaunchArgument('pointcloud_topic',
                              default_value='/zed/zed_node/point_cloud/cloud_registered'),
        DeclareLaunchArgument('odometry_topic', default_value='/mavros/local_position/odom'),
        DeclareLaunchArgument(
            'auto_start', default_value='false',
            description='true: otomatis GUIDED/arm/takeoff setelah semua sensor preflight sehat'),
        Node(package='beehive_drone', executable='vision_to_mavros',
             parameters=[config], output='screen'),
        Node(package='point-cloud-test', executable='pcl_proc_node', output='screen',
             remappings=[('/input_cloud', cloud), ('/odom', odom),
                         ('/output_cloud', '/perception/pcl/non_ground'),
                         ('/clusters', '/perception/pcl/clusters'),
                         ('/cylinders', '/perception/pcl/cylinders'),
                         ('/global/cylinders', '/global_cylinders')],
             # Parameter transform ZED dan filter PCL berada di real.yaml.
             parameters=[config]),
        Node(package='beehive_drone', executable='tree_mapper', parameters=[config], output='screen'),
        Node(package='beehive_drone', executable='vortex_avoidance_controller', output='screen'),
        Node(package='beehive_drone', executable='dynamic_orbit_controller', parameters=[config], output='screen'),
        Node(package='beehive_drone', executable='position_setpoint_controller', parameters=[config], output='screen'),
        Node(package='beehive_drone', executable='flight_manager',
             parameters=[config], output='screen'),
        Node(package='beehive_drone', executable='mission_safety_monitor',
             parameters=[config, {'pointcloud_topic': cloud}], output='screen'),
        Node(package='beehive_drone', executable='mission_analyzer',
             parameters=[config], output='screen'),
        Node(package='beehive_drone', executable='mission_state_machine',
             parameters=[config, {'auto_start': auto_start}], output='screen'),
    ])
