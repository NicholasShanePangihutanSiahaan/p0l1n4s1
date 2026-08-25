from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(get_package_share_directory('beehive_drone'), 'config', 'sim.yaml')
    pcl = Node(package='point-cloud-test', executable='pcl_proc_node', output='screen',
               remappings=[('/input_cloud', '/zed2i/depth/points'),
                           ('/odom', '/simulation/ground_truth/odom'),
                           ('/output_cloud', '/perception/pcl/non_ground'),
                           ('/clusters', '/perception/pcl/clusters'),
                           ('/cylinders', '/perception/pcl/cylinders'),
                           ('/global/cylinders', '/global_cylinders')],
               parameters=[config])
    sim_pose_remap = [('/mavros/local_position/pose', '/simulation/local_position/pose')]
    mission_state_machine = Node(
        package='beehive_drone', executable='mission_state_machine',
        parameters=[config], remappings=sim_pose_remap, output='screen')
    return LaunchDescription([
        Node(package='beehive_drone', executable='sim_external_odometry', output='screen'),
        Node(package='beehive_drone', executable='sim_rangefinder_bridge',
             parameters=[config], output='screen'),
        pcl,
        Node(package='beehive_drone', executable='tree_mapper', parameters=[config], output='screen'),
        # Gunakan jalur controller yang sama dengan drone nyata agar respons
        # gerak simulasi mewakili real_mission.launch.py.
        Node(package='beehive_drone', executable='position_setpoint_controller',
             parameters=[config], remappings=sim_pose_remap, output='screen'),
        Node(package='beehive_drone', executable='vortex_avoidance_controller', remappings=sim_pose_remap, output='screen'),
        Node(package='beehive_drone', executable='dynamic_orbit_controller', parameters=[config], remappings=sim_pose_remap, output='screen'),
        Node(package='beehive_drone', executable='flight_manager',
             parameters=[config], remappings=sim_pose_remap, output='screen'),
        Node(package='beehive_drone', executable='mission_safety_monitor', parameters=[config], remappings=sim_pose_remap, output='screen'),
        Node(package='beehive_drone', executable='mission_analyzer', parameters=[config],
             remappings=sim_pose_remap, output='screen'),
        # Hindari command GUIDED pertama terbit sebelum flight_manager selesai
        # membuat subscription dan service client MAVROS.
        TimerAction(period=2.0, actions=[mission_state_machine]),
    ])
