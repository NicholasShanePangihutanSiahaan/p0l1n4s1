from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    resolution = DeclareLaunchArgument(
        'octomap_resolution', default_value='0.1',
        description='OctoMap resolution in meters')
    max_range = DeclareLaunchArgument(
        'octomap_max_range', default_value='10.0',
        description='Maximum sensor range for OctoMap')
    publish_free_space = DeclareLaunchArgument(
        'publish_free_space', default_value='false',
        description='Publish free space in OctoMap')

    return LaunchDescription([
        resolution,
        max_range,
        publish_free_space,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('point-cloud-test'),
                    'launch',
                    'pcl_proc_node.launch.py',
                ]),
            ]),
            launch_arguments={
                'input_cloud_topic': '/zed2i/depth/points',
                'odom_topic': '/mavros/odometry/out',
                'output_cloud_topic': '/output_cloud',
            }.items(),
        ),

        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('point-cloud-test'),
                    'config',
                    'octomap_params.yaml',
                ]),
                {
                    'resolution': LaunchConfiguration('octomap_resolution'),
                    'publish_free_space': LaunchConfiguration('publish_free_space'),
                },
            ],
            remappings=[
                ('cloud_in', '/zed2i/depth/points'),
            ],
        ),
    ])