from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_cloud = DeclareLaunchArgument(
        'input_cloud_topic', default_value='/zed2i/depth/points',
        description='Input point cloud topic')
    odom = DeclareLaunchArgument(
        'odom_topic', default_value='/mavros/odometry/out',
        description='Odometry topic')
    output_cloud = DeclareLaunchArgument(
        'output_cloud_topic', default_value='/output_cloud',
        description='Output global point cloud topic')
    output_cluster = DeclareLaunchArgument(
        'output_cluster', default_value='/clusters',
        description='Output clusters of trees')
    output_cylinders = DeclareLaunchArgument(
        'output_cylinders', default_value='/cylinders',
    description='Output clusters of trees')

    return LaunchDescription([
        input_cloud,
        odom,
        output_cloud,
        output_cluster,
        output_cylinders,
        Node(
            package='point-cloud-test',
            executable='pcl_proc_node',
            name='pcl_proc_node',
            output='screen',
            remappings=[
                ('/input_cloud', LaunchConfiguration('input_cloud_topic')),
                ('/odom', LaunchConfiguration('odom_topic')),
                ('/output_cloud', LaunchConfiguration('output_cloud_topic')),
                ('/clusters', LaunchConfiguration('output_cluster')),
                ('/cylinders', LaunchConfiguration('output_cylinders')),
                ('/global/cylinders', '/global_cylinders'),
            ],
        ),
    ])