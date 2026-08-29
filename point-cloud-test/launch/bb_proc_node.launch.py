"""Convert ZED 3D object bounding boxes into tracked tree landmarks."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare('point-cloud-test'), 'config', 'bb_node_params.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'objects_topic',
            default_value='/zed/zed_node/obj_det/objects',
            description='ZED ObjectsStamped topic'),
        DeclareLaunchArgument(
            'pose_topic', default_value='/zed/zed_node/pose',
            description='ZED global PoseStamped topic'),
        DeclareLaunchArgument(
            'odom_topic', default_value='/mavros/local_position/odom',
            description='Alternative odometry when use_odom_pose is true'),
        DeclareLaunchArgument(
            'output_cylinders',
            default_value='/perception/bb/cylinders'),
        DeclareLaunchArgument(
            'output_global_cylinders', default_value='/global_cylinders'),
        DeclareLaunchArgument(
            'object_label_target', default_value='pohon',
            description='Exact label emitted by ZED custom object detection'),
        DeclareLaunchArgument(
            'yaml_params_file', default_value=params),
        Node(
            package='point-cloud-test',
            executable='bb_pcl_proc_node',
            name='bb_pcl_proc_node',
            output='screen',
            remappings=[
                ('/objects', LaunchConfiguration('objects_topic')),
                ('/pose', LaunchConfiguration('pose_topic')),
                ('/odom', LaunchConfiguration('odom_topic')),
                ('/cylinders', LaunchConfiguration('output_cylinders')),
                ('/global/cylinders',
                 LaunchConfiguration('output_global_cylinders')),
            ],
            parameters=[
                LaunchConfiguration('yaml_params_file'),
                {'object_label_target':
                 LaunchConfiguration('object_label_target')},
            ],
        ),
    ])
