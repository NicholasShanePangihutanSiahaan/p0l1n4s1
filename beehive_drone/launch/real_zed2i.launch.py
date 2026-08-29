"""Start ZED2i tracking and the packaged palm-tree detector."""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import OpaqueFunction


def launch_setup(context):
    model_path = LaunchConfiguration('model_path').perform(context)
    labels_path = LaunchConfiguration('labels_path').perform(context)
    override = {
        '/**': {
            'ros__parameters': {
                'general': {
                    'camera_model': 'zed2i',
                    'camera_name': 'zed',
                    'grab_resolution': 'HD720',
                    'grab_frame_rate': 15,
                },
                'depth': {
                    'min_depth': 0.3,
                    'max_depth': 10.0,
                },
                'positional_tracking': {
                    'pos_tracking_enabled': True,
                    'publish_tf': True,
                    'publish_map_tf': True,
                },
                'object_detection': {
                    'od_enabled': True,
                    'model': 'CUSTOM_YOLOLIKE_BOX_OBJECTS',
                    'custom_onnx_file': model_path,
                    'custom_onnx_input_size': 512,
                    'custom_label_yaml': labels_path,
                    'allow_reduced_precision_inference': True,
                    'max_range': 10.0,
                    'confidence_threshold': 25.0,
                    'object_tracking_enabled': True,
                    'enable_tracking': True,
                    'filtering_mode': 1,
                },
            }
        }
    }
    override_path = os.path.join(
        tempfile.gettempdir(), 'beehive_zed2i_pohon.yaml')
    with open(override_path, 'w', encoding='utf-8') as stream:
        yaml.safe_dump(override, stream, sort_keys=False)

    zed_launch = os.path.join(
        get_package_share_directory('zed_wrapper'),
        'launch', 'zed_camera.launch.py')
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(zed_launch),
        launch_arguments={
            'camera_model': 'zed2i',
            'ros_params_override_path': override_path,
        }.items())]


def generate_launch_description():
    share = get_package_share_directory('beehive_drone')
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.join(
                share, 'models', 'best_detection_palm_oil.onnx')),
        DeclareLaunchArgument(
            'labels_path',
            default_value=os.path.join(share, 'config', 'pohon_labels.yaml')),
        OpaqueFunction(function=launch_setup),
    ])
