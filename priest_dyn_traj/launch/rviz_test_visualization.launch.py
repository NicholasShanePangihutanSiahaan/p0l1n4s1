#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def declare_arguments(default_priest_config, default_rviz_config):
    """Declare planner, test-selection, RViz, and test-script arguments."""
    return [
        DeclareLaunchArgument(
            'priest_config_yaml',
            default_value=default_priest_config,
            description='PRIEST ROS parameter YAML file',
        ),
        DeclareLaunchArgument(
            'JIT_Delay',
            default_value='20.0',
            description='Seconds to wait before starting the selected test publisher',
        ),
        DeclareLaunchArgument(
            'type',
            default_value='linear',
            choices=['linear', 'by_ref_trajectory'],
            description='Synthetic input type to launch',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=default_rviz_config,
            description='RViz configuration file',
        ),

        # test_linear.py arguments
        DeclareLaunchArgument('linear_goal_x', default_value='5.0'),
        DeclareLaunchArgument('linear_goal_y', default_value='0.0'),
        DeclareLaunchArgument('linear_goal_z', default_value='0.0'),
        DeclareLaunchArgument('linear_state', default_value='LINEAR_GOAL'),
        DeclareLaunchArgument('linear_period', default_value='0.1'),
        DeclareLaunchArgument('linear_obstacle_period', default_value='2.0'),
        DeclareLaunchArgument('linear_min_obstacles', default_value='5'),
        DeclareLaunchArgument('linear_max_obstacles', default_value='10'),
        DeclareLaunchArgument('linear_obstacle_spread', default_value='0.60'),
        DeclareLaunchArgument('linear_seed', default_value='0'),

        # test_ref_traj.py arguments
        DeclareLaunchArgument('ref_radius', default_value='2.0'),
        DeclareLaunchArgument('ref_samples', default_value='100'),
        DeclareLaunchArgument('ref_state', default_value='WAIT_ORBIT'),
        DeclareLaunchArgument('ref_period', default_value='1.0'),
        DeclareLaunchArgument('ref_obstacle_period', default_value='2.0'),
        DeclareLaunchArgument('ref_min_obstacles', default_value='5'),
        DeclareLaunchArgument('ref_max_obstacles', default_value='10'),
        DeclareLaunchArgument('ref_obstacle_radial_spread', default_value='0.75'),
        DeclareLaunchArgument('ref_seed', default_value='0'),
    ]


def generate_launch_description():
    package_share = get_package_share_directory('priest_dyn_traj')
    default_priest_config = os.path.join(
        package_share,
        'config',
        'priest_node_config.yaml',
    )
    default_rviz_config = os.path.join(
        package_share,
        'config',
        'rviz_test_visualization.yaml',
    )

    planner_type = LaunchConfiguration('type')

    priest_node = Node(
        package='priest_dyn_traj',
        executable='priest_dyn_node',
        name='priest_dyn_node',
        output='screen',
        parameters=[LaunchConfiguration('priest_config_yaml')],
    )

    linear_test = Node(
        package='priest_dyn_traj',
        executable='test_linear.py',
        name='priest_linear_reference_test',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", planner_type, "' == 'linear'"])
        ),
        arguments=[
            '--goal-x', LaunchConfiguration('linear_goal_x'),
            '--goal-y', LaunchConfiguration('linear_goal_y'),
            '--goal-z', LaunchConfiguration('linear_goal_z'),
            '--state', LaunchConfiguration('linear_state'),
            '--period', LaunchConfiguration('linear_period'),
            '--obstacle-period', LaunchConfiguration('linear_obstacle_period'),
            '--min-obstacles', LaunchConfiguration('linear_min_obstacles'),
            '--max-obstacles', LaunchConfiguration('linear_max_obstacles'),
            '--obstacle-spread', LaunchConfiguration('linear_obstacle_spread'),
            '--seed', LaunchConfiguration('linear_seed'),
        ],
    )

    reference_test = Node(
        package='priest_dyn_traj',
        executable='test_ref_traj.py',
        name='priest_circular_reference_test',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", planner_type, "' == 'by_ref_trajectory'"])
        ),
        arguments=[
            '--radius', LaunchConfiguration('ref_radius'),
            '--samples', LaunchConfiguration('ref_samples'),
            '--state', LaunchConfiguration('ref_state'),
            '--period', LaunchConfiguration('ref_period'),
            '--obstacle-period', LaunchConfiguration('ref_obstacle_period'),
            '--min-obstacles', LaunchConfiguration('ref_min_obstacles'),
            '--max-obstacles', LaunchConfiguration('ref_max_obstacles'),
            '--obstacle-radial-spread',
            LaunchConfiguration('ref_obstacle_radial_spread'),
            '--seed', LaunchConfiguration('ref_seed'),
        ],
    )

    # Delay only the synthetic publisher. PRIEST and RViz start immediately,
    # giving JAX time to finish its startup compilation first.
    delayed_test = TimerAction(
        period=LaunchConfiguration('JIT_Delay'),
        actions=[linear_test, reference_test],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='priest_test_rviz',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    return LaunchDescription(
        declare_arguments(default_priest_config, default_rviz_config)
        + [priest_node, rviz, delayed_test]
    )
