from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'beehive_drone'

    # 1. Perception & Mapping Nodes
    detector_node = Node(
        package=pkg_name,
        executable='tree_detector',
        name='tree_detector',
        output='screen'
    )
    
    localizer_node = Node(
        package=pkg_name,
        executable='tree_localizer',
        name='tree_localizer',
        output='screen'
    )

    mapper_node = Node(
        package=pkg_name,
        executable='tree_mapper',
        name='tree_mapper',
        output='screen'
    )

    # 2. Navigation & Control Nodes
    velocity_node = Node(
        package=pkg_name,
        executable='velocity_controller',
        name='velocity_controller',
        output='screen'
    )

    vortex_node = Node(
        package=pkg_name,
        executable='vortex_avoidance_controller',
        name='vortex_avoidance_controller',
        output='screen'
    )

    orbit_node = Node(
        package=pkg_name,
        executable='dynamic_orbit_controller',
        name='dynamic_orbit_controller',
        output='screen'
    )

    # 3. Hardware Abstraction Layer (HAL) & Analyzer
    # INI YANG SEBELUMNYA HILANG
    flight_manager_node = Node(
        package=pkg_name,
        executable='flight_manager',
        name='flight_manager',
        output='screen'
    )
    
    analyzer_node = Node(
        package=pkg_name,
        executable='mission_analyzer',
        name='mission_analyzer',
        output='screen'
    )

    # 4. The Brain (FSM)
    fsm_node = Node(
        package=pkg_name,
        executable='mission_state_machine_single_tree',
        name='mission_state_machine_single_tree',
        output='screen'
    )

    return LaunchDescription([
        detector_node,
        localizer_node,
        mapper_node,
        velocity_node,
        vortex_node,
        orbit_node,
        flight_manager_node,
        analyzer_node,
        fsm_node
    ])