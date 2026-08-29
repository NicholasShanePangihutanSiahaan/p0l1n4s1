import os
from glob import glob
from setuptools import setup

package_name = 'beehive_drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['REAL_FLIGHT.md']),
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx')),
        # DAFTARKAN FOLDER LAUNCH DI SINI:
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shane',
    maintainer_email='shane@todo.todo',
    description='Autonomous Plantation Drone Navigation System',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # DAFTARKAN SEMUA NODE DI SINI (nama_eksekusi = nama_folder.nama_file:main)
            'mission_state_machine = beehive_drone.mission_state_machine:main',
            'dynamic_orbit_controller = beehive_drone.dynamic_orbit_controller:main',
            'velocity_controller = beehive_drone.velocity_controller:main',
            'vortex_avoidance_controller = beehive_drone.vortex_avoidance_controller:main',
            'tree_mapper = beehive_drone.tree_mapper:main',
            'flight_manager = beehive_drone.flight_manager:main',
            'mission_analyzer = beehive_drone.mission_analyzer:main',
            'position_setpoint_controller = beehive_drone.position_setpoint_controller:main',
            'mission_safety_monitor = beehive_drone.mission_safety_monitor:main',
            'sim_external_odometry = beehive_drone.sim_external_odometry:main',
            'sim_rangefinder_bridge = beehive_drone.sim_rangefinder_bridge:main',
            'vision_to_mavros = beehive_drone.vision_to_mavros:main',
        ],
    },
)
