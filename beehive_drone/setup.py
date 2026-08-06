from glob import glob
import os
from setuptools import find_packages, setup

package_name = "beehive_drone"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="shane",
    maintainer_email="shane@todo.todo",
    description="ZED 2i VSLAM + PCL single-tree orbit mission for MAVROS/ArduPilot.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vision_to_mavros = beehive_drone.vision_to_mavros:main",
            "flight_manager = beehive_drone.flight_manager:main",
            "pcl_tree_mapper = beehive_drone.pcl_tree_mapper:main",
            "simple_single_tree_mission = beehive_drone.simple_single_tree_mission:main",
            "mission_analyzer = beehive_drone.mission_analyzer:main",
            # Retained real-drone utilities / legacy perception nodes.
            "tree_localizer = beehive_drone.tree_localizer:main",
            "odom_tester = beehive_drone.odom_tester:main",
            "pratesting_works = beehive_drone.pratesting_works:main",
            "tes_kiri_kanan = beehive_drone.tes_kiri_kanan:main",
        ]
    },
)
