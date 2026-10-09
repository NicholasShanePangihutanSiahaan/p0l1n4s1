from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'priest_dyn_traj'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    scripts=[
        'test_scripts/test_linear.py',
        'test_scripts/test_ref_traj.py',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wafi',
    maintainer_email='wafialfaruqhi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'priest_dyn_node = priest_dyn_traj.priest_dyn_node:main'
        ],
    },
)
