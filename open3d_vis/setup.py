from setuptools import setup

package_name = 'open3d_vis'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            ['launch/voxel_visualizer.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wafi',
    maintainer_email='wafialfaruqhi@gmail.com',
    description='Open3D point cloud voxel visualization node',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'voxel_visualizer = open3d_vis.voxel_visualizer:main',
        ],
    },
)