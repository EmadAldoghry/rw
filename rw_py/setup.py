from setuptools import find_packages, setup

package_name = 'rw_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Emad Aldoghry',
    maintainer_email='Aldoghry@isac.rwth-aachen.de',
    description='Python nodes',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'follow_waypoints = rw_py.follow_waypoints:main',

            'waypoint_server = rw_py.waypoint_server:main',
            'proximity_monitor = rw_py.proximity_monitor:main',
            'waypoint_visualizer = rw_py.waypoint_visualizer:main',

            'image_segmenter_node = rw_py.image_segmenter_node:main',
            'pointcloud_fuser_node = rw_py.pointcloud_fuser_node:main',
            'goal_calculator_node = rw_py.goal_calculator_node:main',
            'fusion_visualizer_node = rw_py.fusion_visualizer_node:main',
        ],
    },
)
