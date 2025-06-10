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
            'fusion_segmentation_node = rw_py.lidar_camera_fusion_node:main',
        ],
    },
)
