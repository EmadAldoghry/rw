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
            'send_initialpose = rw_py.send_initialpose:main',
            'slam_toolbox_load_map = rw_py.slam_toolbox_load_map:main',
            'follow_waypoints = rw_py.follow_waypoints:main',
            'lidar_camera_fuser = rw_py.lidar_camera_fusion_node:main',
            'chase_the_ball = rw_py.chase_the_ball:main',
            'roi_cropper = rw_py.roi_cropper_node:main',
            'fusion_segmentation_node = rw_py.lidar_camera_fusion_segmentation_node:main',
            'object_global_localizer_node = rw_py.object_global_localizer_node:main',
            'waypoint_corrector_navigator = rw_py.waypoint_corrector_navigator:main',

        ],
    },
)
