from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rw'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Config files
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*'))),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*'))),
        # Mesh files
        (os.path.join('share', package_name, 'meshes'),
         glob(os.path.join('meshes', '*'))),
        (os.path.join('share', package_name, 'robot_meshes'),
         glob(os.path.join('robot_meshes', '*'))),
        # Model files (URDF)
        (os.path.join('share', package_name, 'urdf'),
         glob(os.path.join('urdf', '*'))),
        # RViz config files
        (os.path.join('share', package_name, 'rviz'),
         glob(os.path.join('rviz', '*'))),

        # --- World files ---
        # Install files directly under 'worlds/' (e.g., empty.sdf, home.sdf, etc.)
        # We filter to ensure we only pick up files, not the subdirectory itself.
        (os.path.join('share', package_name, 'worlds'),
         [f for f in glob(os.path.join('worlds', '*')) if os.path.isfile(f)]),

        # Install the contents of 'worlds/pipeline_road_model/'
        # into 'share/rw/worlds/pipeline_road_model/'
        (os.path.join('share', package_name, 'worlds', 'pipeline_road_model'),
         glob(os.path.join('worlds', 'pipeline_road_model', '*'))),
        # --- End World files ---

        (os.path.join('share', package_name, 'maps'),
         glob(os.path.join('maps', '*'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aldoghry',
    maintainer_email='aldoghry@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 'executable_name = package_name.module_name:main_function',
        ],
    },
)