import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, TextSubstitution # Added TextSubstitution
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get package share directories
    pkg_bme_gazebo_sensors = get_package_share_directory('rw')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim') # Added this

    # --- Environment Variable Setup ---
    # Path to your custom gazebo models
    gazebo_models_path = "/home/aldoghry/gazebo_models" # Make sure this path is correct for your system

    # Get the current value of GZ_SIM_RESOURCE_PATH, or an empty string if it's not set
    # Using os.getenv() within the launch file is generally preferred over os.environ
    current_gz_path = os.getenv('GZ_SIM_RESOURCE_PATH', '')

    # Construct the new path: append your custom path, handling the case where the variable was initially empty
    new_gz_path = current_gz_path + os.pathsep + gazebo_models_path if current_gz_path else gazebo_models_path

    # Action to set the environment variable for the entire launch scope
    set_gazebo_path_action = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', new_gz_path)


    # --- Launch Arguments ---
    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='navigation.rviz',
        description='RViz config file'
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value='pipeline_generated.world', # Kept default from main file
        description='Name of the Gazebo world file to load (in rw/worlds)'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='rw_bot.urdf',
        description='Name of the URDF description to load (in rw/urdf)'
    )

    # x_arg = DeclareLaunchArgument(
    #     'x', default_value='0.0',
    #     description='x coordinate of spawned robot'
    # )

    # y_arg = DeclareLaunchArgument(
    #     'y', default_value='0.0',
    #     description='y coordinate of spawned robot'
    # )

    # yaw_arg = DeclareLaunchArgument(
    #     'yaw', default_value='0.0',
    #     description='yaw angle of spawned robot'
    # )

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )

     # Path to the Slam Toolbox launch file
    nav2_localization_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'localization_launch.py'
    )

    nav2_navigation_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py'
    )

    localization_params_path = os.path.join(
        get_package_share_directory('rw'),
        'config',
        'amcl_localization.yaml'
    )

    navigation_params_path = os.path.join(
        get_package_share_directory('rw'),
        'config',
        'navigation.yaml'
    )

    map_file_path = os.path.join(
        get_package_share_directory('rw'),
        'maps',
        'alpha_shape_nav2_map.yaml'
    )

    # --- Paths ---
    urdf_file_path = PathJoinSubstitution([
        pkg_bme_gazebo_sensors,
        "urdf",
        LaunchConfiguration('model')
    ])

    gz_bridge_params_path = os.path.join(
        get_package_share_directory('rw'),
        'config',
        'gz_bridge.yaml'
    )

    # --- Gazebo Sim Launch (Merged from world.launch.py) ---
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={
            'gz_args': [ # Pass world file path and arguments to Gazebo
                PathJoinSubstitution([
                    pkg_bme_gazebo_sensors,
                    'worlds',
                    LaunchConfiguration('world')
                ]),
                TextSubstitution(text=' -r -v -v1') # Flags like -r (run), -v (verbose)
                # Original commented flags: TextSubstitution(text=' -r -v -v1 --render-engine ogre --render-engine-gui-api-backend opengl')
            ],
            'on_exit_shutdown': 'true' # Shutdown nodes when Gazebo exits
        }.items()
    )

    # --- RViz Launch ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution([pkg_bme_gazebo_sensors, 'rviz', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # --- Robot Description Publisher ---
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': Command(['xacro', ' ', urdf_file_path]),
             'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    # --- Spawn Robot ---
    # Spawn the URDF model using the `/world/<world_name>/create` service
    # Note: The world name in the service call is implicitly the one loaded by Gazebo.
    # The `-topic "robot_description"` tells it to get the model from the topic published by robot_state_publisher
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "rw_bot",  # Name of the model in Gazebo
            "-topic", "robot_description", # Use the robot_description topic
            "-x", "100.72",
            "-y", "5.53",
            "-z", "0.31",
            "-Y", "-2.90"
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # --- Bridges ---
    # Node to bridge common topics
    # gz_bridge_node = Node(
    #     package="ros_gz_bridge",
    #     executable="parameter_bridge",
    #     arguments=[
    #         # Clock
    #         "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    #         # Control & Odometry
    #         "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
    #         "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
    #         # Joints & TF (consider if Gazebo publishes TF directly or if RSP is sufficient)
    #         "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
    #         #"/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V", # Usually handled by RSP + Gazebo ground truth odom
    #         # Sensors
    #         "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
    #         "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
    #         "/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
    #         "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
    #         "/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
    #         "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
    #         "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
    #         "/camera2/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
    #     ],
    #     output="screen",
    #     parameters=[
    #         {'use_sim_time': LaunchConfiguration('use_sim_time')},
    #     ]
    # )

    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args', '-p',
            f'config_file:={gz_bridge_params_path}'
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # Node to bridge camera image using image_transport
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image"], # Gz input topic
        output="screen",
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        # remappings=[
        #     ("image", "camera/image"),       # ROS output image topic
        #     ("camera_info", "camera/camera_info")  # ROS output CameraInfo topic
        # ]
    )

    # Relay node to republish /camera/camera_info to enable image_transport pairing
    # Needed because image_transport subscribers often expect CameraInfo on image_topic/camera_info
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['camera/camera_info', 'camera/image/camera_info'], # Relay from bridged topic to image_transport expected topic
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # --- Localization ---
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_bme_gazebo_sensors, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
        # Add remappings if your EKF config expects topics different from defaults (e.g., /odom, /imu)
        # remappings=[('odometry/filtered', '/odom')]
    )

    # --- Trajectory Servers (Keep both or choose one based on needs) ---
    trajectory_node = Node( # Assumes service-based?
        package='trajectory_server',
        executable='trajectory_server',
        name='trajectory_server',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}] # Add use_sim_time if needed
    )

    trajectory_odom_topic_node = Node( # Topic-based
        package='trajectory_server',
        executable='trajectory_server_topic_based',
        name='trajectory_server_odom_topic',
        parameters=[
            {'trajectory_topic': 'trajectory_raw'},
            {'odometry_topic': 'odom'}, # Make sure this matches your EKF output or bridged odom topic
            {'use_sim_time': LaunchConfiguration('use_sim_time')} # Add use_sim_time
            ]
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_localization_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': localization_params_path,
                'map': map_file_path,
        }.items()
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_navigation_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': navigation_params_path,
        }.items()
    )

    delayed_nav2_launch = RegisterEventHandler(
    event_handler=OnProcessExit(
        target_action=spawn_urdf_node,
        on_exit=[localization_launch, navigation_launch]
    )
)

    # --- Assemble Launch Description ---
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(set_gazebo_path_action)
    ld.add_action(sim_time_arg)
    ld.add_action(rviz_launch_arg)
    ld.add_action(rviz_config_arg)
    ld.add_action(world_arg)
    ld.add_action(model_arg)
    # ld.add_action(x_arg)
    # ld.add_action(y_arg)
    # ld.add_action(yaw_arg)

    # ld.add_action(nav2_localization_launch_path)
    # ld.add_action(nav2_navigation_launch_path)
    # ld.add_action(localization_params_path)
    # ld.add_action(navigation_params_path)
    # ld.add_action(map_file_path)

    # Add actions
    ld.add_action(gazebo_launch)          # Launch Gazebo
    ld.add_action(robot_state_publisher_node) # Publish robot description
    ld.add_action(spawn_urdf_node)        # Spawn robot in Gazebo (needs robot_description)
    ld.add_action(gz_bridge_node)         # Start bridges
    ld.add_action(gz_image_bridge_node)   # Start image bridge
    ld.add_action(relay_camera_info_node) # Start camera_info relay
    ld.add_action(ekf_node)               # Start EKF
    ld.add_action(rviz_node)              # Start RViz (conditionally)
    ld.add_action(trajectory_node)        # Start trajectory server (service)
    ld.add_action(trajectory_odom_topic_node) # Start trajectory server (topic)
    #ld.add_action(localization_launch)
    #ld.add_action(navigation_launch)
    ld.add_action(delayed_nav2_launch)

    return ld