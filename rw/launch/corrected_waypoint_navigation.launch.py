import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get package share directories
    pkg_rw_dir = get_package_share_directory('rw')
    pkg_rw_py_dir = get_package_share_directory('rw_py')

    # --- Declare Launch Arguments (for configuration flexibility) ---

    # Common arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level for all nodes (debug, info, warn, error, fatal)'
    )

    # Arguments for WaypointServer node
    waypoints_yaml_path_arg = DeclareLaunchArgument(
        'waypoints_yaml_path',
        default_value=PathJoinSubstitution([pkg_rw_dir, 'config', 'nav_waypoints.yaml']),
        description='Full path to the waypoints YAML file'
    )

    # Arguments for ProximityMonitor node
    correction_activation_distance_arg = DeclareLaunchArgument(
        'correction_activation_distance',
        default_value='7.0',  # Meters
        description='Distance to global waypoint to activate local correction'
    )
    robot_base_frame_arg = DeclareLaunchArgument(
        'robot_base_frame',
        default_value='base_footprint', # Use 'base_footprint' as it's the standard for Nav2 TF
        description='Robot base frame for TF lookups in ProximityMonitor'
    )
    global_frame_arg = DeclareLaunchArgument(
        'global_frame',
        default_value='map',
        description='Global frame for navigation and TF lookups'
    )

    # Arguments for WaypointFollower (Orchestrator) node
    local_target_arrival_threshold_arg = DeclareLaunchArgument(
        'local_target_arrival_threshold',
        default_value='0.35',  # Meters
        description='Threshold to consider a local corrected target as reached'
    )
    local_goal_update_threshold_arg = DeclareLaunchArgument(
        'local_goal_update_threshold',
        default_value='0.2',  # Meters, how much the local goal must change to trigger a new NavToPose
        description='Minimum distance change for updating local NavToPose goal'
    )

    # Arguments for ROILidarFusionNode (The "Eyes")
    segmentation_node_name_arg = DeclareLaunchArgument(
        'segmentation_node_name',
        default_value='roi_lidar_fusion_node',
        description='Name of the ROI Lidar Fusion node'
    )
    corrected_goal_topic_arg = DeclareLaunchArgument(
        'corrected_local_goal_topic',
        default_value='/corrected_local_goal',  # Global topic name
        description='Topic for corrected local goals from the segmentation node'
    )
    camera_optical_frame_arg = DeclareLaunchArgument(
        'camera_optical_frame',
        default_value='front_camera_link_optical',  # From your URDF
        description='TF frame of the camera optical center'
    )
    lidar_optical_frame_arg = DeclareLaunchArgument(
        'lidar_optical_frame',
        default_value='front_lidar_link_optical',  # From your URDF
        description='TF frame of the Lidar sensor used for correction'
    )


    # --- Node Definitions ---

    # Node 1: Waypoint Server (Manages the mission plan)
    waypoint_server_node = Node(
        package='rw_py',
        executable='waypoint_server',
        name='waypoint_server_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'waypoints_yaml_path': LaunchConfiguration('waypoints_yaml_path')}
        ]
    )

    # Node 2: Proximity Monitor (Calculates distance and triggers correction)
    proximity_monitor_node = Node(
        package='rw_py',
        executable='proximity_monitor',
        name='proximity_monitor_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_base_frame': LaunchConfiguration('robot_base_frame')},
            {'global_frame': LaunchConfiguration('global_frame')},
            {'correction_activation_distance': LaunchConfiguration('correction_activation_distance')}
        ]
    )

    # Node 3: Waypoint Visualizer (Handles all RViz markers)
    waypoint_visualizer_node = Node(
        package='rw_py',
        executable='waypoint_visualizer',
        name='waypoint_visualizer_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Node 4: The Orchestrator (Refactored follow_waypoints.py)
    waypoint_follower_corrected_node = Node(
        package='rw_py',
        executable='follow_waypoints',
        name='waypoint_follower_corrected_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'local_target_arrival_threshold': LaunchConfiguration('local_target_arrival_threshold')},
            {'local_goal_update_threshold': LaunchConfiguration('local_goal_update_threshold')},
            {'segmentation_node_name': LaunchConfiguration('segmentation_node_name')},
            {'corrected_local_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
        ]
    )

    # Node 5: The Fusion/Sensor Node (The "Eyes")
    roi_lidar_fusion_node = Node(
        package='rw_py',
        executable='fusion_segmentation_node',
        name=LaunchConfiguration('segmentation_node_name'),
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_image_topic': '/camera/image'},
            {'input_pc_topic': '/scan_02/points'},
            {'output_corrected_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
            {'navigation_frame': LaunchConfiguration('global_frame')}, # The goal it publishes is in the global frame
            {'camera_optical_frame': LaunchConfiguration('camera_optical_frame')},
            {'lidar_optical_frame': LaunchConfiguration('lidar_optical_frame')},
            {'output_window': 'Fused View'}, # Set to "" to disable CV window
            # Other fusion node parameters from original file
            {'img_w': 1920},
            {'img_h': 1200},
            {'hfov': 1.25},
            {'enable_black_segmentation': True},
            {'black_v_max': 0}, # Original was 0, but 50 is more practical for segmentation
            {'min_dist_colorize': 1.0},
            {'max_dist_colorize': 10.0},
            {'point_display_mode': 2},
        ]
    )

    return LaunchDescription([
        # Add all launch arguments
        use_sim_time_arg,
        log_level_arg,
        waypoints_yaml_path_arg,
        correction_activation_distance_arg,
        robot_base_frame_arg,
        global_frame_arg,
        local_target_arrival_threshold_arg,
        local_goal_update_threshold_arg,
        segmentation_node_name_arg,
        corrected_goal_topic_arg,
        camera_optical_frame_arg,
        lidar_optical_frame_arg,

        # Log info about the launch setup
        LogInfo(msg=["--- Launching Modular Waypoint Navigation System ---"]),
        LogInfo(msg=["- Waypoint Server: Reading from ", LaunchConfiguration('waypoints_yaml_path')]),
        LogInfo(msg=["- Proximity Monitor: Activating at ", LaunchConfiguration('correction_activation_distance'), "m"]),
        LogInfo(msg=["- Orchestrator: Managing the mission flow."]),
        LogInfo(msg=["- Fusion Node: Looking for targets."]),
        LogInfo(msg=["- Visualizer: Publishing markers to RViz."]),

        # Add all the nodes to be launched
        waypoint_server_node,
        proximity_monitor_node,
        waypoint_visualizer_node,
        waypoint_follower_corrected_node,
        roi_lidar_fusion_node,
    ])