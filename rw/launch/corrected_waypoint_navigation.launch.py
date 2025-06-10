import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # Get package share directory for rw_py (where your Python nodes are)
    pkg_rw_py_dir = get_package_share_directory('rw_py')
    # Get package share directory for rw (where waypoints.yaml might be, if not overridden)
    pkg_rw_dir = get_package_share_directory('rw')

    # --- Declare Launch Arguments (optional, but good practice for configurability) ---
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true', # Set to 'false' for a real robot
        description='Use simulation (Gazebo) clock if true'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal)'
    )

    # Waypoints file path argument
    waypoints_yaml_path_arg = DeclareLaunchArgument(
        'waypoints_yaml_path',
        default_value=PathJoinSubstitution([pkg_rw_dir, 'config', 'nav_waypoints.yaml']),
        description='Full path to the waypoints YAML file'
    )

    # Segmentation Node Parameters
    segmentation_node_name_arg = DeclareLaunchArgument(
        'segmentation_node_name',
        default_value='roi_lidar_fusion_node_activated', # Matches class name if not overridden
        description='Name of the ROI Lidar Fusion node'
    )
    corrected_goal_topic_arg = DeclareLaunchArgument(
        'corrected_local_goal_topic',
        default_value='/corrected_local_goal', # Global topic name
        description='Topic for corrected local goals from segmentation node'
    )
    # Frame parameters for segmentation node
    camera_optical_frame_arg = DeclareLaunchArgument(
        'camera_optical_frame',
        default_value='front_camera_link_optical', # From your URDF
        description='TF frame of the camera optical center'
    )
    lidar_optical_frame_arg = DeclareLaunchArgument(
        'lidar_optical_frame',
        default_value='front_lidar_link_optical', # From your URDF ('front_lidar_link_optical' seems more appropriate than 'laser_frame' if it has an optical joint)
                                                # Check your URDF rw_bot.urdf: <joint name="lidar_optical_joint" type="fixed"><parent link="laser_frame"/><child link="lidar_link_optical"/>
        description='TF frame of the Lidar sensor'
    )
    navigation_frame_arg = DeclareLaunchArgument( # For RLFN corrected goal
        'navigation_frame_rlfn',
        default_value='map',
        description='Navigation frame for corrected goals published by RLFN'
    )

    # Waypoint Follower Parameters
    correction_activation_distance_arg = DeclareLaunchArgument(
        'correction_activation_distance',
        default_value='7.0', # Meters
        description='Distance to global waypoint to activate correction'
    )
    local_target_arrival_threshold_arg = DeclareLaunchArgument(
        'local_target_arrival_threshold',
        default_value='0.35', # Meters
        description='Threshold to consider local target reached'
    )

    local_goal_update_threshold_arg = DeclareLaunchArgument(
        'local_goal_update_threshold',
        default_value='0.2', # Meters, how much the local goal must change to trigger a new NavToPose
        description='Minimum distance change for updating local NavToPose goal'
    )

    robot_base_frame_arg = DeclareLaunchArgument(
        'robot_base_frame',
        default_value='base_link', # Often base_footprint for Nav2, but base_link if that's your robot's root moving frame in TF
        description='Robot base frame for TF lookups in WaypointFollower'
    )
    global_frame_arg = DeclareLaunchArgument( # For WLF
        'global_frame_wlf',
        default_value='map',
        description='Global frame for navigation and waypoints in WaypointFollower'
    )


    # --- Node Definitions ---

    # ROI Lidar Fusion Node (Segmentation and Target Localization)
    # Ensure the executable name matches what's in your rw_py/setup.py
    roi_lidar_fusion_node = Node(
        package='rw_py',
        executable='fusion_segmentation_node', # Name from setup.py entry_points
        name=LaunchConfiguration('segmentation_node_name'), # Node name in ROS graph
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_image_topic': '/camera/image'}, # From your spawner.launch.py it seems /camera/image_raw is bridged as /camera/image
            {'input_pc_topic': '/scan_02/points'},    # From your URDF and gz_bridge.yaml
            {'output_corrected_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
            {'navigation_frame': LaunchConfiguration('navigation_frame_rlfn')}, # For the goal it publishes
            {'camera_optical_frame': LaunchConfiguration('camera_optical_frame')},
            {'lidar_optical_frame': LaunchConfiguration('lidar_optical_frame')},
            {'output_window': 'Fused View'}, # Set to "" to disable CV window if running headless
            {'img_w': 1920}, # Match your URDF/gz_bridge settings if specific
            {'img_h': 1200}, # Match your URDF/gz_bridge settings if specific
            {'hfov': 1.25},  # From your URDF
            # Add other parameters for ROILidarFusionNode as needed:
            # e.g., black_h_min, black_s_min, etc.
            {'enable_black_segmentation': True},
            {'black_h_max': 180},
            {'black_s_max': 255},
            {'black_v_max': 0}, # Example, tune this for your black surface
            {'min_dist_colorize': 1.0},
            {'max_dist_colorize': 10.0},
            {'point_display_mode': 2}, # Show all projected points for debugging
        ]
    )

    # Waypoint Follower Corrected Node (Orchestrator)
    # Ensure the executable name matches what's in your rw_py/setup.py
    waypoint_follower_corrected_node = Node(
        package='rw_py',
        executable='follow_waypoints', # Name from setup.py entry_points
        name='waypoint_follower_corrected_node',   # Node name in ROS graph
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'waypoints_yaml_path': LaunchConfiguration('waypoints_yaml_path')},
            {'correction_activation_distance': LaunchConfiguration('correction_activation_distance')},
            {'local_target_arrival_threshold': LaunchConfiguration('local_target_arrival_threshold')},
            {'local_goal_update_threshold': LaunchConfiguration('local_goal_update_threshold')}, # Add this
            {'segmentation_node_name': LaunchConfiguration('segmentation_node_name')}, 
            {'corrected_local_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
            {'robot_base_frame': LaunchConfiguration('robot_base_frame')},
            {'global_frame': LaunchConfiguration('global_frame_wlf')}
        ]
    )

    return LaunchDescription([
        # Launch Arguments
        use_sim_time_arg,
        log_level_arg,
        waypoints_yaml_path_arg,
        segmentation_node_name_arg,
        corrected_goal_topic_arg,
        camera_optical_frame_arg,
        lidar_optical_frame_arg,
        navigation_frame_arg,
        correction_activation_distance_arg,
        local_target_arrival_threshold_arg,
        local_goal_update_threshold_arg, # Add this
        robot_base_frame_arg,
        global_frame_arg,

        # Log info about parameters being used
        LogInfo(msg=["Launching corrected waypoint navigation with parameters:"]),
        LogInfo(msg=["  use_sim_time: ", LaunchConfiguration('use_sim_time')]),
        LogInfo(msg=["  waypoints_yaml_path: ", LaunchConfiguration('waypoints_yaml_path')]),
        LogInfo(msg=["  corrected_local_goal_topic: ", LaunchConfiguration('corrected_local_goal_topic')]),
        LogInfo(msg=["  segmentation_node_name: ", LaunchConfiguration('segmentation_node_name')]),


        # Nodes
        roi_lidar_fusion_node,
        waypoint_follower_corrected_node,
    ])