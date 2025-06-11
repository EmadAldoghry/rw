import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # --- Get Package Share Directories ---
    pkg_rw_dir = get_package_share_directory('rw')

    # ===================================================================================
    # === LAUNCH ARGUMENTS DEFINITION
    # ===================================================================================

    # --- Common Arguments ---
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )
    log_level_arg = DeclareLaunchArgument(
        'log_level', default_value='info',
        description='Logging level for all nodes (debug, info, warn, error, fatal)'
    )

    # --- Waypoint Server Arguments ---
    waypoints_yaml_path_arg = DeclareLaunchArgument(
        'waypoints_yaml_path',
        default_value=os.path.join(pkg_rw_dir, 'config', 'nav_waypoints.yaml'),
        description='Full path to the waypoints YAML file'
    )

    # --- Proximity Monitor Arguments ---
    correction_activation_distance_arg = DeclareLaunchArgument(
        'correction_activation_distance', default_value='7.0',
        description='Distance to global waypoint to activate local correction'
    )
    robot_base_frame_arg = DeclareLaunchArgument(
        'robot_base_frame', default_value='base_footprint',
        description='Robot base frame for TF lookups'
    )
    global_frame_arg = DeclareLaunchArgument(
        'global_frame', default_value='map',
        description='Global frame for navigation and TF'
    )

    # --- Orchestrator (follow_waypoints) Arguments ---
    local_target_arrival_threshold_arg = DeclareLaunchArgument(
        'local_target_arrival_threshold', default_value='0.35',
        description='Threshold to consider a local corrected target as reached'
    )
    local_goal_update_threshold_arg = DeclareLaunchArgument(
        'local_goal_update_threshold', default_value='0.2',
        description='Minimum distance change for updating local NavToPose goal'
    )

    # --- Sensor Fusion Pipeline Arguments ---

    # Shared Arguments for the Pipeline
    img_w_arg = DeclareLaunchArgument('img_w', default_value='1920', description='Camera image width')
    img_h_arg = DeclareLaunchArgument('img_h', default_value='1200', description='Camera image height')
    hfov_arg = DeclareLaunchArgument('hfov', default_value='1.25', description='Camera horizontal field of view (radians)')
    camera_optical_frame_arg = DeclareLaunchArgument('camera_optical_frame', default_value='front_camera_link_optical', description='TF frame of the camera optical center')
    lidar_optical_frame_arg = DeclareLaunchArgument('lidar_optical_frame', default_value='front_lidar_link_optical', description='TF frame of the LiDAR sensor')

    # Arguments for Goal Calculator & Orchestrator
    segmentation_node_name_arg = DeclareLaunchArgument(
        'segmentation_node_name', default_value='goal_calculator_node',
        description='Name of the node hosting the activation service'
    )
    corrected_local_goal_topic_arg = DeclareLaunchArgument(
        'corrected_local_goal_topic', default_value='/corrected_local_goal',
        description='Topic for corrected local goals for Nav2'
    )
    
    # Arguments for Image Segmenter Node
    input_image_topic_arg = DeclareLaunchArgument('input_image_topic', default_value='/camera/image', description='Topic for the raw input image')
    roi_x_start_arg = DeclareLaunchArgument('roi_x_start', default_value='0', description='ROI start pixel on x-axis')
    roi_y_start_arg = DeclareLaunchArgument('roi_y_start', default_value='228', description='ROI start pixel on y-axis (19% of 1200)')
    roi_x_end_arg = DeclareLaunchArgument('roi_x_end', default_value='1920', description='ROI end pixel on x-axis')
    roi_y_end_arg = DeclareLaunchArgument('roi_y_end', default_value='912', description='ROI end pixel on y-axis (76% of 1200)')
    black_h_min_arg = DeclareLaunchArgument('black_h_min', default_value='0', description='HSV H-min for segmentation')
    black_s_min_arg = DeclareLaunchArgument('black_s_min', default_value='0', description='HSV S-min for segmentation')
    black_v_min_arg = DeclareLaunchArgument('black_v_min', default_value='0', description='HSV V-min for segmentation')
    black_h_max_arg = DeclareLaunchArgument('black_h_max', default_value='180', description='HSV H-max for segmentation')
    black_s_max_arg = DeclareLaunchArgument('black_s_max', default_value='255', description='HSV S-max for segmentation')
    black_v_max_arg = DeclareLaunchArgument('black_v_max', default_value='0', description='HSV V-max for segmentation')

    # Arguments for PointCloud Fuser Node
    input_pc_topic_arg = DeclareLaunchArgument('input_pc_topic', default_value='/scan_02/points', description='Topic for the raw input point cloud')
    static_transform_lookup_timeout_sec_arg = DeclareLaunchArgument('static_transform_lookup_timeout_sec', default_value='5.0', description='Timeout for getting static TF between sensors')

    # Arguments for Fusion Visualizer Node
    output_window_arg = DeclareLaunchArgument('output_window', default_value='Fused View', description='Name of the OpenCV window. Empty string to disable.')
    point_radius_viz_arg = DeclareLaunchArgument('point_radius_viz', default_value='2', description='Radius of points in the visualizer window')

    # ===================================================================================
    # === NODE DEFINITIONS
    # ===================================================================================

    # Node 1: Waypoint Server (Manages the mission plan)
    waypoint_server_node = Node(
        package='rw_py', executable='waypoint_server', name='waypoint_server_node', output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'waypoints_yaml_path': LaunchConfiguration('waypoints_yaml_path')}
        ]
    )

    # Node 2: Proximity Monitor (Calculates distance and triggers correction)
    proximity_monitor_node = Node(
        package='rw_py', executable='proximity_monitor', name='proximity_monitor_node', output='screen',
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
        package='rw_py', executable='waypoint_visualizer', name='waypoint_visualizer_node', output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Node 4: The Orchestrator (Refactored follow_waypoints.py)
    waypoint_follower_corrected_node = Node(
        package='rw_py', executable='follow_waypoints', name='waypoint_follower_corrected_node', output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'local_target_arrival_threshold': LaunchConfiguration('local_target_arrival_threshold')},
            {'local_goal_update_threshold': LaunchConfiguration('local_goal_update_threshold')},
            {'segmentation_node_name': LaunchConfiguration('segmentation_node_name')},
            {'corrected_local_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
        ]
    )

    # --- Node 5: The Refactored Sensor Fusion Pipeline ---

    # 5a. Image Segmenter: Finds the target in the 2D image
    image_segmenter_node = Node(
        package='rw_py', executable='image_segmenter_node', name='image_segmenter_node', output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_image_topic': LaunchConfiguration('input_image_topic')},
            {'output_mask_topic': '/target_mask'}, # Internal topic
            {'img_w': LaunchConfiguration('img_w')},
            {'img_h': LaunchConfiguration('img_h')},
            {'roi_x_start': LaunchConfiguration('roi_x_start')},
            {'roi_y_start': LaunchConfiguration('roi_y_start')},
            {'roi_x_end': LaunchConfiguration('roi_x_end')},
            {'roi_y_end': LaunchConfiguration('roi_y_end')},
            {'black_h_min': LaunchConfiguration('black_h_min')},
            {'black_s_min': LaunchConfiguration('black_s_min')},
            {'black_v_min': LaunchConfiguration('black_v_min')},
            {'black_h_max': LaunchConfiguration('black_h_max')},
            {'black_s_max': LaunchConfiguration('black_s_max')},
            {'black_v_max': LaunchConfiguration('black_v_max')},
        ]
    )

    # 5b. PointCloud Fuser: Filters the point cloud based on the image mask
    pointcloud_fuser_node = Node(
        package='rw_py', executable='pointcloud_fuser_node', name='pointcloud_fuser_node', output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_pc_topic': LaunchConfiguration('input_pc_topic')},
            {'input_mask_topic': '/target_mask'}, # Internal topic
            {'output_pc_topic': '/target_points'}, # Internal topic
            {'img_w': LaunchConfiguration('img_w')},
            {'img_h': LaunchConfiguration('img_h')},
            {'hfov': LaunchConfiguration('hfov')},
            {'camera_optical_frame': LaunchConfiguration('camera_optical_frame')},
            {'lidar_optical_frame': LaunchConfiguration('lidar_optical_frame')},
            {'static_transform_lookup_timeout_sec': LaunchConfiguration('static_transform_lookup_timeout_sec')},
        ]
    )

    # 5c. Goal Calculator: Calculates the final goal and hosts the activation service
    goal_calculator_node = Node(
        package='rw_py', executable='goal_calculator_node',
        name=LaunchConfiguration('segmentation_node_name'), # Name matches what orchestrator expects for service call
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'input_pc_topic': '/target_points'}, # Internal topic
            {'output_corrected_goal_topic': LaunchConfiguration('corrected_local_goal_topic')},
            {'navigation_frame': LaunchConfiguration('global_frame')},
        ]
    )

    # 5d. Fusion Visualizer: A passive node for debugging the fusion pipeline
    fusion_visualizer_node = Node(
        package='rw_py', executable='fusion_visualizer_node', name='fusion_visualizer_node', output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'output_window': LaunchConfiguration('output_window')},
            {'raw_image_topic': LaunchConfiguration('input_image_topic')},
            {'mask_topic': '/target_mask'}, # Internal topic
            {'raw_pc_topic': LaunchConfiguration('input_pc_topic')},
            {'target_pc_topic': '/target_points'}, # Internal topic
            {'img_w': LaunchConfiguration('img_w')},
            {'img_h': LaunchConfiguration('img_h')},
            {'hfov': LaunchConfiguration('hfov')},
            {'camera_optical_frame': LaunchConfiguration('camera_optical_frame')},
            {'lidar_optical_frame': LaunchConfiguration('lidar_optical_frame')},
            {'point_radius_viz': LaunchConfiguration('point_radius_viz')},
        ]
    )

    # ===================================================================================
    # === LAUNCH DESCRIPTION ASSEMBLY
    # ===================================================================================
    return LaunchDescription([
        # --- Add all launch arguments ---
        use_sim_time_arg, log_level_arg,
        waypoints_yaml_path_arg,
        correction_activation_distance_arg, robot_base_frame_arg, global_frame_arg,
        local_target_arrival_threshold_arg, local_goal_update_threshold_arg,
        # Pipeline Arguments
        img_w_arg, img_h_arg, hfov_arg, camera_optical_frame_arg, lidar_optical_frame_arg,
        segmentation_node_name_arg, corrected_local_goal_topic_arg,
        # Segmenter Arguments
        input_image_topic_arg, roi_x_start_arg, roi_y_start_arg, roi_x_end_arg, roi_y_end_arg,
        black_h_min_arg, black_s_min_arg, black_v_min_arg, black_h_max_arg, black_s_max_arg, black_v_max_arg,
        # Fuser Arguments
        input_pc_topic_arg, static_transform_lookup_timeout_sec_arg,
        # Visualizer Arguments
        output_window_arg, point_radius_viz_arg,

        # --- Log info about the launch setup ---
        LogInfo(msg="--- Launching MODULAR Waypoint Navigation System ---"),
        LogInfo(msg=["- Waypoint Server: Reading from ", LaunchConfiguration('waypoints_yaml_path')]),
        LogInfo(msg=["- Proximity Monitor: Activating at ", LaunchConfiguration('correction_activation_distance'), "m"]),
        LogInfo(msg="- Orchestrator: Managing the mission flow."),
        LogInfo(msg="- Fusion Pipeline: Segmenter -> Fuser -> Goal Calculator"),
        LogInfo(msg="- Visualizer: Publishing markers to RViz and fused view to OpenCV."),

        # --- Add all the nodes to be launched ---
        waypoint_server_node,
        proximity_monitor_node,
        waypoint_visualizer_node,
        waypoint_follower_corrected_node,
        
        # The new sensor fusion pipeline
        image_segmenter_node,
        pointcloud_fuser_node,
        goal_calculator_node,
        fusion_visualizer_node,
    ])