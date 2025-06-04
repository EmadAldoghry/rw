#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import PoseStamped as GeometryPoseStamped, Point, Quaternion
from std_msgs.msg import Header, ColorRGBA
from nav2_msgs.action import NavigateToPose
import yaml
from pathlib import Path
import sys
import os
from ament_index_python.packages import get_package_share_directory
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException 
import math
from enum import Enum, auto
import traceback 
from action_msgs.msg import GoalStatus
import time
from visualization_msgs.msg import Marker, MarkerArray

from std_srvs.srv import SetBool
import copy # IMPORTANT: Added for deepcopy

class CrackNavState(Enum):
    IDLE = auto()
    LOADING_WAYPOINTS = auto()
    PROCESSING_NEXT_CRACK_DEFINITION = auto()
    NAVIGATING_TO_PREDEFINED_START_AREA = auto()
    DETECTING_CRACK_INITIAL_POINT = auto()
    NAVIGATING_TO_FIXED_CRACK_START = auto()
    FOLLOWING_CRACK_BODY = auto()
    CRACK_NAVIGATION_CONCLUDED = auto()
    ALL_CRACKS_COMPLETED = auto()
    MISSION_FAILED = auto()

class WaypointFollowerCorrected(Node):
    def __init__(self):
        super().__init__('waypoint_follower_corrected_node')

        default_yaml_path = self._get_default_yaml_path('rw', 'config', 'nav_waypoints.yaml')
        self.declare_parameter('waypoints_yaml_path', default_yaml_path)
        self.declare_parameter('crack_start_detection_proximity', 7.0)
        self.declare_parameter('fixed_start_arrival_threshold', 0.35)
        self.declare_parameter('crack_segment_arrival_threshold', 0.5)
        self.declare_parameter('crack_segment_update_threshold', 0.25)
        self.declare_parameter('segmentation_node_name', 'roi_lidar_fusion_node_activated') 
        self.declare_parameter('corrected_local_goal_topic', '/corrected_local_goal')
        self.declare_parameter('robot_base_frame', 'base_link') # Check if base_footprint is more appropriate for your Nav2 setup
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('crack_detection_timeout', 10.0)
        self.declare_parameter('crack_segment_timeout', 3.0) #Timeout for waiting for next segment
        self.declare_parameter('nav_goal_send_retry_delay', 1.0) # Delay before retrying a failed NavToPose send
        self.declare_parameter('tf_initial_wait_attempts', 30) # Increased TF wait attempts
        self.declare_parameter('tf_initial_wait_interval_sec', 0.5) # Interval for TF wait

        self.yaml_path_ = self.get_parameter('waypoints_yaml_path').value
        self.crack_start_detection_proximity_ = self.get_parameter('crack_start_detection_proximity').value
        self.fixed_start_arrival_thresh_ = self.get_parameter('fixed_start_arrival_threshold').value
        self.crack_segment_arrival_thresh_ = self.get_parameter('crack_segment_arrival_threshold').value
        self.crack_segment_update_thresh_ = self.get_parameter('crack_segment_update_threshold').value
        segmentation_node_name = self.get_parameter('segmentation_node_name').value
        self.corrected_goal_topic_ = self.get_parameter('corrected_local_goal_topic').value
        self.robot_base_frame_ = self.get_parameter('robot_base_frame').value
        self.global_frame_ = self.get_parameter('global_frame').value
        self.crack_detection_timeout_sec_ = self.get_parameter('crack_detection_timeout').value
        self.crack_segment_timeout_sec_ = self.get_parameter('crack_segment_timeout').value
        self.nav_goal_send_retry_delay_ = self.get_parameter('nav_goal_send_retry_delay').value
        self.tf_initial_wait_attempts_ = self.get_parameter('tf_initial_wait_attempts').value
        self.tf_initial_wait_interval_sec_ = self.get_parameter('tf_initial_wait_interval_sec').value
        self.activation_srv_name_ = f"/{segmentation_node_name}/activate_segmentation"

        self.all_waypoints_ = []; self.current_waypoint_pair_idx_ = -2
        self.current_target_predefined_start_ = None; self.fixed_crack_start_pose_ = None
        self.last_crack_segment_goal_ = None; self.state_ = CrackNavState.IDLE
        self.nav_goal_handle_ = None; self.lidar_correction_sub_ = None
        self.detection_or_segment_timeout_timer_ = None

        self.tf_buffer_ = tf2_ros.Buffer(cache_time=RclpyDuration(seconds=10.0)) # Give buffer more time
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self, spin_thread=True) # Ensure listener spins
        
        tf_ready = False
        self.get_logger().info(f"Waiting for TF transform from '{self.global_frame_}' to '{self.robot_base_frame_}'...")
        for i in range(self.tf_initial_wait_attempts_): 
            try:
                # Use a longer timeout for the actual check within the loop
                if self.tf_buffer_.can_transform(self.global_frame_, self.robot_base_frame_, RclpyTime(), timeout=RclpyDuration(seconds=1.0)):
                    tf_ready = True
                    self.get_logger().info(f"TF tree ready ({self.global_frame_} -> {self.robot_base_frame_}).")
                    break
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF exception during check: {e}")
            self.get_logger().info(f"Waiting for TF... try {i + 1}/{self.tf_initial_wait_attempts_}"); 
            time.sleep(self.tf_initial_wait_interval_sec_)
        
        if not tf_ready: 
            self.get_logger().error(f"TF tree from '{self.global_frame_}' to '{self.robot_base_frame_}' not available after {self.tf_initial_wait_attempts_} attempts. Functionality impaired.")
            # Decide if you want to fail hard:
            # self.state_ = CrackNavState.MISSION_FAILED 
            # self._process_state_actions() 
            # return


        self._nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.segmentation_activation_client_ = self.create_client(SetBool, self.activation_srv_name_)
        self.marker_publisher_ = self.create_publisher(MarkerArray, '~/debug_waypoints_markers', 10)
        self.marker_publish_timer_ = self.create_timer(1.0, self.publish_state_markers)

        self.get_logger().info("Crack Following Navigator initialized.")
        if self.state_ != CrackNavState.MISSION_FAILED: # Only proceed if TF didn't cause an early fail
            self._change_state_and_process(CrackNavState.LOADING_WAYPOINTS)

    def _get_default_yaml_path(self, package_name='rw', config_dir='config', file_name='nav_waypoints.yaml'):
        try: share_dir = get_package_share_directory(package_name); return os.path.join(share_dir, config_dir, file_name)
        except Exception: return ""

    def _change_state_and_process(self, new_state: CrackNavState):
        if self.state_ == new_state and new_state not in [CrackNavState.LOADING_WAYPOINTS, CrackNavState.PROCESSING_NEXT_CRACK_DEFINITION]:
             self.get_logger().debug(f"Already in state {new_state.name}. No transition needed.")
             return
        self.get_logger().info(f"STATE: {self.state_.name} -> {new_state.name}")
        old_state = self.state_; self.state_ = new_state
        if old_state in [CrackNavState.DETECTING_CRACK_INITIAL_POINT, CrackNavState.FOLLOWING_CRACK_BODY]:
            self._destroy_corrected_goal_subscriber(); self._destroy_timeout_timer()
        self._process_state_actions()

    def _process_state_actions(self):
        self.get_logger().info(f"--- DEBUG Z: _process_state_actions ENTERED for state {self.state_.name} ---")
        if self.all_waypoints_:
            for idx_debug, wp_debug in enumerate(self.all_waypoints_):
                self.get_logger().info(f"  Pre-Action WP {idx_debug}: Z = {wp_debug.pose.position.z:.3f}")
        else:
            self.get_logger().info("  Pre-Action: self.all_waypoints_ is empty.")
        self.get_logger().info(f"--- END DEBUG Z PRE-ACTION ---")

        self.get_logger().info(f"Processing actions for state: {self.state_.name}")
        if self.state_ == CrackNavState.LOADING_WAYPOINTS:
            if self.load_waypoints_from_yaml(Path(self.yaml_path_)):
                if self.all_waypoints_ and len(self.all_waypoints_) > 0 and len(self.all_waypoints_) % 2 == 0 : # Check for non-empty and even
                    self.current_waypoint_pair_idx_ = -2
                    self._change_state_and_process(CrackNavState.PROCESSING_NEXT_CRACK_DEFINITION)
                elif not self.all_waypoints_: 
                    self.get_logger().info("No waypoints loaded. Mission complete.")
                    self._change_state_and_process(CrackNavState.ALL_CRACKS_COMPLETED)
                else: 
                    self.get_logger().error(f"Waypoints must be in pairs (start, end) and non-empty. Found {len(self.all_waypoints_)} waypoints."); 
                    self._change_state_and_process(CrackNavState.MISSION_FAILED)
            else: self._change_state_and_process(CrackNavState.MISSION_FAILED)
        
        elif self.state_ == CrackNavState.PROCESSING_NEXT_CRACK_DEFINITION:
            self.current_waypoint_pair_idx_ += 2
            if self.current_waypoint_pair_idx_ < len(self.all_waypoints_):
                idx_to_access = self.current_waypoint_pair_idx_
                original_wp_in_list = self.all_waypoints_[idx_to_access]
                self.get_logger().info(
                    f"DEBUG Z PRE-ASSIGN: Accessing all_waypoints_[{idx_to_access}]. "
                    f"Original Z value from list: {original_wp_in_list.pose.position.z:.4f}"
                )
                new_ps = GeometryPoseStamped()
                new_ps.header.stamp.sec = original_wp_in_list.header.stamp.sec
                new_ps.header.stamp.nanosec = original_wp_in_list.header.stamp.nanosec
                new_ps.header.frame_id = original_wp_in_list.header.frame_id
                new_ps.pose.position.x = original_wp_in_list.pose.position.x
                new_ps.pose.position.y = original_wp_in_list.pose.position.y
                new_ps.pose.position.z = original_wp_in_list.pose.position.z 
                self.get_logger().info(
                    f"DEBUG Z DURING MANUAL COPY: original_wp_in_list.pose.position.z = {original_wp_in_list.pose.position.z:.4f}, "
                    f"Assigned to new_ps.pose.position.z = {new_ps.pose.position.z:.4f}"
                )
                new_ps.pose.orientation.x = original_wp_in_list.pose.orientation.x
                new_ps.pose.orientation.y = original_wp_in_list.pose.orientation.y
                new_ps.pose.orientation.z = original_wp_in_list.pose.orientation.z
                new_ps.pose.orientation.w = original_wp_in_list.pose.orientation.w
                self.current_target_predefined_start_ = new_ps
                
                self.get_logger().info(
                    f"DEBUG Z POST-ASSIGN (current_target_predefined_start_ manually copied): "
                    f"X={self.current_target_predefined_start_.pose.position.x:.2f}, "
                    f"Y={self.current_target_predefined_start_.pose.position.y:.2f}, "
                    f"Z={self.current_target_predefined_start_.pose.position.z:.4f}"
                )
                self.get_logger().info(f"Next target: Predefined crack start area (index {self.current_waypoint_pair_idx_}).")
                self.fixed_crack_start_pose_ = None; self.last_crack_segment_goal_ = None
                self._send_nav_to_pose_goal(self.current_target_predefined_start_)
                self._change_state_and_process(CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA)
            else: self._change_state_and_process(CrackNavState.ALL_CRACKS_COMPLETED)
        
        elif self.state_ == CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA:
            self.get_logger().debug("In NAVIGATING_TO_PREDEFINED_START_AREA, awaiting NavToPose feedback/result.")

        elif self.state_ == CrackNavState.DETECTING_CRACK_INITIAL_POINT:
            self._activate_segmentation_node(True); self._create_corrected_goal_subscriber()
            self._create_timeout_timer(self.crack_detection_timeout_sec_, self.initial_crack_detection_timeout_cb)
        
        elif self.state_ == CrackNavState.NAVIGATING_TO_FIXED_CRACK_START:
            self.get_logger().debug("In NAVIGATING_TO_FIXED_CRACK_START, awaiting NavToPose result.")

        elif self.state_ == CrackNavState.FOLLOWING_CRACK_BODY:
            self.get_logger().info("Transitioned to FOLLOWING_CRACK_BODY. Activating segmentation and waiting for first segment goal.")
            self._activate_segmentation_node(True); self._create_corrected_goal_subscriber()
            self._create_timeout_timer(self.crack_segment_timeout_sec_, self.crack_segment_timeout_cb)
        
        elif self.state_ == CrackNavState.CRACK_NAVIGATION_CONCLUDED:
            self.get_logger().info(f"Crack navigation concluded for predefined start index {self.current_waypoint_pair_idx_}.")
            self._activate_segmentation_node(False)
            self._change_state_and_process(CrackNavState.PROCESSING_NEXT_CRACK_DEFINITION)
        
        elif self.state_ == CrackNavState.ALL_CRACKS_COMPLETED:
            self.get_logger().info("All cracks completed!"); self._activate_segmentation_node(False)
            # Consider shutting down or going to an idle monitoring state.
            # rclpy.shutdown() 
        
        elif self.state_ == CrackNavState.MISSION_FAILED:
            self.get_logger().error("Mission failed."); self._activate_segmentation_node(False); self._cancel_current_navigation_goal()
            # rclpy.shutdown()
    
    def load_waypoints_from_yaml(self, yaml_file_path: Path) -> bool:
        try:
            with open(yaml_file_path, 'r') as file: yaml_data = yaml.safe_load(file)
            if not yaml_data or 'poses' not in yaml_data: 
                self.get_logger().error(f"YAML '{yaml_file_path}' empty or no 'poses' key.")
                return False
            loaded_waypoints = []
            for i, pose_entry in enumerate(yaml_data['poses']):
                ps_msg = GeometryPoseStamped()
                ps_msg.header.frame_id = self.global_frame_ 
                ps_msg.header.stamp = self.get_clock().now().to_msg()
                pose_block = pose_entry.get('pose', {})
                pos_data = pose_block.get('position', {}); orient_data = pose_block.get('orientation', {})
                ps_msg.pose.position.x = float(pos_data.get('x',0.0))
                ps_msg.pose.position.y = float(pos_data.get('y',0.0))
                ps_msg.pose.position.z = float(pos_data.get('z',0.0)) 
                ps_msg.pose.orientation.x=float(orient_data.get('x',0.0))
                ps_msg.pose.orientation.y=float(orient_data.get('y',0.0))
                ps_msg.pose.orientation.z=float(orient_data.get('z',0.0))
                ps_msg.pose.orientation.w=float(orient_data.get('w',1.0))
                self.get_logger().debug(f"Loaded WP {i} from YAML: X={ps_msg.pose.position.x:.2f}, Y={ps_msg.pose.position.y:.2f}, Z={ps_msg.pose.position.z:.3f} in frame '{ps_msg.header.frame_id}'")
                loaded_waypoints.append(ps_msg)
            self.all_waypoints_ = loaded_waypoints
            self.get_logger().info(f"--- DEBUG Z POST-LOAD (from load_waypoints_from_yaml) ---")
            for idx, wp in enumerate(self.all_waypoints_):
                self.get_logger().info(f"  WP {idx} in self.all_waypoints_: Z = {wp.pose.position.z:.3f}")
            self.get_logger().info(f"--- END DEBUG Z POST-LOAD ---")
            return True
        except Exception as e: self.get_logger().error(f"Error loading YAML '{yaml_file_path}': {e}\n{traceback.format_exc()}"); return False

    def _create_corrected_goal_subscriber(self):
        if self.lidar_correction_sub_ is None:
            self.get_logger().info(f"Creating corrected goal subscriber on '{self.corrected_goal_topic_}'")
            _qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.lidar_correction_sub_ = self.create_subscription(GeometryPoseStamped, self.corrected_goal_topic_, self.corrected_goal_cb, _qos)
    
    def _destroy_corrected_goal_subscriber(self):
        if self.lidar_correction_sub_ is not None: 
            self.get_logger().info("Destroying corrected goal subscriber.")
            self.destroy_subscription(self.lidar_correction_sub_); self.lidar_correction_sub_ = None
    
    def _create_timeout_timer(self, period_sec: float, callback_func):
        self._destroy_timeout_timer()
        self.get_logger().info(f"Starting timeout timer for {period_sec}s for {callback_func.__name__}.")
        self.detection_or_segment_timeout_timer_ = self.create_timer(period_sec, callback_func)

    def _destroy_timeout_timer(self):
        if self.detection_or_segment_timeout_timer_ is not None: 
            if not self.detection_or_segment_timeout_timer_.is_canceled():
                self.get_logger().info("Cancelling active timeout timer.")
                self.detection_or_segment_timeout_timer_.cancel()
            self.destroy_timer(self.detection_or_segment_timeout_timer_) 
            self.detection_or_segment_timeout_timer_ = None
            self.get_logger().info("Timeout timer destroyed/cleared.")


    def initial_crack_detection_timeout_cb(self):
        self.get_logger().warn("Timeout: No initial crack point detected via corrected_goal_cb.")
        self._destroy_timeout_timer() 
        if self.state_ == CrackNavState.DETECTING_CRACK_INITIAL_POINT: 
            self._activate_segmentation_node(False)
            self.get_logger().info("Skipping current crack due to detection timeout.")
            self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)

    def crack_segment_timeout_cb(self):
        self.get_logger().warn("Timeout: No new crack segment received while following.")
        self._destroy_timeout_timer()
        if self.state_ == CrackNavState.FOLLOWING_CRACK_BODY:
             self._activate_segmentation_node(False) 
             self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)

    def _activate_segmentation_node(self, activate: bool):
        if not self.segmentation_activation_client_.service_is_ready(): 
            self.get_logger().warn(f"Segmentation service '{self.activation_srv_name_}' not ready.")
            return False
        self.segmentation_activation_client_.call_async(SetBool.Request(data=activate))
        self.get_logger().info(f"Requested segmentation node {'activation' if activate else 'deactivation'}.")
        return True

    def _send_nav_to_pose_goal(self, target_pose: GeometryPoseStamped):
        self.get_logger().info(
            f"DEBUG Z: _send_nav_to_pose_goal received target_pose: "
            f"X={target_pose.pose.position.x:.2f}, Y={target_pose.pose.position.y:.2f}, Z={target_pose.pose.position.z:.3f}, "
            f"Frame='{target_pose.header.frame_id}'"
        )
        
        active_goal_cancelled_or_none = True
        if self.nav_goal_handle_:
            # Check if the handle is still valid and the goal is in an active state
            # GoalStatus: 1=ACCEPTED, 2=EXECUTING
            # These might vary slightly across ROS 2 versions if not careful
            is_active_custom_check = False
            if hasattr(self.nav_goal_handle_, 'status'): # Foxy and later should have this
                 is_active_custom_check = self.nav_goal_handle_.status == GoalStatus.STATUS_ACCEPTED or \
                                          self.nav_goal_handle_.status == GoalStatus.STATUS_EXECUTING
            elif hasattr(self.nav_goal_handle_, 'is_active'): # Older way, might not be present
                 is_active_custom_check = self.nav_goal_handle_.is_active
            
            if is_active_custom_check:
                self.get_logger().info("Actively cancelling previous NavToPose goal before sending new one.")
                cancel_future = self.nav_goal_handle_.cancel_goal_async()
                # It's better not to block with time.sleep().
                # We'll send the new goal, and Nav2 should handle preemption.
                # The result callback of the *cancelled* goal will eventually fire.
                active_goal_cancelled_or_none = False # Will be set to true if cancel is confirmed or fails
                
                def _cancel_done(future):
                    nonlocal active_goal_cancelled_or_none
                    try:
                        cancel_response = future.result()
                        if cancel_response and len(cancel_response.goals_canceling) > 0:
                            self.get_logger().info('Previous goal cancellation confirmed by server.')
                        else:
                            self.get_logger().warn('Previous goal cancellation request might not have been processed or no goal was canceling.')
                    except Exception as e:
                        self.get_logger().error(f"Exception in cancel_done callback: {e}")
                    active_goal_cancelled_or_none = True # Allow new goal sending

                cancel_future.add_done_callback(_cancel_done)
                # Loop for a short time waiting for active_goal_cancelled_or_none or timeout
                # This is a way to pseudo-block without stopping the executor entirely.
                wait_start_time = self.get_clock().now()
                while not active_goal_cancelled_or_none and (self.get_clock().now() - wait_start_time) < RclpyDuration(seconds=0.7):
                    rclpy.spin_once(self, timeout_sec=0.05) # Allow callbacks to process
                if not active_goal_cancelled_or_none:
                    self.get_logger().warn("Timeout waiting for previous goal cancellation confirmation. Proceeding with new goal anyway.")

            self.nav_goal_handle_ = None


        if not self._nav_to_pose_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("'navigate_to_pose' action server not available."); 
            self._change_state_and_process(CrackNavState.MISSION_FAILED); 
            return
        
        self.get_logger().info(f"Sending NavToPose: P({target_pose.pose.position.x:.2f},y={target_pose.pose.position.y:.2f},z={target_pose.pose.position.z:.3f}) in {target_pose.header.frame_id}")
        nav_goal_marker_array = MarkerArray()
        single_nav_goal_marker = self._create_single_marker(target_pose, "current_nav_goal", 0, r=1.0, g=0.5, b=0.0)
        nav_goal_marker_array.markers.append(single_nav_goal_marker)
        self.marker_publisher_.publish(nav_goal_marker_array)

        send_goal_future = self._nav_to_pose_client.send_goal_async(NavigateToPose.Goal(pose=target_pose), feedback_callback=self.nav_to_pose_feedback_cb)
        send_goal_future.add_done_callback(self.nav_to_pose_goal_response_cb)

    def _cancel_current_navigation_goal(self): 
        if self.nav_goal_handle_: 
            self.get_logger().info("Requesting cancellation of current NavToPose goal (explicit call).")
            is_active_custom_check = False
            if hasattr(self.nav_goal_handle_, 'status'): 
                 is_active_custom_check = self.nav_goal_handle_.status == GoalStatus.STATUS_ACCEPTED or \
                                          self.nav_goal_handle_.status == GoalStatus.STATUS_EXECUTING
            elif hasattr(self.nav_goal_handle_, 'is_active'): 
                 is_active_custom_check = self.nav_goal_handle_.is_active
            
            if is_active_custom_check:
                self.nav_goal_handle_.cancel_goal_async()
            else:
                self.get_logger().info(f"Goal not in a cancelable state (status: {self.nav_goal_handle_.status if hasattr(self.nav_goal_handle_, 'status') else 'N/A'}). Not sending cancel request.")
            self.nav_goal_handle_ = None 

    def nav_to_pose_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle : # If future.result() is None (e.g. server unavailable during send_goal_async)
            self.get_logger().error("NavToPose goal_handle is None in response_cb. Server might have been unavailable.")
            self.nav_goal_handle_ = None
            if self.state_ in [CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA, CrackNavState.NAVIGATING_TO_FIXED_CRACK_START, CrackNavState.FOLLOWING_CRACK_BODY]:
                self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)
            return

        if not goal_handle.accepted:
            self.get_logger().error("NavToPose REJECTED."); self.nav_goal_handle_ = None
            if self.state_ in [CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA, CrackNavState.NAVIGATING_TO_FIXED_CRACK_START, CrackNavState.FOLLOWING_CRACK_BODY]:
                 self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED) 
            return
        self.nav_goal_handle_ = goal_handle; self.get_logger().info("NavToPose ACCEPTED.")
        self.nav_goal_handle_.get_result_async().add_done_callback(self.nav_to_pose_result_cb)

    def nav_to_pose_feedback_cb(self, feedback_msg: NavigateToPose.Feedback):
        current_feedback_pose_geom = feedback_msg.feedback.current_pose.pose.position
        self.get_logger().debug(f"Nav Feedback: Pos(x={current_feedback_pose_geom.x:.2f}, y={current_feedback_pose_geom.y:.2f}, z={current_feedback_pose_geom.z:.2f})")

        if self.state_ == CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA:
            if self.current_target_predefined_start_ and \
               self._check_arrival(self.current_target_predefined_start_, self.crack_start_detection_proximity_, current_pos_geom=current_feedback_pose_geom):
                self.get_logger().info(f"Near predefined crack start area. Cancelling current nav and switching to crack detection.")
                self._cancel_current_navigation_goal() 
        
        elif self.state_ == CrackNavState.NAVIGATING_TO_FIXED_CRACK_START and self.fixed_crack_start_pose_:
             if self._check_arrival(self.fixed_crack_start_pose_, self.fixed_start_arrival_thresh_, current_pos_geom=current_feedback_pose_geom):
                self.get_logger().info("Proximity arrival at fixed crack start. Cancelling current nav.")
                self._cancel_current_navigation_goal() 
        elif self.state_ == CrackNavState.FOLLOWING_CRACK_BODY and self.last_crack_segment_goal_:
            if self._check_arrival(self.last_crack_segment_goal_, self.crack_segment_arrival_thresh_, current_pos_geom=current_feedback_pose_geom):
                self.get_logger().info("Proximity arrival at crack segment. Cancelling current nav.")
                self._cancel_current_navigation_goal() 

    def nav_to_pose_result_cb(self, future):
        action_result_wrapper = future.result()
        if not action_result_wrapper:
            self.get_logger().warn("NavToPose result future was None, possibly during shutdown or error.")
            if self.state_ not in [CrackNavState.MISSION_FAILED, CrackNavState.ALL_CRACKS_COMPLETED, CrackNavState.IDLE]:
                 self._change_state_and_process(CrackNavState.MISSION_FAILED)
            return
        self.nav_goal_handle_ = None 
        status = action_result_wrapper.status
        status_str = "UNKNOWN"
        if status == GoalStatus.STATUS_SUCCEEDED: status_str = "SUCCEEDED"
        elif status == GoalStatus.STATUS_ABORTED: status_str = "ABORTED"
        elif status == GoalStatus.STATUS_CANCELED: status_str = "CANCELED"
        self.get_logger().info(f"NavToPose result: {status_str} (raw_status: {status}) while in state {self.state_.name}")

        current_processing_state = self.state_ 

        if current_processing_state == CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA:
            if status == GoalStatus.STATUS_SUCCEEDED: 
                self.get_logger().info("Reached predefined crack start area successfully. Switching to detection.")
                self._change_state_and_process(CrackNavState.DETECTING_CRACK_INITIAL_POINT)
            elif status == GoalStatus.STATUS_CANCELED:
                self.get_logger().info(f"NavToPredefinedStart CANCELED. Current state {self.state_.name}. Assuming proximity trigger and switching to detection.")
                if self.state_ != CrackNavState.DETECTING_CRACK_INITIAL_POINT: 
                    self._change_state_and_process(CrackNavState.DETECTING_CRACK_INITIAL_POINT)
            else: 
                self.get_logger().error(f"Fail nav to predefined start area (status: {status_str}). Concluding this crack."); 
                self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)
        
        elif current_processing_state == CrackNavState.NAVIGATING_TO_FIXED_CRACK_START:
            is_at_target = self._check_arrival(self.fixed_crack_start_pose_, self.fixed_start_arrival_thresh_)
            if status == GoalStatus.STATUS_SUCCEEDED or (status == GoalStatus.STATUS_CANCELED and is_at_target):
                self.get_logger().info(f"Successfully arrived at fixed crack start (status: {status_str}).")
                self.last_crack_segment_goal_ = self.fixed_crack_start_pose_ 
                self._change_state_and_process(CrackNavState.FOLLOWING_CRACK_BODY)
            else: 
                self.get_logger().error(f"Fail nav to fixed crack start (status: {status_str}). Concluding this crack."); 
                self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)
        
        elif current_processing_state == CrackNavState.FOLLOWING_CRACK_BODY:
            is_at_target = self._check_arrival(self.last_crack_segment_goal_, self.crack_segment_arrival_thresh_)
            if status == GoalStatus.STATUS_SUCCEEDED or (status == GoalStatus.STATUS_CANCELED and is_at_target):
                self.get_logger().info("Reached current crack segment or canceled by proximity. Waiting for next segment or timeout.")
                if self.detection_or_segment_timeout_timer_ is None or self.detection_or_segment_timeout_timer_.is_canceled():
                     self.get_logger().debug("Segment reached, (re)starting segment timeout timer.")
                     self._create_timeout_timer(self.crack_segment_timeout_sec_, self.crack_segment_timeout_cb)
            else: 
                self.get_logger().error(f"Fail nav to crack segment (status: {status_str}). Concluding this crack."); 
                self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)
        else:
            self.get_logger().warn(f"NavToPose result received in unhandled or terminal state: {current_processing_state.name}. Status was: {status_str}")

    def _check_arrival(self, target_pose: GeometryPoseStamped, threshold: float, current_pos_geom: Point = None) -> bool:
        if not target_pose: return False
        robot_current_pos = None
        if current_pos_geom: robot_current_pos = current_pos_geom
        else:
            robot_pose_stamped = self.get_robot_pose()
            if not robot_pose_stamped: return False
            robot_current_pos = robot_pose_stamped.pose.position
        dist_sq = (robot_current_pos.x - target_pose.pose.position.x)**2 + (robot_current_pos.y - target_pose.pose.position.y)**2
        return dist_sq < threshold**2

    def corrected_goal_cb(self, msg: GeometryPoseStamped):
        self.get_logger().info(
            f"DEBUG Z: corrected_goal_cb received goal: "
            f"X={msg.pose.position.x:.2f}, Y={msg.pose.position.y:.2f}, Z={msg.pose.position.z:.3f} "
            f"in frame '{msg.header.frame_id}' while in state {self.state_.name}"
        )
        if not msg.header.frame_id: 
            if self.state_ == CrackNavState.DETECTING_CRACK_INITIAL_POINT: self.get_logger().debug("Invalid corrected goal while detecting initial point.")
            elif self.state_ == CrackNavState.FOLLOWING_CRACK_BODY:
                self.get_logger().info("Invalid corrected goal (no target) while following. Assuming crack end/lost.")
                self._cancel_current_navigation_goal(); self._change_state_and_process(CrackNavState.CRACK_NAVIGATION_CONCLUDED)
            return

        if self.state_ == CrackNavState.DETECTING_CRACK_INITIAL_POINT:
            self._destroy_timeout_timer(); self.fixed_crack_start_pose_ = msg
            self.get_logger().info(f"Crack initial point DETECTED at G(x={msg.pose.position.x:.2f},y={msg.pose.position.y:.2f}, Z={msg.pose.position.z:.3f})")
            self._destroy_corrected_goal_subscriber() 
            self._send_nav_to_pose_goal(self.fixed_crack_start_pose_)
            self._change_state_and_process(CrackNavState.NAVIGATING_TO_FIXED_CRACK_START)
        elif self.state_ == CrackNavState.FOLLOWING_CRACK_BODY:
            new_segment_goal = msg
            if self.last_crack_segment_goal_:
                dist_sq_diff = (new_segment_goal.pose.position.x - self.last_crack_segment_goal_.pose.position.x)**2 + \
                               (new_segment_goal.pose.position.y - self.last_crack_segment_goal_.pose.position.y)**2
                if math.sqrt(dist_sq_diff) < self.crack_segment_update_thresh_: 
                    self.get_logger().debug("New crack segment goal too close to previous. Ignoring.")
                    return 
            
            self.get_logger().info(f"New crack segment goal received: X={new_segment_goal.pose.position.x:.2f}, Y={new_segment_goal.pose.position.y:.2f}, Z={new_segment_goal.pose.position.z:.3f}. Updating navigation.")
            self._destroy_timeout_timer() 
            self._send_nav_to_pose_goal(new_segment_goal) 
            self.last_crack_segment_goal_ = new_segment_goal
            self._create_timeout_timer(self.crack_segment_timeout_sec_, self.crack_segment_timeout_cb) 
        else:
            self.get_logger().debug(f"Corrected goal received in non-processing state {self.state_.name}. Ignored.")

    def get_robot_pose(self) -> GeometryPoseStamped | None:
        try:
            transform = self.tf_buffer_.lookup_transform(self.global_frame_, self.robot_base_frame_, RclpyTime(), timeout=RclpyDuration(seconds=0.2))
            pose = GeometryPoseStamped(); pose.header.stamp = transform.header.stamp; pose.header.frame_id = self.global_frame_
            pose.pose.position = transform.transform.translation; pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as e:
            return None

    def _create_single_marker(self, pose_stamped: GeometryPoseStamped, ns: str, id_val: int, type=Marker.ARROW, r=0.0, g=1.0, b=0.0, a=0.8, scale_x=0.5, scale_y=0.1, scale_z=0.1) -> Marker:
        marker = Marker(header=pose_stamped.header, ns=ns, id=id_val, type=type, action=Marker.ADD, pose=pose_stamped.pose)
        marker.scale.x=scale_x; marker.scale.y=scale_y; marker.scale.z=scale_z
        marker.color.r=float(r); marker.color.g=float(g); marker.color.b=float(b); marker.color.a=float(a)
        marker.lifetime = RclpyDuration(seconds=2.0).to_msg()
        return marker

    def publish_state_markers(self):
        marker_array = MarkerArray(); now = self.get_clock().now().to_msg()
        for i in range(0, len(self.all_waypoints_), 2):
            original_wp_start_in_list = self.all_waypoints_[i]
            wp_start = copy.deepcopy(original_wp_start_in_list) 

            r,g,b,a = 0.0,0.0,1.0,0.7
            if i < self.current_waypoint_pair_idx_: r,g,b = 0.5,0.5,0.5
            elif i == self.current_waypoint_pair_idx_ and self.state_ == CrackNavState.NAVIGATING_TO_PREDEFINED_START_AREA: r,g,b=0.0,1.0,1.0
            marker_array.markers.append(self._create_single_marker(wp_start, "predefined_crack_starts", i, r=r,g=g,b=b,a=a, type=Marker.SPHERE, scale_x=0.4,scale_y=0.4,scale_z=0.4))
            
            text_marker_header = Header(stamp=wp_start.header.stamp, frame_id=wp_start.header.frame_id) 
            text_marker = Marker(header=text_marker_header, ns="predefined_labels", id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, text=f"P{i//2}_Start")
            text_marker.pose = copy.deepcopy(wp_start.pose) 
            text_marker.pose.position.z += 0.5 
            text_marker.scale.z=0.3
            text_marker.color.r=1.0;text_marker.color.g=1.0;text_marker.color.b=1.0;text_marker.color.a=1.0; text_marker.lifetime = RclpyDuration(seconds=2.0).to_msg()
            marker_array.markers.append(text_marker)

        if self.fixed_crack_start_pose_: marker_array.markers.append(self._create_single_marker(self.fixed_crack_start_pose_, "fixed_crack_start", 0, type=Marker.CUBE, r=1.0,g=0.0,b=0.0,a=0.9, scale_x=0.3,scale_y=0.3,scale_z=0.3))
        if self.last_crack_segment_goal_ and self.state_ == CrackNavState.FOLLOWING_CRACK_BODY:
            marker_array.markers.append(self._create_single_marker(self.last_crack_segment_goal_, "current_crack_segment_goal", 0, type=Marker.SPHERE, r=0.0,g=1.0,b=0.0,a=0.9, scale_x=0.25,scale_y=0.25,scale_z=0.25))
            if self.fixed_crack_start_pose_:
                line_marker_header = Header(frame_id=self.global_frame_, stamp=now) 
                line_marker = Marker(header=line_marker_header, ns="crack_follow_path", id=0, type=Marker.LINE_STRIP, action=Marker.ADD)
                line_marker.scale.x = 0.05 
                line_marker.color = ColorRGBA(r=1.0,g=1.0,b=0.0,a=0.8) 
                line_marker.points.extend([self.fixed_crack_start_pose_.pose.position, self.last_crack_segment_goal_.pose.position])
                line_marker.lifetime = RclpyDuration(seconds=2.0).to_msg()
                marker_array.markers.append(line_marker)
        if marker_array.markers: self.marker_publisher_.publish(marker_array)

    def destroy_node(self):
        self.get_logger().info("Destroying Crack Following Navigator..."); self._cancel_current_navigation_goal(); self._activate_segmentation_node(False)
        self._destroy_corrected_goal_subscriber(); self._destroy_timeout_timer()
        if self.marker_publish_timer_ and not self.marker_publish_timer_.is_canceled(): self.marker_publish_timer_.cancel()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args); node = None
    try: node = WaypointFollowerCorrected(); rclpy.spin(node)
    except KeyboardInterrupt: 
        if node: node.get_logger().info("Node interrupted by user.")
    except Exception as e: 
        if node and node.context.ok(): node.get_logger().error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        else: print(f"Unhandled exception during node init or after context invalidation: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok() and node.context.ok(): # Check if node and context are valid
            node.destroy_node() 
        if rclpy.ok(): 
            rclpy.shutdown()
        print("Crack Following Navigator shutdown.")

if __name__ == '__main__':
    main()