#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import PoseStamped as GeometryPoseStamped, Point, Quaternion
from nav2_msgs.action import FollowWaypoints, NavigateToPose
import yaml
from pathlib import Path
import sys
import os
from ament_index_python.packages import get_package_share_directory
import tf2_ros
# from tf_transformations import euler_from_quaternion # Only if needed for extreme debugging
import math
from enum import Enum, auto
import traceback 
from action_msgs.msg import GoalStatus # For checking action status
import time # For time.sleep in TF wait loop
from visualization_msgs.msg import Marker, MarkerArray # For visualization

from std_srvs.srv import SetBool # For activating ROILidarFusionNode

class NavState(Enum):
    IDLE = auto()
    LOADING_WAYPOINTS = auto()
    FOLLOWING_GLOBAL_WAYPOINTS = auto()
    AWAITING_LOCAL_CORRECTION_CONFIRM = auto() # Segmentation active, waiting for first corrected goal
    FOLLOWING_LOCAL_TARGET = auto()
    WAYPOINT_SEQUENCE_COMPLETE = auto()
    MISSION_FAILED = auto()

class WaypointFollowerCorrected(Node):
    def __init__(self):
        super().__init__('waypoint_follower_corrected_node')

        # --- Parameters ---
        default_yaml_path = self._get_default_yaml_path('rw', 'config', 'nav_waypoints.yaml')
        self.declare_parameter('waypoints_yaml_path', default_yaml_path)
        self.declare_parameter('correction_activation_distance', 7.0) 
        self.declare_parameter('local_target_arrival_threshold', 0.35)
        self.declare_parameter('local_goal_update_threshold', 0.25) # Debounce for local goal updates
        self.declare_parameter('segmentation_node_name', 'roi_lidar_fusion_node_activated') 
        self.declare_parameter('corrected_local_goal_topic', '/corrected_local_goal')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')

        self.yaml_path_ = self.get_parameter('waypoints_yaml_path').get_parameter_value().string_value
        self.correction_activation_dist_ = self.get_parameter('correction_activation_distance').get_parameter_value().double_value
        self.local_arrival_thresh_ = self.get_parameter('local_target_arrival_threshold').get_parameter_value().double_value
        self.local_goal_update_threshold_ = self.get_parameter('local_goal_update_threshold').get_parameter_value().double_value
        segmentation_node_name = self.get_parameter('segmentation_node_name').get_parameter_value().string_value
        self.corrected_goal_topic_ = self.get_parameter('corrected_local_goal_topic').get_parameter_value().string_value
        self.robot_base_frame_ = self.get_parameter('robot_base_frame').get_parameter_value().string_value
        self.global_frame_ = self.get_parameter('global_frame').get_parameter_value().string_value
        self.activation_srv_name_ = f"/{segmentation_node_name}/activate_segmentation"

        # --- Initialize Member Variables ---
        self.all_waypoints_ = []
        self.last_truly_completed_global_idx_ = -1
        self.correction_active_for_wp_idx_ = -1 
        self.current_global_waypoint_idx_nav2_ = -1 # For marker coloring
        self.last_sent_local_goal_pose_ = None 
        self.state_ = NavState.IDLE

        # --- Initialize TF Buffer and Listener ---
        self.tf_buffer_ = tf2_ros.Buffer() 
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)

        # --- TF Ready Check ---
        self.get_logger().info("Waiting for initial TF tree to become available...")
        tf_ready = False
        for i in range(10): 
            if self.tf_buffer_.can_transform(self.global_frame_, self.robot_base_frame_, RclpyTime(), timeout=RclpyDuration(seconds=0.05)):
                tf_ready = True
                self.get_logger().info(f"TF tree is ready ({self.global_frame_} -> {self.robot_base_frame_}).")
                break
            self.get_logger().info(f"Waiting for TF ({self.global_frame_} -> {self.robot_base_frame_})... try {i + 1}/10")
            time.sleep(0.25) 
        if not tf_ready:
            self.get_logger().error("TF tree did not become available. Node functionality will be impaired (pose lookups will fail).")
            # self.state_ = NavState.MISSION_FAILED # Option: Fail fast if TF is critical early on

        # --- Initialize Action Clients ---
        self._follow_waypoints_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_follow_wp_goal_handle = None
        self._current_nav_to_pose_goal_handle = None

        # --- Initialize Service Client ---
        self.segmentation_activation_client_ = self.create_client(SetBool, self.activation_srv_name_)
        # No wait_for_service here to prevent blocking __init__ if segmentation node is slow
        # Service readiness will be checked before calls.

        # --- Initialize Subscriber Variable (created on demand) ---
        self.corrected_goal_sub_ = None 
        self.latest_corrected_goal_ = None
        
        # --- Initialize Marker Publisher and Timer ---
        self.marker_publisher_ = self.create_publisher(MarkerArray, '~/debug_waypoints_markers', 10)
        self.marker_publish_timer_ = self.create_timer(1.0, self.publish_waypoint_markers)

        # --- Final Initialization Log and Start State Machine ---
        self.get_logger().info(f"WaypointFollowerCorrected initialized. Initial state: {self.state_.name}")
        if self.state_ != NavState.MISSION_FAILED: # Only start if not already failed (e.g., by TF check)
            self._change_state_and_process(NavState.LOADING_WAYPOINTS)
        else:
            self.get_logger().error("Node initialization determined a MISSION_FAILED state. Not starting waypoint loading.")

    def _get_default_yaml_path(self, package_name, config_dir, file_name):
        try:
            share_dir = get_package_share_directory(package_name)
            return os.path.join(share_dir, config_dir, file_name)
        except Exception: return ""

    def _change_state_and_process(self, new_state: NavState):
        if self.state_ == new_state and new_state not in [NavState.LOADING_WAYPOINTS]:
             self.get_logger().debug(f"Already in state {new_state.name}. No transition.")
             return
        self.get_logger().info(f"STATE: {self.state_.name} -> {new_state.name}")
        old_state = self.state_
        self.state_ = new_state
        
        if old_state == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM and new_state != NavState.FOLLOWING_LOCAL_TARGET:
            self._destroy_corrected_goal_subscriber() # Cleanup if moving away from correction
        self._process_state_actions()

    def _process_state_actions(self):
        action_description = f"Processing actions for state: {self.state_.name}"
        if self.state_ == NavState.LOADING_WAYPOINTS:
            self.get_logger().info(f"{action_description} - Loading waypoints from YAML...")
            if not self.yaml_path_ or not Path(self.yaml_path_).is_file():
                self.get_logger().error(f"Waypoint YAML file not found or path invalid: '{self.yaml_path_}'")
                self._change_state_and_process(NavState.MISSION_FAILED); return
            if self.load_waypoints_from_yaml(Path(self.yaml_path_)):
                if self.all_waypoints_:
                    self.last_truly_completed_global_idx_ = -1 # Reset progress
                    self._send_follow_waypoints_goal_from_index(0) # Start from the beginning
                else: 
                    self.get_logger().info("No waypoints loaded from file. Mission considered complete.")
                    self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
            else: 
                self.get_logger().error("Failed to load waypoints from YAML.")
                self._change_state_and_process(NavState.MISSION_FAILED)

        elif self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM:
            self.get_logger().info(f"{action_description} - Activating subscriber for corrected local goals.")
            self._create_corrected_goal_subscriber()

        elif self.state_ == NavState.WAYPOINT_SEQUENCE_COMPLETE:
            self.get_logger().info(f"{action_description} - All waypoints processed. Mission successful!")
            self._activate_segmentation_node(False) # Ensure segmentation is off
            # rclpy.shutdown() # Let main handle shutdown, or use a self.destroy_node() then rclpy.shutdown()

        elif self.state_ == NavState.MISSION_FAILED:
            self.get_logger().error(f"{action_description} - Mission failed. Attempting cleanup.")
            self._activate_segmentation_node(False)
            self._cancel_all_navigation_actions()
            # rclpy.shutdown() # Let main handle shutdown

        else: # FOLLOWING_GLOBAL_WAYPOINTS, FOLLOWING_LOCAL_TARGET, IDLE
            self.get_logger().debug(f"{action_description} - No immediate actions, driven by callbacks or other events.")


    def load_waypoints_from_yaml(self, yaml_file_path: Path) -> bool:
        # (This method remains the same as your last provided version)
        try:
            with open(yaml_file_path, 'r') as file: yaml_data = yaml.safe_load(file)
            if not yaml_data or 'poses' not in yaml_data:
                self.get_logger().error(f"YAML '{yaml_file_path}' empty or no 'poses' key."); return False
            loaded_waypoints = []
            for i, pose_entry in enumerate(yaml_data['poses']):
                try:
                    ps_msg = GeometryPoseStamped()
                    header_data = pose_entry.get('header', {})
                    ps_msg.header.frame_id = header_data.get('frame_id', self.global_frame_)
                    ps_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_block = pose_entry.get('pose', {})
                    pos_data = pose_block.get('position', {}); orient_data = pose_block.get('orientation', {})
                    ps_msg.pose.position.x = float(pos_data.get('x',0.0)); ps_msg.pose.position.y = float(pos_data.get('y',0.0)); ps_msg.pose.position.z = float(pos_data.get('z',0.0))
                    ps_msg.pose.orientation.x=float(orient_data.get('x',0.0)); ps_msg.pose.orientation.y=float(orient_data.get('y',0.0)); ps_msg.pose.orientation.z=float(orient_data.get('z',0.0)); ps_msg.pose.orientation.w=float(orient_data.get('w',1.0))
                    loaded_waypoints.append(ps_msg)
                except Exception as e: self.get_logger().error(f"Error parsing waypoint {i+1}: {e}"); return False
            self.all_waypoints_ = loaded_waypoints
            self.get_logger().info(f"Successfully loaded {len(self.all_waypoints_)} waypoints from '{yaml_file_path}'.")
            return True
        except Exception as e: self.get_logger().error(f"Error loading YAML file '{yaml_file_path}': {e}"); return False

    def _create_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is None:
            self.get_logger().info(f"Creating subscriber for corrected goals on '{self.corrected_goal_topic_}'.")
            _qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, 
                              depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.corrected_goal_sub_ = self.create_subscription(
                GeometryPoseStamped, self.corrected_goal_topic_, self.corrected_goal_callback, _qos)
    
    def _destroy_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is not None:
            self.get_logger().info("Destroying subscriber for corrected goals.")
            self.destroy_subscription(self.corrected_goal_sub_); self.corrected_goal_sub_ = None
            self.latest_corrected_goal_ = None # Clear any stale data

    def get_robot_pose(self) -> GeometryPoseStamped | None:
        # (This method remains the same)
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.global_frame_, self.robot_base_frame_, RclpyTime(), timeout=RclpyDuration(seconds=0.2))
            pose = GeometryPoseStamped(); pose.header.stamp = transform.header.stamp
            pose.header.frame_id = self.global_frame_; pose.pose.position = transform.transform.translation
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().warn(f"TF Robot Pose Error looking up {self.global_frame_} -> {self.robot_base_frame_}: {e}", throttle_duration_sec=2)
            return None

    def _activate_segmentation_node(self, activate: bool):
        if not self.segmentation_activation_client_.service_is_ready():
            self.get_logger().warn(f"Service '{self.activation_srv_name_}' not ready to {'activate' if activate else 'deactivate'} segmentation."); return False
        req = SetBool.Request(); req.data = activate
        future = self.segmentation_activation_client_.call_async(req)
        self.get_logger().info(f"Requested segmentation node {'activation' if activate else 'deactivation'}.")
        # Optional: Add a callback to future to log actual service response, but don't block.
        return True

    def _cancel_all_navigation_actions(self):
        # (This method remains the same)
        if self._current_follow_wp_goal_handle:
            self._current_follow_wp_goal_handle.cancel_goal_async(); self._current_follow_wp_goal_handle = None 
        if self._current_nav_to_pose_goal_handle:
            self._current_nav_to_pose_goal_handle.cancel_goal_async(); self._current_nav_to_pose_goal_handle = None
        self.get_logger().info("Requested cancellation of any active navigation goals.")

    def _send_follow_waypoints_goal_from_index(self, start_index: int):
        self._cancel_all_navigation_actions() # Ensure other nav actions are stopped
        if start_index >= len(self.all_waypoints_):
            self.get_logger().info("All global waypoints have been processed or skipped.")
            self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE); return
        
        waypoints_to_send = self.all_waypoints_[start_index:]
        if not waypoints_to_send: # Should be caught by above check, but defensive
            self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE); return

        if not self._follow_waypoints_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("'follow_waypoints' action server not available.")
            self._change_state_and_process(NavState.MISSION_FAILED); return

        goal_msg = FollowWaypoints.Goal(); goal_msg.poses = waypoints_to_send
        self.get_logger().info(f"Sending {len(waypoints_to_send)} global waypoints (starting from original_idx {start_index}) to FollowWaypoints.")
        send_goal_future = self._follow_waypoints_client.send_goal_async(
            goal_msg, feedback_callback=self.follow_waypoints_feedback_cb)
        send_goal_future.add_done_callback(self.follow_waypoints_goal_response_cb)
        self._change_state_and_process(NavState.FOLLOWING_GLOBAL_WAYPOINTS)

    def follow_waypoints_goal_response_cb(self, future):
        # (This method remains the same)
        if self.state_ == NavState.MISSION_FAILED: return
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("FollowWaypoints goal was REJECTED."); self._current_follow_wp_goal_handle = None
            if self.state_ != NavState.MISSION_FAILED: self._change_state_and_process(NavState.MISSION_FAILED)
            return
        self._current_follow_wp_goal_handle = goal_handle
        self.get_logger().info("FollowWaypoints goal ACCEPTED.")
        goal_handle.get_result_async().add_done_callback(self.follow_waypoints_result_cb)

    def follow_waypoints_feedback_cb(self, feedback_msg: FollowWaypoints.Feedback):
        if self.state_ != NavState.FOLLOWING_GLOBAL_WAYPOINTS: return

        self.current_global_waypoint_idx_nav2_ = feedback_msg.feedback.current_waypoint
        actual_approaching_global_idx = self.last_truly_completed_global_idx_ + 1 + self.current_global_waypoint_idx_nav2_
        
        if actual_approaching_global_idx < len(self.all_waypoints_):
            self.get_logger().info(f"Following global waypoint {actual_approaching_global_idx + 1}/{len(self.all_waypoints_)} "
                                   f"(Nav2 processing its internal index {self.current_global_waypoint_idx_nav2_}).")
            robot_pose = self.get_robot_pose()
            if robot_pose:
                target_wp = self.all_waypoints_[actual_approaching_global_idx]
                dist_sq = (robot_pose.pose.position.x - target_wp.pose.position.x)**2 + \
                          (robot_pose.pose.position.y - target_wp.pose.position.y)**2
                if dist_sq < self.correction_activation_dist_**2:
                    self.get_logger().info(f"ACTION: Near global_idx {actual_approaching_global_idx} "
                                           f"(dist: {math.sqrt(dist_sq):.2f}m). Activating local correction.")
                    if self._activate_segmentation_node(True):
                        self.correction_active_for_wp_idx_ = actual_approaching_global_idx
                        self._change_state_and_process(NavState.AWAITING_LOCAL_CORRECTION_CONFIRM)
        else:
             self.get_logger().warn(f"Feedback for actual_approaching_global_idx {actual_approaching_global_idx} is out of bounds for all_waypoints_ (len {len(self.all_waypoints_)}). This might occur if FollowWaypoints completes its batch.")


    def follow_waypoints_result_cb(self, future): 
        if self.state_ == NavState.MISSION_FAILED: return

        action_result_wrapper = future.result() 
        if not action_result_wrapper: # Check if future.result() itself is None (should not happen if goal was accepted)
            self.get_logger().error("FollowWaypoints result future had no wrapper object.")
            if self.state_ != NavState.MISSION_FAILED: self._change_state_and_process(NavState.MISSION_FAILED)
            return

        status = action_result_wrapper.status
        actual_result_message = action_result_wrapper.result
        self._current_follow_wp_goal_handle = None 

        # Use GoalStatus directly for human-readable names
        status_name = "UNKNOWN_STATUS"
        if status == GoalStatus.STATUS_SUCCEEDED: status_name = "SUCCEEDED"
        elif status == GoalStatus.STATUS_CANCELED: status_name = "CANCELED"
        elif status == GoalStatus.STATUS_ABORTED: status_name = "ABORTED"
        # Add other statuses if needed (EXECUTING, etc., though they shouldn't appear in a result_cb often)


        if self.state_ in [NavState.AWAITING_LOCAL_CORRECTION_CONFIRM, NavState.FOLLOWING_LOCAL_TARGET]:
            self.get_logger().info(f"FollowWaypoints action ended with status {status_name}, likely due to correction switch.")
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("FollowWaypoints action SUCCEEDED (all sent waypoints reached).")
            self.last_truly_completed_global_idx_ = len(self.all_waypoints_) - 1 
            self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
        elif status == GoalStatus.STATUS_CANCELED:
             self.get_logger().warn(f"FollowWaypoints CANCELED unexpectedly (status: {status_name}).")
             self._change_state_and_process(NavState.MISSION_FAILED)
        else: 
            missed_wps_info = str(actual_result_message.missed_waypoints) if actual_result_message else "N/A"
            self.get_logger().error(f"FollowWaypoints action FAILED/ABORTED. Status: {status_name}. Missed: {missed_wps_info}")
            self._change_state_and_process(NavState.MISSION_FAILED)
    
    def corrected_goal_callback(self, msg: GeometryPoseStamped):
        if not (self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM or \
                self.state_ == NavState.FOLLOWING_LOCAL_TARGET):
            self.get_logger().debug(f"Corrected goal rcvd in wrong state ({self.state_.name}). Ignoring.")
            return

        if not msg.header.frame_id: 
            self.get_logger().info("Corrected goal rcvd with empty frame_id (no target / segmentation deactivated).")
            self.latest_corrected_goal_ = None
            if self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM:
                 self.get_logger().info("No valid local target found during AWAITING_CONFIRM. Resuming global.")
                 self._resume_global_waypoints_after_failed_correction()
            elif self.state_ == NavState.FOLLOWING_LOCAL_TARGET:
                if self._current_nav_to_pose_goal_handle:
                    self.get_logger().info("Local target became invalid while following. Cancelling current NavToPose.")
                    self._current_nav_to_pose_goal_handle.cancel_goal_async()
                    # Result cb for NavToPose will then call _resume_global_waypoints_after_failed_correction
            return

        new_local_goal = msg
        self.latest_corrected_goal_ = new_local_goal 
        self.get_logger().info(f"CORRECTION: Received corrected local target at P(x={new_local_goal.pose.position.x:.2f}, y={new_local_goal.pose.position.y:.2f}) in '{new_local_goal.header.frame_id}'.")

        if self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM:
            self.get_logger().info("ACTION: First valid corrected goal received. Switching to local target following.")
            if self._current_follow_wp_goal_handle: 
                self.get_logger().info("Cancelling FollowWaypoints to switch to local target.")
                self._current_follow_wp_goal_handle.cancel_goal_async()
            self._change_state_and_process(NavState.FOLLOWING_LOCAL_TARGET) 
            self.last_sent_local_goal_pose_ = self.latest_corrected_goal_
            self._send_navigate_to_pose_goal(self.latest_corrected_goal_)

        elif self.state_ == NavState.FOLLOWING_LOCAL_TARGET:
            significant_change = True
            if self.last_sent_local_goal_pose_:
                dist_sq_diff = (new_local_goal.pose.position.x - self.last_sent_local_goal_pose_.pose.position.x)**2 + \
                               (new_local_goal.pose.position.y - self.last_sent_local_goal_pose_.pose.position.y)**2
                if math.sqrt(dist_sq_diff) < self.local_goal_update_threshold_:
                    significant_change = False
            
            if not self._current_nav_to_pose_goal_handle: 
                self.get_logger().info("ACTION: No active local NavToPose. Sending new corrected goal.")
                self.last_sent_local_goal_pose_ = self.latest_corrected_goal_
                self._send_navigate_to_pose_goal(self.latest_corrected_goal_)
            elif significant_change: 
                self.get_logger().info("ACTION: Local target UPDATED significantly. Preempting current NavToPose.")
                if self._current_nav_to_pose_goal_handle: # Ensure handle exists
                    self._current_nav_to_pose_goal_handle.cancel_goal_async()
                # New goal will be sent by navigate_to_pose_result_cb for the CANCELLED status
            else:
                self.get_logger().debug("New local goal not significantly different from last sent. Not preempting NavToPose.")

    def _send_navigate_to_pose_goal(self, target_pose: GeometryPoseStamped):
        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("'navigate_to_pose' action server not available for local target.")
            self._resume_global_waypoints_after_failed_correction(); return
            
        goal_msg = NavigateToPose.Goal(); goal_msg.pose = target_pose
        self.get_logger().info(f"ACTION: Sending local target to NavigateToPose: P(x={target_pose.pose.position.x:.2f}, y={target_pose.pose.position.y:.2f}).")
        send_future = self._navigate_to_pose_client.send_goal_async(goal_msg, self.navigate_to_pose_feedback_cb)
        send_future.add_done_callback(self.navigate_to_pose_goal_response_cb)

    def navigate_to_pose_goal_response_cb(self, future):
        if self.state_ != NavState.FOLLOWING_LOCAL_TARGET and self.state_ != NavState.MISSION_FAILED: 
             self.get_logger().debug(f"NavToPose goal response in unexpected state {self.state_.name}. Ignoring.")
             return
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("NavigateToPose goal for local target REJECTED."); self._current_nav_to_pose_goal_handle = None
            if self.state_ != NavState.MISSION_FAILED: self._resume_global_waypoints_after_failed_correction()
            return
        self._current_nav_to_pose_goal_handle = goal_handle
        self.get_logger().info("NavigateToPose goal for local target ACCEPTED.")
        goal_handle.get_result_async().add_done_callback(self.navigate_to_pose_result_cb)

    def navigate_to_pose_feedback_cb(self, feedback_msg: NavigateToPose.Feedback):
        if self.state_ != NavState.FOLLOWING_LOCAL_TARGET: return
        robot_pose = self.get_robot_pose()
        if robot_pose and self.latest_corrected_goal_ and self.latest_corrected_goal_.header.frame_id: # Check if goal is valid
            dist_sq = (robot_pose.pose.position.x - self.latest_corrected_goal_.pose.position.x)**2 + \
                      (robot_pose.pose.position.y - self.latest_corrected_goal_.pose.position.y)**2
            self.get_logger().debug(f"Distance to local target: {math.sqrt(dist_sq):.2f}m (Threshold: {self.local_arrival_thresh_}m)")
            if dist_sq < self.local_arrival_thresh_**2:
                self.get_logger().info(f"EVENT: Local target arrival by proximity (dist {math.sqrt(dist_sq):.2f}m).")
                if self._current_nav_to_pose_goal_handle: 
                    self.get_logger().info("Cancelling active NavigateToPose due to proximity arrival.")
                    self._current_nav_to_pose_goal_handle.cancel_goal_async()
                else: # Handle already gone (e.g. previous cancel, or succeeded very fast) but we are close
                    self.get_logger().info("Proximity arrival, but no active NavToPose handle. Processing completion directly.")
                    self._handle_local_target_completion()

    def navigate_to_pose_result_cb(self, future):
        if self.state_ != NavState.FOLLOWING_LOCAL_TARGET and self.state_ != NavState.MISSION_FAILED :
             self.get_logger().warn(f"NavigateToPose result callback in unexpected state {self.state_.name}. Ignoring result unless failed.")
             if self.state_ == NavState.MISSION_FAILED: return
             return

        action_result_wrapper = future.result()
        if not action_result_wrapper:
             self.get_logger().error("NavigateToPose result future had no wrapper object.");
             if self.state_ != NavState.MISSION_FAILED: self._resume_global_waypoints_after_failed_correction()
             return
        status = action_result_wrapper.status
        self._current_nav_to_pose_goal_handle = None 

        is_prox_arrival = self.latest_corrected_goal_ and \
                          self.latest_corrected_goal_.header.frame_id and \
                          self._check_arrival_at_local_target(self.latest_corrected_goal_)
        
        status_name = "UNKNOWN_STATUS"
        if status == GoalStatus.STATUS_SUCCEEDED: status_name = "SUCCEEDED"
        elif status == GoalStatus.STATUS_CANCELED: status_name = "CANCELED"
        elif status == GoalStatus.STATUS_ABORTED: status_name = "ABORTED"

        self.get_logger().info(f"NavigateToPose for local target finished with status: {status_name}")

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Local target navigation SUCCEEDED.")
            self._handle_local_target_completion()
        elif status == GoalStatus.STATUS_CANCELED:
            if is_prox_arrival:
                self.get_logger().info("Local target navigation CANCELED due to proximity arrival.")
                self._handle_local_target_completion()
            else:
                self.get_logger().warn("Local target navigation CANCELED (not by proximity).")
                if self.latest_corrected_goal_ and self.latest_corrected_goal_.header.frame_id:
                    self.get_logger().info("Attempting to navigate to the latest updated local goal after cancellation.")
                    self.last_sent_local_goal_pose_ = self.latest_corrected_goal_
                    self._send_navigate_to_pose_goal(self.latest_corrected_goal_)
                else:
                    self.get_logger().info("No valid new local goal after cancellation. Resuming global waypoints.")
                    self._resume_global_waypoints_after_failed_correction()
        else: 
            self.get_logger().error(f"Local target navigation FAILED or ABORTED. Status: {status_name}.")
            self._resume_global_waypoints_after_failed_correction()
            
    def _handle_local_target_completion(self):
        self.get_logger().info(f"EVENT: Local target for global_idx {self.correction_active_for_wp_idx_} successfully processed.")
        self.last_truly_completed_global_idx_ = self.correction_active_for_wp_idx_
        
        self.latest_corrected_goal_ = None # Clear the completed local target
        self._activate_segmentation_node(False) # Turn off segmentation
        self._destroy_corrected_goal_subscriber() # Stop listening for corrections
        self.correction_active_for_wp_idx_ = -1 # Reset for the next global waypoint

        next_global_idx_to_process = self.last_truly_completed_global_idx_ + 1
        self.get_logger().info(f"Proceeding to next global waypoint, index {next_global_idx_to_process}.")
        self._send_follow_waypoints_goal_from_index(next_global_idx_to_process)

    def _check_arrival_at_local_target(self, target_pose: GeometryPoseStamped):
        # (This method remains the same)
        robot_pose = self.get_robot_pose()
        if not robot_pose or not target_pose: return False
        dist_sq = (robot_pose.pose.position.x - target_pose.pose.position.x)**2 + \
                  (robot_pose.pose.position.y - target_pose.pose.position.y)**2
        return dist_sq < self.local_arrival_thresh_**2

    def _resume_global_waypoints_after_failed_correction(self):
        self.get_logger().info("Attempting to resume global waypoints after a local navigation/correction issue.")
        self._activate_segmentation_node(False)
        self._destroy_corrected_goal_subscriber()
        self.latest_corrected_goal_ = None
        
        # Logic to decide next global waypoint:
        # If correction was active for a waypoint, we mark that one as "attempted and failed locally"
        # and move to the one AFTER it.
        if self.correction_active_for_wp_idx_ != -1 and \
           self.correction_active_for_wp_idx_ > self.last_truly_completed_global_idx_:
            self.last_truly_completed_global_idx_ = self.correction_active_for_wp_idx_
            self.get_logger().info(f"Marking global_idx {self.correction_active_for_wp_idx_} as processed due to local correction failure.")
        
        next_wp_idx_to_process = self.last_truly_completed_global_idx_ + 1
        self.correction_active_for_wp_idx_ = -1 # Reset
        
        self.get_logger().info(f"Resuming global waypoints from original index {next_wp_idx_to_process}.")
        self._send_follow_waypoints_goal_from_index(next_wp_idx_to_process)

    def publish_waypoint_markers(self):
        # (This method remains the same as your last provided version)
        marker_array = MarkerArray(); now = self.get_clock().now().to_msg()
        for i, pose_stamped in enumerate(self.all_waypoints_):
            marker = Marker(); marker.header.frame_id = self.global_frame_; marker.header.stamp = now
            marker.ns = "global_waypoints"; marker.id = i; marker.type = Marker.ARROW; marker.action = Marker.ADD
            marker.pose = pose_stamped.pose
            marker.scale.x = 0.5; marker.scale.y = 0.08; marker.scale.z = 0.08
            active_global_target_idx = self.last_truly_completed_global_idx_ + 1 + self.current_global_waypoint_idx_nav2_
            if i == active_global_target_idx and self.state_ == NavState.FOLLOWING_GLOBAL_WAYPOINTS:
                marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 1.0; marker.color.a = 1.0 # Cyan
            elif i <= self.last_truly_completed_global_idx_:
                marker.color.r = 0.5; marker.color.g = 0.5; marker.color.b = 0.5; marker.color.a = 0.7 # Gray
            else: marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 0.8 # Blue
            marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(marker)
        if self.state_ == NavState.FOLLOWING_LOCAL_TARGET and self.latest_corrected_goal_ and self.latest_corrected_goal_.header.frame_id:
            local_marker = Marker(); local_marker.header.frame_id = self.global_frame_; local_marker.header.stamp = now
            local_marker.ns = "local_corrected_target"; local_marker.id = 0; local_marker.type = Marker.SPHERE; local_marker.action = Marker.ADD
            local_marker.pose = self.latest_corrected_goal_.pose
            local_marker.scale.x = 0.4; local_marker.scale.y = 0.4; local_marker.scale.z = 0.4
            local_marker.color.r = 1.0; local_marker.color.g = 0.0; local_marker.color.b = 0.0; local_marker.color.a = 1.0 # Red
            local_marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(local_marker)
            if self.correction_active_for_wp_idx_ >= 0 and self.correction_active_for_wp_idx_ < len(self.all_waypoints_):
                og_pose = self.all_waypoints_[self.correction_active_for_wp_idx_].pose
                line_marker = Marker(); line_marker.header.frame_id = self.global_frame_; line_marker.header.stamp = now
                line_marker.ns = "correction_line"; line_marker.id = self.correction_active_for_wp_idx_
                line_marker.type = Marker.LINE_STRIP; line_marker.action = Marker.ADD; line_marker.scale.x = 0.05
                line_marker.color.r = 1.0; line_marker.color.g = 1.0; line_marker.color.b = 0.0; line_marker.color.a = 0.8 # Yellow
                p1 = Point(); p1.x = og_pose.position.x; p1.y = og_pose.position.y; p1.z = og_pose.position.z
                p2 = Point(); p2.x = self.latest_corrected_goal_.pose.position.x; p2.y = self.latest_corrected_goal_.pose.position.y; p2.z = self.latest_corrected_goal_.pose.position.z
                line_marker.points.extend([p1,p2]); line_marker.lifetime = RclpyDuration(seconds=2.0).to_msg(); marker_array.markers.append(line_marker)
        if marker_array.markers: self.marker_publisher_.publish(marker_array)


    def destroy_node(self):
        self.get_logger().info("Destroying WaypointFollowerCorrected node...")
        self._activate_segmentation_node(False) 
        self._cancel_all_navigation_actions()
        if self.marker_publish_timer_: # Check if timer object exists
            if not self.marker_publish_timer_.is_canceled(): # Check if not already canceled
                self.get_logger().info("Cancelling marker publish timer.")
                self.marker_publish_timer_.cancel()
            else:
                self.get_logger().debug("Marker publish timer was already cancelled.")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointFollowerCorrected()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Node interrupted by user (KeyboardInterrupt).')
    except SystemExit: # This can be raised by rclpy.shutdown() being called elsewhere or sys.exit()
        if node: node.get_logger().info('Node is shutting down (SystemExit).')
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception in WaypointFollowerCorrected: {e}\n{traceback.format_exc()}")
        else: print(f"Unhandled exception during WaypointFollowerCorrected init: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok(): # Check rclpy.ok() before destroying if node is still valid
            node.destroy_node() 
        if rclpy.ok(): # Check again before final rclpy.shutdown()
            rclpy.shutdown()
        print("WaypointFollowerCorrected main finally block finished.")

if __name__ == '__main__':
    main()