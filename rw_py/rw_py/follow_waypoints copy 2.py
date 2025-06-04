#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.parameter import Parameter # Not directly used but good to have if needed
from rcl_interfaces.msg import ParameterDescriptor # Not directly used
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
from tf_transformations import euler_from_quaternion # Only if needed for debugging/logging yaw
import math
from enum import Enum, auto
import traceback 
from action_msgs.msg import GoalStatus
import time

from std_srvs.srv import SetBool # For activating ROILidarFusionNode

class NavState(Enum):
    IDLE = auto()
    LOADING_WAYPOINTS = auto()
    FOLLOWING_GLOBAL_WAYPOINTS = auto()
    AWAITING_LOCAL_CORRECTION_CONFIRM = auto()
    FOLLOWING_LOCAL_TARGET = auto()
    WAYPOINT_SEQUENCE_COMPLETE = auto()
    MISSION_FAILED = auto()

class WaypointFollowerCorrected(Node): # Renamed class for clarity
    def __init__(self):
        super().__init__('waypoint_follower_corrected_node') # Renamed node for clarity

        # --- Parameters ---
        default_yaml_path = self._get_default_yaml_path('rw', 'config', 'nav_waypoints.yaml')
        self.declare_parameter('waypoints_yaml_path', default_yaml_path)
        self.declare_parameter('correction_activation_distance', 3.0) 
        self.declare_parameter('local_target_arrival_threshold', 0.35)
        self.declare_parameter('local_goal_update_threshold', 0.2) # Added for debouncing local goal updates
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

        self.all_waypoints_ = []
        self.last_truly_completed_global_idx_ = -1
        self.correction_active_for_wp_idx_ = -1 
        self.last_sent_local_goal_pose_ = None # For debouncing

        # --- Initialize TF Buffer and Listener EARLIER ---
        self.tf_buffer_ = tf2_ros.Buffer() 
        self.tf_listener_ = tf2_ros.TransformListener(self.tf_buffer_, self)
        # --- END TF Buffer Initialization ---

        self.get_logger().info("Waiting for initial TF tree to become available...")
        tf_ready = False
        for i in range(40): # Increased attempts, shorter sleep
            if self.tf_buffer_.can_transform(self.global_frame_, self.robot_base_frame_, RclpyTime(), timeout=RclpyDuration(seconds=0.05)): # Shorter timeout for individual check
                tf_ready = True
                self.get_logger().info("TF tree is ready.")
                break
            self.get_logger().info(f"Waiting for TF ({self.global_frame_} -> {self.robot_base_frame_})... try {i + 1}/40")
            time.sleep(0.25) 
        if not tf_ready:
            self.get_logger().error("TF tree did not become available after multiple attempts. Critical TF issue. Node may not function correctly.")
            # Decide if you want to sys.exit(1) here or let it try and fail later

        self.state_ = NavState.IDLE
        self.get_logger().info(f"Initial state: {self.state_.name}")

        self._follow_waypoints_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self._navigate_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_follow_wp_goal_handle = None
        self._current_nav_to_pose_goal_handle = None

        self.segmentation_activation_client_ = self.create_client(SetBool, self.activation_srv_name_)
        self.get_logger().info(f"Waiting for segmentation activation service '{self.activation_srv_name_}'...")
        if not self.segmentation_activation_client_.wait_for_service(timeout_sec=5.0): # This can block, consider if it's desired
            self.get_logger().error(f"Service '{self.activation_srv_name_}' not available! Local correction will fail.")
        else:
            self.get_logger().info(f"Service '{self.activation_srv_name_}' is available.")

        self.corrected_goal_sub_ = None 
        self.latest_corrected_goal_ = None

        # TF Buffer and Listener are already initialized above this point.

        self.get_logger().info("WaypointFollower (with correction logic) initialized.")
        self._change_state_and_process(NavState.LOADING_WAYPOINTS)

    def _get_default_yaml_path(self, package_name, config_dir, file_name):
        try:
            share_dir = get_package_share_directory(package_name)
            return os.path.join(share_dir, config_dir, file_name)
        except Exception: return ""

    def _change_state_and_process(self, new_state: NavState):
        if self.state_ == new_state and new_state not in [NavState.LOADING_WAYPOINTS]:
             self.get_logger().debug(f"Already in state {new_state.name}. No transition.")
             return
        self.get_logger().info(f"State: {self.state_.name} -> {new_state.name}")
        old_state = self.state_
        self.state_ = new_state
        
        if old_state == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM and new_state != NavState.FOLLOWING_LOCAL_TARGET:
            self._destroy_corrected_goal_subscriber()
        self._process_state_actions()

    def _process_state_actions(self):
        if self.state_ == NavState.LOADING_WAYPOINTS:
            if not self.yaml_path_ or not Path(self.yaml_path_).is_file():
                self._change_state_and_process(NavState.MISSION_FAILED); return
            if self.load_waypoints_from_yaml(Path(self.yaml_path_)):
                if self.all_waypoints_:
                    self.last_truly_completed_global_idx_ = -1
                    self._send_follow_waypoints_goal_from_index(0) # Start from the beginning
                else: self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
            else: self._change_state_and_process(NavState.MISSION_FAILED)
        elif self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM:
            self._create_corrected_goal_subscriber()
        elif self.state_ == NavState.WAYPOINT_SEQUENCE_COMPLETE:
            self._activate_segmentation_node(False); rclpy.shutdown()
        elif self.state_ == NavState.MISSION_FAILED:
            self._activate_segmentation_node(False)
            self._cancel_all_navigation_actions(); rclpy.shutdown()

    def load_waypoints_from_yaml(self, yaml_file_path: Path) -> bool:
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
                    ps_msg.header.stamp = self.get_clock().now().to_msg() # Use current time for Nav2
                    
                    pose_block = pose_entry.get('pose', {})
                    pos_data = pose_block.get('position', {})
                    orient_data = pose_block.get('orientation', {})
                    ps_msg.pose.position.x = float(pos_data.get('x',0.0))
                    ps_msg.pose.position.y = float(pos_data.get('y',0.0))
                    ps_msg.pose.position.z = float(pos_data.get('z',0.0)) # Keep Z if specified
                    ps_msg.pose.orientation.x=float(orient_data.get('x',0.0))
                    ps_msg.pose.orientation.y=float(orient_data.get('y',0.0))
                    ps_msg.pose.orientation.z=float(orient_data.get('z',0.0))
                    ps_msg.pose.orientation.w=float(orient_data.get('w',1.0)) # Default to valid quaternion
                    loaded_waypoints.append(ps_msg)
                except Exception as e: self.get_logger().error(f"Error parsing waypoint {i+1}: {e}"); return False
            self.all_waypoints_ = loaded_waypoints
            self.get_logger().info(f"Loaded {len(self.all_waypoints_)} waypoints.")
            return True
        except Exception as e: self.get_logger().error(f"Error loading YAML: {e}"); return False

    def _create_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is None:
            self.get_logger().info(f"Creating subscriber for corrected goals on '{self.corrected_goal_topic_}'")
            _qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, 
                              depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.corrected_goal_sub_ = self.create_subscription(
                GeometryPoseStamped, self.corrected_goal_topic_, self.corrected_goal_callback, _qos)
    
    def _destroy_corrected_goal_subscriber(self):
        if self.corrected_goal_sub_ is not None:
            self.destroy_subscription(self.corrected_goal_sub_); self.corrected_goal_sub_ = None
            self.latest_corrected_goal_ = None

    def get_robot_pose(self) -> GeometryPoseStamped | None:
        try:
            transform = self.tf_buffer_.lookup_transform(
                self.global_frame_, self.robot_base_frame_, 
                rclpy.time.Time(), # Request latest available transform
                timeout=RclpyDuration(seconds=0.2) 
            )
            pose = GeometryPoseStamped()
            pose.header.stamp = transform.header.stamp
            pose.header.frame_id = self.global_frame_
            pose.pose.position = transform.transform.translation
            pose.pose.orientation = transform.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().warn(f"TF Robot Pose Error: {e}", throttle_duration_sec=2)
            return None

    def _activate_segmentation_node(self, activate: bool):
        if not self.segmentation_activation_client_.service_is_ready():
            self.get_logger().warn(f"Service '{self.activation_srv_name_}' not ready."); return False
        req = SetBool.Request(); req.data = activate
        self.segmentation_activation_client_.call_async(req) # Fire and forget for now
        self.get_logger().info(f"Requested segmentation {'activation' if activate else 'deactivation'}.")
        return True

    def _cancel_all_navigation_actions(self):
        if self._current_follow_wp_goal_handle:
            self._current_follow_wp_goal_handle.cancel_goal_async()
            self._current_follow_wp_goal_handle = None 
        if self._current_nav_to_pose_goal_handle:
            self._current_nav_to_pose_goal_handle.cancel_goal_async()
            self._current_nav_to_pose_goal_handle = None
        self.get_logger().debug("Requested cancellation of all active navigation goals.")

    def _send_follow_waypoints_goal_from_index(self, start_index: int):
        self._cancel_all_navigation_actions()
        if start_index >= len(self.all_waypoints_):
            self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE); return
        
        waypoints_to_send = self.all_waypoints_[start_index:]
        if not waypoints_to_send:
            self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE); return

        if not self._follow_waypoints_client.wait_for_server(timeout_sec=5.0):
            self._change_state_and_process(NavState.MISSION_FAILED); return

        goal_msg = FollowWaypoints.Goal(); goal_msg.poses = waypoints_to_send
        self.get_logger().info(f"Sending {len(waypoints_to_send)} waypoints to FollowWaypoints (from original_idx {start_index}).")
        send_goal_future = self._follow_waypoints_client.send_goal_async(
            goal_msg, feedback_callback=self.follow_waypoints_feedback_cb)
        send_goal_future.add_done_callback(self.follow_waypoints_goal_response_cb)
        self._change_state_and_process(NavState.FOLLOWING_GLOBAL_WAYPOINTS)

    def follow_waypoints_goal_response_cb(self, future):
        if self.state_ == NavState.MISSION_FAILED: return
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("FollowWaypoints goal REJECTED."); self._current_follow_wp_goal_handle = None
            if self.state_ != NavState.MISSION_FAILED: self._change_state_and_process(NavState.MISSION_FAILED)
            return
        self._current_follow_wp_goal_handle = goal_handle
        self.get_logger().info("FollowWaypoints goal ACCEPTED.")
        goal_handle.get_result_async().add_done_callback(self.follow_waypoints_result_cb)

    def follow_waypoints_feedback_cb(self, feedback_msg: FollowWaypoints.Feedback):
        if self.state_ != NavState.FOLLOWING_GLOBAL_WAYPOINTS: return

        # feedback.current_waypoint is 0-indexed for the list Nav2 currently has
        # actual_approaching_global_idx is the index in self.all_waypoints_
        actual_approaching_global_idx = self.last_truly_completed_global_idx_ + 1 + feedback_msg.feedback.current_waypoint
        self.get_logger().debug(f"FW Feedback: Nav2 on its_idx {feedback_msg.feedback.current_waypoint} (Overall_idx {actual_approaching_global_idx})")

        if actual_approaching_global_idx < len(self.all_waypoints_):
            robot_pose = self.get_robot_pose()
            if robot_pose:
                target_wp = self.all_waypoints_[actual_approaching_global_idx]
                dist_sq = (robot_pose.pose.position.x - target_wp.pose.position.x)**2 + \
                          (robot_pose.pose.position.y - target_wp.pose.position.y)**2
                if dist_sq < self.correction_activation_dist_**2:
                    self.get_logger().info(f"Near global_idx {actual_approaching_global_idx}. Activating local correction.")
                    if self._activate_segmentation_node(True):
                        self.correction_active_for_wp_idx_ = actual_approaching_global_idx
                        self._change_state_and_process(NavState.AWAITING_LOCAL_CORRECTION_CONFIRM)

    def follow_waypoints_result_cb(self, future): # Modified
        if self.state_ == NavState.MISSION_FAILED: return

        action_result_wrapper = future.result() # Get the top-level result wrapper
        status = action_result_wrapper.status
        actual_result_message = action_result_wrapper.result # This is FollowWaypoints.Result

        self._current_follow_wp_goal_handle = None # Clear handle

        if self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM or \
           self.state_ == NavState.FOLLOWING_LOCAL_TARGET:
            self.get_logger().info(f"FollowWaypoints action finished with status {status} (likely cancelled by us for local correction).")
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("FollowWaypoints action SUCCEEDED (all given waypoints reached).")
            # If FollowWaypoints succeeds, it means all waypoints in the *current batch sent to it* were completed.
            # The logic to update last_truly_completed_global_idx_ should reflect this.
            # Since we send remaining waypoints, success means all remaining are done.
            self.last_truly_completed_global_idx_ = len(self.all_waypoints_) - 1 
            self._change_state_and_process(NavState.WAYPOINT_SEQUENCE_COMPLETE)
        elif status == GoalStatus.STATUS_CANCELED:
             self.get_logger().warn(f"FollowWaypoints CANCELED unexpectedly. Status: {status}")
             self._change_state_and_process(NavState.MISSION_FAILED)
        else: # ABORTED etc.
            missed_wps_info = "N/A"
            if actual_result_message: # Check if actual_result_message is not None
                missed_wps_info = str(actual_result_message.missed_waypoints)

            self.get_logger().error(f"FollowWaypoints action FAILED/ABORTED. Status: {status}. Missed: {missed_wps_info}")
            self._change_state_and_process(NavState.MISSION_FAILED)

    def corrected_goal_callback(self, msg: GeometryPoseStamped):
        # ... (existing initial checks for state and empty frame_id) ...

        new_local_goal = msg
        self.get_logger().info(f"Received corrected local goal: P({new_local_goal.pose.position.x:.2f}, {new_local_goal.pose.position.y:.2f})")

        if self.state_ == NavState.AWAITING_LOCAL_CORRECTION_CONFIRM:
            # ... (existing logic to cancel FollowWaypoints and send first NavigateToPose) ...
            self.latest_corrected_goal_ = new_local_goal
            self.last_sent_local_goal_pose_ = new_local_goal # Store what was sent
            self._send_navigate_to_pose_goal(self.latest_corrected_goal_)

        elif self.state_ == NavState.FOLLOWING_LOCAL_TARGET:
            # Check if the new goal is significantly different from the last one sent
            significant_change = True # Assume change unless proven otherwise
            if self.last_sent_local_goal_pose_:
                dist_sq_diff = (new_local_goal.pose.position.x - self.last_sent_local_goal_pose_.pose.position.x)**2 + \
                            (new_local_goal.pose.position.y - self.last_sent_local_goal_pose_.pose.position.y)**2
                if math.sqrt(dist_sq_diff) < self.local_goal_update_threshold_: # Use a threshold
                    significant_change = False
                    self.get_logger().debug("New local goal is too close to the previously sent one. Not updating NavToPose.")
            
            if significant_change:
                self.get_logger().info("Local goal UPDATED significantly. Sending new NavigateToPose goal.")
                if self._current_nav_to_pose_goal_handle:
                    self._current_nav_to_pose_goal_handle.cancel_goal_async()
                    # Important: Wait for cancellation or have a flag before sending new one to avoid race
                    # For now, simple preemption:
                    self._current_nav_to_pose_goal_handle = None
                self.latest_corrected_goal_ = new_local_goal
                self.last_sent_local_goal_pose_ = new_local_goal # Store what was sent
                self._send_navigate_to_pose_goal(self.latest_corrected_goal_)
            else:
                self.latest_corrected_goal_ = new_local_goal # Still update our knowledge of the latest target

    def _send_navigate_to_pose_goal(self, target_pose: GeometryPoseStamped):
        if not self._navigate_to_pose_client.wait_for_server(timeout_sec=3.0):
            self._resume_global_waypoints_after_failed_correction(); return
        goal_msg = NavigateToPose.Goal(); goal_msg.pose = target_pose
        self.get_logger().info(f"Sending NavigateToPose goal: P({target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f})")
        send_future = self._navigate_to_pose_client.send_goal_async(goal_msg, self.navigate_to_pose_feedback_cb)
        send_future.add_done_callback(self.navigate_to_pose_goal_response_cb)

    def navigate_to_pose_goal_response_cb(self, future):
        if self.state_ != NavState.FOLLOWING_LOCAL_TARGET and self.state_ != NavState.MISSION_FAILED: return
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("NavigateToPose goal REJECTED."); self._current_nav_to_pose_goal_handle = None
            if self.state_ != NavState.MISSION_FAILED: self._resume_global_waypoints_after_failed_correction()
            return
        self._current_nav_to_pose_goal_handle = goal_handle
        self.get_logger().info("NavigateToPose goal ACCEPTED.")
        goal_handle.get_result_async().add_done_callback(self.navigate_to_pose_result_cb)

    def navigate_to_pose_feedback_cb(self, feedback_msg: NavigateToPose.Feedback):
        if self.state_ != NavState.FOLLOWING_LOCAL_TARGET: return
        robot_pose = self.get_robot_pose()
        if robot_pose and self.latest_corrected_goal_:
            dist_sq = (robot_pose.pose.position.x - self.latest_corrected_goal_.pose.position.x)**2 + \
                      (robot_pose.pose.position.y - self.latest_corrected_goal_.pose.position.y)**2
            if dist_sq < self.local_arrival_thresh_**2:
                self.get_logger().info(f"Local target arrival by proximity (dist {math.sqrt(dist_sq):.2f}m).")
                if self._current_nav_to_pose_goal_handle: self._current_nav_to_pose_goal_handle.cancel_goal_async()
                else: self._handle_local_target_completion() # If handle already gone but we are close

    def navigate_to_pose_result_cb(self, future):
        if self.state_ != NavState.FOLLOWING_LOCAL_TARGET and self.state_ != NavState.MISSION_FAILED :
             self.get_logger().warn(f"NavigateToPose result in unexpected state {self.state_.name}. Ignoring result unless failed.")
             if self.state_ == NavState.MISSION_FAILED: return # Already handled

        action_result_wrapper = future.result() # Get the top-level result wrapper
        status = action_result_wrapper.status
        # actual_result_message = action_result_wrapper.result # This would be NavigateToPose.Result (contains std_msgs.Empty)
                                                            # Not strictly needed here if you don't use its content.

        self._current_nav_to_pose_goal_handle = None 

        is_prox_arrival = self.latest_corrected_goal_ and self._check_arrival_at_local_target(self.latest_corrected_goal_)

        if status == GoalStatus.STATUS_SUCCEEDED or \
           (status == GoalStatus.STATUS_CANCELED and is_prox_arrival):
            self.get_logger().info("Local target reached successfully (or by proximity cancellation).")
            self._handle_local_target_completion()
        elif status == GoalStatus.STATUS_CANCELED: # Cancelled not by proximity
            self.get_logger().warn("NavigateToPose for local target CANCELED (not by proximity).")
            self._resume_global_waypoints_after_failed_correction()
        else: # ABORTED, etc.
            self.get_logger().error(f"NavigateToPose for local target FAILED/ABORTED. Status: {status}.")
            self._resume_global_waypoints_after_failed_correction()
            
    def _handle_local_target_completion(self):
        self.get_logger().info("Local target processing completed successfully.")
        # The waypoint for which correction was active is now truly completed.
        self.last_truly_completed_global_idx_ = self.correction_active_for_wp_idx_
        
        self.latest_corrected_goal_ = None
        self._activate_segmentation_node(False)
        self._destroy_corrected_goal_subscriber()
        self.correction_active_for_wp_idx_ = -1 # Reset

        next_global_idx = self.last_truly_completed_global_idx_ + 1
        self._send_follow_waypoints_goal_from_index(next_global_idx)

    def _check_arrival_at_local_target(self, target_pose: GeometryPoseStamped):
        robot_pose = self.get_robot_pose()
        if not robot_pose or not target_pose: return False
        dist_sq = (robot_pose.pose.position.x - target_pose.pose.position.x)**2 + \
                  (robot_pose.pose.position.y - target_pose.pose.position.y)**2
        return dist_sq < self.local_arrival_thresh_**2

    def _resume_global_waypoints_after_failed_correction(self):
        self.get_logger().info("Attempting to recover after local navigation issue.")
        self._activate_segmentation_node(False)
        self._destroy_corrected_goal_subscriber()
        self.latest_corrected_goal_ = None

        # If a correction was attempted for a specific waypoint,
        # and that local navigation failed, we should definitively move past that global waypoint.
        # The `self.correction_active_for_wp_idx_` holds the index of the global waypoint
        # for which the correction was active.
        
        # Mark the waypoint for which correction was attempted as "processed" in some way,
        # even if not perfectly reached.
        if self.correction_active_for_wp_idx_ > self.last_truly_completed_global_idx_:
            self.last_truly_completed_global_idx_ = self.correction_active_for_wp_idx_
            self.get_logger().info(f"Marking global_idx {self.correction_active_for_wp_idx_} as processed due to failed local correction.")
        
        next_wp_idx_to_process = self.last_truly_completed_global_idx_ + 1
        
        self.get_logger().info(f"Resuming global waypoints from original index {next_wp_idx_to_process}.")
        self.correction_active_for_wp_idx_ = -1 # Reset for next attempt
        self._send_follow_waypoints_goal_from_index(next_wp_idx_to_process)

    def destroy_node(self): # Override to ensure cleanup
        self.get_logger().info("Destroying WaypointFollowerCorrected node.")
        self._activate_segmentation_node(False) # Try to turn off segmentation
        self._cancel_all_navigation_actions()
        # Add a small delay or spin_once to allow cancellations to be processed
        # This is hard to do perfectly without blocking shutdown too long.
        # Consider using a dedicated shutdown sequence if problems persist.
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointFollowerCorrected()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node: node.get_logger().info('Node interrupted by user.')
    except SystemExit:
        if node: node.get_logger().info('Node is shutting down via SystemExit.')
    except Exception as e:
        if node: node.get_logger().error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
        else: print(f"Unhandled exception during node init: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.destroy_node() # This will call the overridden destroy_node
        if rclpy.ok():
            rclpy.shutdown()
        print("WaypointFollowerCorrected shutdown complete.")

if __name__ == '__main__':
    main()